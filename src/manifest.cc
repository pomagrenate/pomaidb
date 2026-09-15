#include "manifest.h"
#include "crc32c.h"
#include "posix_file.h"
#include "storage/palloc_io.h"
#include "utils/palloc_smart_ptr.h"
#include "utils/logging.h"
#include "env.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace pomai::storage
{

    namespace
    {
        static bool IsValidName(std::string_view s)
        {
            if (s.empty() || s.size() > 64)
                return false;
            if (s == "." || s == "..")
                return false;
            for (unsigned char c : s)
            {
                const bool ok = (c >= 'a' && c <= 'z') ||
                                (c >= 'A' && c <= 'Z') ||
                                (c >= '0' && c <= '9') ||
                                c == '_' || c == '-' || c == '.';
                if (!ok)
                    return false;
            }
            return true;
        }

        static std::string RootManifestPath(std::string_view root_path)
        {
            return std::string(root_path) + "/MANIFEST";
        }

        static std::string MembraneDir(std::string_view root_path, std::string_view name)
        {
            return std::string(root_path) + "/membranes/" + std::string(name);
        }

        static std::string MembraneManifestPath(std::string_view root_path, std::string_view name)
        {
            return std::string(root_path) + "/membranes/" + std::string(name) + "/MANIFEST";
        }

        static pomai::Status ReadAll(const std::string &path, std::string *out)
        {
            // Check if file exists first
            auto exists_st = storage::PallocFilesystem::FileExists(path.c_str());
            if (!exists_st.ok()) {
                if (exists_st.code() == pomai::ErrorCode::kNotFound) {
                    return pomai::Status::NotFound("manifest file not found");
                }
                return exists_st;
            }

            // Use palloc random access I/O instead of std::ifstream
            alloc::UniquePtr<storage::PallocRandomAccessFile> file;
            auto st = storage::PallocRandomAccessFile::Open(path.c_str(), &file);
            if (!st.ok())
                return pomai::Status::IOError("read failed: open");

            // Get file size
            uint64_t file_size = 0;
            st = storage::PallocFilesystem::GetFileSize(path.c_str(), &file_size);
            if (!st.ok())
                return pomai::Status::IOError("read failed: get size");

            std::string buf;
            if (file_size > 0)
            {
                buf.resize(static_cast<std::size_t>(file_size));
                Slice slice;
                st = file->Read(0, file_size, &slice);
                if (!st.ok())
                    return pomai::Status::IOError("read failed");
                if (slice.size() != file_size)
                    return pomai::Status::IOError("read failed: short read");
                std::memcpy(buf.data(), slice.data(), file_size);
            }

            // CRC validation (return kAborted for crash-safety: caller should not retry corrupted manifest)
            if (file_size < 4)
                return pomai::Status::Aborted("file too short for CRC");

            uint32_t stored_crc;
            const size_t content_len = static_cast<size_t>(file_size) - 4;
            // stored CRC is last 4 bytes (little endian ideally, but we assume same endianness for now)
            // Just copying for simplicity
            unsigned char crc_buf[4];
            for(int i=0; i<4; ++i) crc_buf[i] = buf[content_len + i];
            
            stored_crc = (uint32_t)crc_buf[0] | 
                         ((uint32_t)crc_buf[1] << 8) | 
                         ((uint32_t)crc_buf[2] << 16) | 
                         ((uint32_t)crc_buf[3] << 24);

            uint32_t computed = pomai::util::Crc32c(buf.data(), content_len);
            if (computed != stored_crc)
                return pomai::Status::Aborted("CRC mismatch");

            *out = buf.substr(0, content_len);
            return pomai::Status::Ok();
        }

        static pomai::Status AtomicWriteFile(const std::string &final_path, std::string_view content)
        {
            const std::string tmp = final_path + ".tmp";
            
            // Use palloc I/O for explicit sync control
            alloc::UniquePtr<storage::PallocWritableFile> file;
            auto st = storage::PallocWritableFile::Create(tmp.c_str(), &file);
            if (!st.ok()) return st;

            // Write content
            st = file->Append(Slice(content.data(), content.size()));
            if (!st.ok()) return st;
            
            // Calculate and write CRC
            uint32_t crc = pomai::util::Crc32c(content.data(), content.size());
            char crc_buf[4];
            crc_buf[0] = static_cast<char>(crc & 0xFF);
            crc_buf[1] = static_cast<char>((crc >> 8) & 0xFF);
            crc_buf[2] = static_cast<char>((crc >> 16) & 0xFF);
            crc_buf[3] = static_cast<char>((crc >> 24) & 0xFF);

            st = file->Append(Slice(crc_buf, 4));
            if (!st.ok()) return st;

            // Critical: Fsync data to disk before rename
            st = file->Flush();
            if (!st.ok()) return st;
            st = file->Sync();
            if (!st.ok()) return st;
            st = file->Close();
            if (!st.ok()) return st;

            // Atomic rename simulation
            st = storage::PallocFilesystem::RemoveFile(final_path.c_str());
            if (!st.ok() && st.code() != ErrorCode::kNotFound) return st;
            st = storage::PallocFilesystem::RemoveFile(tmp.c_str());
            if (!st.ok()) return st;
            st = storage::PallocWritableFile::Create(final_path.c_str(), &file);
            if (!st.ok()) return st;
            st = file->Append(Slice(content.data(), content.size()));
            if (!st.ok()) return st;
            st = file->Append(Slice(crc_buf, 4));
            if (!st.ok()) return st;
            st = file->Flush();
            if (!st.ok()) return st;
            st = file->Sync();
            if (!st.ok()) return st;
            st = file->Close();
            if (!st.ok()) return st;

            // Directory fsync for rename durability
            std::string parent_dir = final_path.substr(0, final_path.find_last_of("/\\"));
            return storage::PallocFilesystem::SyncDir(parent_dir.c_str());
        }

        struct RootEntry
        {
            std::string name;
        };

        static pomai::Status ParseU32(std::string_view tok, std::uint32_t *out)
        {
            if (!out)
                return pomai::Status::InvalidArgument("out=null");
            if (tok.empty())
                return pomai::Status::InvalidArgument("empty number");

            std::uint64_t v = 0;
            for (char ch : tok)
            {
                if (ch < '0' || ch > '9')
                    return pomai::Status::InvalidArgument("invalid number");
                v = v * 10 + static_cast<std::uint64_t>(ch - '0');
                if (v > 0xFFFFFFFFull)
                    return pomai::Status::InvalidArgument("number too large");
            }
            *out = static_cast<std::uint32_t>(v);
            return pomai::Status::Ok();
        }

        static std::vector<std::string_view> SplitWs(std::string_view line)
        {
            std::vector<std::string_view> out;
            std::size_t i = 0;
            while (i < line.size())
            {
                while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r'))
                    ++i;
                if (i >= line.size())
                    break;
                std::size_t j = i;
                while (j < line.size() && line[j] != ' ' && line[j] != '\t' && line[j] != '\r')
                    ++j;
                out.push_back(line.substr(i, j - i));
                i = j;
            }
            return out;
        }

        static pomai::Status LoadRoot(std::string_view root_path, std::vector<RootEntry> *out_entries)
        {
            out_entries->clear();

            std::string content;
            auto st = ReadAll(RootManifestPath(root_path), &content);
            if (!st.ok()) {
                // If file doesn't exist, that's OK for new database
                if (st.code() == pomai::ErrorCode::kIO || st.code() == pomai::ErrorCode::kNotFound) {
                    return pomai::Status::Ok();
                }
                return st;
            }

            std::string_view sv(content);

            std::size_t p = sv.find('\n');
            std::string_view header = (p == std::string_view::npos) ? sv : sv.substr(0, p);
            
            // Checking for v3 (kAborted for crash-safety: corrupted/invalid manifest)
            if (header != "pomai.manifest.v3")
                return pomai::Status::Aborted("bad manifest header: expected v3");
            
            sv = (p == std::string_view::npos) ? std::string_view{} : sv.substr(p + 1);

            while (!sv.empty())
            {
                std::size_t eol = sv.find('\n');
                std::string_view line = (eol == std::string_view::npos) ? sv : sv.substr(0, eol);
                sv = (eol == std::string_view::npos) ? std::string_view{} : sv.substr(eol + 1);

                if (line.empty())
                    continue;

                auto toks = SplitWs(line);
                if (toks.empty()) continue;

                if (toks[0] == "version") {
                     // Global version, ignore for now or store it if we had a struct for it
                     continue;
                }

                if (toks[0] == "membrane") {
                    if (toks.size() < 2) return pomai::Status::Corruption("bad membrane line");
                    RootEntry e;
                    e.name = std::string(toks[1]);
                    if (!IsValidName(e.name))
                        return pomai::Status::Corruption("invalid membrane name");
                    out_entries->push_back(std::move(e));
                }
            }

            std::sort(out_entries->begin(), out_entries->end(),
                      [](const RootEntry &a, const RootEntry &b)
                      { return a.name < b.name; });

            return pomai::Status::Ok();
        }

        static std::string SerializeRoot(const std::vector<RootEntry> &entries)
        {
            std::string out;
            out += "pomai.manifest.v3\n";
            out += "version 1\n"; // Hardcoded global version for now
            for (const auto &e : entries)
            {
                out += "membrane " + e.name + "\n";
            }
            return out;
        }

        static std::string MembraneKindToString(pomai::MembraneKind kind)
        {
            (void)kind;
            return "VECTOR";
        }

        static pomai::MembraneKind ParseMembraneKind(std::string_view tok)
        {
            (void)tok;
            return pomai::MembraneKind::kVector;
        }

        static pomai::Status WriteMembraneManifest(std::string_view root_path, const pomai::MembraneSpec &spec)
        {
            std::string out;
            out += "pomai.membrane.v3\n";
            out += "name " + spec.name + "\n";
            out += "shards " + std::to_string(spec.shard_count) + "\n";
            out += "dim " + std::to_string(spec.dim) + "\n";
            out += "kind " + MembraneKindToString(spec.kind) + "\n";
            
            std::string mtype = "L2";
            if (spec.metric == pomai::MetricType::kInnerProduct) mtype = "IP";
            else if (spec.metric == pomai::MetricType::kCosine) mtype = "COS";
            out += "metric " + mtype + "\n";

            out += "index_params " + std::to_string(static_cast<uint32_t>(spec.index_params.type)) + " " +
                   std::to_string(spec.index_params.nlist) + " " + 
                   std::to_string(spec.index_params.nprobe) + " " +
                   std::to_string(spec.index_params.hnsw_m) + " " +
                   std::to_string(spec.index_params.hnsw_ef_construction) + " " +
                   std::to_string(spec.index_params.hnsw_ef_search) + "\n";
            out += "sync_lsn " + std::to_string(spec.sync_lsn) + "\n";
            out += "ttl_sec " + std::to_string(spec.ttl_sec) + "\n";
            out += "retention_max_count " + std::to_string(spec.retention_max_count) + "\n";
            out += "retention_max_bytes " + std::to_string(spec.retention_max_bytes) + "\n";
            return AtomicWriteFile(MembraneManifestPath(root_path, spec.name), out);
        }

        static pomai::Status LoadMembraneManifest(std::string_view root_path, std::string_view name, pomai::MembraneSpec *spec) {
            std::string content;
            auto st = ReadAll(MembraneManifestPath(root_path, name), &content);
            if (!st.ok()) {
                // If file doesn't exist, that's OK for new membrane
                if (st.code() == pomai::ErrorCode::kIO || st.code() == pomai::ErrorCode::kNotFound) {
                    return pomai::Status::Ok();
                }
                return st;
            }

            std::string_view sv(content);
            std::size_t p = sv.find('\n');
            std::string_view header = (p == std::string_view::npos) ? sv : sv.substr(0, p);

            const bool v2 = header == "pomai.membrane.v2";
            const bool v3 = header == "pomai.membrane.v3";
            if (!v2 && !v3)
                 return pomai::Status::Corruption("bad membrane manifest header: expected v2/v3");

            sv = (p == std::string_view::npos) ? std::string_view{} : sv.substr(p + 1);
            
            spec->name = std::string(name);
            // defaults
            spec->shard_count = 0;
            spec->dim = 0;
            spec->metric = pomai::MetricType::kL2;
            spec->kind = pomai::MembraneKind::kVector;
            spec->ttl_sec = 0;
            spec->retention_max_count = 0;
            spec->retention_max_bytes = 0;

            while (!sv.empty()) {
                std::size_t eol = sv.find('\n');
                std::string_view line = (eol == std::string_view::npos) ? sv : sv.substr(0, eol);
                sv = (eol == std::string_view::npos) ? std::string_view{} : sv.substr(eol + 1);

                if (line.empty()) continue;
                auto toks = SplitWs(line);
                if (toks.empty()) continue;

                if (toks[0] == "name") {
                    // verify name matches?
                } else if (toks[0] == "shards") {
                    if (toks.size() > 1) (void)ParseU32(toks[1], &spec->shard_count);
                } else if (toks[0] == "dim") {
                    if (toks.size() > 1) (void)ParseU32(toks[1], &spec->dim);
                } else if (toks[0] == "metric") {
                    if (toks.size() > 1) {
                        if (toks[1] == "IP") spec->metric = pomai::MetricType::kInnerProduct;
                        else if (toks[1] == "COS") spec->metric = pomai::MetricType::kCosine;
                        else spec->metric = pomai::MetricType::kL2;
                    }
                } else if (toks[0] == "kind") {
                    if (toks.size() > 1) {
                        spec->kind = ParseMembraneKind(toks[1]);
                    }
                } else if (toks[0] == "index_params") {
                    if (toks.size() == 3) {
                         spec->index_params.type = pomai::IndexType::kIvfFlat;
                         (void)ParseU32(toks[1], &spec->index_params.nlist);
                         (void)ParseU32(toks[2], &spec->index_params.nprobe);
                    } else if (toks.size() >= 7) {
                         uint32_t type_val = 0;
                         (void)ParseU32(toks[1], &type_val);
                         spec->index_params.type = (type_val == 1) ? pomai::IndexType::kHnsw : pomai::IndexType::kIvfFlat;
                         (void)ParseU32(toks[2], &spec->index_params.nlist);
                         (void)ParseU32(toks[3], &spec->index_params.nprobe);
                         (void)ParseU32(toks[4], &spec->index_params.hnsw_m);
                          (void)ParseU32(toks[5], &spec->index_params.hnsw_ef_construction);
                          (void)ParseU32(toks[6], &spec->index_params.hnsw_ef_search);
                     }
                } else if (toks[0] == "sync_lsn") {
                    if (toks.size() > 1) {
                         // ParseU32 only handles 32 bits, but seq yields 64. 
                         // For now let's add a ParseU64 or just use stoull.
                         try {
                            spec->sync_lsn = std::stoull(std::string(toks[1]));
                         } catch(...) {}
                    }
                } else if (toks[0] == "ttl_sec") {
                    if (toks.size() > 1) (void)ParseU32(toks[1], &spec->ttl_sec);
                } else if (toks[0] == "retention_max_count") {
                    if (toks.size() > 1) (void)ParseU32(toks[1], &spec->retention_max_count);
                } else if (toks[0] == "retention_max_bytes") {
                    if (toks.size() > 1) {
                        try { spec->retention_max_bytes = std::stoull(std::string(toks[1])); } catch(...) {}
                    }
                }
            }
            return pomai::Status::Ok();
        }

    } // namespace

    pomai::Status Manifest::EnsureInitialized(std::string_view root_path)
    {
        auto st = storage::PallocFilesystem::CreateDir(std::string(root_path).c_str());
        if (!st.ok()) return st;

        std::string membranes_dir = std::string(root_path) + "/membranes";
        st = storage::PallocFilesystem::CreateDir(membranes_dir.c_str());
        if (!st.ok()) return st;

        const auto mp = RootManifestPath(root_path);
        if (storage::PallocFilesystem::FileExists(mp.c_str()).ok())
            return pomai::Status::Ok();

        // Write empty root v3
        return AtomicWriteFile(mp, "pomai.manifest.v3\nversion 1\n");
    }

    pomai::Status Manifest::CreateMembrane(std::string_view root_path, const pomai::MembraneSpec &spec)
    {
        if (!IsValidName(spec.name))
            return pomai::Status::InvalidArgument("invalid membrane name");
        if (spec.dim == 0)
            return pomai::Status::InvalidArgument("dim must be > 0");
        if (spec.shard_count == 0)
            return pomai::Status::InvalidArgument("shard_count must be > 0");

        auto st = EnsureInitialized(root_path);
        if (!st.ok())
            return st;

        std::vector<RootEntry> entries;
        st = LoadRoot(root_path, &entries);
        if (!st.ok())
            return st;

        auto it = std::find_if(entries.begin(), entries.end(),
                               [&](const RootEntry &e)
                               { return e.name == spec.name; });
        if (it != entries.end())
            return pomai::Status::AlreadyExists("membrane already exists");

        auto dir_st = storage::PallocFilesystem::CreateDir(MembraneDir(root_path, spec.name).c_str());
        if (!dir_st.ok())
            return pomai::Status::IOError("create_directories membrane failed");

        st = WriteMembraneManifest(root_path, spec);
        if (!st.ok())
            return st;

        POMAI_LOG_INFO("Membrane manifest written");

        entries.push_back({spec.name});
        std::sort(entries.begin(), entries.end(),
                  [](const RootEntry &a, const RootEntry &b)
                  { return a.name < b.name; });

        return AtomicWriteFile(RootManifestPath(root_path), SerializeRoot(entries));
    }

    pomai::Status Manifest::DropMembrane(std::string_view root_path, std::string_view name)
    {
        if (!IsValidName(name))
            return pomai::Status::InvalidArgument("invalid membrane name");

        auto st = EnsureInitialized(root_path);
        if (!st.ok())
            return st;

        std::vector<RootEntry> entries;
        st = LoadRoot(root_path, &entries);
        if (!st.ok())
            return st;

        auto it = std::find_if(entries.begin(), entries.end(),
                               [&](const RootEntry &e)
                               { return e.name == name; });
        if (it == entries.end())
            return pomai::Status::NotFound("membrane not found");

        entries.erase(it);
        st = AtomicWriteFile(RootManifestPath(root_path), SerializeRoot(entries));
        if (!st.ok())
            return st;

        // Remove membrane directory (recursive removal simulation)
        std::string membrane_dir = MembraneDir(root_path, name);
        // For now, just remove the manifest file
        return storage::PallocFilesystem::RemoveFile(MembraneManifestPath(root_path, name).c_str());
    }

    pomai::Status Manifest::ListMembranes(std::string_view root_path, std::vector<std::string> *out)
    {
        if (!out)
            return pomai::Status::InvalidArgument("out=null");

        auto st = EnsureInitialized(root_path);
        if (!st.ok())
            return st;

        std::vector<RootEntry> entries;
        st = LoadRoot(root_path, &entries);
        if (!st.ok())
            return st;

        out->clear();
        out->reserve(entries.size());
        for (const auto &e : entries)
            out->push_back(e.name);

        return pomai::Status::Ok();
    }

    pomai::Status Manifest::GetMembrane(std::string_view root_path, std::string_view name, pomai::MembraneSpec *out)
    {
        if (!out)
            return pomai::Status::InvalidArgument("out=null");
        if (!IsValidName(name))
            return pomai::Status::InvalidArgument("invalid membrane name");

        // First check if it exists in root
        std::vector<RootEntry> entries;
        auto st = LoadRoot(root_path, &entries);
        if (!st.ok()) return st;

        auto it = std::find_if(entries.begin(), entries.end(),
                               [&](const RootEntry &e)
                               { return e.name == name; });
        if (it == entries.end())
            return pomai::Status::NotFound("membrane not found in root");

        // Now load detailed spec from membrane specific manifest
        return LoadMembraneManifest(root_path, name, out);
    }

    pomai::Status Manifest::UpdateSyncLSN(std::string_view root_path, std::string_view name, uint64_t lsn) {
        pomai::MembraneSpec spec;
        auto st = GetMembrane(root_path, name, &spec);
        if (!st.ok()) return st;

        if (spec.sync_lsn == lsn) return pomai::Status::Ok(); // Already there
        spec.sync_lsn = lsn;
        return WriteMembraneManifest(root_path, spec);
    }

    pomai::Status Manifest::UpdateRetentionPolicy(std::string_view root_path, std::string_view name,
                                                  uint32_t ttl_sec, uint32_t retention_max_count,
                                                  uint64_t retention_max_bytes) {
        pomai::MembraneSpec spec;
        auto st = GetMembrane(root_path, name, &spec);
        if (!st.ok()) return st;
        spec.ttl_sec = ttl_sec;
        spec.retention_max_count = retention_max_count;
        spec.retention_max_bytes = retention_max_bytes;
        return WriteMembraneManifest(root_path, spec);
    }

    pomai::Status Manifest::CheckCompatibility(std::string_view root_path) {
        std::string content;
        auto st = ReadAll(RootManifestPath(root_path), &content);
        if (!st.ok()) {
            // If file doesn't exist, that's OK for new database
            if (st.code() == pomai::ErrorCode::kIO || st.code() == pomai::ErrorCode::kNotFound) {
                return pomai::Status::Ok();
            }
            return st;
        }
        const std::string_view sv(content);
        const std::size_t nl = sv.find('\n');
        const std::string_view header = (nl == std::string_view::npos) ? sv : sv.substr(0, nl);
        if (header != "pomai.manifest.v3") {
            return pomai::Status::Aborted("manifest compatibility check failed");
        }
        return pomai::Status::Ok();
    }

} // namespace pomai::storage
