#include "core/keyvalue/keyvalue_engine.h"

#include <filesystem>
#include <fstream>

namespace pomai::core {

KeyValueEngine::KeyValueEngine(std::string path, std::size_t max_entries)
    : path_(std::move(path)), max_entries_(max_entries) {}

Status KeyValueEngine::Open() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path_, ec);
    if (ec) return Status::IOError("kv create dir failed");
    std::ifstream in(path_ + "/kv.log", std::ios::binary);
    if (!in.good()) return Status::Ok();
    while (true) {
        uint8_t op = 0;
        uint32_t ksz = 0, vsz = 0;
        if (!in.read(reinterpret_cast<char*>(&op), sizeof(op))) break;
        if (!in.read(reinterpret_cast<char*>(&ksz), sizeof(ksz))) break;
        if (!in.read(reinterpret_cast<char*>(&vsz), sizeof(vsz))) break;
        std::string k(ksz, '\0');
        if (!in.read(k.data(), static_cast<std::streamsize>(ksz))) break;
        std::string v(vsz, '\0');
        if (vsz > 0 && !in.read(v.data(), static_cast<std::streamsize>(vsz))) break;
        if (op == 1) kv_[std::move(k)] = std::move(v);
        else if (op == 2) kv_.erase(k);
        if (max_entries_ > 0 && kv_.size() > max_entries_) kv_.erase(kv_.begin());
    }
    return Status::Ok();
}

Status KeyValueEngine::Close() { return Status::Ok(); }

Status KeyValueEngine::Put(std::string_view key, std::string_view value) {
    if (key.empty()) return Status::InvalidArgument("kv key empty");
    if (max_entries_ > 0 && kv_.size() >= max_entries_ && kv_.find(std::string(key)) == kv_.end()) {
        kv_.erase(kv_.begin());
    }
    kv_[std::string(key)] = std::string(value);
    std::ofstream out(path_ + "/kv.log", std::ios::binary | std::ios::app);
    if (!out.good()) return Status::IOError("kv append failed");
    uint8_t op = 1;
    uint32_t ksz = static_cast<uint32_t>(key.size());
    uint32_t vsz = static_cast<uint32_t>(value.size());
    out.write(reinterpret_cast<const char*>(&op), sizeof(op));
    out.write(reinterpret_cast<const char*>(&ksz), sizeof(ksz));
    out.write(reinterpret_cast<const char*>(&vsz), sizeof(vsz));
    out.write(key.data(), static_cast<std::streamsize>(key.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
    return Status::Ok();
}

Status KeyValueEngine::Get(std::string_view key, std::string* out) const {
    if (!out) return Status::InvalidArgument("kv out null");
    auto it = kv_.find(std::string(key));
    if (it == kv_.end()) return Status::NotFound("kv not found");
    *out = it->second;
    return Status::Ok();
}

Status KeyValueEngine::Delete(std::string_view key) {
    kv_.erase(std::string(key));
    std::ofstream out(path_ + "/kv.log", std::ios::binary | std::ios::app);
    if (!out.good()) return Status::IOError("kv append failed");
    uint8_t op = 2;
    uint32_t ksz = static_cast<uint32_t>(key.size());
    uint32_t vsz = 0;
    out.write(reinterpret_cast<const char*>(&op), sizeof(op));
    out.write(reinterpret_cast<const char*>(&ksz), sizeof(ksz));
    out.write(reinterpret_cast<const char*>(&vsz), sizeof(vsz));
    out.write(key.data(), static_cast<std::streamsize>(key.size()));
    return Status::Ok();
}

} // namespace pomai::core

