// PomaiDB Embedded: Single-instance, non-thread-safe by design for maximum raw throughput.
// Shared-nothing monolithic pattern: one StorageEngine, one Arena, one append-only WAL.
// Data flow is strictly sequential (Input -> Database -> Single Arena -> Single Append-Only File)
// for optimal performance on MicroSD and low-end embedded hardware.

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "pomai/iterator.h"
#include "pomai/metadata.h"
#include "pomai/options.h"
#include "pomai/search.h"
#include "pomai/snapshot.h"
#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai {

// Forward declaration: single-instance storage (one Arena, one WAL, one index).
class StorageEngine;

// Embedded database options: no sharding, no routing, no thread count.
struct EmbeddedOptions {
    std::string path;
    std::uint32_t dim = 512;
    MetricType metric = MetricType::kL2;
    FsyncPolicy fsync = FsyncPolicy::kNever;
    IndexParams index_params;
};

/**
 * Database: thin wrapper around one StorageEngine and one vector index.
 * Single-threaded only; caller serializes access or runs on one thread.
 */
class Database {
public:
    Database();
    ~Database();

    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;

    /** Open at the given path with embedded options. */
    Status Open(const EmbeddedOptions& options);

    /** Close and release storage. */
    Status Close();

    /** Flush WAL to storage (sequential append). */
    Status Flush();

    /** Freeze: move active memtable to segment (single-instance). */
    Status Freeze();

    /** Append one vector; direct path to storage_engine_->append(). */
    Status AddVector(VectorId id, std::span<const float> vec);

    /** Append one vector with metadata. */
    Status AddVector(VectorId id, std::span<const float> vec,
                    const Metadata& meta);

    /** Batch append (sequential, single WAL segment). */
    Status AddVectorBatch(const std::vector<VectorId>& ids,
                         const std::vector<std::span<const float>>& vectors);

    /** Get vector by id. */
    Status Get(VectorId id, std::vector<float>* out);
    Status Get(VectorId id, std::vector<float>* out,
              Metadata* out_meta);

    /** Check existence. */
    Status Exists(VectorId id, bool* exists);

    /** Delete by id (tombstone). */
    Status Delete(VectorId id);

    /** Search: direct path to index_->search(). */
    Status Search(std::span<const float> query, std::uint32_t topk,
                  SearchResult* out);

    Status Search(std::span<const float> query, std::uint32_t topk,
                  const SearchOptions& opts,
                  SearchResult* out);

    /** Batch search (multiple queries). */
    Status SearchBatch(std::span<const float> queries, std::uint32_t num_queries,
                       std::uint32_t topk, const SearchOptions& opts,
                       std::vector<SearchResult>* out);

    /** Snapshot (point-in-time view). */
    Status GetSnapshot(std::shared_ptr<Snapshot>* out);
    /** Iterator over snapshot; snap must be from GetSnapshot(). */
    Status NewIterator(const std::shared_ptr<Snapshot>& snap,
                      std::unique_ptr<SnapshotIterator>* out);

    [[nodiscard]] bool IsOpen() const { return opened_; }

private:
    std::unique_ptr<StorageEngine> storage_engine_;
    bool opened_ = false;
};

} // namespace pomai
