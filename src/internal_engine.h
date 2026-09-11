#pragma once
#include <memory>
#include <optional>
#include <vector>
#include <span>
#include <string_view>

#include "status.h"
#include "types.h"
#include "metadata.h"
#include "search.h"
#include "options.h"
#include "snapshot.h"
#include "hooks.h"
#include "scheduler.h"
#include "micro_kernel.h"

namespace pomai {

class StorageEngine {
public:
    Status Open(const EmbeddedOptions& options);
    void Close();

    Status Flush();
    Status Freeze();

    Status Append(VectorId id, std::span<const float> vec);
    Status Append(VectorId id, std::span<const float> vec, const Metadata& meta);
    Status AppendBatch(const std::vector<VectorId>& ids, const std::vector<std::span<const float>>& vectors);

    Status Get(VectorId id, std::vector<float>* out, Metadata* meta);
    Status Exists(VectorId id, bool* exists);
    Status Delete(VectorId id);
    
    Status Search(std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult* out);

    Status PushSync(core::SyncReceiver* receiver);

    Status GetSnapshot(std::shared_ptr<Snapshot>* out);
    Status NewIterator(const std::shared_ptr<Snapshot>& snap, std::unique_ptr<SnapshotIterator>* out);

    std::size_t GetMemTableBytesUsed() const;
    void AddPostPutHook(std::shared_ptr<PostPutHook> hook);

private:
    core::MicroKernel kernel_;
    std::vector<std::shared_ptr<PostPutHook>> hooks_;
};

} // namespace pomai

