#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "status.h"
#include "types.h"
#include "metadata.h"

namespace pomai {
class Env;
class WritableFile;
}

namespace pomai::core {

/**
 * @brief Represents a single operation in the WAL.
 */
struct WalEntry {
    uint64_t lsn;
    uint8_t op; // 1=Put, 2=Del, 3=PutMeta, 4=RawKV, 5=BatchStart, 6=BatchEnd
    VectorId id;
    uint32_t dim;
    std::span<const float> vec;
    std::string raw_data; // For RawKV, metadata blobs, etc.
    Metadata meta;
};

/**
 * @brief Interface for receiving synced WAL entries.
 * This can be implemented as a gRPC client, HTTP pusher, etc.
 */
class SyncReceiver {
public:
    virtual ~SyncReceiver() = default;
    virtual Status Receive(const WalEntry& entry) = 0;
};

/**
 * @brief Mock receiver for local testing and demonstration.
 */
class MockSyncReceiver : public SyncReceiver {
public:
    Status Receive(const WalEntry& entry) override {
        // In a real implementation, this would send data over the network.
        // For now, we just acknowledge receipt.
        last_received_lsn = entry.lsn;
        return Status::Ok();
    }
    uint64_t last_received_lsn = 0;
};

/**
 * @brief In-process callback receiver for custom stream handlers or hooks.
 */
class CallbackSyncReceiver : public SyncReceiver {
public:
    using Callback = std::function<Status(const WalEntry&)>;
    explicit CallbackSyncReceiver(Callback cb) : cb_(std::move(cb)) {}
    Status Receive(const WalEntry& entry) override {
        if (!cb_) return Status::Ok();
        return cb_(entry);
    }
private:
    Callback cb_;
};

/**
 * @brief File-based WAL receiver that persists replicated entries to a replica file.
 */
class FileWalSyncReceiver : public SyncReceiver {
public:
    explicit FileWalSyncReceiver(const std::string& path, Env* env = nullptr);
    ~FileWalSyncReceiver() override;

    Status Receive(const WalEntry& entry) override;
    Status Flush();

    uint64_t entries_received() const noexcept { return count_; }
    uint64_t last_lsn() const noexcept { return last_lsn_; }

private:
    std::string path_;
    Env* env_;
    std::unique_ptr<WritableFile> file_;
    uint64_t count_ = 0;
    uint64_t last_lsn_ = 0;
};

/**
 * @brief Handles streaming of WAL entries from a specific LSN.
 */
class WalStreamer {
public:
    WalStreamer(const std::string& db_path, uint32_t shard_id);
    
    /**
     * @brief Pushes all entries since last_lsn to the receiver.
     * @return Number of entries pushed.
     */
    Status PushSince(uint64_t last_lsn, SyncReceiver* receiver, uint64_t* new_last_lsn);

private:
    std::string db_path_;
    uint32_t shard_id_;
    
    std::string SegmentPath(uint64_t gen) const;
};

} // namespace pomai::core
