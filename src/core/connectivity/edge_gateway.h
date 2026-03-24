#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "pomai/status.h"

namespace pomai::core {
class MembraneManager;

class EdgeGateway {
public:
    explicit EdgeGateway(MembraneManager* manager);
    ~EdgeGateway();

    Status Start(uint16_t http_port, uint16_t ingest_port, std::string auth_token = {});
    Status Stop();
    bool Running() const { return running_.load(); }

private:
    bool IsAuthorized(const std::string& request) const;
    bool AllowRequest();
    bool HasIdempotencyKey(const std::string& key);
    void RememberIdempotencyKey(const std::string& key);
    Status LoadIdempotencyLog();
    void AppendIdempotencyLog(const std::string& key);
    Status CompactIdempotencyLogIfNeeded();
    Status RotateAuditLogIfNeeded();
    void AppendAuditLog(std::string_view event, std::string_view detail);
    std::string JsonResponse(int code, std::string_view reason, std::string_view status, std::string_view message) const;
    void HttpLoop(uint16_t port);
    void IngestLoop(uint16_t port);

    MembraneManager* manager_ = nullptr;
    std::atomic<bool> running_{false};
    std::string auth_token_;
    std::atomic<int> http_fd_{-1};
    std::atomic<int> ingest_fd_{-1};
    std::atomic<uint64_t> http_requests_total_{0};
    std::atomic<uint64_t> http_errors_total_{0};
    std::atomic<uint64_t> ingest_requests_total_{0};
    std::atomic<uint64_t> ingest_errors_total_{0};
    std::atomic<uint64_t> ingest_duplicates_total_{0};
    std::atomic<uint64_t> rate_limited_total_{0};
    std::atomic<uint64_t> idempotency_compactions_total_{0};
    std::atomic<uint64_t> idempotency_log_errors_total_{0};
    std::atomic<uint64_t> audit_log_errors_total_{0};
    std::atomic<uint64_t> audit_log_rotations_total_{0};
    std::atomic<uint64_t> rate_window_sec_{0};
    std::atomic<uint32_t> rate_window_count_{0};
    uint32_t rate_limit_per_sec_ = 2000;
    std::string idempotency_log_path_;
    std::string audit_log_path_;
    uint64_t idempotency_append_count_ = 0;
    uint64_t idempotency_compact_every_ = 1000;
    uint64_t audit_append_count_ = 0;
    uint64_t audit_rotate_every_ = 5000;
    mutable std::mutex idem_mu_;
    std::unordered_map<std::string, uint64_t> seen_idempotency_keys_;
    uint64_t idempotency_ttl_sec_ = 300;
    std::thread http_thread_;
    std::thread ingest_thread_;
};

} // namespace pomai::core
