#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

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
    void HttpLoop(uint16_t port);
    void IngestLoop(uint16_t port);

    MembraneManager* manager_ = nullptr;
    std::atomic<bool> running_{false};
    std::string auth_token_;
    std::atomic<int> http_fd_{-1};
    std::atomic<int> ingest_fd_{-1};
    std::thread http_thread_;
    std::thread ingest_thread_;
};

} // namespace pomai::core
