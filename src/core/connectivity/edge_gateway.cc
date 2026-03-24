#include "core/connectivity/edge_gateway.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <string_view>
#include <vector>

#include "core/membrane/manager.h"

namespace pomai::core {

namespace {
int OpenListenSocket(uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int on = 1;
    (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return -1;
    }
    if (::listen(fd, 16) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}
} // namespace

EdgeGateway::EdgeGateway(MembraneManager* manager) : manager_(manager) {}
EdgeGateway::~EdgeGateway() { (void)Stop(); }

Status EdgeGateway::Start(uint16_t http_port, uint16_t ingest_port, std::string auth_token) {
    if (!manager_) return Status::InvalidArgument("manager is null");
    if (running_.exchange(true)) return Status::AlreadyExists("edge gateway already running");
    auth_token_ = std::move(auth_token);
    idempotency_log_path_ = manager_->GetOptions().path + "/gateway/idempotency.log";
    audit_log_path_ = manager_->GetOptions().path + "/gateway/audit.log";
    const auto load_st = LoadIdempotencyLog();
    if (!load_st.ok()) {
        running_.store(false);
        return load_st;
    }
    http_thread_ = std::thread([this, http_port]() { HttpLoop(http_port); });
    ingest_thread_ = std::thread([this, ingest_port]() { IngestLoop(ingest_port); });
    return Status::Ok();
}

Status EdgeGateway::Stop() {
    if (!running_.exchange(false)) return Status::Ok();
    const int hfd = http_fd_.exchange(-1);
    if (hfd >= 0) {
        ::shutdown(hfd, SHUT_RDWR);
        ::close(hfd);
    }
    const int ifd = ingest_fd_.exchange(-1);
    if (ifd >= 0) {
        ::shutdown(ifd, SHUT_RDWR);
        ::close(ifd);
    }
    if (http_thread_.joinable()) http_thread_.join();
    if (ingest_thread_.joinable()) ingest_thread_.join();
    return Status::Ok();
}

bool EdgeGateway::IsAuthorized(const std::string& request) const {
    if (auth_token_.empty()) return true;
    const std::string expected = "Authorization: Bearer " + auth_token_;
    return request.find(expected) != std::string::npos;
}

bool EdgeGateway::AllowRequest() {
    const uint64_t now_sec = static_cast<uint64_t>(std::time(nullptr));
    uint64_t prev = rate_window_sec_.load();
    if (prev != now_sec) {
        rate_window_sec_.store(now_sec);
        rate_window_count_.store(0);
    }
    const uint32_t c = rate_window_count_.fetch_add(1) + 1;
    return c <= rate_limit_per_sec_;
}

bool EdgeGateway::HasIdempotencyKey(const std::string& key) {
    if (key.empty()) return false;
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    std::lock_guard<std::mutex> lock(idem_mu_);
    for (auto it = seen_idempotency_keys_.begin(); it != seen_idempotency_keys_.end();) {
        if (it->second <= now) {
            it = seen_idempotency_keys_.erase(it);
        } else {
            ++it;
        }
    }
    return seen_idempotency_keys_.find(key) != seen_idempotency_keys_.end();
}

void EdgeGateway::RememberIdempotencyKey(const std::string& key) {
    if (key.empty()) return;
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    bool inserted = false;
    {
        std::lock_guard<std::mutex> lock(idem_mu_);
        const auto it = seen_idempotency_keys_.find(key);
        if (it == seen_idempotency_keys_.end()) {
            inserted = true;
        }
        seen_idempotency_keys_[key] = now + idempotency_ttl_sec_;
    }
    if (inserted) {
        AppendIdempotencyLog(key);
        (void)CompactIdempotencyLogIfNeeded();
    }
}

Status EdgeGateway::LoadIdempotencyLog() {
    namespace fs = std::filesystem;
    const fs::path p(idempotency_log_path_);
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    if (ec) return Status::IOError("failed to create gateway dir");

    std::ifstream in(idempotency_log_path_);
    if (!in.good()) return Status::Ok();
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    std::string line;
    std::lock_guard<std::mutex> lock(idem_mu_);
    while (std::getline(in, line)) {
        // Keep only sane keys to tolerate partial/corrupt lines.
        if (!line.empty() && line.size() <= 256 && line.find('\n') == std::string::npos && line.find('\r') == std::string::npos) {
            seen_idempotency_keys_[line] = now + idempotency_ttl_sec_;
        } else if (!line.empty()) {
            idempotency_log_errors_total_.fetch_add(1);
        }
    }
    return Status::Ok();
}

void EdgeGateway::AppendIdempotencyLog(const std::string& key) {
    if (idempotency_log_path_.empty()) return;
    std::ofstream out(idempotency_log_path_, std::ios::app);
    if (!out.good()) {
        idempotency_log_errors_total_.fetch_add(1);
        return;
    }
    out << key << "\n";
    out.flush();
    ++idempotency_append_count_;
}

Status EdgeGateway::CompactIdempotencyLogIfNeeded() {
    if (idempotency_append_count_ < idempotency_compact_every_) return Status::Ok();
    idempotency_append_count_ = 0;
    const std::string tmp_path = idempotency_log_path_ + ".tmp";
    std::ofstream out(tmp_path, std::ios::trunc);
    if (!out.good()) return Status::IOError("failed to compact idempotency log");
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    {
        std::lock_guard<std::mutex> lock(idem_mu_);
        for (auto it = seen_idempotency_keys_.begin(); it != seen_idempotency_keys_.end();) {
            if (it->second <= now) {
                it = seen_idempotency_keys_.erase(it);
                continue;
            }
            out << it->first << "\n";
            ++it;
        }
    }
    out.flush();
    out.close();
    std::error_code ec;
    std::filesystem::rename(tmp_path, idempotency_log_path_, ec);
    if (ec) {
        idempotency_log_errors_total_.fetch_add(1);
        return Status::IOError("failed to rotate idempotency log");
    }
    idempotency_compactions_total_.fetch_add(1);
    return Status::Ok();
}

Status EdgeGateway::RotateAuditLogIfNeeded() {
    if (audit_append_count_ < audit_rotate_every_) return Status::Ok();
    audit_append_count_ = 0;
    namespace fs = std::filesystem;
    const fs::path p(audit_log_path_);
    std::error_code ec;
    const fs::path rotated = p.string() + ".1";
    if (fs::exists(rotated, ec)) {
        fs::remove(rotated, ec);
    }
    fs::rename(p, rotated, ec);
    if (ec) {
        audit_log_errors_total_.fetch_add(1);
        return Status::IOError("failed to rotate audit log");
    }
    audit_log_rotations_total_.fetch_add(1);
    return Status::Ok();
}

void EdgeGateway::AppendAuditLog(std::string_view event, std::string_view detail) {
    if (audit_log_path_.empty()) return;
    std::ofstream out(audit_log_path_, std::ios::app);
    if (!out.good()) {
        audit_log_errors_total_.fetch_add(1);
        return;
    }
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    out << now << "|" << event << "|" << detail << "\n";
    out.flush();
    ++audit_append_count_;
    (void)RotateAuditLogIfNeeded();
}

std::string EdgeGateway::JsonResponse(int code, std::string_view reason, std::string_view status, std::string_view message) const {
    std::ostringstream body;
    body << "{\"status\":\"" << status << "\",\"message\":\"" << message << "\"}";
    const std::string body_s = body.str();
    std::ostringstream head;
    head << "HTTP/1.1 " << code << " " << reason << "\r\n";
    head << "Content-Type: application/json\r\n";
    head << "Content-Length: " << body_s.size() << "\r\n\r\n";
    return head.str() + body_s;
}

void EdgeGateway::HttpLoop(uint16_t port) {
    const int server_fd = OpenListenSocket(port);
    if (server_fd < 0) {
        running_.store(false);
        return;
    }
    http_fd_.store(server_fd);
    while (running_.load()) {
        int client = ::accept(server_fd, nullptr, nullptr);
        if (client < 0) continue;
        http_requests_total_.fetch_add(1);
        if (!AllowRequest()) {
            const std::string too_many = JsonResponse(429, "Too Many Requests", "error", "rate_limited");
            (void)::send(client, too_many.data(), too_many.size(), 0);
            ::close(client);
            http_errors_total_.fetch_add(1);
            rate_limited_total_.fetch_add(1);
            AppendAuditLog("http_rate_limited", "request");
            continue;
        }
        char buf[2048];
        const ssize_t n = ::recv(client, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = '\0';
            std::string req(buf, static_cast<size_t>(n));
            std::string response = JsonResponse(200, "OK", "ok", "healthy");
            if (req.rfind("GET /health ", 0) == 0) {
                (void)::send(client, response.data(), response.size(), 0);
                ::close(client);
                continue;
            }
            if (req.rfind("GET /metrics ", 0) == 0) {
                std::ostringstream oss;
                oss << "http_requests_total " << http_requests_total_.load() << "\n";
                oss << "http_errors_total " << http_errors_total_.load() << "\n";
                oss << "ingest_requests_total " << ingest_requests_total_.load() << "\n";
                oss << "ingest_errors_total " << ingest_errors_total_.load() << "\n";
                oss << "ingest_duplicates_total " << ingest_duplicates_total_.load() << "\n";
                oss << "rate_limited_total " << rate_limited_total_.load() << "\n";
                oss << "idempotency_compactions_total " << idempotency_compactions_total_.load() << "\n";
                oss << "idempotency_log_errors_total " << idempotency_log_errors_total_.load() << "\n";
                oss << "audit_log_errors_total " << audit_log_errors_total_.load() << "\n";
                oss << "audit_log_rotations_total " << audit_log_rotations_total_.load() << "\n";
                const std::string body = oss.str();
                const std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n";
                const std::string full = head + body;
                (void)::send(client, full.data(), full.size(), 0);
                ::close(client);
                continue;
            }
            if (!IsAuthorized(req)) {
                response = JsonResponse(401, "Unauthorized", "error", "unauthorized");
                (void)::send(client, response.data(), response.size(), 0);
                ::close(client);
                http_errors_total_.fetch_add(1);
                AppendAuditLog("http_unauthorized", "request");
                continue;
            }
            const auto key_pos = req.find("X-Idempotency-Key:");
            std::string idempotency_key;
            if (key_pos != std::string::npos) {
                const auto eol = req.find("\r\n", key_pos);
                if (eol != std::string::npos) {
                    idempotency_key = req.substr(key_pos + 18, eol - (key_pos + 18));
                    while (!idempotency_key.empty() && (idempotency_key.front() == ' ' || idempotency_key.front() == '\t')) {
                        idempotency_key.erase(idempotency_key.begin());
                    }
                }
            }
            if (HasIdempotencyKey(idempotency_key)) {
                response = JsonResponse(200, "OK", "duplicate", "idempotency_key_seen");
                (void)::send(client, response.data(), response.size(), 0);
                ::close(client);
                ingest_duplicates_total_.fetch_add(1);
                AppendAuditLog("http_duplicate", idempotency_key);
                continue;
            }
            // Tiny endpoints:
            // - GET /health
            // - POST /ingest/meta/<membrane>/<gid> body=value
            // - POST /ingest/vector/<membrane>/<id> body=f1,f2,f3
            if (req.rfind("POST /ingest/meta/", 0) == 0) {
                const auto sp1 = req.find(' ');
                const auto sp2 = req.find(' ', sp1 + 1);
                const std::string path = req.substr(sp1 + 1, sp2 - (sp1 + 1));
                const auto body_pos = req.find("\r\n\r\n");
                const std::string body = (body_pos == std::string::npos) ? std::string() : req.substr(body_pos + 4);
                const std::string prefix = "/ingest/meta/";
                const auto rem = path.substr(prefix.size());
                const auto slash = rem.find('/');
                if (slash != std::string::npos) {
                    const auto membrane = rem.substr(0, slash);
                    const auto gid = rem.substr(slash + 1);
                    const auto st = manager_->MetaPut(membrane, gid, body);
                    if (st.ok()) {
                        RememberIdempotencyKey(idempotency_key);
                        AppendAuditLog("http_meta_ok", std::string(membrane) + "/" + std::string(gid));
                        response = JsonResponse(200, "OK", "ok", "meta_ingested");
                    } else {
                        AppendAuditLog("http_meta_err", st.message());
                        response = JsonResponse(400, "Bad Request", "error", st.message());
                        http_errors_total_.fetch_add(1);
                    }
                } else {
                    AppendAuditLog("http_meta_err", "invalid_meta_path");
                    response = JsonResponse(400, "Bad Request", "error", "invalid_meta_path");
                    http_errors_total_.fetch_add(1);
                }
            } else if (req.rfind("POST /ingest/vector/", 0) == 0) {
                const auto sp1 = req.find(' ');
                const auto sp2 = req.find(' ', sp1 + 1);
                const std::string path = req.substr(sp1 + 1, sp2 - (sp1 + 1));
                const auto body_pos = req.find("\r\n\r\n");
                const std::string body = (body_pos == std::string::npos) ? std::string() : req.substr(body_pos + 4);
                const std::string prefix = "/ingest/vector/";
                const auto rem = path.substr(prefix.size());
                const auto slash = rem.find('/');
                if (slash != std::string::npos) {
                    const auto membrane = rem.substr(0, slash);
                    const auto id_s = rem.substr(slash + 1);
                    const uint64_t id = static_cast<uint64_t>(std::strtoull(id_s.c_str(), nullptr, 10));
                    std::vector<float> v;
                    std::istringstream viss(body);
                    std::string tok;
                    while (std::getline(viss, tok, ',')) {
                        v.push_back(static_cast<float>(std::strtod(tok.c_str(), nullptr)));
                    }
                    if (!v.empty()) {
                        const auto st = manager_->PutVector(membrane, id, v);
                        if (st.ok()) {
                            RememberIdempotencyKey(idempotency_key);
                            AppendAuditLog("http_vector_ok", std::string(membrane) + "/" + std::to_string(id));
                            response = JsonResponse(200, "OK", "ok", "vector_ingested");
                        } else {
                            AppendAuditLog("http_vector_err", st.message());
                            response = JsonResponse(400, "Bad Request", "error", st.message());
                            http_errors_total_.fetch_add(1);
                        }
                    } else {
                        AppendAuditLog("http_vector_err", "empty_vector");
                        response = JsonResponse(400, "Bad Request", "error", "empty_vector");
                        http_errors_total_.fetch_add(1);
                    }
                } else {
                    AppendAuditLog("http_vector_err", "invalid_vector_path");
                    response = JsonResponse(400, "Bad Request", "error", "invalid_vector_path");
                    http_errors_total_.fetch_add(1);
                }
            } else {
                AppendAuditLog("http_not_found", "endpoint");
                response = JsonResponse(404, "Not Found", "error", "endpoint_not_found");
                http_errors_total_.fetch_add(1);
            }
            (void)::send(client, response.data(), response.size(), 0);
        }
        ::close(client);
    }
    if (http_fd_.exchange(-1) >= 0) ::close(server_fd);
}

void EdgeGateway::IngestLoop(uint16_t port) {
    const int server_fd = OpenListenSocket(port);
    if (server_fd < 0) {
        running_.store(false);
        return;
    }
    ingest_fd_.store(server_fd);
    while (running_.load()) {
        int client = ::accept(server_fd, nullptr, nullptr);
        if (client < 0) continue;
        ingest_requests_total_.fetch_add(1);
        if (!AllowRequest()) {
            static const std::string nack = "ERR|rate_limited\n";
            (void)::send(client, nack.data(), nack.size(), 0);
            ::close(client);
            ingest_errors_total_.fetch_add(1);
            rate_limited_total_.fetch_add(1);
            AppendAuditLog("ingest_rate_limited", "request");
            continue;
        }
        char buf[4096];
        const ssize_t n = ::recv(client, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = '\0';
            // Simple line protocol for MQTT/WebSocket gateway compatibility:
            // No-auth: MQTT|<membrane>|<id>|f1,f2,f3
            // Auth:    MQTT|<token>|<membrane>|<id>|f1,f2,f3
            // Same shape for WS|...
            std::string line(buf, static_cast<size_t>(n));
            std::istringstream iss(line);
            std::string proto;
            if (!std::getline(iss, proto, '|')) {
                ::close(client);
                ingest_errors_total_.fetch_add(1);
                continue;
            }
            std::string token_or_membrane;
            std::string membrane;
            std::string id_s;
            std::string vec_s;
            if (!std::getline(iss, token_or_membrane, '|')) {
                ::close(client);
                continue;
            }
            if (auth_token_.empty()) {
                membrane = token_or_membrane;
                if (!std::getline(iss, id_s, '|') || !std::getline(iss, vec_s, '|')) {
                    ::close(client);
                    ingest_errors_total_.fetch_add(1);
                    continue;
                }
            } else {
                if (token_or_membrane != auth_token_) {
                    ::close(client);
                    ingest_errors_total_.fetch_add(1);
                    continue;
                }
                if (!std::getline(iss, membrane, '|') || !std::getline(iss, id_s, '|') || !std::getline(iss, vec_s, '|')) {
                    ::close(client);
                    ingest_errors_total_.fetch_add(1);
                    continue;
                }
            }
            if ((proto == "MQTT" || proto == "WS") && !membrane.empty() && !id_s.empty()) {
                const uint64_t id = static_cast<uint64_t>(std::strtoull(id_s.c_str(), nullptr, 10));
                std::vector<float> v;
                std::istringstream viss(vec_s);
                std::string tok;
                while (std::getline(viss, tok, ',')) {
                    v.push_back(static_cast<float>(std::strtod(tok.c_str(), nullptr)));
                }
                if (!v.empty()) {
                    // Optional fields:
                    // 5th token: idempotency key
                    // 6th token: durability ("D")
                    std::string idem_key;
                    std::string durability;
                    (void)std::getline(iss, idem_key, '|');
                    (void)std::getline(iss, durability, '|');
                    if (HasIdempotencyKey(idem_key)) {
                        const std::string reply = "ACK|duplicate\n";
                        ingest_duplicates_total_.fetch_add(1);
                        AppendAuditLog("ingest_duplicate", idem_key);
                        (void)::send(client, reply.data(), reply.size(), 0);
                        ::close(client);
                        continue;
                    }
                    auto st = manager_->PutVector(membrane, id, v);
                    if (st.ok() && durability == "D") {
                        st = manager_->FlushAll();
                    }
                    const std::string reply = st.ok() ? "ACK|ok\n" : "ERR|write_failed\n";
                    if (!st.ok()) {
                        ingest_errors_total_.fetch_add(1);
                        AppendAuditLog("ingest_err", st.message());
                    } else {
                        RememberIdempotencyKey(idem_key);
                        AppendAuditLog("ingest_ok", std::string(membrane) + "/" + std::to_string(id));
                    }
                    (void)::send(client, reply.data(), reply.size(), 0);
                } else {
                    const std::string reply = "ERR|bad_vector\n";
                    ingest_errors_total_.fetch_add(1);
                    (void)::send(client, reply.data(), reply.size(), 0);
                }
            } else {
                const std::string reply = "ERR|bad_protocol\n";
                ingest_errors_total_.fetch_add(1);
                (void)::send(client, reply.data(), reply.size(), 0);
            }
        }
        ::close(client);
    }
    if (ingest_fd_.exchange(-1) >= 0) ::close(server_fd);
}

} // namespace pomai::core
