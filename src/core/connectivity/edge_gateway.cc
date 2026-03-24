#include "core/connectivity/edge_gateway.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <sstream>
#include <string>
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
        char buf[2048];
        const ssize_t n = ::recv(client, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = '\0';
            std::string req(buf, static_cast<size_t>(n));
            std::string response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nOK";
            if (req.rfind("GET /health ", 0) == 0) {
                (void)::send(client, response.data(), response.size(), 0);
                ::close(client);
                continue;
            }
            if (!IsAuthorized(req)) {
                response = "HTTP/1.1 401 Unauthorized\r\nContent-Type: text/plain\r\nContent-Length: 12\r\n\r\nunauthorized";
                (void)::send(client, response.data(), response.size(), 0);
                ::close(client);
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
                    (void)manager_->MetaPut(membrane, gid, body);
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
                    if (!v.empty()) (void)manager_->PutVector(membrane, id, v);
                }
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
                if (!std::getline(iss, id_s, '|') || !std::getline(iss, vec_s)) {
                    ::close(client);
                    continue;
                }
            } else {
                if (token_or_membrane != auth_token_) {
                    ::close(client);
                    continue;
                }
                if (!std::getline(iss, membrane, '|') || !std::getline(iss, id_s, '|') || !std::getline(iss, vec_s)) {
                    ::close(client);
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
                    (void)manager_->PutVector(membrane, id, v);
                }
            }
        }
        ::close(client);
    }
    if (ingest_fd_.exchange(-1) >= 0) ::close(server_fd);
}

} // namespace pomai::core
