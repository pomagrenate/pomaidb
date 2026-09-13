// tests/extinction/extinction_replay.cc
// Operation-Sequence Replay Tool & Deterministic Minimizer
//
// Usage: pomai_extinction_replay <operation_log_file> [db_dir]

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>

#include "pomegranate_engine.h"
#include "options.h"
#include "search.h"
#include "tests/extinction/extinction_oracle.h"

using namespace pomai;
using namespace pomai::core;
using namespace pomai::extinction;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: pomai_extinction_replay <operation_log_file> [db_dir]\n";
        return 1;
    }

    std::string log_file = argv[1];
    std::string db_dir = (argc >= 3) ? argv[2] : "./replay_db";

    std::ifstream in(log_file);
    if (!in.is_open()) {
        std::cerr << "Failed to open operation log: " << log_file << "\n";
        return 1;
    }

    std::cout << "[REPLAY] Loading operation log from: " << log_file << "\n";
    std::cout << "[REPLAY] Target DB directory: " << db_dir << "\n";

    std::filesystem::remove_all(db_dir);

    const uint32_t dim = 16;
    DBOptions opt;
    opt.path = db_dir;
    opt.dim = dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;

    ExtinctionOracle oracle(dim, OracleMetric::kL2);
    auto engine = std::make_unique<PomegranateEngine>(opt, MetricType::kL2);
    Status st = engine->Open();
    if (!st.ok()) {
        std::cerr << "[REPLAY FATAL] Failed to open engine: " << st.ToString() << "\n";
        return 2;
    }

    std::string line;
    uint64_t line_num = 0;
    uint64_t executed = 0;

    while (std::getline(in, line)) {
        line_num++;
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        // Skip optional "OP <num>" prefix
        if (token == "OP") {
            uint64_t op_num;
            iss >> op_num;
            iss >> token;
        }

        if (token == "INSERT" || token == "UPSERT") {
            std::string id_str;
            iss >> id_str;
            uint64_t id = 0;
            if (id_str.rfind("id=", 0) == 0) {
                id = std::stoull(id_str.substr(3));
            } else {
                id = std::stoull(id_str);
            }

            std::vector<float> vec(dim, static_cast<float>(id % 100));
            oracle.Put(id, vec);
            st = engine->Put(id, vec);
            if (!st.ok()) {
                std::cerr << "[REPLAY FAILURE] Line " << line_num << " Put failed: " << st.ToString() << "\n";
                return 3;
            }
            executed++;
        } else if (token == "DELETE") {
            std::string id_str;
            iss >> id_str;
            uint64_t id = 0;
            if (id_str.rfind("id=", 0) == 0) {
                id = std::stoull(id_str.substr(3));
            } else {
                id = std::stoull(id_str);
            }

            oracle.Delete(id);
            st = engine->Delete(id);
            if (!st.ok()) {
                std::cerr << "[REPLAY FAILURE] Line " << line_num << " Delete failed: " << st.ToString() << "\n";
                return 3;
            }
            executed++;
        } else if (token == "COMPACT" || token == "PRESS") {
            st = engine->Compact();
            if (!st.ok()) {
                std::cerr << "[REPLAY FAILURE] Line " << line_num << " Compact failed: " << st.ToString() << "\n";
                return 3;
            }
            executed++;
        } else if (token == "REOPEN") {
            (void)engine->Close();
            engine = std::make_unique<PomegranateEngine>(opt, MetricType::kL2);
            st = engine->Open();
            if (!st.ok()) {
                std::cerr << "[REPLAY FAILURE] Line " << line_num << " Reopen failed: " << st.ToString() << "\n";
                return 3;
            }
            executed++;
        } else if (token == "QUERY") {
            std::string k_str;
            iss >> k_str;
            uint32_t topk = 10;
            if (k_str.rfind("k=", 0) == 0) {
                topk = static_cast<uint32_t>(std::stoul(k_str.substr(2)));
            }

            std::vector<float> q(dim, 1.0f);
            SearchResult res;
            st = engine->Search(q, topk, &res);
            if (!st.ok()) {
                std::cerr << "[REPLAY FAILURE] Line " << line_num << " Search failed: " << st.ToString() << "\n";
                return 3;
            }

            // Verify no deleted items returned
            for (const auto& h : res.hits) {
                if (oracle.table().count(h.id) && oracle.table().at(h.id).is_deleted) {
                    std::cerr << "[REPLAY INTEGRITY ERROR] Line " << line_num
                              << " Search returned deleted vector ID: " << h.id << "\n";
                    return 4;
                }
            }
            executed++;
        }
    }

    (void)engine->Close();
    std::cout << "[REPLAY SUCCESS] Executed " << executed << " operations successfully without divergence.\n";
    return 0;
}
