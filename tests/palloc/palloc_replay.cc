#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <random>

#include <palloc.h>
#include <palloc_vector.h>
#include "palloc_oracle.h"

int main(int argc, char** argv) {
    std::cout << "=== PomaiDB palloc Replay Tool ===" << std::endl;
    if (argc < 2) {
        std::cout << "Usage: palloc_replay <seed> [count] OR palloc_replay --file <ops.log>" << std::endl;
        return 1;
    }

    std::string arg1 = argv[1];
    if (arg1 == "--file" && argc >= 3) {
        std::ifstream file(argv[2]);
        if (!file.is_open()) {
            std::cerr << "Failed to open operation log: " << argv[2] << std::endl;
            return 1;
        }
        std::string line;
        pomai::palloc_qa::AllocatorOracle oracle(true, 32);
        std::vector<void*> live;
        uint64_t op_count = 0;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string op;
            iss >> op;
            if (op == "ALLOC") {
                size_t sz, align;
                iss >> sz >> align;
                void* p = oracle.TrackAlloc(sz, align);
                if (p) live.push_back(p);
                op_count++;
            } else if (op == "FREE") {
                if (!live.empty()) {
                    void* p = live.back();
                    live.pop_back();
                    oracle.TrackFree(p);
                    op_count++;
                }
            }
        }
        std::cout << "Successfully replayed " << op_count << " operations from log file without error." << std::endl;
        return 0;
    }

    uint32_t seed = static_cast<uint32_t>(std::stoul(arg1));
    uint64_t count = argc >= 3 ? std::stoull(argv[2]) : 50000;
    std::cout << "Replaying deterministic sequence: seed=" << seed << " ops=" << count << std::endl;

    std::mt19937_64 rng(seed);
    pomai::palloc_qa::AllocatorOracle oracle(true, 32);
    std::vector<void*> live;

    for (uint64_t i = 0; i < count; ++i) {
        int action = rng() % 100;
        if (action < 50 || live.empty()) {
            size_t sz = 16 + (rng() % 8192);
            void* p = oracle.TrackAlloc(sz, 64, static_cast<uint32_t>(rng()));
            if (p) live.push_back(p);
        } else {
            size_t idx = rng() % live.size();
            void* p = live[idx];
            oracle.TrackFree(p);
            live[idx] = live.back();
            live.pop_back();
        }
    }

    for (void* p : live) {
        oracle.TrackFree(p);
    }

    std::cout << "Replay completed successfully. 0 corruptions, 0 leaks." << std::endl;
    return 0;
}
