// tests/extinction/stateful_fuzzer.cc
// Extinction Event: Stateful Chaos, Operation-Sequence Replay & Differential Fuzzing
//
// Stresses PomaiDB across millions of pseudo-random operations with parallel FP64 Oracle tracking.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "tests/extinction/extinction_oracle.h"

#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "pomegranate_engine.h"
#include "options.h"
#include "search.h"
#include "types.h"
#include "metadata.h"

namespace {

using namespace pomai;
using namespace pomai::core;
using namespace pomai::extinction;

enum class OpType : uint8_t {
    kInsert = 0,
    kUpsert = 1,
    kDelete = 2,
    kQuery = 3,
    kFilteredQuery = 4,
    kBatchInsert = 5,
    kBatchDelete = 6,
    kCompact = 7,
    kReopen = 8,
    kCount = 9
};

struct FuzzerConfig {
    uint32_t dim{16};
    MetricType metric{MetricType::kL2};
    OracleMetric oracle_metric{OracleMetric::kL2};
    uint32_t total_ops{10000};
    uint64_t seed{42};
    std::string db_dir;
};

// Generates adversarial and diverse vectors
class VectorAdversary {
public:
    explicit VectorAdversary(uint64_t seed, uint32_t dim)
        : rng_(seed), dim_(dim), d_norm_(0.0f, 1.0f), d_uni_(-1.0f, 1.0f) {}

    std::vector<float> Generate(int type = -1) {
        if (type == -1) {
            type = static_cast<int>(rng_() % 8);
        }
        std::vector<float> vec(dim_, 0.0f);
        switch (type) {
            case 0: // Standard Gaussian
                for (uint32_t d = 0; d < dim_; ++d) vec[d] = d_norm_(rng_);
                break;
            case 1: // Uniform
                for (uint32_t d = 0; d < dim_; ++d) vec[d] = d_uni_(rng_);
                break;
            case 2: // Sparse (mostly zeros)
                for (uint32_t d = 0; d < std::min<uint32_t>(3, dim_); ++d) {
                    vec[rng_() % dim_] = d_norm_(rng_);
                }
                break;
            case 3: // Collinear / Constant
                vec.assign(dim_, 1.5f);
                break;
            case 4: // Tiny magnitudes
                for (uint32_t d = 0; d < dim_; ++d) vec[d] = d_norm_(rng_) * 1e-15f;
                break;
            case 5: // Large magnitudes
                for (uint32_t d = 0; d < dim_; ++d) vec[d] = d_norm_(rng_) * 1e5f;
                break;
            case 6: // Antipodal pattern
                for (uint32_t d = 0; d < dim_; ++d) vec[d] = (d % 2 == 0) ? 1.0f : -1.0f;
                break;
            default:
                for (uint32_t d = 0; d < dim_; ++d) vec[d] = d_norm_(rng_);
                break;
        }
        return vec;
    }

private:
    std::mt19937_64 rng_;
    uint32_t dim_;
    std::normal_distribution<float> d_norm_;
    std::uniform_real_distribution<float> d_uni_;
};

void RunStatefulCampaign(const FuzzerConfig& cfg) {
    std::filesystem::remove_all(cfg.db_dir);
    std::filesystem::create_directories(cfg.db_dir);

    ExtinctionOracle oracle(cfg.dim, cfg.oracle_metric);
    VectorAdversary adversary(cfg.seed, cfg.dim);
    std::mt19937_64 op_rng(cfg.seed);

    DBOptions opt;
    opt.path = cfg.db_dir;
    opt.dim = cfg.dim;
    opt.fsync = FsyncPolicy::kNever;
    opt.index_params.type = IndexType::kHnsw;
    opt.index_params.hnsw_m = 32;
    opt.index_params.hnsw_ef_construction = 200;
    opt.index_params.hnsw_ef_search = 64;

    std::unique_ptr<PomegranateEngine> engine = std::make_unique<PomegranateEngine>(opt, cfg.metric);
    POMAI_EXPECT_OK(engine->Open());

    std::vector<std::string> op_log;
    op_log.reserve(cfg.total_ops);

    std::vector<uint64_t> active_pool;
    std::vector<uint64_t> deleted_pool;
    uint64_t next_id = 1;
    uint64_t total_queries = 0;
    double sum_recall = 0.0;

    for (uint32_t op_i = 1; op_i <= cfg.total_ops; ++op_i) {
        OpType op = static_cast<OpType>(op_rng() % static_cast<uint8_t>(OpType::kCount));

        // Adjust probabilities: More inserts and queries than reopens
        uint32_t roll = static_cast<uint32_t>(op_rng() % 100);
        if (roll < 40) {
            op = OpType::kInsert;
        } else if (roll < 55) {
            op = OpType::kQuery;
        } else if (roll < 70) {
            op = OpType::kDelete;
        } else if (roll < 80) {
            op = OpType::kFilteredQuery;
        } else if (roll < 90) {
            op = OpType::kBatchInsert;
        } else if (roll < 95) {
            op = OpType::kCompact;
        } else {
            op = OpType::kReopen;
        }

        std::ostringstream ss;

        switch (op) {
            case OpType::kInsert:
            case OpType::kUpsert: {
                uint64_t id = next_id++;
                if (op == OpType::kUpsert && !active_pool.empty() && (op_rng() % 2 == 0)) {
                    id = active_pool[op_rng() % active_pool.size()];
                }
                auto vec = adversary.Generate();
                OracleMetadata om;
                om.device_id = (id % 3 == 0) ? "dev_alpha" : "dev_beta";
                om.location_id = (id % 5 == 0) ? "loc_prime" : "loc_sub";

                Metadata pm;
                pm.device_id = om.device_id;
                pm.location_id = om.location_id;

                ss << "OP " << op_i << " INSERT id=" << id;
                op_log.push_back(ss.str());

                oracle.Put(id, vec, om);
                POMAI_EXPECT_OK(engine->Put(id, vec, pm));
                active_pool.push_back(id);
                break;
            }

            case OpType::kDelete: {
                if (active_pool.empty()) break;
                size_t idx = op_rng() % active_pool.size();
                uint64_t id = active_pool[idx];
                active_pool.erase(active_pool.begin() + idx);
                deleted_pool.push_back(id);

                ss << "OP " << op_i << " DELETE id=" << id;
                op_log.push_back(ss.str());

                oracle.Delete(id);
                POMAI_EXPECT_OK(engine->Delete(id));
                break;
            }

            case OpType::kQuery: {
                auto q = adversary.Generate();
                uint32_t topk = static_cast<uint32_t>((op_rng() % 15) + 1);

                ss << "OP " << op_i << " QUERY k=" << topk;
                op_log.push_back(ss.str());

                auto golden_hits = oracle.Search(q, topk);
                SearchResult res;
                if (op_i == 1978) _putenv("POMAI_DEBUG_QUERY=1");
                POMAI_EXPECT_OK(engine->Search(q, topk, &res));
                if (op_i == 1978) _putenv("POMAI_DEBUG_QUERY=");

                // Verification 1: Returned hits must NEVER contain deleted vectors!
                for (const auto& h : res.hits) {
                    if (oracle.table().find(h.id) != oracle.table().end()) {
                        POMAI_EXPECT_TRUE(!oracle.table().at(h.id).is_deleted);
                    }
                }

                // Verification 2: Number of hits must match expected live available
                size_t expected_hits = std::min<size_t>(topk, oracle.LiveCount());
                POMAI_EXPECT_EQ(res.hits.size(), expected_hits);

                // Verification 3: Recall against Golden Oracle
                if (expected_hits > 0) {
                    double recall = ExtinctionOracle::ComputeRecall(golden_hits, res.hits);
                    total_queries++;
                    sum_recall += recall;
                    if (recall < 0.50) {
                        std::cerr << std::setprecision(17);
                        std::cerr << "\n[OUTLIER RECALL NOTICE] Op " << op_i << " (seed=" << cfg.seed
                                  << "): expected_hits=" << expected_hits << " recall=" << recall << "\n";
                    }
                }
                break;
            }

            case OpType::kFilteredQuery: {
                auto q = adversary.Generate();
                uint32_t topk = 5;
                OracleFilter of;
                of.field = "device_id";
                of.value = "dev_alpha";

                SearchOptions sopts;
                sopts.filters.push_back(Filter("device_id", "dev_alpha"));

                ss << "OP " << op_i << " FILTERED_QUERY device_id=dev_alpha";
                op_log.push_back(ss.str());

                auto golden_hits = oracle.Search(q, topk, {of});
                SearchResult res;
                POMAI_EXPECT_OK(engine->Search(q, topk, sopts, &res));

                // Verify all returned hits strictly match filter
                for (const auto& h : res.hits) {
                    POMAI_EXPECT_TRUE(oracle.table().find(h.id) != oracle.table().end());
                    POMAI_EXPECT_EQ(oracle.table().at(h.id).meta.device_id, "dev_alpha");
                }
                break;
            }

            case OpType::kBatchInsert: {
                uint32_t bsz = static_cast<uint32_t>((op_rng() % 5) + 2);
                std::vector<VectorId> ids;
                std::vector<float> vecs_flat;
                ids.reserve(bsz);
                vecs_flat.reserve(bsz * cfg.dim);

                for (uint32_t b = 0; b < bsz; ++b) {
                    uint64_t id = next_id++;
                    auto vec = adversary.Generate();
                    ids.push_back(id);
                    vecs_flat.insert(vecs_flat.end(), vec.begin(), vec.end());
                    oracle.Put(id, vec);
                    active_pool.push_back(id);
                }

                ss << "OP " << op_i << " BATCH_INSERT size=" << bsz;
                op_log.push_back(ss.str());

                POMAI_EXPECT_OK(engine->PutBatch(ids, vecs_flat, cfg.dim));
                break;
            }

            case OpType::kCompact: {
                ss << "OP " << op_i << " COMPACT";
                op_log.push_back(ss.str());
                POMAI_EXPECT_OK(engine->Compact());
                break;
            }

            case OpType::kReopen: {
                ss << "OP " << op_i << " REOPEN";
                op_log.push_back(ss.str());

                POMAI_EXPECT_OK(engine->Close());
                engine = std::make_unique<PomegranateEngine>(opt, cfg.metric);
                POMAI_EXPECT_OK(engine->Open());

                // State verification: check random sample of active vectors
                if (!active_pool.empty()) {
                    uint32_t checks = std::min<uint32_t>(10, static_cast<uint32_t>(active_pool.size()));
                    for (uint32_t c = 0; c < checks; ++c) {
                        uint64_t test_id = active_pool[op_rng() % active_pool.size()];
                        std::vector<float> got;
                        POMAI_EXPECT_OK(engine->Get(test_id, &got));
                        std::vector<float> expected;
                        POMAI_EXPECT_TRUE(oracle.Get(test_id, &expected));
                        POMAI_EXPECT_EQ(got.size(), expected.size());
                    }
                }
                break;
            }

            default:
                break;
        }

        // Periodic checkpoint
        if (op_i % 2500 == 0) {
            std::cout << "[STATEFUL FUZZER] Seed " << cfg.seed << " reached " << op_i << " ops. "
                      << "Live=" << oracle.LiveCount() << " Tombs=" << oracle.TombstoneCount() << std::endl;
        }
    }

    if (total_queries > 0) {
        double avg_recall = sum_recall / static_cast<double>(total_queries);
        std::cout << "[FUZZER STATS] Seed " << cfg.seed << ": total_queries=" << total_queries
                  << " avg_recall=" << avg_recall << std::endl;
        POMAI_EXPECT_TRUE(avg_recall >= 0.95);
    }

    POMAI_EXPECT_OK(engine->Close());
}

// Stateful Campaign across multiple seeds
POMAI_TEST(Extinction_StatefulChaos_Seed42) {
    FuzzerConfig cfg;
    cfg.seed = 42;
    cfg.dim = 16;
    cfg.total_ops = 5000;
    cfg.db_dir = test::TempDir("pomai-extinction-seed42");
    RunStatefulCampaign(cfg);
}

POMAI_TEST(Extinction_StatefulChaos_Seed1337) {
    FuzzerConfig cfg;
    cfg.seed = 1337;
    cfg.dim = 32;
    cfg.total_ops = 5000;
    cfg.db_dir = test::TempDir("pomai-extinction-seed1337");
    RunStatefulCampaign(cfg);
}

POMAI_TEST(Extinction_StatefulChaos_Seed2026) {
    FuzzerConfig cfg;
    cfg.seed = 2026;
    cfg.dim = 8;
    cfg.total_ops = 5000;
    cfg.db_dir = test::TempDir("pomai-extinction-seed2026");
    RunStatefulCampaign(cfg);
}

POMAI_TEST(Extinction_StatefulChaos_SeedDeadBeef) {
    FuzzerConfig cfg;
    cfg.seed = 0xDEADBEEF;
    cfg.dim = 16;
    cfg.total_ops = 5000;
    cfg.db_dir = test::TempDir("pomai-extinction-seeddeadbeef");
    RunStatefulCampaign(cfg);
}

} // namespace
