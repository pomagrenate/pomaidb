#include "core/kernel/pods/vector_pod.h"
#include "core/query/lexical_index.h"
#include <cstring>

namespace pomai::core {

    void VectorPod::Handle(Message&& msg) {
        switch (msg.opcode) {
            case 0x01: { // Put
                // Payload is VectorId (8) + vec (dim*4)
                if (msg.payload.size() < 8) return;
                VectorId id = *reinterpret_cast<const VectorId*>(msg.payload.data());
                std::span<const float> vec(
                    reinterpret_cast<const float*>(msg.payload.data() + 8),
                    (msg.payload.size() - 8) / 4
                );
                auto st = runtime_->Put(id, vec);
                if (msg.result_ptr) *static_cast<Status*>(msg.result_ptr) = st;
                break;
            }
            case 0x03: { // Search
                // Payload is topk (4) + query (dim*4)
                if (msg.payload.size() < 4) return;
                uint32_t topk = *reinterpret_cast<const uint32_t*>(msg.payload.data());
                std::span<const float> query(
                    reinterpret_cast<const float*>(msg.payload.data() + 4),
                    (msg.payload.size() - 4) / 4
                );
                if (msg.result_ptr) {
                    auto* out = static_cast<SearchResult*>(msg.result_ptr);
                    std::vector<SearchHit> hits;
                    auto st = runtime_->Search(query, topk, SearchOptions(), &hits);
                    if (st.ok()) {
                        out->hits = std::move(hits);
                        out->routed_shards_count = 1;
                    }
                }
                break;
            }
            case 0x02: { // Get
                if (msg.payload.size() < 8) return;
                VectorId id = *reinterpret_cast<const VectorId*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<std::vector<float>*>(msg.result_ptr);
                    (void)runtime_->Get(id, out, nullptr);
                }
                break;
            }
            case 0x04: { // Delete
                if (msg.payload.size() < 8) return;
                VectorId id = *reinterpret_cast<const VectorId*>(msg.payload.data());
                auto st = runtime_->Delete(id);
                if (msg.result_ptr) *static_cast<Status*>(msg.result_ptr) = st;
                break;
            }
            case 0x05: { // Flush
                auto st = runtime_->Flush();
                if (msg.result_ptr) *static_cast<Status*>(msg.result_ptr) = st;
                break;
            }
            case 0x06: { // Freeze
                auto st = runtime_->Freeze();
                if (msg.result_ptr) *static_cast<Status*>(msg.result_ptr) = st;
                break;
            }
            case 0x09: { // Exists
                if (msg.payload.size() < 8) return;
                VectorId id = *reinterpret_cast<const VectorId*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<bool*>(msg.result_ptr);
                    (void)runtime_->Exists(id, out);
                }
                break;
            }
            case 0x07: { // Sync
                if (msg.result_ptr) {
                    auto* rx = static_cast<SyncReceiver*>(msg.result_ptr);
                    (void)runtime_->PushSync(rx);
                }
                break;
            }
            case 0x0C: { // SearchLexical
                if (msg.payload.size() < 4) return;
                uint32_t topk = *reinterpret_cast<const uint32_t*>(msg.payload.data());
                std::string query(reinterpret_cast<const char*>(msg.payload.data() + 4), msg.payload.size() - 4);
                if (msg.result_ptr) {
                    auto* out = static_cast<std::vector<pomai::core::LexicalHit>*>(msg.result_ptr);
                    (void)runtime_->SearchLexical(query, topk, out);
                }
                break;
            }
            case 0x0D: { // PutBatch
                struct P {
                    const std::vector<VectorId>* ids;
                    const std::vector<std::span<const float>>* vectors;
                };
                if (msg.payload.size() < sizeof(P)) return;
                const auto* p = reinterpret_cast<const P*>(msg.payload.data());
                auto st = runtime_->PutBatch(*p->ids, *p->vectors);
                if (msg.result_ptr) *static_cast<Status*>(msg.result_ptr) = st;
                break;
            }
            case 0x0A: { // GetSnapshot
                if (msg.result_ptr) {
                    auto* out = static_cast<std::shared_ptr<Snapshot>*>(msg.result_ptr);
                    *out = runtime_->GetSnapshot();
                }
                break;
            }
            case 0x0B: { // NewIterator
                if (msg.payload.size() < sizeof(void*)) return;
                const std::shared_ptr<Snapshot>* snap = *reinterpret_cast<const std::shared_ptr<Snapshot>* const*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<std::unique_ptr<SnapshotIterator>*>(msg.result_ptr);
                    auto v_snap = std::static_pointer_cast<VectorSnapshot>(*snap);
                    (void)runtime_->NewIterator(v_snap, out);
                }
                break;
            }
            default:
                break;
        }
    }

} // namespace pomai::core
