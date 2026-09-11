#include "vector_pod.h"
#include <cstring>

namespace pomai::core {
    namespace {
    template <typename T>
    bool PayloadAs(std::span<const uint8_t> payload, const T** out) {
        if (payload.size() != sizeof(T)) return false;
        *out = reinterpret_cast<const T*>(payload.data());
        return true;
    }
    void SetStatus(Status* out, const Status& st) {
        if (out) *out = st;
    }
    } // namespace

    void VectorPod::Handle(Message&& msg) {
        if (!engine_) {
            SetStatus(msg.status_ptr, Status::Corruption("VectorPod has no engine"));
            return;
        }

        switch (msg.opcode) {
            case Op::kPut: {
                if (msg.payload.size() < sizeof(VectorId)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector put payload too small"));
                    return;
                }
                VectorId id = *reinterpret_cast<const VectorId*>(msg.payload.data());
                std::span<const float> vec(
                    reinterpret_cast<const float*>(msg.payload.data() + sizeof(VectorId)),
                    (msg.payload.size() - sizeof(VectorId)) / sizeof(float)
                );
                if (vec.empty()) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector put empty payload"));
                    return;
                }
                auto st = engine_->Put(id, vec);
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kPutWithMeta: {
                struct P {
                    VectorId id;
                    const float* vec_data;
                    size_t vec_size;
                    const Metadata* meta;
                };
                const P* p = nullptr;
                if (!PayloadAs(msg.payload, &p) || !p->vec_data || !p->meta || p->vec_size == 0) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector put-meta invalid payload"));
                    return;
                }
                auto st = engine_->Put(p->id, std::span<const float>(p->vec_data, p->vec_size), *p->meta);
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kSearch: {
                struct P {
                    uint32_t topk;
                    const float* query_data;
                    size_t query_size;
                    const SearchOptions* opts;
                };
                struct SearchResultEnvelope {
                    SearchResult* out;
                    Status* st;
                };
                const P* p = nullptr;
                auto* env = static_cast<SearchResultEnvelope*>(msg.result_ptr);
                if (!PayloadAs(msg.payload, &p) || !p->query_data || p->query_size == 0 || !p->opts || !env ||
                    !env->out || !env->st) {
                    if (env && env->st) {
                        *env->st = Status::InvalidArgument("vector search invalid payload");
                    }
                    return;
                }
                std::span<const float> query(p->query_data, p->query_size);
                auto st = engine_->Search(query, p->topk, *p->opts, env->out);
                *env->st = st;
                break;
            }
            case Op::kGet: {
                if (msg.payload.size() < sizeof(VectorId)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector get payload too small"));
                    return;
                }
                VectorId id = *reinterpret_cast<const VectorId*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<std::vector<float>*>(msg.result_ptr);
                    Status st = engine_->Get(id, out, nullptr);
                    SetStatus(msg.status_ptr, st);
                }
                break;
            }
            case Op::kGetWithMeta: {
                struct P {
                    VectorId id;
                    std::vector<float>* out_vec;
                    pomai::Metadata* out_meta;
                };
                const P* p = nullptr;
                if (!PayloadAs(msg.payload, &p) || !p->out_vec) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector get-meta invalid payload"));
                    return;
                }
                auto st = engine_->Get(p->id, p->out_vec, p->out_meta);
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kDelete: {
                if (msg.payload.size() < sizeof(VectorId)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector delete payload too small"));
                    return;
                }
                VectorId id = *reinterpret_cast<const VectorId*>(msg.payload.data());
                auto st = engine_->Delete(id);
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kFlush: {
                auto st = engine_->Flush();
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kFreeze: {
                auto st = engine_->Freeze();
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kExists: {
                if (msg.payload.size() < sizeof(VectorId)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector exists payload too small"));
                    return;
                }
                VectorId id = *reinterpret_cast<const VectorId*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<bool*>(msg.result_ptr);
                    Status st = engine_->Exists(id, out);
                    SetStatus(msg.status_ptr, st);
                }
                break;
            }
            case Op::kSync: {
                if (msg.result_ptr) {
                    auto* rx = static_cast<SyncReceiver*>(msg.result_ptr);
                    (void)engine_->PushSync(rx);
                }
                break;
            }

            case Op::kPutBatch: {
                struct P {
                    const std::vector<VectorId>* ids;
                    const std::vector<std::span<const float>>* vectors;
                };
                const P* p = nullptr;
                if (!PayloadAs(msg.payload, &p) || !p->ids || !p->vectors) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector put-batch invalid payload"));
                    return;
                }
                auto st = engine_->PutBatch(*p->ids, *p->vectors);
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kGetSnapshot: {
                if (msg.result_ptr) {
                    auto* out = static_cast<std::shared_ptr<Snapshot>*>(msg.result_ptr);
                    Status st = engine_->GetSnapshot(out);
                    SetStatus(msg.status_ptr, st);
                }
                break;
            }
            case Op::kNewIterator: {
                if (msg.payload.size() < sizeof(void*)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector new-iterator payload too small"));
                    return;
                }
                const std::shared_ptr<Snapshot>* snap = *reinterpret_cast<const std::shared_ptr<Snapshot>* const*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<std::unique_ptr<SnapshotIterator>*>(msg.result_ptr);
                    Status st = engine_->NewIterator(*snap, out);
                    SetStatus(msg.status_ptr, st);
                }
                break;
            }
            default:
                SetStatus(msg.status_ptr, Status::InvalidArgument("vector pod unsupported opcode"));
                break;
        }
    }

} // namespace pomai::core
