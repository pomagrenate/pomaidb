#include "core/kernel/pods/graph_pod.h"
#include <cstring>

namespace pomai::core {

    struct AddVertexPayload {
        VertexId id;
        TagId tag;
        // Metadata handled separately or as tail
    };

    struct AddEdgePayload {
        VertexId src;
        VertexId dst;
        EdgeType type;
        uint32_t rank;
    };

    void GraphPod::Handle(Message&& msg) {
        switch (msg.opcode) {
            case 0x10: { // AddVertex
                if (msg.payload.size() < sizeof(AddVertexPayload)) return;
                const auto* p = reinterpret_cast<const AddVertexPayload*>(msg.payload.data());
                // Metadata is empty for now or extracted from tail
                auto st = runtime_->AddVertex(p->id, p->tag, Metadata());
                if (msg.result_ptr) *static_cast<Status*>(msg.result_ptr) = st;
                break;
            }
            case 0x11: { // AddEdge
                if (msg.payload.size() < sizeof(AddEdgePayload)) return;
                const auto* p = reinterpret_cast<const AddEdgePayload*>(msg.payload.data());
                auto st = runtime_->AddEdge(p->src, p->dst, p->type, p->rank, Metadata());
                if (msg.result_ptr) *static_cast<Status*>(msg.result_ptr) = st;
                break;
            }
            case 0x12: { // GetNeighbors
                if (msg.payload.size() < sizeof(VertexId)) return;
                VertexId src = *reinterpret_cast<const VertexId*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<std::vector<Neighbor>*>(msg.result_ptr);
                    (void)runtime_->GetNeighbors(src, out);
                }
                break;
            }
            case 0x13: { // GetNeighborsWithType
                struct P { VertexId src; EdgeType type; };
                if (msg.payload.size() < sizeof(P)) return;
                const auto* p = reinterpret_cast<const P*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<std::vector<Neighbor>*>(msg.result_ptr);
                    (void)runtime_->GetNeighbors(p->src, p->type, out);
                }
                break;
            }
            default:
                break;
        }
    }

} // namespace pomai::core
