#pragma once

#include <mutex>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/kernel/message.h"
#include "core/kernel/pod.h"
#include "pomai/status.h"
#include "ai/analytical_engine.h"

namespace pomai::core {

    /**
     * Pomegranate MicroKernel: The central coordinator for Pods.
     * Manages sequential execution of tasks via an in-memory message queue.
     */
    class MicroKernel {
    public:
        MicroKernel() = default;
        ~MicroKernel() { Stop(); }

        // Non-copyable
        MicroKernel(const MicroKernel&) = delete;
        MicroKernel& operator=(const MicroKernel&) = delete;

        /** Register a service pod. kernel takes ownership. */
        Status RegisterPod(std::unique_ptr<Pod> pod) {
            if (!pod) return Status::InvalidArgument("pod is null");
            std::lock_guard<std::recursive_mutex> lock(mu_);
            PodId id = pod->Id();
            if (pods_.count(id)) return Status::AlreadyExists("pod already registered");
            
            pod->OnStart();
            pods_[id] = std::move(pod);
            return Status::Ok();
        }

        /** Unregister and stop a pod. */
        void UnregisterPod(PodId id) {
            std::lock_guard<std::recursive_mutex> lock(mu_);
            auto it = pods_.find(id);
            if (it != pods_.end()) {
                it->second->OnStop();
                pods_.erase(it);
            }
        }

        /** Post a message for later execution. */
        void Enqueue(Message&& msg) {
            // Apply ELM-based Backpressure
            auto* elm = AnalyticalEngine::Global().GetModel("kernel_pressure");
            if (!elm) {
                // Lazily Train strict mathematical relationship: latency ~ queue_size * 5 + payload_size * 0.1
                (void)AnalyticalEngine::Global().CreateELMModel("kernel_pressure", 2, 8, 1);
                elm = AnalyticalEngine::Global().GetModel("kernel_pressure");
                if (elm) {
                    float X[8] = {
                        0.0f, 0.0f,
                        10.0f, 100.0f,
                        100.0f, 1000.0f,
                        200.0f, 5000.0f
                    };
                    float Y[4] = {
                        0.0f,
                        60.0f,
                        600.0f,
                        1500.0f
                    };
                    (void)elm->Train(std::span<const float>(X, 8), std::span<const float>(Y, 4), 4);
                }
            }

            std::lock_guard<std::recursive_mutex> lock(mu_);
            if (elm && elm->InputDim() == 2 && elm->OutputDim() >= 1) {
                float x[2] = { static_cast<float>(queue_.size()), static_cast<float>(msg.payload.size()) };
                float pred[1] = { 0.0f };
                elm->Predict(std::span<const float>(x, 2), std::span<float>(pred, 1));
                if (pred[0] > 1000.0f) { // Threshold for system stall
                    if (msg.result_ptr) {
                        *static_cast<Status*>(msg.result_ptr) = Status::ResourceExhausted("ELM predicted system timeout under load");
                    }
                    return; // Drop message (Load shedding)
                }
            }
            queue_.push_back(std::move(msg));
        }

        /** 
         * Synchronously execute one message from the queue. 
         * Returns true if a message was processed.
         */
        bool DispatchOne() {
            Message msg;
            {
                std::lock_guard<std::recursive_mutex> lock(mu_);
                if (queue_.empty()) return false;

                msg = std::move(queue_.front());
                queue_.pop_front();
            }

            std::lock_guard<std::recursive_mutex> lock(mu_);
            auto it = pods_.find(msg.target);
            if (it != pods_.end()) {
                // Check memory quota before execution
                auto quota = it->second->GetQuota();
                if (quota.is_exceeded()) {
                     // TODO: Log warning or reject if strict
                }
                
                it->second->Handle(std::move(msg));
            } else {
                // Target pod not found
                // TODO: Handle orphaned messages
            }

            return true;
        }

        /** Drain the entire queue. */
        void ProcessAll() {
            while (DispatchOne());
        }

        /** Shutdown all pods. */
        void Stop() {
            for (auto& kv : pods_) {
                kv.second->OnStop();
            }
            pods_.clear();
            queue_.clear();
        }

        /** Direct access to a pod (use sparingly, prefer message passing). */
        Pod* GetPod(PodId id) {
            auto it = pods_.find(id);
            return (it != pods_.end()) ? it->second.get() : nullptr;
        }

    private:
        mutable std::recursive_mutex mu_;
        std::unordered_map<PodId, std::unique_ptr<Pod>> pods_;
        std::deque<Message> queue_;
    };

} // namespace pomai::core
