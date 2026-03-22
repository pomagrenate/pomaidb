#pragma once

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/kernel/message.h"
#include "core/kernel/pod.h"
#include "pomai/status.h"

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
            PodId id = pod->Id();
            if (pods_.count(id)) return Status::AlreadyExists("pod already registered");
            
            pod->OnStart();
            pods_[id] = std::move(pod);
            return Status::Ok();
        }

        /** Unregister and stop a pod. */
        void UnregisterPod(PodId id) {
            auto it = pods_.find(id);
            if (it != pods_.end()) {
                it->second->OnStop();
                pods_.erase(it);
            }
        }

        /** Post a message for later execution. */
        void Enqueue(Message&& msg) {
            queue_.push_back(std::move(msg));
        }

        /** 
         * Synchronously execute one message from the queue. 
         * Returns true if a message was processed.
         */
        bool DispatchOne() {
            if (queue_.empty()) return false;

            Message msg = std::move(queue_.front());
            queue_.pop_front();

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
        std::unordered_map<PodId, std::unique_ptr<Pod>> pods_;
        std::deque<Message> queue_;
    };

} // namespace pomai::core
