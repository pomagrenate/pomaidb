#pragma once

#include "core/kernel/pod.h"
#include "core/shard/runtime.h"

namespace pomai::core {

    /**
     * VectorPod: Wraps VectorRuntime to provide Pod capabilities.
     */
    class VectorPod : public Pod {
    public:
        explicit VectorPod(std::unique_ptr<VectorRuntime> runtime)
            : runtime_(std::move(runtime)) {}

        void Handle(Message&& msg) override;

        PodId Id() const override { return PodId::kIndex; }
        std::string Name() const override { return "VectorService"; }

        MemoryQuota GetQuota() const override {
            MemoryQuota q;
            q.used_bytes = runtime_->MemTableBytesUsed();
            q.max_bytes = 0; // Configured at kernel level
            return q;
        }

        std::size_t GetMemTableBytesUsed() const {
            return runtime_->MemTableBytesUsed();
        }

        void OnStart() override { (void)runtime_->Start(); }
        void OnStop() override {}

    private:
        std::unique_ptr<VectorRuntime> runtime_;
    };

} // namespace pomai::core
