#pragma once

#include "pod.h"
#include "vector_engine.h"

namespace pomai::core {

    /**
     * VectorPod: Wraps VectorEngine to provide Pod capabilities.
     */
    class VectorPod : public Pod {
    public:
        explicit VectorPod(std::unique_ptr<VectorEngine> engine)
            : engine_(std::move(engine)) {}

        void Handle(Message&& msg) override;

        PodId Id() const override { return PodId::kIndex; }
        std::string Name() const override { return "VectorService"; }

        MemoryQuota GetQuota() const override {
            MemoryQuota q;
            q.used_bytes = engine_ ? engine_->MemTableBytesUsed() : 0;
            q.max_bytes = 0; // Configured at kernel level
            return q;
        }

        std::size_t GetMemTableBytesUsed() const {
            return engine_ ? engine_->MemTableBytesUsed() : 0;
        }

        void OnStart() override {
            if (engine_) (void)engine_->Open();
        }
        void OnStop() override {
            if (engine_) (void)engine_->Close();
        }

    private:
        std::unique_ptr<VectorEngine> engine_;
    };

} // namespace pomai::core
