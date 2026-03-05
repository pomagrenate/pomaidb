#pragma once
#include <memory>
#include <vector>
#include "pomai/snapshot.h"
#include "core/shard/snapshot.h"

namespace pomai::core {
    
    // Concrete implementation of public opaque Snapshot.
    class SnapshotWrapper : public pomai::Snapshot {
    public:
        explicit SnapshotWrapper(std::shared_ptr<VectorSnapshot> s) : snap_(std::move(s)) {}
        
        std::shared_ptr<VectorSnapshot> GetInternal() const { return snap_; }

    private:
        std::shared_ptr<VectorSnapshot> snap_;
    };

} // namespace pomai::core
