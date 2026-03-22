#pragma once
#include <string>
#include "core/kernel/message.h"

namespace pomai::core {

    /**
     * Bounded resource info for a Pod.
     */
    struct MemoryQuota {
        size_t used_bytes = 0;
        size_t max_bytes = 0;
        
        bool is_exceeded() const {
            return max_bytes > 0 && used_bytes > max_bytes;
        }
    };

    /**
     * Pod Interface: A single-threaded, message-driven service.
     * All Pods run sequentially in the Kernel's event loop.
     */
    class Pod {
    public:
        virtual ~Pod() = default;

        /** 
         * The heartbeat of a Pod. All logic happens here.
         * Must be non-blocking and return quickly.
         */
        virtual void Handle(Message&& msg) = 0;

        /** Unique identifier for the Pod. */
        virtual PodId Id() const = 0;
        
        /** Descriptive name for logging. */
        virtual std::string Name() const = 0;

        /** Resource monitoring. */
        virtual MemoryQuota GetQuota() const = 0;
        
        /** Lifecycle hooks. */
        virtual void OnStart() {}
        virtual void OnStop() {}
    };

} // namespace pomai::core
