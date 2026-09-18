// scratch_buffer.h — Thread-local scratch buffers for zero-allocation hot paths
//
// Eliminates heap allocations in critical query paths by providing
// thread-local pre-allocated buffers that are reused across calls.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstddef>
#include <cstring>
#include <vector>
#include <atomic>

namespace pomai::util {

// Thread-local scratch buffer manager
template <typename T>
class ScratchBuffer {
public:
    // Get a thread-local buffer of at least the requested size
    // The buffer is reused across calls and grows as needed
    static T* Get(size_t required_size) {
        thread_local ScratchBuffer instance;
        return instance.GetBuffer(required_size);
    }
    
    // Get the current buffer size (for debugging/monitoring)
    static size_t CurrentSize() {
        thread_local ScratchBuffer instance;
        return instance.buffer_.size();
    }
    
private:
    ScratchBuffer() = default;
    
    T* GetBuffer(size_t required_size) {
        if (buffer_.size() < required_size) {
            buffer_.resize(required_size);
        }
        return buffer_.data();
    }
    
    std::vector<T> buffer_;
};

// Specialized scratch buffers for common types
using FloatScratch = ScratchBuffer<float>;
using UInt8Scratch = ScratchBuffer<uint8_t>;
using UInt16Scratch = ScratchBuffer<uint16_t>;

// RAII guard for scratch buffer usage (optional, for scope tracking)
class ScratchBufferGuard {
public:
    ScratchBufferGuard() : active_(true) {
        ++active_count_;
    }
    
    ~ScratchBufferGuard() {
        if (active_) {
            --active_count_;
        }
    }
    
    ScratchBufferGuard(const ScratchBufferGuard&) = delete;
    ScratchBufferGuard& operator=(const ScratchBufferGuard&) = delete;
    
    static size_t ActiveCount() { return active_count_.load(); }
    
private:
    bool active_;
    static std::atomic<size_t> active_count_;
};

inline std::atomic<size_t> ScratchBufferGuard::active_count_{0};

// Helper function to zero a scratch buffer (for security/privacy)
template <typename T>
void ZeroScratchBuffer(T* buffer, size_t count) {
    std::memset(buffer, 0, count * sizeof(T));
}

} // namespace pomai::util