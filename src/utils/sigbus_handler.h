// sigbus_handler.h — Cross-platform SIGBUS/EXCEPTION_IN_PAGE_ERROR protection for mmap reads
//
// Protects against crashes when memory-mapped segment files are truncated or deleted
// while mapped. Provides graceful error handling instead of process termination.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstdint>
#include <stdexcept>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <excpt.h>
#else
#include <csignal>
#include <csetjmp>
#include <atomic>
#endif

namespace pomai::util {

class SigBusException : public std::runtime_error {
public:
    SigBusException(const char* msg) : std::runtime_error(msg) {}
};

// Cross-platform SIGBUS protection wrapper
class SigBusGuard {
public:
    SigBusGuard();
    ~SigBusGuard();
    
    // Returns true if a SIGBUS was caught during the protected region
    bool CaughtSignal() const { return caught_signal_; }
    
    // Reset the signal state
    void Reset() { caught_signal_ = false; }
    
private:
    bool caught_signal_{false};
    
#if !defined(_WIN32) && !defined(_WIN64)
    static std::atomic<SigBusGuard*> active_guard_;
    static std::jmp_buf jump_buffer_;
    static struct sigaction old_action_;
    
    static void SignalHandler(int sig, siginfo_t* info, void* context);
#endif
};

// RAII wrapper for protected memory access
template <typename Fn>
auto WithSigBusProtection(Fn&& fn) -> decltype(fn()) {
    SigBusGuard guard;
    try {
        return fn();
    } catch (const SigBusException& e) {
        // Log the error and return appropriate error handling
        // For now, re-throw to let the caller handle it
        throw;
    }
}

} // namespace pomai::util