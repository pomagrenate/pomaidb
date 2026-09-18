// sigbus_handler.cc — Cross-platform SIGBUS/EXCEPTION_IN_PAGE_ERROR protection implementation
//
// Copyright 2026 PomaiDB authors. MIT License.

#include "sigbus_handler.h"
#include <iostream>

#if defined(_WIN32) || defined(_WIN64)

namespace pomai::util {

SigBusGuard::SigBusGuard() {
    caught_signal_ = false;
}

SigBusGuard::~SigBusGuard() {
    // Windows uses SEH, no cleanup needed
}

} // namespace pomai::util

#else

#include <cstring>
#include <unistd.h>

namespace pomai::util {

std::atomic<SigBusGuard*> SigBusGuard::active_guard_{nullptr};
std::jmp_buf SigBusGuard::jump_buffer_;
struct sigaction SigBusGuard::old_action_;

void SigBusGuard::SignalHandler(int sig, siginfo_t* info, void* context) {
    (void)context; // Unused
    
    if (sig == SIGBUS || sig == SIGSEGV) {
        SigBusGuard* guard = active_guard_.load(std::memory_order_acquire);
        if (guard) {
            guard->caught_signal_ = true;
            std::longjmp(jump_buffer_, 1);
        }
    }
    
    // If no active guard, call the old handler
    if (old_action_.sa_handler) {
        if (old_action_.sa_flags & SA_SIGINFO) {
            old_action_.sa_sigaction(sig, info, context);
        } else {
            old_action_.sa_handler(sig);
        }
    } else {
        // Default action: terminate
        std::signal(sig, SIG_DFL);
        std::raise(sig);
    }
}

SigBusGuard::SigBusGuard() {
    caught_signal_ = false;
    
    // Install signal handler
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = SignalHandler;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    
    // Save old action
    sigaction(SIGBUS, nullptr, &old_action_);
    
    // Install new handler
    if (sigaction(SIGBUS, &sa, nullptr) != 0) {
        std::cerr << "Warning: Failed to install SIGBUS handler" << std::endl;
    }
    
    // Also protect against SIGSEGV (can occur on some systems)
    struct sigaction sa_segv;
    std::memset(&sa_segv, 0, sizeof(sa_segv));
    sa_segv.sa_sigaction = SignalHandler;
    sa_segv.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&sa_segv.sa_mask);
    sigaction(SIGSEGV, &sa_segv, nullptr);
    
    active_guard_.store(this, std::memory_order_release);
}

SigBusGuard::~SigBusGuard() {
    active_guard_.store(nullptr, std::memory_order_release);
    
    // Restore old handler
    sigaction(SIGBUS, &old_action_, nullptr);
}

} // namespace pomai::util

#endif