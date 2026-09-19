#pragma once

// psync_locks.h
// Generic RAII lock guards and wrappers for psync synchronization primitives.
// Fully reusable across any BasicLockable and SharedLockable types.

#include <utility>

namespace psync {

// ============================================================================
// LOCK TAGS
// ============================================================================

struct defer_lock_t { explicit defer_lock_t() = default; };
struct try_to_lock_t { explicit try_to_lock_t() = default; };
struct adopt_lock_t { explicit adopt_lock_t() = default; };

inline constexpr defer_lock_t defer_lock{};
inline constexpr try_to_lock_t try_to_lock{};
inline constexpr adopt_lock_t adopt_lock{};

// ============================================================================
// LOCK GUARD (RAII Scoped Lock)
// ============================================================================

template <typename MutexType>
class LockGuard {
public:
    using mutex_type = MutexType;

    explicit LockGuard(MutexType& m) : mutex_(m) {
        mutex_.lock();
    }

    LockGuard(MutexType& m, adopt_lock_t) noexcept : mutex_(m) {}

    ~LockGuard() {
        mutex_.unlock();
    }

    LockGuard(const LockGuard&) = delete;
    LockGuard& operator=(const LockGuard&) = delete;

private:
    MutexType& mutex_;
};

// ============================================================================
// UNIQUE LOCK (Exclusive Ownership RAII Wrapper)
// ============================================================================

template <typename MutexType>
class UniqueLock {
public:
    using mutex_type = MutexType;

    UniqueLock() noexcept : mutex_(nullptr), owns_(false) {}

    explicit UniqueLock(MutexType& m) : mutex_(&m), owns_(true) {
        mutex_->lock();
    }

    UniqueLock(MutexType& m, defer_lock_t) noexcept : mutex_(&m), owns_(false) {}

    UniqueLock(MutexType& m, try_to_lock_t) : mutex_(&m), owns_(mutex_->try_lock()) {}

    UniqueLock(MutexType& m, adopt_lock_t) noexcept : mutex_(&m), owns_(true) {}

    ~UniqueLock() {
        if (owns_ && mutex_) {
            mutex_->unlock();
        }
    }

    UniqueLock(const UniqueLock&) = delete;
    UniqueLock& operator=(const UniqueLock&) = delete;

    UniqueLock(UniqueLock&& other) noexcept : mutex_(other.mutex_), owns_(other.owns_) {
        other.mutex_ = nullptr;
        other.owns_ = false;
    }

    UniqueLock& operator=(UniqueLock&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (owns_ && mutex_) {
            mutex_->unlock();
        }
        mutex_ = other.mutex_;
        owns_ = other.owns_;
        other.mutex_ = nullptr;
        other.owns_ = false;
        return *this;
    }

    void lock() {
        if (mutex_ && !owns_) {
            mutex_->lock();
            owns_ = true;
        }
    }

    bool try_lock() {
        if (mutex_ && !owns_) {
            owns_ = mutex_->try_lock();
            return owns_;
        }
        return false;
    }

    void unlock() {
        if (mutex_ && owns_) {
            mutex_->unlock();
            owns_ = false;
        }
    }

    MutexType* release() noexcept {
        MutexType* ret = mutex_;
        mutex_ = nullptr;
        owns_ = false;
        return ret;
    }

    bool owns_lock() const noexcept {
        return owns_;
    }

    explicit operator bool() const noexcept {
        return owns_;
    }

    MutexType* mutex() const noexcept {
        return mutex_;
    }

    void swap(UniqueLock& other) noexcept {
        std::swap(mutex_, other.mutex_);
        std::swap(owns_, other.owns_);
    }

private:
    MutexType* mutex_;
    bool owns_;
};

template <typename MutexType>
inline void swap(UniqueLock<MutexType>& lhs, UniqueLock<MutexType>& rhs) noexcept {
    lhs.swap(rhs);
}

// ============================================================================
// SHARED LOCK (Shared Ownership RAII Wrapper)
// ============================================================================

template <typename MutexType>
class SharedLock {
public:
    using mutex_type = MutexType;

    SharedLock() noexcept : mutex_(nullptr), owns_(false) {}

    explicit SharedLock(MutexType& m) : mutex_(&m), owns_(true) {
        mutex_->lock_shared();
    }

    SharedLock(MutexType& m, defer_lock_t) noexcept : mutex_(&m), owns_(false) {}

    SharedLock(MutexType& m, try_to_lock_t) : mutex_(&m), owns_(mutex_->try_lock_shared()) {}

    SharedLock(MutexType& m, adopt_lock_t) noexcept : mutex_(&m), owns_(true) {}

    ~SharedLock() {
        if (owns_ && mutex_) {
            mutex_->unlock_shared();
        }
    }

    SharedLock(const SharedLock&) = delete;
    SharedLock& operator=(const SharedLock&) = delete;

    SharedLock(SharedLock&& other) noexcept : mutex_(other.mutex_), owns_(other.owns_) {
        other.mutex_ = nullptr;
        other.owns_ = false;
    }

    SharedLock& operator=(SharedLock&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (owns_ && mutex_) {
            mutex_->unlock_shared();
        }
        mutex_ = other.mutex_;
        owns_ = other.owns_;
        other.mutex_ = nullptr;
        other.owns_ = false;
        return *this;
    }

    void lock() {
        if (mutex_ && !owns_) {
            mutex_->lock_shared();
            owns_ = true;
        }
    }

    bool try_lock() {
        if (mutex_ && !owns_) {
            owns_ = mutex_->try_lock_shared();
            return owns_;
        }
        return false;
    }

    void unlock() {
        if (mutex_ && owns_) {
            mutex_->unlock_shared();
            owns_ = false;
        }
    }

    MutexType* release() noexcept {
        MutexType* ret = mutex_;
        mutex_ = nullptr;
        owns_ = false;
        return ret;
    }

    bool owns_lock() const noexcept {
        return owns_;
    }

    explicit operator bool() const noexcept {
        return owns_;
    }

    MutexType* mutex() const noexcept {
        return mutex_;
    }

    void swap(SharedLock& other) noexcept {
        std::swap(mutex_, other.mutex_);
        std::swap(owns_, other.owns_);
    }

private:
    MutexType* mutex_;
    bool owns_;
};

template <typename MutexType>
inline void swap(SharedLock<MutexType>& lhs, SharedLock<MutexType>& rhs) noexcept {
    lhs.swap(rhs);
}

} // namespace psync
