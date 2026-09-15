// palloc_smart_ptr.h — Palloc-based smart pointer wrappers
//
// Zero-dependency, bare-metal memory architecture replacements for
// std::unique_ptr and std::shared_ptr using palloc memory subsystem.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <atomic>
#include <cstddef>
#include <utility>
#include <type_traits>
#include "palloc_compat.h"

namespace pomai::alloc {

// ============================================================================
// UniquePtr — Exclusive ownership wrapper (replaces std::unique_ptr)
// ============================================================================

template <typename T>
class UniquePtr {
public:
    using element_type = T;
    using pointer = T*;

    template <typename U>
    friend class UniquePtr;

    // Constructors
    constexpr UniquePtr() noexcept : ptr_(nullptr), is_palloc_owned_(false) {}
    constexpr UniquePtr(std::nullptr_t) noexcept : ptr_(nullptr), is_palloc_owned_(false) {}

    explicit UniquePtr(T* ptr) noexcept : ptr_(ptr), is_palloc_owned_(false) {}

    // Constructor for externally-allocated pointers (e.g., from upstream libraries)
    // These will be deleted with operator delete, not palloc_free
    explicit UniquePtr(T* ptr, bool is_palloc_owned) noexcept
        : ptr_(ptr), is_palloc_owned_(is_palloc_owned) {}

    // Move constructor/assignment
    UniquePtr(UniquePtr&& other) noexcept : ptr_(other.ptr_), is_palloc_owned_(other.is_palloc_owned_) {
        other.ptr_ = nullptr;
        other.is_palloc_owned_ = false;
    }

    // Move from different type (base class conversion)
    template <typename U>
    UniquePtr(UniquePtr<U>&& other) noexcept
        : ptr_(other.release()), is_palloc_owned_(other.is_palloc_owned_release()) {
    }

    UniquePtr& operator=(UniquePtr&& other) noexcept {
        if (this != &other) {
            reset(other.ptr_, other.is_palloc_owned_);
            other.ptr_ = nullptr;
            other.is_palloc_owned_ = false;
        }
        return *this;
    }

    // Move assignment from different type
    template <typename U>
    UniquePtr& operator=(UniquePtr<U>&& other) noexcept {
        reset(other.release(), other.is_palloc_owned_release());
        return *this;
    }

    // Disallow copy
    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(const UniquePtr&) = delete;

    // Destructor
    ~UniquePtr() {
        reset();
    }

    // Accessors
    T* get() const noexcept { return ptr_; }
    T& operator*() const noexcept { return *ptr_; }
    T* operator->() const noexcept { return ptr_; }

    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    // Comparison with nullptr
    bool operator==(std::nullptr_t) const noexcept { return ptr_ == nullptr; }
    bool operator!=(std::nullptr_t) const noexcept { return ptr_ != nullptr; }

    // Modifiers
    T* release() noexcept {
        T* tmp = ptr_;
        ptr_ = nullptr;
        is_palloc_owned_ = false;
        return tmp;
    }

    bool is_palloc_owned_release() noexcept {
        bool tmp = is_palloc_owned_;
        is_palloc_owned_ = false;
        return tmp;
    }

    void reset(T* ptr = nullptr) noexcept {
        reset(ptr, false);
    }

    void reset(T* ptr, bool is_palloc_owned) noexcept {
        if (ptr_) {
            if (is_palloc_owned_) {
                ptr_->~T();
                palloc_free(ptr_);
            } else {
                delete ptr_;
            }
        }
        ptr_ = ptr;
        is_palloc_owned_ = is_palloc_owned;
    }

    void swap(UniquePtr& other) noexcept {
        T* tmp_ptr = ptr_;
        bool tmp_owned = is_palloc_owned_;
        ptr_ = other.ptr_;
        is_palloc_owned_ = other.is_palloc_owned_;
        other.ptr_ = tmp_ptr;
        other.is_palloc_owned_ = tmp_owned;
    }

    // Palloc-specific factory methods
    template <typename... Args>
    static UniquePtr Make(palloc_heap_t* heap, Args&&... args) {
        void* mem = heap ? palloc_heap_malloc(heap, sizeof(T), alignof(T))
                        : palloc_malloc(sizeof(T), alignof(T));
        if (!mem) return UniquePtr();
        T* obj = new (mem) T(std::forward<Args>(args)...);
        return UniquePtr(obj, true);  // Mark as palloc-owned
    }

    template <typename... Args>
    static UniquePtr MakeAligned(palloc_heap_t* heap, size_t alignment, Args&&... args) {
        void* mem = heap ? palloc_heap_malloc_aligned(heap, sizeof(T), alignment)
                        : palloc_malloc_aligned(sizeof(T), alignment);
        if (!mem) return UniquePtr();
        T* obj = new (mem) T(std::forward<Args>(args)...);
        return UniquePtr(obj, true);  // Mark as palloc-owned
    }

    // Adopt externally-allocated pointer (from new, malloc, etc.)
    // Will be deleted with delete, not palloc_free
    static UniquePtr Adopt(T* ptr) noexcept {
        return UniquePtr(ptr, false);
    }

    // Adopt palloc-allocated pointer (from palloc_malloc)
    // Will be destroyed with palloc_free
    static UniquePtr AdoptPalloc(T* ptr) noexcept {
        return UniquePtr(ptr, true);
    }

private:
    T* ptr_;
    bool is_palloc_owned_;
};

// ============================================================================
// SharedPtr — Shared ownership wrapper (replaces std::shared_ptr)
// ============================================================================

template <typename T>
class SharedPtr {
public:
    using element_type = T;
    using pointer = T*;

    // Control block for reference counting
    struct ControlBlock {
        std::atomic<int> ref_count;
        std::atomic<int> weak_count;
        palloc_heap_t* heap;
        bool is_palloc_owned;  // Track if the object is palloc-owned
        
        ControlBlock(palloc_heap_t* h, bool is_palloc) : ref_count(1), weak_count(1), heap(h), is_palloc_owned(is_palloc) {}
        
        void add_ref() noexcept {
            ref_count.fetch_add(1, std::memory_order_relaxed);
        }
        
        bool release() noexcept {
            if (ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                // Last strong reference
                return true;
            }
            return false;
        }
        
        void add_weak_ref() noexcept {
            weak_count.fetch_add(1, std::memory_order_relaxed);
        }
        
        void release_weak() noexcept {
            if (weak_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                // Last weak reference, free control block
                palloc_free(this);
            }
        }
    };

    // Constructors
    constexpr SharedPtr() noexcept : ptr_(nullptr), ctrl_(nullptr) {}
    constexpr SharedPtr(std::nullptr_t) noexcept : ptr_(nullptr), ctrl_(nullptr) {}
    
    explicit SharedPtr(T* ptr, ControlBlock* ctrl) noexcept : ptr_(ptr), ctrl_(ctrl) {}
    
    // Copy constructor/assignment
    SharedPtr(const SharedPtr& other) noexcept : ptr_(other.ptr_), ctrl_(other.ctrl_) {
        if (ctrl_) ctrl_->add_ref();
    }
    
    SharedPtr& operator=(const SharedPtr& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            ctrl_ = other.ctrl_;
            if (ctrl_) ctrl_->add_ref();
        }
        return *this;
    }
    
    // Move constructor/assignment
    SharedPtr(SharedPtr&& other) noexcept : ptr_(other.ptr_), ctrl_(other.ctrl_) {
        other.ptr_ = nullptr;
        other.ctrl_ = nullptr;
    }
    
    SharedPtr& operator=(SharedPtr&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            ctrl_ = other.ctrl_;
            other.ptr_ = nullptr;
            other.ctrl_ = nullptr;
        }
        return *this;
    }
    
    // Destructor
    ~SharedPtr() {
        reset();
    }
    
    // Accessors
    T* get() const noexcept { return ptr_; }
    T& operator*() const noexcept { return *ptr_; }
    T* operator->() const noexcept { return ptr_; }
    
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    // Comparison with nullptr
    bool operator==(std::nullptr_t) const noexcept { return ptr_ == nullptr; }
    bool operator!=(std::nullptr_t) const noexcept { return ptr_ != nullptr; }

    int use_count() const noexcept {
        return ctrl_ ? ctrl_->ref_count.load(std::memory_order_relaxed) : 0;
    }
    
    // Modifiers
    void reset() noexcept {
        if (ctrl_ && ctrl_->release()) {
            // Destroy object
            if (ptr_) {
                ptr_->~T();
                if (ctrl_->is_palloc_owned) {
                    palloc_free(const_cast<void*>(static_cast<const void*>(ptr_)));
                } else {
                    delete const_cast<T*>(ptr_);
                }
            }
            ctrl_->release_weak();
        }
        ptr_ = nullptr;
        ctrl_ = nullptr;
    }
    
    void swap(SharedPtr& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(ctrl_, other.ctrl_);
    }
    
    // Palloc-specific factory methods
    template <typename... Args>
    static SharedPtr Make(palloc_heap_t* heap, Args&&... args) {
        // Allocate control block
        void* ctrl_mem = palloc_malloc(sizeof(ControlBlock), alignof(ControlBlock));
        if (!ctrl_mem) return SharedPtr();

        ControlBlock* ctrl = new (ctrl_mem) ControlBlock(heap, true);

        // Allocate object
        void* obj_mem = heap ? palloc_heap_malloc(heap, sizeof(T), alignof(T))
                            : palloc_malloc(sizeof(T), alignof(T));
        if (!obj_mem) {
            ctrl->~ControlBlock();
            palloc_free(ctrl);
            return SharedPtr();
        }

        T* obj = new (obj_mem) T(std::forward<Args>(args)...);
        return SharedPtr(obj, ctrl);
    }

    template <typename... Args>
    static SharedPtr MakeAligned(palloc_heap_t* heap, size_t alignment, Args&&... args) {
        // Allocate control block
        void* ctrl_mem = palloc_malloc(sizeof(ControlBlock), alignof(ControlBlock));
        if (!ctrl_mem) return SharedPtr();

        ControlBlock* ctrl = new (ctrl_mem) ControlBlock(heap, true);

        // Allocate object with alignment
        void* obj_mem = heap ? palloc_heap_malloc_aligned(heap, sizeof(T), alignment)
                            : palloc_malloc_aligned(sizeof(T), alignment);
        if (!obj_mem) {
            ctrl->~ControlBlock();
            palloc_free(ctrl);
            return SharedPtr();
        }

        T* obj = new (obj_mem) T(std::forward<Args>(args)...);
        return SharedPtr(obj, ctrl);
    }

    // Adopt externally-allocated pointer (from new, malloc, etc.)
    // Will be deleted with delete, not palloc_free
    static SharedPtr Adopt(T* ptr) noexcept {
        // Allocate control block
        void* ctrl_mem = palloc_malloc(sizeof(ControlBlock), alignof(ControlBlock));
        if (!ctrl_mem) return SharedPtr();

        ControlBlock* ctrl = new (ctrl_mem) ControlBlock(nullptr, false);
        return SharedPtr(ptr, ctrl);
    }

    // Adopt palloc-allocated pointer (from palloc_malloc)
    // Will be destroyed with palloc_free
    static SharedPtr AdoptPalloc(T* ptr) noexcept {
        // Allocate control block
        void* ctrl_mem = palloc_malloc(sizeof(ControlBlock), alignof(ControlBlock));
        if (!ctrl_mem) return SharedPtr();

        ControlBlock* ctrl = new (ctrl_mem) ControlBlock(nullptr, true);
        return SharedPtr(ptr, ctrl);
    }

private:
    T* ptr_;
    ControlBlock* ctrl_;
};

// ============================================================================
// Utility functions
// ============================================================================

template <typename T, typename U>
bool operator==(const UniquePtr<T>& lhs, const UniquePtr<U>& rhs) noexcept {
    return lhs.get() == rhs.get();
}

template <typename T, typename U>
bool operator!=(const UniquePtr<T>& lhs, const UniquePtr<U>& rhs) noexcept {
    return lhs.get() != rhs.get();
}

template <typename T, typename U>
bool operator==(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept {
    return lhs.get() == rhs.get();
}

template <typename T, typename U>
bool operator!=(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept {
    return lhs.get() != rhs.get();
}

} // namespace pomai::alloc
