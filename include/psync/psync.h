#pragma once

// psync.h
// Umbrella header for the psync zero-dependency synchronization library.
// Provides native futex-backed Mutex, SharedMutex, ConditionVariable,
// generic RAII locks (LockGuard, UniqueLock, SharedLock), OnceFlag, call_once,
// MicroLock, SeqLock, Batch, and TaggedPtr.

#include "psync_platform.h"
#include "psync_locks.h"
#include "psync_mutex.h"
#include "psync_shared.h"
#include "psync_condvar.h"
#include "psync_once.h"
#include "psync_microlock.h"
#include "psync_seqlock.h"
#include "psync_batch.h"
#include "psync_tagged_ptr.h"
#include "psync_cache_padded.h"
#include "psync_baton.h"
#include "psync_latch.h"
#include "psync_semaphore.h"
#include "psync_spinlock.h"
#include "psync_barrier.h"
#include "psync_mpsc_queue.h"
#include "psync_mpsc_intrusive.h"
#include "psync_spsc_queue.h"
#include "psync_epoch.h"

