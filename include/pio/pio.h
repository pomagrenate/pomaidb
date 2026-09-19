// include/pio/pio.h — Zero-dependency, crash-safe, hardware-direct I/O library
// Copyright 2026 PomaiDB / pio authors. MIT License.
//
// Standalone high-performance storage and file I/O primitives:
// - Direct sector-aligned buffer allocation (AlignedBuffer)
// - Sequential, random-access, and buffered appendable file abstractions
// - Crash-safe atomic atomic filesystem operations (rename, sync, directory fsync)
// - Zero-copy memory mapped files with platform prefetch/MADV_WILLNEED hints
// - Full Win32 / POSIX transparent platform compatibility

#pragma once

#include "pio_types.h"
#include "pio_status.h"
#include "pio_platform.h"
#include "pio_direct.h"
#include "pio_file.h"
#include "pio_posix.h"
#include "pio_windows.h"
#include "pio_fs.h"
