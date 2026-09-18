# Real-Time Log Anomaly Detector - Chaos Test Edition

## 🎯 Purpose

This is a **chaos test** application that simulates a real-time log anomaly detection system while deliberately pushing `pomaidb` to its limits. The application acts as a "chaotic normal user" who:

- Makes naive assumptions about the API
- Passes dirty/invalid inputs (NaN, Inf, zeros, unnormalized vectors)
- Uses erratic lifecycle sequences (query before insert, rapid close/reopen)
- Applies memory pressure with weird batch sizes
- Tortures the delete/re-insert functionality
- Tests boundary conditions and edge cases

This is **not** a production-ready application—it's a forensic tool to discover crashes, hangs, assertion failures, and logic flaws in the pomaidb engine.

## 🔥 Chaos Test Coverage

The application runs 12 different chaos tests:

1. **Query Before Insert**: Queries the database before any vectors are inserted
2. **Dirty Vector Insertion**: Inserts vectors containing NaN, Inf, zeros, and unnormalized values
3. **Empty Batch**: Attempts to insert a batch with count=0
4. **Oversized Batch**: Inserts a batch larger than the MemTable capacity
5. **Rapid Close/Reopen**: Opens, immediately closes, and reopens the database
6. **Delete/Re-insert Torture**: Deletes vectors and re-inserts them with different data
7. **Rapid Write Hammer**: Hammers the database with rapid sequential writes without sleeping
8. **Empty Output Buffers**: Queries with null/empty output buffers
9. **Freeze Before Data**: Attempts to checkpoint a database with no data
10. **Concurrent Queries**: Launches multiple threads querying without external locks
11. **Double-Close Handle**: Attempts to close the database handle twice (commented to prevent crash)
12. **Weird Batch Sizes**: Inserts batches with sizes like 1, 13337, 42, 7, 999, etc.

## 🏗️ Building the Application

### Prerequisites

1. **pomaidb must be built first** in the parent directory
2. C++20 compatible compiler (GCC, Clang, or MSVC)
3. CMake 3.20+ (optional, for CMake build)
4. Make (optional, for Makefile build)

### Step 1: Build pomaidb

First, build the main pomaidb library:

```bash
cd E:\GithubProjects\pomaidb
mkdir build
cd build
cmake ..
cmake --build .
```

### Step 2: Build the chaos test

#### Option A: Using CMake (Recommended)

```bash
cd examples\realtime_log_anomaly_detector
mkdir build
cd build
cmake ..
cmake --build .
```

#### Option B: Using Makefile (Linux/Mac)

```bash
cd examples/realtime_log_anomaly_detector
make
```

#### Option C: Using Windows Batch Script

```batch
cd examples\realtime_log_anomaly_detector
build.bat
```

#### Option D: Manual Compilation

If you prefer manual compilation:

**Linux/Mac with GCC:**
```bash
g++ -std=c++20 -I../../include -I../../src -I../../src/capi -I../../src/utils -I../../third_party/palloc/include chaos_anomaly_detector.cpp -o chaos_anomaly_detector -L../../build -lpomai -lpomai_c -lpalloc-static -lpthread
```

**Windows with MSVC:**
```batch
cl.exe /EHsc /std:c++20 /I..\..\include /I..\..\src /I..\..\src\capi /I..\..\src\utils /I..\..\third_party\palloc\include chaos_anomaly_detector.cpp /Fe:chaos_anomaly_detector.exe /link /LIBPATH:..\..\build pomai.lib pomai_c.lib palloc-static.lib ws2_32.lib
```

**Windows with MinGW:**
```batch
g++ -std=c++20 -I../../include -I../../src -I../../src/capi -I../../src/utils -I../../third_party/palloc/include chaos_anomaly_detector.cpp -o chaos_anomaly_detector.exe -L../../build -lpomai -lpomai_c -lpalloc-static -lws2_32
```

## 🚀 Running the Chaos Test

### Basic Execution

```bash
# Linux/Mac
./chaos_anomaly_detector

# Windows
chaos_anomaly_detector.exe
```

### Expected Output

The application will:

1. Print a header indicating the start of chaos testing
2. Run each of the 12 chaos tests sequentially
3. Log each test with timestamp and result (PASS/FAIL/EXPECTED)
4. Print final statistics (total vectors, memtable state, segments, etc.)
5. Complete with a summary message

### Sample Output

```
=== POMAIDB CHAOS TEST: Real-Time Log Anomaly Detector ===
This test deliberately pushes pomaidb to its limits with chaotic inputs
and erratic usage patterns. Crashes, hangs, or unexpected behavior
will be logged as forensic evidence.

[CHAOS 2026-09-14 19:30:00] START - Beginning chaos test sequence
[CHAOS 2026-09-14 19:30:00] TEST1 - Querying database before any vectors inserted
[CHAOS 2026-09-14 19:30:00] TEST1 PASS - Query succeeded before insert (might return empty results)
[CHAOS 2026-09-14 19:30:00] TEST2 - Inserting vectors with NaN, Inf, zeros, and unnormalized values
[CHAOS 2026-09-14 19:30:01] TEST2 PASS - Dirty vector insertion succeeded
...
=== FINAL STATISTICS ===
Total vectors: 15234
Memtable vectors: 8912
Sealed segments: 3
Arena allocated: 45678912 bytes
Arena committed: 23456789 bytes
Current LSN: 15234
[CHAOS 2026-09-14 19:30:15] CLEANUP - Closing database
[CHAOS 2026-09-14 19:30:15] COMPLETE - Chaos test sequence completed

=== CHAOS TEST COMPLETE ===
Review the log above for any crashes, hangs, or unexpected behavior.
```

## 🔍 Forensic Analysis

### What to Look For

When running the chaos test, watch for:

- **Segfaults/Crashes**: The application terminates unexpectedly
- **Hangs**: The application freezes and doesn't progress
- **Assertion Failures**: Internal pomaidb assertions fail
- **Memory Leaks**: Increasing memory usage over time
- **Silent Data Loss**: Vectors inserted but not found in queries
- **Wrong Results**: Query returns incorrect nearest neighbors
- **Status Code Failures**: API returns unexpected error codes

### Logging

Each chaos test logs:
- Timestamp
- Test name
- What action is being performed
- Result (PASS/FAIL/EXPECTED/UNEXPECTED)

This forensic logging helps identify exactly which sequence of actions broke the engine.

## 🛠️ Configuration

You can modify the chaos parameters at the top of `chaos_anomaly_detector.cpp`:

```cpp
const size_t VECTOR_DIM = 128;              // Vector dimensionality
const size_t MEMTABLE_CAPACITY = 10000;    // MemTable capacity for triggering flushes
const char* DB_PATH = "./chaos_anomaly_db"; // Database storage path
```

## 🧹 Cleanup

To clean up after testing:

```bash
# Remove the executable
rm chaos_anomaly_detector        # Linux/Mac
del chaos_anomaly_detector.exe  # Windows

# Remove the database directory
rm -rf chaos_anomaly_db         # Linux/Mac
rmdir /s chaos_anomaly_db       # Windows
```

## 📊 Test Results Interpretation

### PASS
The test completed successfully without crashing or returning unexpected errors.

### FAIL  
The test failed with an unexpected error code or crash.

### EXPECTED
The test failed, but this is expected behavior (e.g., empty batch insertion should fail).

### UNEXPECTED
The test succeeded when it was expected to fail (e.g., null buffer query succeeded).

### WARN
The test completed but with warnings (e.g., some concurrent queries failed).

## 🐛 Reporting Issues

If you discover a crash, hang, or logic flaw:

1. Copy the full output from the chaos test
2. Note which specific test caused the issue
3. Include your platform (OS, compiler, pomaidb version)
4. Report the issue to the pomaidb maintainers

## 🎓 Educational Value

This chaos test demonstrates:

- **Real-world usage patterns** of the pomaidb C API
- **Common mistakes** developers make when using vector databases
- **Edge cases** that vector database engines should handle gracefully
- **Importance of defensive programming** in database engines
- **Value of chaos testing** for finding obscure bugs

## ⚠️ Disclaimer

This is a **chaos test tool**, not a production application. The code deliberately:

- Uses invalid inputs that should be rejected
- Follows incorrect API usage patterns
- Attempts operations that may cause undefined behavior
- Does not represent best practices for using pomaidb

For production usage, refer to the official pomaidb examples in `examples/python/`, `examples/javascript/`, `examples/rust/`, and `examples/go/`.

## 📝 Technical Details

- **Language**: C++20
- **API**: pomaidb C API (`pomaidb.h`)
- **Vector Dimension**: 128 (configurable)
- **Distance Metric**: L2 (Euclidean)
- **Threading**: Uses std::thread for concurrent query test
- **Randomness**: Uses C++11 random number generation for dirty vectors

## 🔗 Related Examples

For proper pomaidb usage examples, see:
- `examples/python/` - Python API example
- `examples/javascript/` - JavaScript/Node.js example  
- `examples/rust/` - Rust API example
- `examples/go/` - Go API example

---

**Remember**: This is a chaos test designed to break things. Use it to find bugs, not as a template for production code! 🚀
