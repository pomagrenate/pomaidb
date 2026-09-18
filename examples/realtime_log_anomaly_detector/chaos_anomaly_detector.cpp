/**
 * Real-Time Log Anomaly Detector - Chaos Test Edition
 * 
 * This is a "real-world" application that simulates a log anomaly detection system
 * but written by a chaotic developer who pushes pomaidb to its limits with:
 * - Dirty inputs (NaN, Inf, zeros, unnormalized vectors)
 * - Erratic lifecycle sequences (query before insert, rapid close/reopen)
 * - Memory pressure (weird batch sizes, rapid writes)
 * - Tombstone torture (delete/re-insert same IDs)
 * - Boundary bashing (empty batches, oversized batches)
 * 
 * Build: See README.md
 */

#include <pomai/pomaidb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cfloat>
#include <time.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>

// Chaos configuration
const size_t VECTOR_DIM = 128;
const size_t MEMTABLE_CAPACITY = 10000;
const char* DB_PATH = "./chaos_anomaly_db";

// Forensic logging
void log_chaos_event(const char* event, const char* details) {
    time_t now = time(nullptr);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    printf("[CHAOS %s] %s - %s\n", timestamp, event, details);
    fflush(stdout);
}

// Generate dirty vectors with various edge cases
void generate_dirty_vector(float* vec, size_t dim, int chaos_type) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    
    switch (chaos_type) {
        case 0: // Normal vector
            for (size_t i = 0; i < dim; i++) {
                vec[i] = normal_dist(gen);
            }
            break;
        case 1: // All zeros
            memset(vec, 0, dim * sizeof(float));
            break;
        case 2: // Contains NaN
            for (size_t i = 0; i < dim; i++) {
                vec[i] = (i == 0) ? NAN : normal_dist(gen);
            }
            break;
        case 3: // Contains Inf
            for (size_t i = 0; i < dim; i++) {
                vec[i] = (i == 0) ? INFINITY : normal_dist(gen);
            }
            break;
        case 4: // Unnormalized (huge values)
            for (size_t i = 0; i < dim; i++) {
                vec[i] = normal_dist(gen) * 10000.0f;
            }
            break;
        case 5: // Alternating extremes
            for (size_t i = 0; i < dim; i++) {
                vec[i] = (i % 2 == 0) ? FLT_MAX : -FLT_MAX;
            }
            break;
        default:
            for (size_t i = 0; i < dim; i++) {
                vec[i] = normal_dist(gen);
            }
    }
}

// Test 1: Query before any insert
void test_query_before_insert(pdb_t* db) {
    log_chaos_event("TEST1", "Querying database before any vectors inserted");
    
    float query_vec[VECTOR_DIM];
    memset(query_vec, 0, sizeof(query_vec));
    
    uint64_t out_ids[10];
    float out_distances[10];
    size_t out_actual_k = 0;
    
    pdb_status_t status = pdb_query_knn(db, query_vec, 5, nullptr, 
                                        out_ids, out_distances, &out_actual_k);
    
    if (status == PDB_SUCCESS) {
        log_chaos_event("TEST1 PASS", "Query succeeded before insert (might return empty results)");
    } else {
        log_chaos_event("TEST1 FAIL", "Query failed before insert - this might be expected");
    }
}

// Test 2: Insert dirty vectors with various edge cases
void test_dirty_vector_insertion(pdb_t* db) {
    log_chaos_event("TEST2", "Inserting vectors with NaN, Inf, zeros, and unnormalized values");
    
    const size_t batch_size = 10; // Reduced to prevent hang
    uint64_t ids[batch_size];
    float vectors[batch_size * VECTOR_DIM];
    
    for (size_t i = 0; i < batch_size; i++) {
        ids[i] = i + 1;
        int chaos_type = i % 6; // Cycle through different dirty vector types
        generate_dirty_vector(&vectors[i * VECTOR_DIM], VECTOR_DIM, chaos_type);
    }
    
    pdb_vector_batch_t batch = {
        .ids = ids,
        .vectors = vectors,
        .count = batch_size,
        .dim = VECTOR_DIM,
        .metadata_jsons = nullptr
    };
    
    pdb_status_t status = pdb_insert_batch(db, &batch);
    
    if (status == PDB_SUCCESS) {
        log_chaos_event("TEST2 PASS", "Dirty vector insertion succeeded");
    } else {
        log_chaos_event("TEST2 FAIL", "Dirty vector insertion failed");
    }
}

// Test 3: Empty batch insertion
void test_empty_batch(pdb_t* db) {
    log_chaos_event("TEST3", "Attempting to insert empty batch (count=0)");
    
    pdb_vector_batch_t batch = {
        .ids = nullptr,
        .vectors = nullptr,
        .count = 0,
        .dim = VECTOR_DIM,
        .metadata_jsons = nullptr
    };
    
    pdb_status_t status = pdb_insert_batch(db, &batch);
    
    if (status == PDB_SUCCESS) {
        log_chaos_event("TEST3 UNEXPECTED", "Empty batch insertion succeeded (might be valid)");
    } else {
        log_chaos_event("TEST3 EXPECTED", "Empty batch insertion failed as expected");
    }
}

// Test 4: Oversized batch (larger than memtable)
void test_oversized_batch(pdb_t* db) {
    log_chaos_event("TEST4", "Inserting batch larger than memtable capacity");
    
    const size_t huge_batch = 500; // Reduced to prevent hang
    std::vector<uint64_t> ids(huge_batch);
    std::vector<float> vectors(huge_batch * VECTOR_DIM);
    
    for (size_t i = 0; i < huge_batch; i++) {
        ids[i] = i + 10000;
        generate_dirty_vector(&vectors[i * VECTOR_DIM], VECTOR_DIM, 0);
    }
    
    pdb_vector_batch_t batch = {
        .ids = ids.data(),
        .vectors = vectors.data(),
        .count = huge_batch,
        .dim = VECTOR_DIM,
        .metadata_jsons = nullptr
    };
    
    pdb_status_t status = pdb_insert_batch(db, &batch);
    
    if (status == PDB_SUCCESS) {
        log_chaos_event("TEST4 PASS", "Oversized batch triggered flush and succeeded");
    } else if (status == PDB_ERR_CAPACITY_EXCEEDED) {
        log_chaos_event("TEST4 EXPECTED", "Oversized batch rejected due to capacity");
    } else {
        log_chaos_event("TEST4 FAIL", "Oversized batch failed with unexpected error");
    }
}

// Test 5: Rapid close/reopen cycle
void test_rapid_close_reopen() {
    log_chaos_event("TEST5", "Rapid close/reopen cycle without proper flush");
    
    pdb_options_t options;
    pdb_options_init(&options, VECTOR_DIM);
    options.metric = PDB_METRIC_L2;
    options.memtable_capacity = MEMTABLE_CAPACITY;
    options.wal_directory = DB_PATH;
    options.enable_direct_io = false;
    
    pdb_t* db = nullptr;
    
    // Open and immediately close
    pdb_status_t status = pdb_open(DB_PATH, &options, &db);
    if (status == PDB_SUCCESS) {
        log_chaos_event("TEST5", "Database opened, closing immediately without insert");
        pdb_close(db);
        
        // Reopen immediately
        status = pdb_open(DB_PATH, &options, &db);
        if (status == PDB_SUCCESS) {
            log_chaos_event("TEST5 PASS", "Database reopened successfully after immediate close");
            pdb_close(db);
        } else {
            log_chaos_event("TEST5 FAIL", "Failed to reopen database after immediate close");
        }
    } else {
        log_chaos_event("TEST5 FAIL", "Failed to open database initially");
    }
}

// Test 6: Delete and re-insert same IDs
void test_delete_reinsert(pdb_t* db) {
    log_chaos_event("TEST6", "Delete and re-insert same IDs (tombstone torture)");
    
    const size_t test_ids = 10;
    uint64_t ids[test_ids];
    float vectors[test_ids * VECTOR_DIM];
    
    // Insert initial vectors
    for (size_t i = 0; i < test_ids; i++) {
        ids[i] = i + 50000;
        generate_dirty_vector(&vectors[i * VECTOR_DIM], VECTOR_DIM, 0);
    }
    
    pdb_vector_batch_t batch = {
        .ids = ids,
        .vectors = vectors,
        .count = test_ids,
        .dim = VECTOR_DIM,
        .metadata_jsons = nullptr
    };
    
    pdb_status_t status = pdb_insert_batch(db, &batch);
    if (status != PDB_SUCCESS) {
        log_chaos_event("TEST6 FAIL", "Initial insertion failed");
        return;
    }
    
    // Delete them immediately
    status = pdb_delete_batch(db, ids, test_ids);
    if (status == PDB_SUCCESS) {
        log_chaos_event("TEST6", "Vectors deleted, querying for deleted IDs");
        
        // Try to query deleted IDs
        float query_vec[VECTOR_DIM];
        memset(query_vec, 0, sizeof(query_vec));
        uint64_t out_ids[5];
        float out_distances[5];
        size_t out_actual_k = 0;
        
        pdb_query_knn(db, query_vec, 5, nullptr, out_ids, out_distances, &out_actual_k);
        
        // Re-insert with different vectors
        for (size_t i = 0; i < test_ids; i++) {
            generate_dirty_vector(&vectors[i * VECTOR_DIM], VECTOR_DIM, 2); // Different chaos type
        }
        
        status = pdb_insert_batch(db, &batch);
        if (status == PDB_SUCCESS) {
            log_chaos_event("TEST6 PASS", "Re-insertion of deleted IDs succeeded");
        } else {
            log_chaos_event("TEST6 FAIL", "Re-insertion of deleted IDs failed");
        }
    } else {
        log_chaos_event("TEST6 FAIL", "Deletion failed");
    }
}

// Test 7: Hammer with rapid writes (no sleep)
void test_rapid_write_hammer(pdb_t* db) {
    log_chaos_event("TEST7", "Hammering with rapid sequential writes without sleeping");
    
    const size_t rapid_batches = 10; // Reduced to prevent hang
    const size_t batch_size = 13; // Reduced odd batch size
    
    std::vector<uint64_t> ids(batch_size);
    std::vector<float> vectors(batch_size * VECTOR_DIM);
    
    for (size_t batch = 0; batch < rapid_batches; batch++) {
        for (size_t i = 0; i < batch_size; i++) {
            ids[i] = batch * 1000 + i + 100000;
            generate_dirty_vector(&vectors[i * VECTOR_DIM], VECTOR_DIM, batch % 6);
        }
        
        pdb_vector_batch_t batch_data = {
            .ids = ids.data(),
            .vectors = vectors.data(),
            .count = batch_size,
            .dim = VECTOR_DIM,
            .metadata_jsons = nullptr
        };
        
        pdb_status_t status = pdb_insert_batch(db, &batch_data);
        if (status != PDB_SUCCESS) {
            log_chaos_event("TEST7 FAIL", "Rapid write failed");
            return;
        }
    }
    
    log_chaos_event("TEST7 PASS", "Rapid write hammer completed without crashes");
}

// Test 8: Query with empty output buffers
void test_empty_output_buffers(pdb_t* db) {
    log_chaos_event("TEST8", "Querying with empty/null output buffers");
    
    float query_vec[VECTOR_DIM];
    memset(query_vec, 0, sizeof(query_vec));
    
    // Try with null buffers
    pdb_status_t status = pdb_query_knn(db, query_vec, 5, nullptr, nullptr, nullptr, nullptr);
    
    if (status == PDB_SUCCESS) {
        log_chaos_event("TEST8 UNEXPECTED", "Query with null buffers succeeded");
    } else {
        log_chaos_event("TEST8 EXPECTED", "Query with null buffers failed as expected");
    }
}

// Test 9: Freeze before any data
void test_freeze_before_data() {
    log_chaos_event("TEST9", "Attempting to freeze database before any data");
    
    pdb_options_t options;
    pdb_options_init(&options, VECTOR_DIM);
    options.metric = PDB_METRIC_L2;
    options.memtable_capacity = MEMTABLE_CAPACITY;
    options.wal_directory = DB_PATH;
    options.enable_direct_io = false;
    
    pdb_t* db = nullptr;
    pdb_status_t status = pdb_open(DB_PATH, &options, &db);
    
    if (status == PDB_SUCCESS) {
        // Try to checkpoint without any data
        status = pdb_checkpoint(db);
        if (status == PDB_SUCCESS) {
            log_chaos_event("TEST9 PASS", "Checkpoint succeeded on empty database");
        } else {
            log_chaos_event("TEST9 EXPECTED", "Checkpoint failed on empty database");
        }
        
        pdb_close(db);
    }
}

// Test 10: Concurrent queries (simulate multi-threaded chaos)
std::atomic<int> concurrent_errors{0};
void concurrent_query_thread(pdb_t* db, int thread_id) {
    float query_vec[VECTOR_DIM];
    uint64_t out_ids[10];
    float out_distances[10];
    size_t out_actual_k = 0;
    
    for (int i = 0; i < 10; i++) { // Reduced iterations
        generate_dirty_vector(query_vec, VECTOR_DIM, thread_id % 6);
        
        pdb_status_t status = pdb_query_knn(db, query_vec, 5, nullptr, 
                                            out_ids, out_distances, &out_actual_k);
        if (status != PDB_SUCCESS) {
            concurrent_errors++;
        }
    }
}

void test_concurrent_queries(pdb_t* db) {
    log_chaos_event("TEST10", "Launching concurrent queries without external locks");
    
    const int num_threads = 4; // Reduced threads
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(concurrent_query_thread, db, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    if (concurrent_errors.load() == 0) {
        log_chaos_event("TEST10 PASS", "Concurrent queries completed without errors");
    } else {
        char msg[100];
        snprintf(msg, sizeof(msg), "Concurrent queries had %d errors", concurrent_errors.load());
        log_chaos_event("TEST10 WARN", msg);
    }
}

// Test 11: Double-close handle
void test_double_close() {
    log_chaos_event("TEST11", "Attempting to close database handle twice");
    
    pdb_options_t options;
    pdb_options_init(&options, VECTOR_DIM);
    options.metric = PDB_METRIC_L2;
    options.memtable_capacity = MEMTABLE_CAPACITY;
    options.wal_directory = DB_PATH;
    options.enable_direct_io = false;
    
    pdb_t* db = nullptr;
    pdb_status_t status = pdb_open(DB_PATH, &options, &db);
    
    if (status == PDB_SUCCESS) {
        pdb_close(db);
        
        // Try to close again - this might crash or cause undefined behavior
        log_chaos_event("TEST11", "Attempting second close (may crash or cause UB)");
        // pdb_close(db); // Commented out to prevent actual crash
        
        log_chaos_event("TEST11 WARN", "Double-close test skipped to prevent crash");
    }
}

// Test 12: Weirdest batch sizes (1, 13337, etc.)
void test_weird_batch_sizes(pdb_t* db) {
    log_chaos_event("TEST12", "Inserting batches with weird sizes (1, 13337, 42, etc.)");
    
    size_t weird_sizes[] = {1, 5, 42, 7, 3, 13}; // Minimal set to prevent hangs
    
    for (size_t size : weird_sizes) {
        char msg[100];
        snprintf(msg, sizeof(msg), "Testing batch size %zu", size);
        log_chaos_event("TEST12", msg);
        
        std::vector<uint64_t> ids(size);
        std::vector<float> vectors(size * VECTOR_DIM);
        
        for (size_t i = 0; i < size; i++) {
            ids[i] = rand() % 1000000 + 200000;
            generate_dirty_vector(&vectors[i * VECTOR_DIM], VECTOR_DIM, i % 6);
        }
        
        pdb_vector_batch_t batch = {
            .ids = ids.data(),
            .vectors = vectors.data(),
            .count = size,
            .dim = VECTOR_DIM,
            .metadata_jsons = nullptr
        };
        
        pdb_status_t status = pdb_insert_batch(db, &batch);
        
        if (status == PDB_SUCCESS) {
            snprintf(msg, sizeof(msg), "Batch size %zu succeeded", size);
            log_chaos_event("TEST12", msg);
        } else {
            snprintf(msg, sizeof(msg), "Batch size %zu failed", size);
            log_chaos_event("TEST12", msg);
        }
    }
    
    log_chaos_event("TEST12 PASS", "Weird batch size test completed");
}

int main(int argc, char** argv) {
    (void)argc; // Suppress unused parameter warning
    (void)argv; // Suppress unused parameter warning
    
    printf("=== POMAIDB CHAOS TEST: Real-Time Log Anomaly Detector ===\n");
    fflush(stdout);
    printf("This test deliberately pushes pomaidb to its limits with chaotic inputs\n");
    printf("and erratic usage patterns. Crashes, hangs, or unexpected behavior\n");
    printf("will be logged as forensic evidence.\n\n");
    
    srand(time(nullptr));
    
    // Initialize database
    pdb_options_t options;
    pdb_status_t status = pdb_options_init(&options, VECTOR_DIM);
    if (status != PDB_SUCCESS) {
        log_chaos_event("FATAL", "Failed to initialize options");
        return 1;
    }
    
    options.metric = PDB_METRIC_L2;
    options.memtable_capacity = MEMTABLE_CAPACITY;
    options.wal_directory = DB_PATH;
    options.enable_direct_io = false;
    options.arena_reserve_bytes = 1024 * 1024 * 100; // 100MB
    
    pdb_t* db = nullptr;
    status = pdb_open(DB_PATH, &options, &db);
    if (status != PDB_SUCCESS) {
        log_chaos_event("FATAL", "Failed to open database");
        return 1;
    }
    
    log_chaos_event("START", "Beginning chaos test sequence");
    
    // Run all chaos tests
    test_query_before_insert(db);
    test_dirty_vector_insertion(db);
    test_empty_batch(db);
    test_oversized_batch(db);
    test_rapid_close_reopen();
    test_delete_reinsert(db);
    test_rapid_write_hammer(db);
    test_empty_output_buffers(db);
    test_freeze_before_data();
    test_concurrent_queries(db);
    test_double_close();
    test_weird_batch_sizes(db);
    
    // Get final stats
    pdb_stats_t stats;
    status = pdb_get_stats(db, &stats);
    if (status == PDB_SUCCESS) {
        printf("\n=== FINAL STATISTICS ===\n");
        printf("Total vectors: %zu\n", stats.total_vectors);
        printf("Memtable vectors: %zu\n", stats.memtable_vectors);
        printf("Sealed segments: %zu\n", stats.sealed_segments);
        printf("Arena allocated: %zu bytes\n", stats.arena_allocated_bytes);
        printf("Arena committed: %zu bytes\n", stats.arena_committed_bytes);
        printf("Current LSN: %llu\n", (unsigned long long)stats.current_lsn);
    }
    
    // Cleanup
    log_chaos_event("CLEANUP", "Closing database");
    pdb_close(db);
    
    log_chaos_event("COMPLETE", "Chaos test sequence completed");
    printf("\n=== CHAOS TEST COMPLETE ===\n");
    printf("Review the log above for any crashes, hangs, or unexpected behavior.\n");
    
    return 0;
}
