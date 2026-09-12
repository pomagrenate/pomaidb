//! PomaiDB Rust Comprehensive Feature Pipeline Example
//! ====================================================
//! Demonstrates 100% of PomaiDB's capabilities:
//!   1. Engine configuration (Quantization: SQ8, Metric: Cosine, Memory budget)
//!   2. Database lifecycle (Open, RAII Drop cleanup)
//!   3. Multi-membrane management (Create, Open, Close, List, Drop)
//!   4. Vector CRUD (Put, Get, Exists, Delete)
//!   5. Arbitrary binary payloads & event timestamps
//!   6. Top-K ANN vector search
//!   7. Membrane-scoped vector search
//!   8. Point-in-time temporal queries (as_of_ts)
//!   9. Maintenance operations (Flush, Freeze, Compact)
//!  10. Engine telemetry and statistics (get_stats)

use pomaidb::{Database, MetricType, Options, PutOptions, QuantType, SearchOptions};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const DB_PATH: &str = "./pomaidb_rust_example_store";
const DIMENSION: usize = 64;

fn generate_vector(seed_val: f32) -> Vec<f32> {
    let mut vec = Vec::with_capacity(DIMENSION);
    let mut sum_sq = 0.0f32;
    for i in 0..DIMENSION {
        let v = seed_val + (i as f32) * 0.01;
        vec.push(v);
        sum_sq += v * v;
    }
    let norm = sum_sq.sqrt();
    vec.into_iter().map(|x| x / norm).collect()
}

fn clean_db_dir() {
    if Path::new(DB_PATH).exists() {
        let _ = fs::remove_dir_all(DB_PATH);
    }
}

fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(70));
    println!(" PomaiDB Rust Comprehensive Feature Pipeline");
    println!("{}", "=".repeat(70));

    clean_db_dir();

    // -------------------------------------------------------------------------
    // 1. DATABASE INITIALIZATION & CONFIGURATION
    // -------------------------------------------------------------------------
    println!("\n[Step 1] Opening PomaiDB with SQ8 Quantization & Cosine Metric...");
    let opts = Options::new(DB_PATH, DIMENSION)
        .shards(2)
        .metric(MetricType::Cosine)
        .quant_type(QuantType::SQ8)
        .memory_budget_bytes(64 * 1024 * 1024); // 64MB

    let mut db = Database::open(&opts)?;
    println!("  -> Database opened at: {}", DB_PATH);

    // -------------------------------------------------------------------------
    // 2. MULTI-MEMBRANE LIFECYCLE
    // -------------------------------------------------------------------------
    println!("\n[Step 2] Multi-Membrane Architecture Management...");
    println!("  -> Creating 'realtime_telemetry' membrane (dim=64, shards=2)...");
    db.create_membrane("realtime_telemetry", DIMENSION, 2)?;

    println!("  -> Creating 'knowledge_base' membrane (dim=64, shards=1)...");
    db.create_membrane("knowledge_base", DIMENSION, 1)?;

    let membranes = db.list_membranes()?;
    println!("  -> Active membranes: {:?}", membranes);

    // -------------------------------------------------------------------------
    // 3. VECTOR INSERTION WITH PAYLOADS AND TIMESTAMPS
    // -------------------------------------------------------------------------
    println!("\n[Step 3] Vector Insertion with Payloads and Timestamps...");
    let now_ts = current_timestamp_ms();

    // Insert into default membrane
    let vec1 = generate_vector(1.0);
    let payload1 = b"{\"user\":\"charlie\",\"action\":\"checkout\"}";
    db.put_with_options(
        101,
        &vec1,
        &PutOptions {
            membrane: None,
            timestamp: now_ts,
            payload: Some(payload1),
        },
    )?;
    println!("  -> Inserted ID 101 into default membrane with timestamp {}", now_ts);

    // Insert into 'realtime_telemetry' membrane
    let vec2 = generate_vector(2.0);
    let payload2 = b"{\"sensor\":\"drone_gyro\",\"altitude_m\":120.5}";
    db.put_with_options(
        201,
        &vec2,
        &PutOptions {
            membrane: Some("realtime_telemetry"),
            timestamp: now_ts + 20,
            payload: Some(payload2),
        },
    )?;
    println!("  -> Inserted ID 201 into 'realtime_telemetry' membrane with payload");

    // Insert into 'knowledge_base'
    for i in 0..3 {
        let id = 301 + i;
        let v = generate_vector(3.0 + i as f32);
        let p = format!("{{\"doc_id\":\"rust_doc_{}\",\"tier\":\"gold\"}}", id);
        db.put_with_options(
            id,
            &v,
            &PutOptions {
                membrane: Some("knowledge_base"),
                timestamp: now_ts + i,
                payload: Some(p.as_bytes()),
            },
        )?;
    }
    println!("  -> Inserted IDs [301, 302, 303] into 'knowledge_base' membrane");

    // -------------------------------------------------------------------------
    // 4. EXISTENCE & RECORD RETRIEVAL
    // -------------------------------------------------------------------------
    println!("\n[Step 4] Checking Existence and Retrieving Full Records...");
    let exists_101 = db.exists(101)?;
    let exists_201 = db.exists_membrane(Some("realtime_telemetry"), 201)?;
    println!("  -> Exists 101 (default): {}", exists_101);
    println!("  -> Exists 201 (realtime_telemetry): {}", exists_201);

    let rec = db.get_membrane(Some("realtime_telemetry"), 201)?;
    if let Some(r) = rec {
        println!("  -> Retrieved Record 201:");
        println!("     Dim: {}", r.dim);
        println!("     Timestamp: {}", r.timestamp);
        if let Some(ref p) = r.payload {
            println!("     Payload: {}", String::from_utf8_lossy(p));
        }
        let snippet: Vec<f32> = r.vector.iter().take(4).cloned().collect();
        println!("     Vector snippet: {:?}...", snippet);
    }

    // -------------------------------------------------------------------------
    // 5. VECTOR SEARCH (ANN) & MEMBRANE-SCOPED SEARCH
    // -------------------------------------------------------------------------
    println!("\n[Step 5] Top-K Vector Search...");
    let query_vec = generate_vector(1.05); // Closest to 101

    // Search in default membrane
    let default_hits = db.search(&query_vec, 3)?;
    println!("  -> Search in default membrane (Top 3):");
    for hit in default_hits {
        println!("     Hit ID: {}, Cosine Score: {:.4}", hit.id, hit.score);
    }

    // Scoped search in 'knowledge_base'
    let query_kb = generate_vector(3.1);
    let kb_hits = db.search_with_options(
        &query_kb,
        3,
        &SearchOptions {
            membrane: Some("knowledge_base"),
            ..Default::default()
        },
    )?;
    println!("  -> Scoped Search in 'knowledge_base' membrane:");
    for hit in kb_hits {
        println!("     Hit ID: {}, Cosine Score: {:.4}", hit.id, hit.score);
    }

    // -------------------------------------------------------------------------
    // 6. TEMPORAL TIME-TRAVEL QUERY
    // -------------------------------------------------------------------------
    println!("\n[Step 6] Temporal Query (Point-in-Time as_of_ts)...");
    let past_hits = db.search_with_options(
        &query_vec,
        3,
        &SearchOptions {
            as_of_ts: now_ts + 10,
            ..Default::default()
        },
    )?;
    println!("  -> Search as_of_ts={}: Found {} hit(s)", now_ts + 10, past_hits.len());

    // -------------------------------------------------------------------------
    // 7. ENGINE MAINTENANCE (FLUSH, FREEZE, COMPACT)
    // -------------------------------------------------------------------------
    println!("\n[Step 7] Engine Maintenance Operations...");
    println!("  -> Flushing memtable to disk...");
    db.flush()?;

    println!("  -> Freezing active write membrane...");
    db.freeze(Some("realtime_telemetry"))?;

    println!("  -> Triggering background compaction...");
    db.compact(Some("realtime_telemetry"))?;
    println!("  -> Maintenance cycle complete.");

    // -------------------------------------------------------------------------
    // 8. ENGINE TELEMETRY & STATS
    // -------------------------------------------------------------------------
    println!("\n[Step 8] Fetching Engine Health & Statistics...");
    let stats = db.get_stats()?;
    println!("  -> Version: {}", stats.get("version").unwrap_or(&serde_json::Value::Null));
    println!("  -> ABI Version: {}", stats.get("abi_version").unwrap_or(&serde_json::Value::Null));
    println!("  -> Active Membranes: {:?}", stats.get("membranes").unwrap_or(&serde_json::Value::Null));

    // -------------------------------------------------------------------------
    // 9. DELETION & CLEANUP
    // -------------------------------------------------------------------------
    println!("\n[Step 9] Deletion & Clean Shutdown...");
    db.delete(101)?;
    println!("  -> Deleted ID 101. Exists now: {}", db.exists(101)?);

    db.drop_membrane("knowledge_base")?;
    println!("  -> Dropped 'knowledge_base'. Current membranes: {:?}", db.list_membranes()?);

    db.close()?;
    println!("  -> Database closed successfully.");

    clean_db_dir();

    println!("\n{}", "=".repeat(70));
    println!(" [SUCCESS] All PomaiDB Rust pipeline features executed cleanly!");
    println!("{}", "=".repeat(70));

    Ok(())
}
