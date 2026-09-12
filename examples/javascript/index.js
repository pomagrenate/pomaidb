/**
 * PomaiDB JavaScript / Node.js Comprehensive Feature Pipeline Example
 * ===================================================================
 * Demonstrates 100% of PomaiDB's capabilities:
 *   1. Engine configuration (Quantization: SQ8, Metric: Cosine, Memory budget)
 *   2. Database lifecycle (Open, Close)
 *   3. Multi-membrane management (Create, Open, Close, List, Drop)
 *   4. Vector CRUD (Put, Get, Exists, Delete)
 *   5. Arbitrary binary payloads (Buffer) & event timestamps
 *   6. Top-K ANN vector search
 *   7. Membrane-scoped vector search
 *   8. Point-in-time temporal queries (asOfTs)
 *   9. Maintenance operations (Flush, Freeze, Compact)
 *  10. Engine telemetry and statistics (getStats)
 */

import { Database, MetricType, QuantType } from "pomaidb";
import fs from "fs";
import path from "path";

const DB_PATH = "./pomaidb_js_example_store";
const DIMENSION = 64;

function generateVector(seedVal) {
  const vec = [];
  let sumSq = 0.0;
  for (let i = 0; i < DIMENSION; i++) {
    const v = seedVal + i * 0.01;
    vec.push(v);
    sumSq += v * v;
  }
  const norm = Math.sqrt(sumSq);
  return vec.map(x => x / norm);
}

function cleanDbDir() {
  if (fs.existsSync(DB_PATH)) {
    fs.rmSync(DB_PATH, { recursive: true, force: true });
  }
}

async function main() {
  console.log("=".repeat(70));
  console.log(" PomaiDB JavaScript Comprehensive Feature Pipeline");
  console.log("=".repeat(70));

  cleanDbDir();

  // -------------------------------------------------------------------------
  // 1. DATABASE INITIALIZATION & CONFIGURATION
  // -------------------------------------------------------------------------
  console.log("\n[Step 1] Opening PomaiDB with SQ8 Quantization & Cosine Metric...");
  const db = Database.open({
    path: DB_PATH,
    dim: DIMENSION,
    shards: 2,
    metric: MetricType.Cosine,
    quantType: QuantType.SQ8,
    memoryBudgetBytes: 64 * 1024 * 1024 // 64MB
  });
  console.log(`  -> Database opened at: ${DB_PATH}`);

  // -------------------------------------------------------------------------
  // 2. MULTI-MEMBRANE LIFECYCLE
  // -------------------------------------------------------------------------
  console.log("\n[Step 2] Multi-Membrane Architecture Management...");
  console.log("  -> Creating 'realtime_telemetry' membrane (dim=64, shards=2)...");
  db.createMembrane("realtime_telemetry", DIMENSION, 2);

  console.log("  -> Creating 'knowledge_base' membrane (dim=64, shards=1)...");
  db.createMembrane("knowledge_base", DIMENSION, 1);

  let membranes = db.listMembranes();
  console.log(`  -> Active membranes:`, membranes);

  // -------------------------------------------------------------------------
  // 3. VECTOR INSERTION WITH PAYLOADS AND TIMESTAMPS
  // -------------------------------------------------------------------------
  console.log("\n[Step 3] Vector Insertion with Payloads and Timestamps...");
  const nowTs = Date.now();

  // Insert into default membrane
  const vec1 = generateVector(1.0);
  const payload1 = Buffer.from(JSON.stringify({ user: "bob", action: "page_view" }));
  db.put(101, vec1, { timestamp: nowTs, payload: payload1 });
  console.log(`  -> Inserted ID 101 into default membrane with timestamp ${nowTs}`);

  // Insert into 'realtime_telemetry' membrane
  const vec2 = generateVector(2.0);
  const payload2 = Buffer.from(JSON.stringify({ device: "edge_camera_01", fps: 60 }));
  db.put(201, vec2, {
    membrane: "realtime_telemetry",
    timestamp: nowTs + 15,
    payload: payload2
  });
  console.log(`  -> Inserted ID 201 into 'realtime_telemetry' membrane with payload`);

  // Insert multiple into 'knowledge_base'
  for (let i = 0; i < 3; i++) {
    const id = 301 + i;
    const v = generateVector(3.0 + i);
    const p = Buffer.from(JSON.stringify({ doc_id: `doc_${id}`, author: "PomaiDB Team" }));
    db.put(id, v, { membrane: "knowledge_base", timestamp: nowTs + i, payload: p });
  }
  console.log("  -> Inserted IDs [301, 302, 303] into 'knowledge_base' membrane");

  // -------------------------------------------------------------------------
  // 4. EXISTENCE & RECORD RETRIEVAL
  // -------------------------------------------------------------------------
  console.log("\n[Step 4] Checking Existence and Retrieving Full Records...");
  const exists101 = db.exists(101);
  const exists201 = db.exists(201, "realtime_telemetry");
  console.log(`  -> Exists 101 (default): ${exists101}`);
  console.log(`  -> Exists 201 (realtime_telemetry): ${exists201}`);

  const rec = db.get(201, "realtime_telemetry");
  if (rec) {
    console.log(`  -> Retrieved Record 201:`);
    console.log(`     Dim: ${rec.dim}`);
    console.log(`     Timestamp: ${rec.timestamp}`);
    console.log(`     Payload: ${rec.payload ? rec.payload.toString("utf-8") : "none"}`);
    console.log(`     Vector snippet: [${rec.vector.slice(0, 4).map(x => x.toFixed(4)).join(", ")}]...`);
  }

  // -------------------------------------------------------------------------
  // 5. VECTOR SEARCH (ANN) & MEMBRANE-SCOPED SEARCH
  // -------------------------------------------------------------------------
  console.log("\n[Step 5] Top-K Vector Search...");
  const queryVec = generateVector(1.05); // Closest to 101

  // Search in default membrane
  const defaultHits = db.search(queryVec, 3);
  console.log(`  -> Search in default membrane (Top 3):`);
  for (const hit of defaultHits) {
    console.log(`     Hit ID: ${hit.id}, Cosine Score: ${hit.score.toFixed(4)}`);
  }

  // Scoped search in 'knowledge_base'
  const queryKb = generateVector(3.1);
  const kbHits = db.search(queryKb, 3, { membrane: "knowledge_base" });
  console.log(`  -> Scoped Search in 'knowledge_base' membrane:`);
  for (const hit of kbHits) {
    console.log(`     Hit ID: ${hit.id}, Cosine Score: ${hit.score.toFixed(4)}`);
  }

  // -------------------------------------------------------------------------
  // 6. TEMPORAL TIME-TRAVEL QUERY
  // -------------------------------------------------------------------------
  console.log("\n[Step 6] Temporal Query (Point-in-Time asOfTs)...");
  const pastHits = db.search(queryVec, 3, { asOfTs: nowTs + 5 });
  console.log(`  -> Search asOfTs=${nowTs + 5}: Found ${pastHits.length} hit(s)`);

  // -------------------------------------------------------------------------
  // 7. ENGINE MAINTENANCE (FLUSH, FREEZE, COMPACT)
  // -------------------------------------------------------------------------
  console.log("\n[Step 7] Engine Maintenance Operations...");
  console.log("  -> Flushing memtable to disk...");
  db.flush();

  console.log("  -> Freezing active write membrane...");
  db.freeze("realtime_telemetry");

  console.log("  -> Triggering background compaction...");
  db.compact("realtime_telemetry");
  console.log("  -> Maintenance cycle complete.");

  // -------------------------------------------------------------------------
  // 8. ENGINE TELEMETRY & STATS
  // -------------------------------------------------------------------------
  console.log("\n[Step 8] Fetching Engine Health & Statistics...");
  const stats = db.getStats();
  console.log(`  -> Version: ${stats.version}`);
  console.log(`  -> ABI Version: ${stats.abi_version}`);
  console.log(`  -> Active Membranes:`, stats.membranes);

  // -------------------------------------------------------------------------
  // 9. DELETION & CLEANUP
  // -------------------------------------------------------------------------
  console.log("\n[Step 9] Deletion & Clean Shutdown...");
  db.delete(101);
  console.log(`  -> Deleted ID 101. Exists now: ${db.exists(101)}`);

  db.dropMembrane("knowledge_base");
  console.log(`  -> Dropped 'knowledge_base'. Current membranes:`, db.listMembranes());

  db.close();
  console.log("  -> Database closed successfully.");

  cleanDbDir();

  console.log("\n" + "=".repeat(70));
  console.log(" [SUCCESS] All PomaiDB JavaScript pipeline features executed cleanly!");
  console.log("=".repeat(70));
}

main().catch(err => {
  console.error("Pipeline failed with error:", err);
  process.exit(1);
});
