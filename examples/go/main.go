// PomaiDB Go Comprehensive Feature Pipeline Example
// ==================================================
// Demonstrates 100% of PomaiDB's capabilities:
//   1. Engine configuration (Quantization: SQ8, Metric: Cosine, Memory budget)
//   2. Database lifecycle (Open, Close, Finalizer safety)
//   3. Multi-membrane management (Create, Open, Close, List, Drop)
//   4. Vector CRUD (Put, Get, Exists, Delete)
//   5. Arbitrary binary payloads & event timestamps
//   6. Top-K ANN vector search
//   7. Membrane-scoped vector search
//   8. Point-in-time temporal queries (AsOfTs)
//   9. Maintenance operations (Flush, Freeze, Compact)
//  10. Engine telemetry and statistics (GetStats)

package main

import (
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/pomagrenate/pomaidb/bindings/go"
)

const (
	dbPath    = "./pomaidb_go_example_store"
	dimension = 64
)

func generateVector(seedVal float32) []float32 {
	vec := make([]float32, dimension)
	var sumSq float32
	for i := 0; i < dimension; i++ {
		v := seedVal + float32(i)*0.01
		vec[i] = v
		sumSq += v * v
	}
	norm := float32(math.Sqrt(float64(sumSq)))
	for i := range vec {
		vec[i] /= norm
	}
	return vec
}

func cleanDbDir() {
	_ = os.RemoveAll(dbPath)
}

func main() {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println(" PomaiDB Go Comprehensive Feature Pipeline")
	fmt.Println(strings.Repeat("=", 70))

	cleanDbDir()

	// -------------------------------------------------------------------------
	// 1. DATABASE INITIALIZATION & CONFIGURATION
	// -------------------------------------------------------------------------
	fmt.Println("\n[Step 1] Opening PomaiDB with SQ8 Quantization & Cosine Metric...")
	opts := pomaidb.Options{
		Path:              dbPath,
		Dim:               dimension,
		Shards:            2,
		Metric:            pomaidb.MetricCosine,
		QuantType:         pomaidb.QuantSQ8,
		MemoryBudgetBytes: 64 * 1024 * 1024, // 64MB
	}

	db, err := pomaidb.Open(opts)
	if err != nil {
		panic(fmt.Sprintf("Failed to open PomaiDB: %v", err))
	}
	defer db.Close()
	fmt.Printf("  -> Database opened at: %s\n", dbPath)

	// -------------------------------------------------------------------------
	// 2. MULTI-MEMBRANE LIFECYCLE
	// -------------------------------------------------------------------------
	fmt.Println("\n[Step 2] Multi-Membrane Architecture Management...")
	fmt.Println("  -> Creating 'realtime_telemetry' membrane (dim=64, shards=2)...")
	if err := db.CreateMembrane("realtime_telemetry", dimension, 2); err != nil {
		panic(err)
	}

	fmt.Println("  -> Creating 'knowledge_base' membrane (dim=64, shards=1)...")
	if err := db.CreateMembrane("knowledge_base", dimension, 1); err != nil {
		panic(err)
	}

	membranes, err := db.ListMembranes()
	if err != nil {
		panic(err)
	}
	fmt.Printf("  -> Active membranes: %v\n", membranes)

	// -------------------------------------------------------------------------
	// 3. VECTOR INSERTION WITH PAYLOADS AND TIMESTAMPS
	// -------------------------------------------------------------------------
	fmt.Println("\n[Step 3] Vector Insertion with Payloads and Timestamps...")
	nowTs := uint64(time.Now().UnixMilli())

	// Insert into default membrane
	vec1 := generateVector(1.0)
	payload1 := []byte(`{"user":"david","action":"sync"}`)
	err = db.PutWithOptions(101, vec1, pomaidb.PutOptions{
		Timestamp: nowTs,
		Payload:   payload1,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("  -> Inserted ID 101 into default membrane with timestamp %d\n", nowTs)

	// Insert into 'realtime_telemetry' membrane
	vec2 := generateVector(2.0)
	payload2 := []byte(`{"sensor":"lidar_rear","battery_pct":88}`)
	err = db.PutWithOptions(201, vec2, pomaidb.PutOptions{
		Membrane:  "realtime_telemetry",
		Timestamp: nowTs + 25,
		Payload:   payload2,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println("  -> Inserted ID 201 into 'realtime_telemetry' membrane with payload")

	// Insert into 'knowledge_base'
	for i := 0; i < 3; i++ {
		id := uint64(301 + i)
		v := generateVector(3.0 + float32(i))
		p := []byte(fmt.Sprintf(`{"doc_id":"go_doc_%d","topic":"microservices"}`, id))
		err := db.PutWithOptions(id, v, pomaidb.PutOptions{
			Membrane:  "knowledge_base",
			Timestamp: nowTs + uint64(i),
			Payload:   p,
		})
		if err != nil {
			panic(err)
		}
	}
	fmt.Println("  -> Inserted IDs [301, 302, 303] into 'knowledge_base' membrane")

	// -------------------------------------------------------------------------
	// 4. EXISTENCE & RECORD RETRIEVAL
	// -------------------------------------------------------------------------
	fmt.Println("\n[Step 4] Checking Existence and Retrieving Full Records...")
	exists101, err := db.Exists(101)
	if err != nil {
		panic(err)
	}
	exists201, err := db.ExistsMembrane("realtime_telemetry", 201)
	if err != nil {
		panic(err)
	}
	fmt.Printf("  -> Exists 101 (default): %v\n", exists101)
	fmt.Printf("  -> Exists 201 (realtime_telemetry): %v\n", exists201)

	rec, err := db.GetMembrane("realtime_telemetry", 201)
	if err != nil {
		panic(err)
	}
	if rec != nil {
		fmt.Printf("  -> Retrieved Record 201:\n")
		fmt.Printf("     Dim: %d\n", rec.Dim)
		fmt.Printf("     Timestamp: %d\n", rec.Timestamp)
		fmt.Printf("     Payload: %s\n", string(rec.Payload))
		fmt.Printf("     Vector snippet: [%.4f, %.4f, %.4f, %.4f]...\n",
			rec.Vector[0], rec.Vector[1], rec.Vector[2], rec.Vector[3])
	}

	// -------------------------------------------------------------------------
	// 5. VECTOR SEARCH (ANN) & MEMBRANE-SCOPED SEARCH
	// -------------------------------------------------------------------------
	fmt.Println("\n[Step 5] Top-K Vector Search...")
	queryVec := generateVector(1.05) // Closest to 101

	// Search in default membrane
	defaultHits, err := db.Search(queryVec, 3)
	if err != nil {
		panic(err)
	}
	fmt.Println("  -> Search in default membrane (Top 3):")
	for _, hit := range defaultHits {
		fmt.Printf("     Hit ID: %d, Cosine Score: %.4f\n", hit.ID, hit.Score)
	}

	// Scoped search in 'knowledge_base'
	queryKb := generateVector(3.1)
	kbHits, err := db.SearchWithOptions(queryKb, 3, pomaidb.SearchOptions{
		Membrane: "knowledge_base",
	})
	if err != nil {
		panic(err)
	}
	fmt.Println("  -> Scoped Search in 'knowledge_base' membrane:")
	for _, hit := range kbHits {
		fmt.Printf("     Hit ID: %d, Cosine Score: %.4f\n", hit.ID, hit.Score)
	}

	// -------------------------------------------------------------------------
	// 6. TEMPORAL TIME-TRAVEL QUERY
	// -------------------------------------------------------------------------
	fmt.Println("\n[Step 6] Temporal Query (Point-in-Time AsOfTs)...")
	pastHits, err := db.SearchWithOptions(queryVec, 3, pomaidb.SearchOptions{
		AsOfTs: nowTs + 10,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("  -> Search AsOfTs=%d: Found %d hit(s)\n", nowTs+10, len(pastHits))

	// -------------------------------------------------------------------------
	// 7. ENGINE MAINTENANCE (FLUSH, FREEZE, COMPACT)
	// -------------------------------------------------------------------------
	fmt.Println("\n[Step 7] Engine Maintenance Operations...")
	fmt.Println("  -> Flushing memtable to disk...")
	if err := db.Flush(); err != nil {
		panic(err)
	}

	fmt.Println("  -> Freezing active write membrane...")
	if err := db.Freeze("realtime_telemetry"); err != nil {
		panic(err)
	}

	fmt.Println("  -> Triggering background compaction...")
	if err := db.Compact("realtime_telemetry"); err != nil {
		panic(err)
	}
	fmt.Println("  -> Maintenance cycle complete.")

	// -------------------------------------------------------------------------
	// 8. ENGINE TELEMETRY & STATS
	// -------------------------------------------------------------------------
	fmt.Println("\n[Step 8] Fetching Engine Health & Statistics...")
	stats, err := db.GetStats()
	if err != nil {
		panic(err)
	}
	fmt.Printf("  -> Version: %v\n", stats["version"])
	fmt.Printf("  -> ABI Version: %v\n", stats["abi_version"])
	fmt.Printf("  -> Active Membranes: %v\n", stats["membranes"])

	// -------------------------------------------------------------------------
	// 9. DELETION & CLEANUP
	// -------------------------------------------------------------------------
	fmt.Println("\n[Step 9] Deletion & Clean Shutdown...")
	if err := db.Delete(101); err != nil {
		panic(err)
	}
	e101, _ := db.Exists(101)
	fmt.Printf("  -> Deleted ID 101. Exists now: %v\n", e101)

	if err := db.DropMembrane("knowledge_base"); err != nil {
		panic(err)
	}
	remaining, _ := db.ListMembranes()
	fmt.Printf("  -> Dropped 'knowledge_base'. Current membranes: %v\n", remaining)

	_ = db.Close()
	fmt.Println("  -> Database closed successfully.")

	cleanDbDir()

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println(" [SUCCESS] All PomaiDB Go pipeline features executed cleanly!")
	fmt.Println(strings.Repeat("=", 70))
}
