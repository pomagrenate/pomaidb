package pomaidb

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDatabaseBasicAndSearch(t *testing.T) {
	testDir := filepath.Join(".", fmt.Sprintf("tmp_go_test_%d", time.Now().UnixNano()))
	defer os.RemoveAll(testDir)

	db, err := Open(Options{
		Path:      testDir,
		Dim:       4,
		Shards:    1,
		Metric:    MetricL2,
		QuantType: QuantNone,
	})
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer db.Close()

	// Put item
	if err := db.Put(1, []float32{1.0, 0.0, 0.0, 0.0}); err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	exists, err := db.Exists(1)
	if err != nil {
		t.Fatalf("Exists failed: %v", err)
	}
	if !exists {
		t.Fatal("Expected ID 1 to exist")
	}

	// Membrane tests
	if err := db.CreateMembrane("test_mem", 4, 1); err != nil {
		t.Fatalf("CreateMembrane failed: %v", err)
	}
	if err := db.OpenMembrane("test_mem"); err != nil {
		t.Fatalf("OpenMembrane failed: %v", err)
	}

	if err := db.PutWithOptions(2, []float32{0.0, 1.0, 0.0, 0.0}, PutOptions{
		Membrane:  "test_mem",
		Timestamp: 1234567,
		Payload:   []byte("hello go"),
	}); err != nil {
		t.Fatalf("PutWithOptions failed: %v", err)
	}

	rec, err := db.GetMembrane("test_mem", 2)
	if err != nil {
		t.Fatalf("GetMembrane failed: %v", err)
	}
	if rec == nil {
		t.Fatal("Record 2 should not be nil")
	}
	if rec.ID != 2 {
		t.Fatalf("Expected ID 2, got %d", rec.ID)
	}
	if rec.Timestamp != 1234567 {
		t.Fatalf("Expected timestamp 1234567, got %d", rec.Timestamp)
	}
	if !bytes.Equal(rec.Payload, []byte("hello go")) {
		t.Fatalf("Payload mismatch: %s", string(rec.Payload))
	}

	// Search
	hits, err := db.Search([]float32{1.0, 0.0, 0.0, 0.0}, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(hits) == 0 || hits[0].ID != 1 {
		t.Fatalf("Search hits unexpected: %+v", hits)
	}

	// Stats
	stats, err := db.GetStats()
	if err != nil {
		t.Fatalf("GetStats failed: %v", err)
	}
	if _, ok := stats["version"]; !ok {
		t.Fatal("Missing version in stats")
	}

	if err := db.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
}