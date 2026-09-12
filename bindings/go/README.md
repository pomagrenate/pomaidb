# PomaiDB Go Client

Official Go bindings for **PomaiDB** - High-performance embedded vector database for Edge AI.

## Installation

```bash
go get github.com/pomagrenate/pomaidb/bindings/go
```

Make sure `libpomai_c.dll` (or `.so`/`.dylib`) is available on your library search path.

## Quick Start

```go
package main

import (
	"fmt"
	"log"

	"github.com/pomagrenate/pomaidb/bindings/go"
)

func main() {
	db, err := pomaidb.Open(pomaidb.Options{
		Path:   "./data_dir",
		Dim:    4,
		Metric: pomaidb.MetricL2,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// Put vector
	if err := db.Put(1, []float32{1.0, 0.0, 0.0, 0.0}); err != nil {
		log.Fatal(err)
	}

	// Vector Search
	hits, err := db.Search([]float32{1.0, 0.0, 0.0, 0.0}, 5)
	if err != nil {
		log.Fatal(err)
	}

	for _, hit := range hits {
		fmt.Printf("ID: %d, Score: %f\n", hit.ID, hit.Score)
	}
}
```

## License

Apache-2.0