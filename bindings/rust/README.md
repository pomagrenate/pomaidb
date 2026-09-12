# PomaiDB Rust Client

Official Rust bindings for **PomaiDB** - High-performance embedded vector database for Edge AI.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pomaidb = "0.1.0"
```

Ensure `libpomai_c.dll` (or `.so`/`.dylib`) is available on your library search path or set `POMAIDB_LIB_DIR`.

## Usage Example

```rust
use pomaidb::{Database, Options, MetricType, QuantType, PutOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Options::new("./data_dir", 4)
        .metric(MetricType::L2)
        .quant_type(QuantType::None);

    let db = Database::open(&opts)?;

    // Upsert vector
    db.put(1, &[1.0, 0.0, 0.0, 0.0])?;

    // Multi-membrane support
    db.create_membrane("analytics", 4, 1)?;
    db.open_membrane("analytics")?;
    db.put_with_options(
        2,
        &[0.0, 1.0, 0.0, 0.0],
        &PutOptions {
            membrane: Some("analytics"),
            timestamp: 1234567,
            payload: Some(b"hello rust"),
        },
    )?;

    // Vector search
    let hits = db.search(&[1.0, 0.0, 0.0, 0.0], 5)?;
    for hit in hits {
        println!("ID: {}, Score: {}", hit.id, hit.score);
    }

    db.flush()?;
    Ok(())
}
```

## License

Apache-2.0