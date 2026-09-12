use pomaidb::{Database, MetricType, Options, PutOptions, QuantType};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn test_basic_crud_and_search() {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let test_dir = format!("tmp_rust_test_{}", now);

    let opts = Options::new(&test_dir, 4)
        .metric(MetricType::L2)
        .quant_type(QuantType::None);

    let mut db = Database::open(&opts).expect("Failed to open db");

    // Put item
    db.put(1, &[1.0, 0.0, 0.0, 0.0]).expect("put failed");
    assert!(db.exists(1).expect("exists check failed"));

    // Multi-membrane
    db.create_membrane("test_mem", 4, 1).expect("create_membrane failed");
    db.open_membrane("test_mem").expect("open_membrane failed");

    db.put_with_options(
        2,
        &[0.0, 1.0, 0.0, 0.0],
        &PutOptions {
            membrane: Some("test_mem"),
            timestamp: 1234567,
            payload: Some(b"hello rust"),
        },
    ).expect("put_with_options failed");

    assert!(db.exists_membrane(Some("test_mem"), 2).expect("exists_membrane failed"));

    let rec = db.get_membrane(Some("test_mem"), 2).expect("get_membrane failed").expect("record 2 missing");
    assert_eq!(rec.id, 2);
    assert_eq!(rec.timestamp, 1234567);
    assert_eq!(rec.payload.as_deref(), Some(b"hello rust".as_slice()));

    // Search
    let hits = db.search(&[1.0, 0.0, 0.0, 0.0], 2).expect("search failed");
    assert!(!hits.is_empty());
    assert_eq!(hits[0].id, 1);

    // Stats
    let stats = db.get_stats().expect("get_stats failed");
    assert!(stats.get("version").is_some());

    db.flush().expect("flush failed");
    db.close().expect("close failed");

    let _ = fs::remove_dir_all(&test_dir);
}