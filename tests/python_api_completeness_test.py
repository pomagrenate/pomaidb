import tempfile
import pomaidb
import shutil
import time

dirpath = tempfile.mkdtemp()
print("Starting DB...")
try:
    db = pomaidb.open_db(dirpath, dim=128, metric="ip")
    print("DB Opened.")
    pomaidb.put_batch(db, ids=[1, 2], vectors=[[0.1]*128, [0.2]*128])
    print("Put batch.")
    pomaidb.freeze(db)
    print("Frozen.")
    
    assert pomaidb.exists(db, 1) == True
    assert pomaidb.exists(db, 99) == False
    print("Exists works.")
    
    rec = pomaidb.get(db, 1)
    assert rec is not None
    assert rec["id"] == 1
    assert rec["dim"] == 128
    print("Get works.")
    
    pomaidb.delete(db, 1)
    print("Delete works.")
    
    # Test vector membrane creation and listing
    pomaidb.create_membrane(db, "vec_memb", dim=128)
    membranes = pomaidb.list_membranes(db)
    assert "vec_memb" in membranes
    print("Membrane list works:", membranes)

    # Put and search in membrane with timestamp & payload
    test_payload = b'{"tenant_doc":"doc42"}'
    pomaidb.put(db, 10, [0.2]*128, membrane="vec_memb", timestamp=987654321, payload=test_payload)
    pomaidb.freeze(db, membrane="vec_memb")

    rec10 = pomaidb.get(db, 10, membrane="vec_memb")
    assert rec10 is not None
    assert rec10["id"] == 10
    assert rec10["timestamp"] == 987654321
    assert rec10["payload"] == test_payload
    print("Membrane get with payload and timestamp works.")

    hits = pomaidb.search(db, [0.2]*128, topk=5, membrane="vec_memb")
    assert len(hits) > 0
    assert hits[0][0] == 10
    print("Membrane search works:", hits)

    # Test batch search
    batch_hits = pomaidb.search_batch(db, [[0.2]*128], topk=5, membrane="vec_memb")
    assert len(batch_hits) == 1
    assert len(batch_hits[0]) > 0
    assert batch_hits[0][0][0] == 10
    print("Membrane batch search works.")

    # Test flush, compact, and stats
    pomaidb.flush(db)
    print("Flush works.")
    pomaidb.compact_membrane(db, "vec_memb")
    print("Compact membrane works.")
    stats = pomaidb.get_stats(db)
    assert "version" in stats
    assert "vec_memb" in stats["membranes"]
    print("Get stats works:", stats)

    # Test drop membrane
    pomaidb.drop_membrane(db, "vec_memb")
    membranes_after_drop = pomaidb.list_membranes(db)
    assert "vec_memb" not in membranes_after_drop
    print("Drop membrane works.")

    pomaidb.close(db)
    print("DB Closed.")

    # Test opening DB with quantization and memory budget
    dirpath2 = tempfile.mkdtemp()
    try:
        db_quant = pomaidb.open_db(dirpath2, dim=64, metric="cosine", quant_type=pomaidb.QUANT_SQ8, memory_budget_bytes=16*1024*1024)
        pomaidb.put(db_quant, 100, [0.5]*64)
        assert pomaidb.exists(db_quant, 100)
        pomaidb.close(db_quant)
        print("Quantized DB open, put, exists works.")
    finally:
        shutil.rmtree(dirpath2, ignore_errors=True)

    print("ALL TESTS PASSED")
    
finally:
    shutil.rmtree(dirpath, ignore_errors=True)
