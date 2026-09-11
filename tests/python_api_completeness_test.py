import tempfile
import pomaidb
import shutil
import time

dirpath = tempfile.mkdtemp()
print("Starting DB...")
try:
    db = pomaidb.open_db(dirpath, dim=128, shards=1, metric="ip")
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
    pomaidb.create_membrane(db, "vec_memb", dim=128, shard_count=1)
    membranes = pomaidb.list_membranes(db)
    assert "vec_memb" in membranes
    print("Membrane list works:", membranes)

    # Put and search in membrane
    pomaidb.put(db, 10, [0.2]*128, membrane="vec_memb")
    pomaidb.freeze(db, membrane="vec_memb")
    hits = pomaidb.search(db, [0.2]*128, topk=5, membrane="vec_memb")
    assert len(hits) > 0
    assert hits[0][0] == 10
    print("Membrane search works:", hits)

    pomaidb.close(db)
    print("DB Closed.")
    print("ALL TESTS PASSED")
    
finally:
    shutil.rmtree(dirpath)
