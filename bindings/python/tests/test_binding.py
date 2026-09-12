import shutil
import tempfile
import unittest
import pomaidb

class TestPomaiDBBinding(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="pomai_py_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_database_class_api(self):
        db = pomaidb.Database.open(self.test_dir, dim=4)
        self.assertIsNotNone(db)

        # Put vector
        db.put(1, [1.0, 0.0, 0.0, 0.0])
        db.create_membrane("mem1", dim=4)
        db.open_membrane("mem1")
        db.put(2, [0.0, 1.0, 0.0, 0.0], membrane="mem1", payload=b"payload_val", timestamp=42)

        # Exists
        self.assertTrue(db.exists(1))
        self.assertTrue(db.exists(2, membrane="mem1"))

        # Get
        v1 = db.get(1)
        self.assertEqual(len(v1), 4)
        self.assertAlmostEqual(v1[0], 1.0, places=3)

        rec2 = db.get(2, membrane="mem1", with_metadata=True)
        self.assertEqual(rec2.payload, b"payload_val")
        self.assertEqual(rec2.timestamp, 42)

        # Search
        res = db.search([1.0, 0.0, 0.0, 0.0], topk=2)
        self.assertTrue(len(res) >= 1)
        self.assertEqual(res[0].id, 1)

        # Stats
        stats = db.get_stats()
        self.assertIn("version", stats)

        db.flush()
        db.close()

    def test_functional_c_api(self):
        dir2 = tempfile.mkdtemp(prefix="pomai_py_c_")
        try:
            h = pomaidb.open_db(dir2, dim=4)
            pomaidb.put(h, 10, [0.5, 0.5, 0.5, 0.5])
            self.assertTrue(pomaidb.exists(h, 10))
            v = pomaidb.get(h, 10)
            self.assertEqual(v["id"], 10)
            pomaidb.flush(h)
            pomaidb.close(h)
        finally:
            shutil.rmtree(dir2, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
