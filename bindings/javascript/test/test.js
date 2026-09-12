import { Database, MetricType, QuantType } from "../index.js";
import fs from "fs";
import path from "path";

const testDir = path.join(process.cwd(), "tmp_js_test_" + Date.now());
if (fs.existsSync(testDir)) {
  fs.rmSync(testDir, { recursive: true, force: true });
}

console.log("Testing PomaiDB JS bindings...");
const db = Database.open({
  path: testDir,
  dim: 4,
  metric: MetricType.L2,
  quantType: QuantType.None
});
console.log("Database opened");

db.put(1, [1.0, 0.0, 0.0, 0.0]);
console.log("Put 1 done");
db.createMembrane("test_mem", 4);
console.log("createMembrane done");
db.openMembrane("test_mem");
console.log("openMembrane done");
db.put(2, [0.0, 1.0, 0.0, 0.0], {
  membrane: "test_mem",
  timestamp: 1234567,
  payload: Buffer.from("hello world")
});
console.log("Put 2 done");

if (!db.exists(1)) throw new Error("ID 1 should exist");
console.log("exists 1 passed");
if (!db.exists(2, "test_mem")) throw new Error("ID 2 should exist in test_mem");
console.log("exists 2 passed");

const rec = db.get(2, "test_mem");
console.log("get 2 returned:", rec);
if (!rec) throw new Error("Record 2 should not be null");
if (rec.id !== 2) throw new Error("Record id mismatch");
if (rec.timestamp !== 1234567) throw new Error("Record timestamp mismatch");
if (!rec.payload || rec.payload.toString() !== "hello world") throw new Error("Record payload mismatch");
console.log("Record assertions passed");

const hits = db.search([1.0, 0.0, 0.0, 0.0], 2);
console.log("Search hits returned:", hits);
if (hits.length === 0 || hits[0].id !== 1) throw new Error("Search hits mismatch");
console.log("Search assertions passed");

const stats = db.getStats();
console.log("Stats returned:", stats);
if (!stats.version) throw new Error("Stats version missing");

db.flush();
console.log("Flush done");
db.close();
console.log("Close done");

try {
  fs.rmSync(testDir, { recursive: true, force: true });
} catch(e) {}
console.log("ALL JS BINDING TESTS PASSED!");