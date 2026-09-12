# PomaiDB JavaScript / Node.js Bindings

Official Node.js bindings for **PomaiDB**, an embedded vector database for Edge AI.

## Installation

```bash
npm install pomaidb
```

## Quick Start

```javascript
import { Database, MetricType, QuantType } from "pomaidb";

// 1. Open Database
const db = Database.open({
  path: "./test_db",
  dim: 4,
  metric: MetricType.L2
});

// 2. Put vectors
db.put(1, [1.0, 0.0, 0.0, 0.0]);
db.createMembrane("docs", 4);
db.openMembrane("docs");
db.put(2, [0.0, 1.0, 0.0, 0.0], {
  membrane: "docs",
  timestamp: Date.now(),
  payload: Buffer.from("metadata_json")
});

// 3. Search
const hits = db.search([1.0, 0.0, 0.0, 0.0], 5);
console.log("Search hits:", hits);

// 4. Retrieve Record
const rec = db.get(2, "docs");
console.log("Retrieved record:", rec);

// 5. Cleanup
db.flush();
db.close();
```