import koffi from "koffi";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function findLibrary() {
  if (process.env.POMAI_C_LIB && fs.existsSync(process.env.POMAI_C_LIB)) {
    return process.env.POMAI_C_LIB;
  }
  const candidates = [
    path.join(__dirname, "lib", "libpomai_c.dll"),
    path.join(__dirname, "lib", "pomai_c.dll"),
    path.join(__dirname, "lib", "libpomai_c.so"),
    path.join(__dirname, "lib", "libpomai_c.dylib"),
    path.join(__dirname, "..", "..", "build", "libpomai_c.dll"),
    path.join(__dirname, "..", "..", "build", "libpomai_c.so"),
    path.join(__dirname, "..", "..", "build", "libpomai_c.dylib")
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  throw new Error("Could not locate PomaiDB native library (libpomai_c). Set POMAI_C_LIB.");
}

const libPath = findLibrary();
const lib = koffi.load(libPath);

// Struct definitions
const PomaiOptions = koffi.struct("pomai_options_t", {
  struct_size: "uint32_t",
  path: "str",
  reserved0: "uint32_t",
  dim: "uint32_t",
  search_threads: "uint32_t",
  fsync_policy: "int",
  memory_budget_bytes: "uint64_t",
  deadline_ms: "uint32_t",
  index_type: "uint8_t",
  hnsw_m: "uint32_t",
  hnsw_ef_construction: "uint32_t",
  hnsw_ef_search: "uint32_t",
  adaptive_threshold: "uint32_t",
  metric: "uint8_t",
  edge_profile: "uint8_t",
  tick_max_ops: "uint32_t",
  tick_max_ms: "uint32_t",
  strict_deterministic: "bool",
  quant_type: "uint8_t",
  pq_m: "uint32_t",
  memtable_flush_threshold_mb: "uint32_t",
  auto_freeze_on_pressure: "bool",
  max_memtable_mb: "uint32_t",
  write_coalesce_window_us: "uint32_t",
  write_coalesce_batch_size: "uint32_t",
  enable_encryption_at_rest: "bool",
  encryption_key_hex: "str"
});

const PomaiUpsert = koffi.struct("pomai_upsert_t", {
  struct_size: "uint32_t",
  id: "uint64_t",
  vector: "float*",
  dim: "uint32_t",
  metadata: "uint8_t*",
  metadata_len: "uint32_t",
  membrane: "str",
  timestamp: "uint64_t",
  payload: "uint8_t*",
  payload_len: "uint32_t"
});

const PomaiQuery = koffi.struct("pomai_query_t", {
  struct_size: "uint32_t",
  vector: "float*",
  dim: "uint32_t",
  topk: "uint32_t",
  filter_expression: "str",
  partition_device_id: "str",
  partition_location_id: "str",
  deadline_ms: "uint32_t",
  flags: "uint32_t",
  membrane: "str",
  as_of_ts: "uint64_t",
  as_of_lsn: "uint64_t"
});

const PomaiRecord = koffi.struct("pomai_record_t", {
  struct_size: "uint32_t",
  id: "uint64_t",
  dim: "uint32_t",
  vector: "float*",
  metadata: "uint8_t*",
  metadata_len: "uint32_t",
  is_deleted: "bool",
  timestamp: "uint64_t",
  payload: "uint8_t*",
  payload_len: "uint32_t"
});

const PomaiSearchResults = koffi.struct("pomai_search_results_t", {
  struct_size: "uint32_t",
  count: "size_t",
  ids: "uint64_t*",
  scores: "float*",
  total_locules_count: "uint32_t",
  pruned_locules_count: "uint32_t",
  zero_copy_pointers: "void*"
});

const PomaiStatusPtr = koffi.opaque("pomai_status_t");

// Function bindings
const pomai_status_free = lib.func("void pomai_status_free(pomai_status_t* status)");
const pomai_status_code = lib.func("int pomai_status_code(pomai_status_t* status)");
const pomai_status_message = lib.func("str pomai_status_message(pomai_status_t* status)");

const pomai_options_init = lib.func("void pomai_options_init(_Out_ pomai_options_t* opts)");
const pomai_open = lib.func("pomai_status_t* pomai_open(pomai_options_t* opts, _Out_ void** out_db)");
const pomai_close = lib.func("pomai_status_t* pomai_close(void* db)");
const pomai_flush = lib.func("pomai_status_t* pomai_flush(void* db)");
const pomai_freeze = lib.func("pomai_status_t* pomai_freeze(void* db)");
const pomai_freeze_membrane = lib.func("pomai_status_t* pomai_freeze_membrane(void* db, str membrane)");
const pomai_compact = lib.func("pomai_status_t* pomai_compact(void* db)");
const pomai_compact_membrane = lib.func("pomai_status_t* pomai_compact_membrane(void* db, str membrane_name)");
const pomai_get_stats_json = lib.func("pomai_status_t* pomai_get_stats_json(void* db, _Out_ void** out_json, _Out_ size_t* out_len)");

const pomai_put = lib.func("pomai_status_t* pomai_put(void* db, pomai_upsert_t* item)");
const pomai_put_membrane = lib.func("pomai_status_t* pomai_put_membrane(void* db, str membrane, pomai_upsert_t* item)");
const pomai_delete = lib.func("pomai_status_t* pomai_delete(void* db, uint64_t id)");
const pomai_delete_membrane = lib.func("pomai_status_t* pomai_delete_membrane(void* db, str membrane, uint64_t id)");
const pomai_exists = lib.func("pomai_status_t* pomai_exists(void* db, uint64_t id, _Out_ bool* out_exists)");
const pomai_exists_membrane = lib.func("pomai_status_t* pomai_exists_membrane(void* db, str membrane, uint64_t id, _Out_ bool* out_exists)");
const pomai_get = lib.func("pomai_status_t* pomai_get(void* db, uint64_t id, _Out_ void** out_record)");
const pomai_get_membrane = lib.func("pomai_status_t* pomai_get_membrane(void* db, str membrane, uint64_t id, _Out_ void** out_record)");
const pomai_record_free = lib.func("void pomai_record_free(void* record)");

const pomai_search = lib.func("pomai_status_t* pomai_search(void* db, pomai_query_t* query, _Out_ void** out)");
const pomai_search_membrane = lib.func("pomai_status_t* pomai_search_membrane(void* db, str membrane, pomai_query_t* query, _Out_ void** out)");
const pomai_search_results_free = lib.func("void pomai_search_results_free(void* results)");

const pomai_create_membrane_kind = lib.func("pomai_status_t* pomai_create_membrane_kind(void* db, str name, uint32_t dim, uint32_t kind)");
const pomai_drop_membrane = lib.func("pomai_status_t* pomai_drop_membrane(void* db, str membrane_name)");
const pomai_open_membrane = lib.func("pomai_status_t* pomai_open_membrane(void* db, str membrane_name)");
const pomai_close_membrane = lib.func("pomai_status_t* pomai_close_membrane(void* db, str membrane_name)");
const pomai_list_membranes_json = lib.func("pomai_status_t* pomai_list_membranes_json(void* db, _Out_ void** out_json, _Out_ size_t* out_len)");

const pomai_free = lib.func("void pomai_free(void* ptr)");

function checkStatus(st) {
  if (st !== null && st !== undefined) {
    const code = pomai_status_code(st);
    const msg = pomai_status_message(st);
    pomai_status_free(st);
    throw new Error(`PomaiDB error (code ${code}): ${msg || "Unknown error"}`);
  }
}

export const MetricType = {
  L2: 0,
  InnerProduct: 1,
  Cosine: 2
};

export const QuantType = {
  None: 0,
  SQ8: 1,
  FP16: 2,
  Bit: 3,
  PQ8: 4
};

export class Database {
  constructor(handle) {
    this._handle = handle;
  }

  static open(options) {
    const opts = { struct_size: koffi.sizeof(PomaiOptions) };
    pomai_options_init(opts);
    opts.struct_size = koffi.sizeof(PomaiOptions);
    opts.path = options.path;
    opts.dim = options.dim;
    opts.shards = options.shards || 1;
    opts.metric = options.metric !== undefined ? options.metric : MetricType.L2;
    opts.quant_type = options.quantType !== undefined ? options.quantType : QuantType.None;
    if (options.memoryBudgetBytes) opts.memory_budget_bytes = BigInt(options.memoryBudgetBytes);

    const outDb = [null];
    checkStatus(pomai_open(opts, outDb));
    return new Database(outDb[0]);
  }

  close() {
    if (this._handle) {
      checkStatus(pomai_close(this._handle));
      this._handle = null;
    }
  }

  put(id, vector, options = {}) {
    const upsert = {
      struct_size: koffi.sizeof(PomaiUpsert),
      id: BigInt(id),
      vector: vector,
      dim: vector.length,
      metadata: null,
      metadata_len: 0,
      membrane: options.membrane || null,
      timestamp: BigInt(options.timestamp || 0),
      payload: options.payload ? Buffer.from(options.payload) : null,
      payload_len: options.payload ? options.payload.length : 0
    };
    if (options.membrane) {
      checkStatus(pomai_put_membrane(this._handle, options.membrane, upsert));
    } else {
      checkStatus(pomai_put(this._handle, upsert));
    }
  }

  get(id, membrane = null) {
    const outRec = [null];
    try {
      if (membrane) {
        checkStatus(pomai_get_membrane(this._handle, membrane, BigInt(id), outRec));
      } else {
        checkStatus(pomai_get(this._handle, BigInt(id), outRec));
      }
    } catch (e) {
      return null;
    }

    const recPtr = outRec[0];
    if (!recPtr) return null;

    try {
function decodeFloats(ptr, count) {
  const arr = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    arr[i] = koffi.decode(ptr, i * 4, "float");
  }
  return arr;
}

function decodeBytes(ptr, len) {
  const buf = Buffer.alloc(len);
  for (let i = 0; i < len; i++) {
    buf[i] = koffi.decode(ptr, i, "uint8_t");
  }
  return buf;
}

      const rec = koffi.decode(recPtr, PomaiRecord);
      const vec = decodeFloats(rec.vector, rec.dim);
      let payload = null;
      if (rec.payload && rec.payload_len > 0) {
        payload = decodeBytes(rec.payload, rec.payload_len);
      }
      return {
        id: Number(rec.id),
        vector: Array.from(vec),
        dim: rec.dim,
        timestamp: Number(rec.timestamp),
        payload: payload,
        isDeleted: rec.is_deleted
      };
    } finally {
      pomai_record_free(recPtr);
    }
  }

  exists(id, membrane = null) {
    const out = [false];
    if (membrane) {
      checkStatus(pomai_exists_membrane(this._handle, membrane, BigInt(id), out));
    } else {
      checkStatus(pomai_exists(this._handle, BigInt(id), out));
    }
    return out[0];
  }

  delete(id, membrane = null) {
    if (membrane) {
      checkStatus(pomai_delete_membrane(this._handle, membrane, BigInt(id)));
    } else {
      checkStatus(pomai_delete(this._handle, BigInt(id)));
    }
  }

  search(queryVector, topK = 10, options = {}) {
    const q = {
      struct_size: koffi.sizeof(PomaiQuery),
      vector: queryVector,
      dim: queryVector.length,
      topk: topK,
      filter_expression: options.filterExpression || null,
      partition_device_id: null,
      partition_location_id: null,
      deadline_ms: 0,
      flags: 0,
      membrane: options.membrane || null,
      as_of_ts: BigInt(options.asOfTs || 0),
      as_of_lsn: BigInt(options.asOfLsn || 0)
    };

    const outResults = [null];
    if (options.membrane) {
      checkStatus(pomai_search_membrane(this._handle, options.membrane, q, outResults));
    } else {
      checkStatus(pomai_search(this._handle, q, outResults));
    }

    const resPtr = outResults[0];
    if (!resPtr) return [];

    try {
      const res = koffi.decode(resPtr, PomaiSearchResults);
      const hits = [];
      const count = Number(res.count);
      if (count > 0 && res.ids && res.scores) {
        for (let i = 0; i < count; ++i) {
          const id = koffi.decode(res.ids, i * 8, "uint64_t");
          const score = koffi.decode(res.scores, i * 4, "float");
          hits.push({ id: Number(id), score: score });
        }
      }
      return hits;
    } finally {
      pomai_search_results_free(resPtr);
    }
  }

  flush() {
    checkStatus(pomai_flush(this._handle));
  }

  freeze(membrane = null) {
    if (membrane) {
      checkStatus(pomai_freeze_membrane(this._handle, membrane));
    } else {
      checkStatus(pomai_freeze(this._handle));
    }
  }

  compact(membrane = null) {
    if (membrane) {
      checkStatus(pomai_compact_membrane(this._handle, membrane));
    } else {
      checkStatus(pomai_compact(this._handle));
    }
  }

  createMembrane(name, dim) {
    checkStatus(pomai_create_membrane_kind(this._handle, name, dim, 0));
  }

  dropMembrane(name) {
    checkStatus(pomai_drop_membrane(this._handle, name));
  }

  openMembrane(name) {
    checkStatus(pomai_open_membrane(this._handle, name));
  }

  closeMembrane(name) {
    checkStatus(pomai_close_membrane(this._handle, name));
  }

  listMembranes() {
    const outJson = [null];
    const outLen = [0];
    checkStatus(pomai_list_membranes_json(this._handle, outJson, outLen));
    if (!outJson[0]) return [];
    try {
      const len = Number(outLen[0]);
      const jsonStr = koffi.decode(outJson[0], "char", len);
      return JSON.parse(jsonStr);
    } finally {
      pomai_free(outJson[0]);
    }
  }

  getStats() {
    const outJson = [null];
    const outLen = [0];
    checkStatus(pomai_get_stats_json(this._handle, outJson, outLen));
    if (!outJson[0]) return {};
    try {
      const len = Number(outLen[0]);
      const jsonStr = koffi.decode(outJson[0], "char", len);
      return JSON.parse(jsonStr);
    } finally {
      pomai_free(outJson[0]);
    }
  }
}

export default Database;