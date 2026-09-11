"""
PomaiDB — embedded vector database for Edge AI.

Use the C library (libpomai_c.so / libpomai_c.dylib / pomai_c.dll) via ctypes.
Set POMAI_C_LIB to the path to the shared library, or build from source
and point to build/libpomai_c.so (Linux) or build/pomai_c.dll (Windows).
"""

import ctypes
import json
import os
import sys
from pathlib import Path

__all__ = [
    "open_db", "close", "put", "put_batch", "delete", "exists", "get",
    "search", "search_batch", "search_zero_copy", "release_zero_copy_session",
    "freeze", "compact_membrane", "create_membrane", "list_membranes",
    "resolve_effective_options", "PomaiDBError",
    "MEMBRANE_KIND_VECTOR",
]

MEMBRANE_KIND_VECTOR = 0

class PomaiDBError(Exception):
    pass

def _find_lib():
    env = os.environ.get("POMAI_C_LIB")
    if env:
        return env
    for base in [Path(__file__).resolve().parents[3], Path(__file__).resolve().parents[2], Path.cwd()]:
        for name in ["libpomai_c.so", "libpomai_c.dylib", "pomai_c.dll", "libpomai_c.dll"]:
            p = base / "build" / name
            if p.exists():
                return str(p)
            p_bin = base / "build" / "bin" / name
            if p_bin.exists():
                return str(p_bin)
    return None

_lib_path = _find_lib()
_lib = None

def _ensure_lib():
    global _lib
    if _lib is not None:
        return
    path = _find_lib()
    if not path or not os.path.isfile(path):
        raise PomaiDBError(
            "PomaiDB C library not found. Set POMAI_C_LIB to path to libpomai_c.so (or .dll/.dylib), "
            "or build the project and run from repo root."
        )
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        dll_dir = os.path.dirname(os.path.abspath(path))
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass
        for p in os.environ.get("PATH", "").split(os.pathsep):
            if p and os.path.isdir(p):
                try:
                    os.add_dll_directory(p)
                except Exception:
                    pass
    _lib = ctypes.CDLL(path)
    _register_api(_lib)

def _register_api(lib):
    class PomaiOptions(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("path", ctypes.c_char_p),
            ("shards", ctypes.c_uint32),
            ("dim", ctypes.c_uint32),
            ("search_threads", ctypes.c_uint32),
            ("fsync_policy", ctypes.c_int),
            ("memory_budget_bytes", ctypes.c_uint64),
            ("deadline_ms", ctypes.c_uint32),
            ("index_type", ctypes.c_uint8),
            ("hnsw_m", ctypes.c_uint32),
            ("hnsw_ef_construction", ctypes.c_uint32),
            ("hnsw_ef_search", ctypes.c_uint32),
            ("adaptive_threshold", ctypes.c_uint32),
            ("metric", ctypes.c_uint8),
            ("edge_profile", ctypes.c_uint8),
            ("tick_max_ops", ctypes.c_uint32),
            ("tick_max_ms", ctypes.c_uint32),
            ("strict_deterministic", ctypes.c_bool),
        ]

    class PomaiUpsert(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("id", ctypes.c_uint64),
            ("vector", ctypes.POINTER(ctypes.c_float)),
            ("dim", ctypes.c_uint32),
            ("metadata", ctypes.POINTER(ctypes.c_uint8)),
            ("metadata_len", ctypes.c_uint32),
        ]

    class PomaiQuery(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("vector", ctypes.POINTER(ctypes.c_float)),
            ("dim", ctypes.c_uint32),
            ("topk", ctypes.c_uint32),
            ("filter_expression", ctypes.c_char_p),
            ("partition_device_id", ctypes.c_char_p),
            ("partition_location_id", ctypes.c_char_p),
            ("deadline_ms", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
        ]

    class PomaiSemanticPointer(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("raw_data_ptr", ctypes.c_void_p),
            ("dim", ctypes.c_uint32),
            ("quant_min", ctypes.c_float),
            ("quant_inv_scale", ctypes.c_float),
            ("session_id", ctypes.c_uint64),
        ]

    class PomaiSearchResults(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("count", ctypes.c_size_t),
            ("ids", ctypes.POINTER(ctypes.c_uint64)),
            ("scores", ctypes.POINTER(ctypes.c_float)),
            ("shard_ids", ctypes.POINTER(ctypes.c_uint32)),
            ("total_shards_count", ctypes.c_uint32),
            ("pruned_shards_count", ctypes.c_uint32),
            ("zero_copy_pointers", ctypes.POINTER(PomaiSemanticPointer)),
        ]

    class PomaiRecord(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("id", ctypes.c_uint64),
            ("dim", ctypes.c_uint32),
            ("vector", ctypes.POINTER(ctypes.c_float)),
            ("metadata", ctypes.POINTER(ctypes.c_uint8)),
            ("metadata_len", ctypes.c_uint32),
            ("is_deleted", ctypes.c_bool),
        ]

    lib.PomaiOptions = PomaiOptions
    lib.PomaiUpsert = PomaiUpsert
    lib.PomaiQuery = PomaiQuery
    lib.PomaiSemanticPointer = PomaiSemanticPointer
    lib.PomaiSearchResults = PomaiSearchResults
    lib.PomaiRecord = PomaiRecord

    lib.pomai_options_init.argtypes = [ctypes.POINTER(PomaiOptions)]
    lib.pomai_options_init.restype = None

    lib.pomai_open.argtypes = [ctypes.POINTER(PomaiOptions), ctypes.POINTER(ctypes.c_void_p)]
    lib.pomai_open.restype = ctypes.c_void_p

    lib.pomai_close.argtypes = [ctypes.c_void_p]
    lib.pomai_close.restype = ctypes.c_void_p

    lib.pomai_freeze.argtypes = [ctypes.c_void_p]
    lib.pomai_freeze.restype = ctypes.c_void_p

    lib.pomai_put.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiUpsert)]
    lib.pomai_put.restype = ctypes.c_void_p

    lib.pomai_put_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiUpsert), ctypes.c_size_t]
    lib.pomai_put_batch.restype = ctypes.c_void_p

    lib.pomai_delete.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.pomai_delete.restype = ctypes.c_void_p

    lib.pomai_exists.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_bool)]
    lib.pomai_exists.restype = ctypes.c_void_p

    lib.pomai_get.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.POINTER(PomaiRecord))]
    lib.pomai_get.restype = ctypes.c_void_p

    lib.pomai_record_free.argtypes = [ctypes.POINTER(PomaiRecord)]
    lib.pomai_record_free.restype = None

    lib.pomai_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiQuery), ctypes.POINTER(ctypes.POINTER(PomaiSearchResults))]
    lib.pomai_search.restype = ctypes.c_void_p

    lib.pomai_search_results_free.argtypes = [ctypes.POINTER(PomaiSearchResults)]
    lib.pomai_search_results_free.restype = None

    lib.pomai_search_batch.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(PomaiQuery), ctypes.c_size_t,
        ctypes.POINTER(ctypes.POINTER(PomaiSearchResults))
    ]
    lib.pomai_search_batch.restype = ctypes.c_void_p

    lib.pomai_search_batch_free.argtypes = [ctypes.POINTER(PomaiSearchResults), ctypes.c_size_t]
    lib.pomai_search_batch_free.restype = None

    lib.pomai_create_membrane_kind.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.pomai_create_membrane_kind.restype = ctypes.c_void_p

    lib.pomai_list_membranes_json.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
    lib.pomai_list_membranes_json.restype = ctypes.c_void_p

    lib.pomai_compact_membrane.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.pomai_compact_membrane.restype = ctypes.c_void_p

    lib.pomai_release_pointer.argtypes = [ctypes.c_uint64]
    lib.pomai_release_pointer.restype = None

    lib.pomai_free.argtypes = [ctypes.c_void_p]
    lib.pomai_free.restype = None

    lib.pomai_status_message.argtypes = [ctypes.c_void_p]
    lib.pomai_status_message.restype = ctypes.c_char_p

    lib.pomai_status_free.argtypes = [ctypes.c_void_p]
    lib.pomai_status_free.restype = None

def _check_status(st):
    if not st:
        return
    msg = _lib.pomai_status_message(st)
    err = msg.decode("utf-8", errors="replace") if msg else "Unknown PomaiDB error"
    _lib.pomai_status_free(st)
    raise PomaiDBError(err)

def open_db(path, dim, shards=1, metric="l2", edge_profile=0):
    _ensure_lib()
    opts = _lib.PomaiOptions()
    _lib.pomai_options_init(ctypes.byref(opts))
    opts.path = path.encode("utf-8")
    opts.dim = dim
    opts.shards = shards
    opts.metric = 1 if metric.lower() in ("ip", "innerproduct", "cosine") else 0
    opts.edge_profile = edge_profile
    db_ptr = ctypes.c_void_p()
    st = _lib.pomai_open(ctypes.byref(opts), ctypes.byref(db_ptr))
    _check_status(st)
    return db_ptr

def close(db):
    if db:
        _check_status(_lib.pomai_close(db))

def freeze(db, membrane=""):
    _check_status(_lib.pomai_freeze(db))

def put(db, id, vector, tenant="", membrane=""):
    _ensure_lib()
    up = _lib.PomaiUpsert()
    up.struct_size = ctypes.sizeof(_lib.PomaiUpsert)
    up.id = id
    up.dim = len(vector)
    c_floats = (ctypes.c_float * len(vector))(*vector)
    up.vector = c_floats
    if tenant:
        t_bytes = tenant.encode("utf-8")
        up.metadata = (ctypes.c_uint8 * len(t_bytes))(*t_bytes)
        up.metadata_len = len(t_bytes)
    _check_status(_lib.pomai_put(db, ctypes.byref(up)))

def put_batch(db, ids, vectors, tenants=None):
    _ensure_lib()
    n = len(ids)
    if n == 0:
        return
    arr_type = _lib.PomaiUpsert * n
    arr = arr_type()
    keep_alive = []
    for i in range(n):
        arr[i].struct_size = ctypes.sizeof(_lib.PomaiUpsert)
        arr[i].id = ids[i]
        arr[i].dim = len(vectors[i])
        c_v = (ctypes.c_float * len(vectors[i]))(*vectors[i])
        arr[i].vector = c_v
        keep_alive.append(c_v)
        if tenants and i < len(tenants) and tenants[i]:
            t_b = tenants[i].encode("utf-8")
            c_m = (ctypes.c_uint8 * len(t_b))(*t_b)
            arr[i].metadata = c_m
            arr[i].metadata_len = len(t_b)
            keep_alive.append(c_m)
    _check_status(_lib.pomai_put_batch(db, arr, n))

def delete(db, id):
    _check_status(_lib.pomai_delete(db, id))

def exists(db, id):
    out = ctypes.c_bool()
    _check_status(_lib.pomai_exists(db, id, ctypes.byref(out)))
    return out.value

class Hit(tuple):
    """Search hit (id, score) supporting tuple indexing, dict keys, and attributes."""
    def __new__(cls, id, score):
        return super(Hit, cls).__new__(cls, (id, score))
    @property
    def id(self):
        return self[0]
    @property
    def score(self):
        return self[1]
    def __getitem__(self, item):
        if item == "id":
            return self[0]
        if item == "score":
            return self[1]
        return super().__getitem__(item)
    def __repr__(self):
        return f"Hit(id={self[0]}, score={self[1]})"

def get(db, id):
    rec_ptr = ctypes.POINTER(_lib.PomaiRecord)()
    _check_status(_lib.pomai_get(db, id, ctypes.byref(rec_ptr)))
    if not rec_ptr:
        return None
    r = rec_ptr.contents
    vec = [r.vector[i] for i in range(r.dim)]
    meta = ""
    if r.metadata and r.metadata_len > 0:
        meta = bytes(r.metadata[:r.metadata_len]).decode("utf-8", errors="replace")
    dim = r.dim
    _lib.pomai_record_free(rec_ptr)
    return {"id": id, "dim": dim, "vector": vec, "tenant": meta}

def search(db, query_vector, topk=10, tenant="", membrane=""):
    _ensure_lib()
    q = _lib.PomaiQuery()
    q.struct_size = ctypes.sizeof(_lib.PomaiQuery)
    c_v = (ctypes.c_float * len(query_vector))(*query_vector)
    q.vector = c_v
    q.dim = len(query_vector)
    q.topk = topk
    if tenant:
        q.filter_expression = f"tenant={tenant}".encode("utf-8")
    res_ptr = ctypes.POINTER(_lib.PomaiSearchResults)()
    _check_status(_lib.pomai_search(db, ctypes.byref(q), ctypes.byref(res_ptr)))
    if not res_ptr:
        return []
    res = res_ptr.contents
    hits = [Hit(res.ids[i], res.scores[i]) for i in range(res.count)]
    _lib.pomai_search_results_free(res_ptr)
    return hits

def search_batch(db, query_vectors, topk=10):
    _ensure_lib()
    n = len(query_vectors)
    if n == 0:
        return []
    arr_type = _lib.PomaiQuery * n
    arr = arr_type()
    keep_alive = []
    dim = len(query_vectors[0])
    for i in range(n):
        arr[i].struct_size = ctypes.sizeof(_lib.PomaiQuery)
        c_v = (ctypes.c_float * dim)(*query_vectors[i])
        arr[i].vector = c_v
        arr[i].dim = dim
        arr[i].topk = topk
        keep_alive.append(c_v)
    res_ptr = ctypes.POINTER(_lib.PomaiSearchResults)()
    _check_status(_lib.pomai_search_batch(db, arr, n, ctypes.byref(res_ptr)))
    if not res_ptr:
        return []
    out = []
    for i in range(n):
        r = res_ptr[i]
        hits = [Hit(r.ids[j], r.scores[j]) for j in range(r.count)] if r.count > 0 else []
        out.append(hits)
    _lib.pomai_search_batch_free(res_ptr, n)
    return out

def search_zero_copy(db, query_vector, topk=10):
    _ensure_lib()
    q = _lib.PomaiQuery()
    q.struct_size = ctypes.sizeof(_lib.PomaiQuery)
    c_v = (ctypes.c_float * len(query_vector))(*query_vector)
    q.vector = c_v
    q.dim = len(query_vector)
    q.topk = topk
    q.flags = 1  # ZERO_COPY
    res_ptr = ctypes.POINTER(_lib.PomaiSearchResults)()
    _check_status(_lib.pomai_search(db, ctypes.byref(q), ctypes.byref(res_ptr)))
    if not res_ptr:
        return {"hits": [], "session_id": 0}
    res = res_ptr.contents
    hits = [{"id": res.ids[i], "score": res.scores[i]} for i in range(res.count)]
    sess_id = res.zero_copy_pointers[0].session_id if res.count > 0 and res.zero_copy_pointers else 0
    _lib.pomai_search_results_free(res_ptr)
    return {"hits": hits, "session_id": sess_id}

def release_zero_copy_session(session_id):
    if session_id:
        _lib.pomai_release_pointer(session_id)

def create_membrane(db, name, dim, shard_count=1):
    _check_status(_lib.pomai_create_membrane_kind(db, name.encode("utf-8"), dim, shard_count, 0))

def compact_membrane(db, name):
    _check_status(_lib.pomai_compact_membrane(db, name.encode("utf-8")))

def list_membranes(db):
    out_json = ctypes.c_char_p()
    out_len = ctypes.c_size_t()
    _check_status(_lib.pomai_list_membranes_json(db, ctypes.byref(out_json), ctypes.byref(out_len)))
    if not out_json.value:
        return []
    s = out_json.value.decode("utf-8")
    _lib.pomai_free(ctypes.cast(out_json, ctypes.c_void_p))
    return json.loads(s)

def resolve_effective_options(path, dim, shards=1, edge_profile=0):
    _ensure_lib()
    opts = _lib.PomaiOptions()
    _lib.pomai_options_init(ctypes.byref(opts))
    opts.path = path.encode("utf-8")
    opts.dim = dim
    opts.shards = shards
    opts.edge_profile = edge_profile
    out_json = ctypes.c_char_p()
    out_len = ctypes.c_size_t()
    _check_status(_lib.pomai_options_resolve_json(ctypes.byref(opts), ctypes.byref(out_json), ctypes.byref(out_len)))
    s = out_json.value.decode("utf-8")
    _lib.pomai_free(ctypes.cast(out_json, ctypes.c_void_p))
    return json.loads(s)
