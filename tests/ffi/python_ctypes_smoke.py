#!/usr/bin/env python3
import ctypes
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
lib_name = 'libpomai_c.dll' if sys.platform == 'win32' else 'libpomai_c.so'
LIB = ROOT / 'build' / lib_name

if not LIB.exists():
    raise SystemExit(f'missing shared library: {LIB}')

if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(str(LIB.parent))
    for p in os.environ.get('PATH', '').split(os.pathsep):
        if p and os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass

lib = ctypes.CDLL(str(LIB))

class PomaiOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("path", ctypes.c_char_p),
        ("reserved0", ctypes.c_uint32),
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
        ("quant_type", ctypes.c_uint8),
        ("pq_m", ctypes.c_uint32),
        ("memtable_flush_threshold_mb", ctypes.c_uint32),
        ("auto_freeze_on_pressure", ctypes.c_bool),
        ("max_memtable_mb", ctypes.c_uint32),
        ("write_coalesce_window_us", ctypes.c_uint32),
        ("write_coalesce_batch_size", ctypes.c_uint32),
        ("enable_encryption_at_rest", ctypes.c_bool),
        ("encryption_key_hex", ctypes.c_char_p),
    ]

class PomaiUpsert(ctypes.Structure):
    _fields_ = [
        ('struct_size', ctypes.c_uint32),
        ('id', ctypes.c_uint64),
        ('vector', ctypes.POINTER(ctypes.c_float)),
        ('dim', ctypes.c_uint32),
        ('metadata', ctypes.POINTER(ctypes.c_uint8)),
        ('metadata_len', ctypes.c_uint32),
        ('membrane', ctypes.c_char_p),
        ('timestamp', ctypes.c_uint64),
        ('payload', ctypes.POINTER(ctypes.c_uint8)),
        ('payload_len', ctypes.c_uint32),
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
        ("membrane", ctypes.c_char_p),
        ("as_of_ts", ctypes.c_uint64),
        ("as_of_lsn", ctypes.c_uint64),
    ]

class PomaiSearchResults(ctypes.Structure):
    _fields_ = [
        ('struct_size', ctypes.c_uint32),
        ('count', ctypes.c_size_t),
        ('ids', ctypes.POINTER(ctypes.c_uint64)),
        ('scores', ctypes.POINTER(ctypes.c_float)),
        ('total_locules_count', ctypes.c_uint32),
        ('pruned_locules_count', ctypes.c_uint32),
        ('zero_copy_pointers', ctypes.c_void_p),  # pomai_semantic_pointer_t*; we ignore
    ]

# Options, Upsert, Query, Results
lib.pomai_options_init.argtypes = [ctypes.POINTER(PomaiOptions)]
lib.pomai_options_init.restype = None
lib.pomai_open.argtypes = [ctypes.POINTER(PomaiOptions), ctypes.POINTER(ctypes.c_void_p)]
lib.pomai_open.restype = ctypes.c_void_p
lib.pomai_put_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiUpsert), ctypes.c_size_t]
lib.pomai_put_batch.restype = ctypes.c_void_p
lib.pomai_freeze.argtypes = [ctypes.c_void_p]
lib.pomai_freeze.restype = ctypes.c_void_p
lib.pomai_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiQuery), ctypes.POINTER(ctypes.POINTER(PomaiSearchResults))]
lib.pomai_search.restype = ctypes.c_void_p
lib.pomai_search_results_free.argtypes = [ctypes.POINTER(PomaiSearchResults)]
lib.pomai_search_results_free.restype = None
lib.pomai_close.argtypes = [ctypes.c_void_p]
lib.pomai_close.restype = ctypes.c_void_p
lib.pomai_status_message.argtypes = [ctypes.c_void_p]
lib.pomai_status_message.restype = ctypes.c_char_p
lib.pomai_status_free.argtypes = [ctypes.c_void_p]
lib.pomai_status_free.restype = None


def check_status(st):
    if st:
        msg = lib.pomai_status_message(st).decode('utf-8', errors='replace')
        lib.pomai_status_free(st)
        raise RuntimeError(msg)


def main():
    with tempfile.TemporaryDirectory(prefix='pomai_ffi_smoke_') as td:
        opts = PomaiOptions()
        lib.pomai_options_init(ctypes.byref(opts))
        opts.struct_size = ctypes.sizeof(PomaiOptions)
        path_buf = ctypes.create_string_buffer(td.encode('utf-8') + b'\0')
        opts.path = ctypes.cast(path_buf, ctypes.c_char_p)
        opts.dim = 8

        db = ctypes.c_void_p()
        check_status(lib.pomai_open(ctypes.byref(opts), ctypes.byref(db)))

        vecs = [(ctypes.c_float * 8)(float(i), 0, 0, 0, 0, 0, 0, 0) for i in range(1, 5)]
        batch = (PomaiUpsert * 4)()
        for i in range(4):
            batch[i].struct_size = ctypes.sizeof(PomaiUpsert)
            batch[i].id = i + 1
            batch[i].vector = vecs[i]
            batch[i].dim = 8
            batch[i].metadata = None
            batch[i].metadata_len = 0

        check_status(lib.pomai_put_batch(db, batch, 4))
        check_status(lib.pomai_freeze(db))

        qv = (ctypes.c_float * 8)(3.0, 0, 0, 0, 0, 0, 0, 0)
        query = PomaiQuery()
        query.struct_size = ctypes.sizeof(PomaiQuery)
        query.vector = qv
        query.dim = 8
        query.topk = 2
        query.filter_expression = None
        query.deadline_ms = 0
        query.flags = 0

        out = ctypes.POINTER(PomaiSearchResults)()
        check_status(lib.pomai_search(db, ctypes.byref(query), ctypes.byref(out)))
        if not out or out.contents.count == 0:
            raise RuntimeError('empty search results')

        first_id = out.contents.ids[0]
        first_score = out.contents.scores[0]
        if first_id == 0:
            raise RuntimeError('invalid first id')
        if not (-1e6 < first_score < 1e6):
            raise RuntimeError('invalid first score')

        lib.pomai_search_results_free(out)
        check_status(lib.pomai_close(db))

    # Package-level import check for SDK
    py_pkg = ROOT / "sdk" / "python"
    sys.path.insert(0, str(py_pkg))
    import pomaidb  # type: ignore

    if not hasattr(pomaidb, "search_zero_copy"):
        raise RuntimeError("missing merged API: search_zero_copy")
    if not hasattr(pomaidb, "release_zero_copy_session"):
        raise RuntimeError("missing merged API: release_zero_copy_session")

    print("python_ctypes_smoke: ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
