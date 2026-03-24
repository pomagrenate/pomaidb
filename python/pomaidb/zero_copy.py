"""
Zero-copy helpers for PomaiDB search results.

This module uses the existing C ABI zero-copy semantic pointer path
(`POMAI_QUERY_FLAG_ZERO_COPY`) and maps the returned memory into NumPy
without copying.
"""

from __future__ import annotations

import ctypes

from . import _ensure_lib, _lib, PomaiDBError

POMAI_QUERY_FLAG_ZERO_COPY = 1


def release_zero_copy_session(session_id: int) -> None:
    """Release a pinned zero-copy session id."""
    _ensure_lib()
    _lib.pomai_release_pointer(int(session_id))


def search_zero_copy(db, query, topk: int = 10):
    """
    Execute one query with zero-copy enabled.

    Returns list of dicts:
      {"id", "score", "raw_u8", "dequant_f32", "session_id"}
    """
    _ensure_lib()
    try:
        import numpy as np
    except Exception as exc:
        raise PomaiDBError("numpy is required for zero-copy helpers") from exc

    dim = len(query)
    qvec = (ctypes.c_float * dim)(*query)
    q = _lib._pomai_query()
    q.struct_size = ctypes.sizeof(_lib._pomai_query())
    q.vector = qvec
    q.dim = dim
    q.topk = int(topk)
    q.filter_expression = None
    q.partition_device_id = None
    q.partition_location_id = None
    q.as_of_ts = 0
    q.as_of_lsn = 0
    q.aggregate_op = 0
    q.aggregate_field = None
    q.aggregate_topk = 0
    q.mesh_detail_preference = 0
    q.alpha = 1.0
    q.deadline_ms = 0
    q.flags = POMAI_QUERY_FLAG_ZERO_COPY

    out = ctypes.POINTER(_lib._pomai_search_results)()
    st = _lib.pomai_search(db, ctypes.byref(q), ctypes.byref(out))
    if st:
        msg = _lib.pomai_status_message(st).decode("utf-8", errors="replace")
        _lib.pomai_status_free(st)
        raise PomaiDBError(msg)

    rows = []
    try:
        count = out.contents.count
        pointers = out.contents.zero_copy_pointers
        for i in range(min(int(topk), count)):
            session_id = 0
            raw_u8 = None
            dequant_f32 = None
            if pointers:
                p = pointers[i]
                session_id = int(p.session_id)
                if p.raw_data_ptr and p.dim > 0:
                    addr = int(ctypes.cast(p.raw_data_ptr, ctypes.c_void_p).value or 0)
                    if addr:
                        buf = (ctypes.c_uint8 * int(p.dim)).from_address(addr)
                        raw_u8 = np.frombuffer(buf, dtype=np.uint8, count=int(p.dim))
                        dequant_f32 = raw_u8.astype(np.float32) * float(p.quant_inv_scale) + float(p.quant_min)
            rows.append(
                {
                    "id": int(out.contents.ids[i]),
                    "score": float(out.contents.scores[i]),
                    "raw_u8": raw_u8,
                    "dequant_f32": dequant_f32,
                    "session_id": session_id,
                }
            )
        return rows
    finally:
        _lib.pomai_search_results_free(out)

