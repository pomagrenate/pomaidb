#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import pomaidb


def http_get(url: str, token: str | None = None) -> str:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read().decode("utf-8", errors="replace")


def http_post(url: str, body: str = "", token: str | None = None, *, idempotency: str | None = None) -> str:
    data = body.encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if idempotency:
        req.add_header("Idempotency-Key", idempotency)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


def cmd_preflight(args) -> int:
    txt = pomaidb.resolve_effective_options(
        args.path,
        args.dim,
        profile=args.profile,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        adaptive_threshold=args.adaptive_threshold,
    )
    print(txt)
    return 0


def cmd_status(args) -> int:
    print(http_get(f"http://{args.host}:{args.port}/v1/health", args.token))
    print(http_get(f"http://{args.host}:{args.port}/v1/healthz", args.token))
    return 0


def cmd_health(args) -> int:
    healthz = json.loads(http_get(f"http://{args.host}:{args.port}/v1/healthz", args.token))
    if args.assert_ok and healthz.get("status") != "ok":
        print(json.dumps({"status": "failed", "healthz": healthz}))
        return 2
    print(json.dumps({"status": "ok", "healthz": healthz}))
    return 0


def cmd_metrics(args) -> int:
    print(http_get(f"http://{args.host}:{args.port}/v1/metrics", args.token))
    return 0


def cmd_doctor(args) -> int:
    health = http_get(f"http://{args.host}:{args.port}/v1/healthz", args.token)
    metrics = http_get(f"http://{args.host}:{args.port}/v1/metrics", args.token)
    print("healthz:", health)
    lines = [ln for ln in metrics.splitlines() if ln.strip()]
    wanted = ("http_errors_total", "ingest_errors_total", "rate_limited_total", "auth_denied_total")
    print("key_metrics:")
    for ln in lines:
        if any(ln.startswith(w) for w in wanted):
            print(" ", ln)
    return 0

def _open_local_db(args):
    return pomaidb.open_db(
        args.path,
        args.dim,
        shards=args.shards,
        metric=args.metric,
        profile=args.profile,
    )

def cmd_membranes(args) -> int:
    db = _open_local_db(args)
    try:
        names = pomaidb.list_membranes(db)
        print(json.dumps({"count": len(names), "membranes": names}))
    finally:
        pomaidb.close(db)
    return 0

def cmd_compact(args) -> int:
    db = _open_local_db(args)
    try:
        pomaidb.compact_membrane(db, args.membrane)
    finally:
        pomaidb.close(db)
    print(json.dumps({"status": "ok", "action": "compact", "membrane": args.membrane}))
    return 0

def cmd_gateway(args) -> int:
    health = http_get(f"http://{args.host}:{args.port}/v1/health", args.token)
    healthz = http_get(f"http://{args.host}:{args.port}/v1/healthz", args.token)
    metrics = http_get(f"http://{args.host}:{args.port}/v1/metrics", args.token)
    out = {
        "health": json.loads(health),
        "healthz": json.loads(healthz),
        "metrics_sample": {},
    }
    wanted = (
        "http_requests_total",
        "http_errors_total",
        "ingest_requests_total",
        "ingest_errors_total",
        "rate_limited_total",
        "auth_denied_total",
        "sync_attempts_total",
        "sync_success_total",
        "sync_fail_total",
        "sync_queue_depth",
        "sync_backlog_drops_total",
    )
    for ln in metrics.splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[0] in wanted:
            out["metrics_sample"][parts[0]] = parts[1]
    print(json.dumps(out))
    return 0

def cmd_ingest_http(args) -> int:
    """POST raw body to a gateway ingest path (e.g. /v1/ingest/vector/default/1)."""
    url = f"http://{args.host}:{args.port}{args.path}"
    body = ""
    if args.body_file:
        body = Path(args.body_file).read_text(encoding="utf-8")
    elif args.body is not None:
        body = args.body
    out = http_post(url, body, args.token, idempotency=args.idempotency)
    print(out)
    return 0

def cmd_query(args) -> int:
    db = _open_local_db(args)
    try:
        vals = [float(x.strip()) for x in args.vector.split(",") if x.strip()]
        out = pomaidb.search_batch(db, [vals], topk=args.topk)
        print(json.dumps({"status": "ok", "membrane": args.membrane, "topk": args.topk, "results": out}))
    finally:
        pomaidb.close(db)
    return 0

def cmd_explain(args) -> int:
    template_key = f"v1:{int(args.enable_graph_rerank)}:{int(args.include_docs)}"
    plan = {
        "schema_version": "v1",
        "template_key": template_key,
        "template_cache_hit": True,
        "query_mode": "multi_modal_lite",
        "steps": [
            "vector_prefilter",
            "graph_rerank" if args.enable_graph_rerank else "graph_rerank_skipped",
            "document_evidence" if args.include_docs else "document_evidence_skipped",
        ],
        "notes": "Planner-lite explain output for operator debugging.",
    }
    print(json.dumps(plan))
    return 0

def cmd_replay(args) -> int:
    src = Path(args.file)
    n = 0
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            et = ev.get("type", "")
            membrane = ev.get("membrane", "")
            if et == "vector_put":
                path = f"/v1/ingest/vector/{membrane}/{ev.get('id', 0)}"
                body = ev.get("aux_v", "")
            elif et == "timeseries_put":
                path = f"/v1/ingest/timeseries/{membrane}/{ev.get('id', 0)}/{ev.get('id2', 0)}"
                body = ev.get("aux_v", "")
            elif et == "kv_put":
                path = f"/v1/ingest/keyvalue/{membrane}/{ev.get('aux_k', '')}"
                body = ev.get("aux_v", "")
            elif et == "document_put":
                path = f"/v1/ingest/document/{membrane}/{ev.get('id', 0)}"
                body = ev.get("aux_v", "")
            else:
                continue
            _ = http_post(f"http://{args.host}:{args.port}{path}", body, args.token, idempotency=f"replay-{n}")
            n += 1
    print(json.dumps({"status": "ok", "replayed_events": n}))
    return 0

def cmd_inspect_segment(args) -> int:
    membrane_dir = Path(args.path) / "membranes" / args.membrane
    segs = sorted(membrane_dir.glob("seg_*.dat"))
    out = []
    for p in segs:
        out.append({"file": p.name, "bytes": p.stat().st_size})
    print(json.dumps({"status": "ok", "membrane": args.membrane, "segments": out}))
    return 0

def cmd_snapshot(args) -> int:
    db_path = Path(args.path).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "export":
        subprocess.run(["tar", "--zstd", "-cf", str(out), "-C", str(db_path.parent), db_path.name], check=True)
        print(json.dumps({"status": "ok", "action": "export", "snapshot": str(out)}))
    else:
        target = Path(args.target).resolve()
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "--zstd", "-xf", str(out), "-C", str(target)], check=True)
        print(json.dumps({"status": "ok", "action": "import", "snapshot": str(out), "target": str(target)}))
    return 0

def cmd_doctor_repair(args) -> int:
    checks = {
        "manifest_exists": (Path(args.path) / "MANIFEST").exists(),
        "gateway_dir_exists": (Path(args.path) / "gateway").exists(),
    }
    repaired = []
    if args.repair and not checks["gateway_dir_exists"]:
        (Path(args.path) / "gateway").mkdir(parents=True, exist_ok=True)
        checks["gateway_dir_exists"] = True
        repaired.append("gateway_dir_created")
    status = "ok" if all(checks.values()) else "degraded"
    print(json.dumps({"status": status, "checks": checks, "repaired": repaired, "dry_run": args.dry_run}))
    return 0


def cmd_verify(args) -> int:
    db_path = Path(args.path)
    checks = {
        "db_path_exists": db_path.exists(),
        "manifest_exists": (db_path / "MANIFEST").exists(),
        "gateway_dir_exists": (db_path / "gateway").exists(),
    }
    ok = all(checks.values())
    print(json.dumps({"status": "ok" if ok else "degraded", "checks": checks}))
    return 0

def cmd_lifecycle(args) -> int:
    db = _open_local_db(args)
    try:
        if args.show:
            info = pomaidb.get_membrane_retention(db, args.membrane)
            print(json.dumps({"status": "ok", "action": "show", **info}))
            return 0
        if args.create:
            pomaidb.create_membrane_kind(
                db, args.membrane, args.dim, args.shards, args.kind,
                ttl_sec=args.ttl_sec,
                retention_max_count=args.retention_max_count,
                retention_max_bytes=args.retention_max_bytes,
            )
            action = "created"
        else:
            pomaidb.update_membrane_retention(
                db, args.membrane,
                ttl_sec=args.ttl_sec,
                retention_max_count=args.retention_max_count,
                retention_max_bytes=args.retention_max_bytes,
            )
            action = "updated"
    finally:
        pomaidb.close(db)
    print(json.dumps({
        "status": "ok",
        "action": action,
        "membrane": args.membrane,
        "ttl_sec": args.ttl_sec,
        "retention_max_count": args.retention_max_count,
        "retention_max_bytes": args.retention_max_bytes,
    }))
    return 0


def cmd_profile(args) -> int:
    db = pomaidb.open_db(args.path, args.dim, shards=args.shards, metric=args.metric, profile=args.set)
    try:
        resolved = pomaidb.resolve_effective_options(args.path, args.dim, shards=args.shards, metric=args.metric, profile=args.set)
        print(resolved)
    finally:
        pomaidb.close(db)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="pomaictl edge operations helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--path", required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--profile", default="user_defined")
    p.add_argument("--hnsw-m", type=int, default=32)
    p.add_argument("--hnsw-ef-construction", type=int, default=200)
    p.add_argument("--hnsw-ef-search", type=int, default=64)
    p.add_argument("--adaptive-threshold", type=int, default=0)
    p.set_defaults(func=cmd_preflight)

    for name, fn in (("status", cmd_status), ("metrics", cmd_metrics), ("doctor", cmd_doctor)):
        c = sub.add_parser(name)
        c.add_argument("--host", default="127.0.0.1")
        c.add_argument("--port", type=int, default=8080)
        c.add_argument("--token", default=None)
        c.set_defaults(func=fn)
    h = sub.add_parser("health")
    h.add_argument("--host", default="127.0.0.1")
    h.add_argument("--port", type=int, default=8080)
    h.add_argument("--token", default=None)
    h.add_argument("--assert", dest="assert_ok", action="store_true", help="Exit non-zero if healthz.status != ok")
    h.set_defaults(func=cmd_health)

    ing = sub.add_parser("ingest-http", help="POST to /v1/ingest/... (body from --body or --body-file)")
    ing.add_argument("--host", default="127.0.0.1")
    ing.add_argument("--port", type=int, default=8080)
    ing.add_argument("--token", default=None)
    ing.add_argument("--path", required=True, help="e.g. /v1/ingest/graph/vertex/default/10/1")
    ing.add_argument("--body", default=None, help="Request body (e.g. CSV floats)")
    ing.add_argument("--body-file", default=None, help="Read body from file")
    ing.add_argument("--idempotency", default=None, help="Idempotency-Key header value")
    ing.set_defaults(func=cmd_ingest_http)

    g = sub.add_parser("gateway")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8080)
    g.add_argument("--token", default=None)
    g.set_defaults(func=cmd_gateway)

    m = sub.add_parser("membranes")
    m.add_argument("--path", required=True)
    m.add_argument("--dim", type=int, required=True)
    m.add_argument("--shards", type=int, default=1)
    m.add_argument("--metric", default="ip")
    m.add_argument("--profile", default="user_defined")
    m.set_defaults(func=cmd_membranes)

    c = sub.add_parser("compact")
    c.add_argument("--path", required=True)
    c.add_argument("--dim", type=int, required=True)
    c.add_argument("--shards", type=int, default=1)
    c.add_argument("--metric", default="ip")
    c.add_argument("--profile", default="user_defined")
    c.add_argument("--membrane", required=True)
    c.set_defaults(func=cmd_compact)

    q = sub.add_parser("query")
    q.add_argument("--path", required=True)
    q.add_argument("--dim", type=int, required=True)
    q.add_argument("--shards", type=int, default=1)
    q.add_argument("--metric", default="ip")
    q.add_argument("--profile", default="user_defined")
    q.add_argument("--membrane", default="__default__")
    q.add_argument("--vector", required=True, help="comma-separated floats")
    q.add_argument("--topk", type=int, default=10)
    q.set_defaults(func=cmd_query)

    ex = sub.add_parser("explain")
    ex.add_argument("--enable-graph-rerank", action="store_true")
    ex.add_argument("--include-docs", action="store_true")
    ex.set_defaults(func=cmd_explain)

    rp = sub.add_parser("replay")
    rp.add_argument("--host", default="127.0.0.1")
    rp.add_argument("--port", type=int, default=8080)
    rp.add_argument("--token", default=None)
    rp.add_argument("--file", required=True, help="ndjson of sync events")
    rp.set_defaults(func=cmd_replay)

    isg = sub.add_parser("inspect-segment")
    isg.add_argument("--path", required=True)
    isg.add_argument("--membrane", required=True)
    isg.set_defaults(func=cmd_inspect_segment)

    v = sub.add_parser("verify")
    v.add_argument("--path", required=True)
    v.set_defaults(func=cmd_verify)

    l = sub.add_parser("lifecycle")
    l.add_argument("--path", required=True)
    l.add_argument("--dim", type=int, required=True)
    l.add_argument("--shards", type=int, default=1)
    l.add_argument("--metric", default="ip")
    l.add_argument("--profile", default="user_defined")
    l.add_argument("--membrane", required=True)
    l.add_argument("--ttl-sec", type=int, default=0)
    l.add_argument("--retention-max-count", type=int, default=0)
    l.add_argument("--retention-max-bytes", type=int, default=0)
    l.add_argument("--create", action="store_true", help="Create membrane with retention policy if missing")
    l.add_argument("--show", action="store_true", help="Show current retention policy for membrane")
    l.add_argument("--kind", type=int, default=12, help="Membrane kind when --create is used (default: 12/meta)")
    l.set_defaults(func=cmd_lifecycle)

    pr = sub.add_parser("profile")
    pr.add_argument("--path", required=True)
    pr.add_argument("--dim", type=int, required=True)
    pr.add_argument("--shards", type=int, default=1)
    pr.add_argument("--metric", default="ip")
    pr.add_argument("--set", required=True, choices=["edge_safe", "edge_balanced", "edge_fast", "user_defined"])
    pr.set_defaults(func=cmd_profile)

    sn = sub.add_parser("snapshot")
    sn.add_argument("--mode", choices=["export", "import"], required=True)
    sn.add_argument("--path", required=True, help="db path for export")
    sn.add_argument("--output", required=True, help="snapshot tar.zst path")
    sn.add_argument("--target", default=".", help="import destination parent directory")
    sn.set_defaults(func=cmd_snapshot)

    dr = sub.add_parser("doctor-repair")
    dr.add_argument("--path", required=True)
    dr.add_argument("--repair", action="store_true")
    dr.add_argument("--dry-run", action="store_true")
    dr.set_defaults(func=cmd_doctor_repair)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
