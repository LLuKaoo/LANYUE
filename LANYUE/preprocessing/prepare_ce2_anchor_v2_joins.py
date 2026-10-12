#!/usr/bin/env python3
"""Create a disk-backed CE2-key join index for DEM and Mini-RF sources.

This is a metadata-only pass.  It connects the CE2 master key to the existing
pair/quad metadata and Mini-RF sidecars without keeping millions of rows in
RAM.  The resulting SQLite file is consumed by build_ce2_anchor_v2.py.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from collections import Counter
from pathlib import Path


def open_lmdb(path):
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError("lmdb is required") from exc
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=512,
        subdir=Path(path).is_dir(),
    )


def text_key(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def patch_records(path):
    env = open_lmdb(path)
    try:
        with env.begin(buffers=False) as txn:
            for key, value in txn.cursor():
                if key.startswith(b"patch_"):
                    yield text_key(key), pickle.loads(value)
    finally:
        env.close()


def build(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript("""
        CREATE TABLE joins (
            ce2_key TEXT PRIMARY KEY,
            ce2_dem_json TEXT,
            minirf_sar_json TEXT
        );
        CREATE TABLE quad_map (
            quad_key TEXT PRIMARY KEY,
            ce2_key TEXT NOT NULL
        );
        """)
    counts = Counter()
    try:
        if args.ce2_pair_meta:
            for row_index, (key, meta) in enumerate(patch_records(args.ce2_pair_meta)):
                if args.limit is not None and row_index >= args.limit:
                    break
                path = meta.get("dem_tif_path") or meta.get("ce2_dem_tif_path")
                if not path:
                    continue
                spec = {"path": path}
                window = meta.get("dem_window") or meta.get("ce2_dem_window")
                if window is not None:
                    spec["window"] = [int(v) for v in window]
                connection.execute(
                    "INSERT INTO joins(ce2_key,ce2_dem_json) VALUES(?,?) "
                    "ON CONFLICT(ce2_key) DO UPDATE SET ce2_dem_json=excluded.ce2_dem_json",
                    (key, json.dumps(spec, separators=(",", ":"))),
                )
                counts["ce2_dem"] += 1
                if counts["ce2_dem"] % args.commit_every == 0:
                    connection.commit()

        if args.quad_meta:
            for row_index, (quad_key, meta) in enumerate(patch_records(args.quad_meta)):
                if args.limit is not None and row_index >= args.limit:
                    break
                ce2_key = text_key(meta["source_ce2_key"])
                connection.execute(
                    "INSERT OR REPLACE INTO quad_map(quad_key,ce2_key) VALUES(?,?)",
                    (quad_key, ce2_key),
                )
                path = meta.get("ce2_dem_tif_path")
                if path:
                    spec = {"path": path}
                    window = meta.get("ce2_dem_window")
                    if window is not None:
                        spec["window"] = [int(v) for v in window]
                    connection.execute(
                        "INSERT INTO joins(ce2_key,ce2_dem_json) VALUES(?,?) "
                        "ON CONFLICT(ce2_key) DO UPDATE SET ce2_dem_json=excluded.ce2_dem_json",
                        (ce2_key, json.dumps(spec, separators=(",", ":"))),
                    )
                    counts["ce2_dem"] += 1
                counts["quad_map"] += 1
                if counts["quad_map"] % args.commit_every == 0:
                    connection.commit()
        connection.commit()

        for sidecar in args.sar_sidecar_meta or ():
            for row_index, (_, meta) in enumerate(patch_records(sidecar)):
                if args.limit is not None and row_index >= args.limit:
                    break
                quad_key = text_key(meta.get("source_meta_key"))
                row = connection.execute(
                    "SELECT ce2_key FROM quad_map WHERE quad_key=?", (quad_key,)
                ).fetchone()
                if row is None or not meta.get("minirf_sar_source_path"):
                    counts["sar_unjoined"] += 1
                    continue
                spec = {
                    "path": meta["minirf_sar_source_path"],
                    "window": [int(v) for v in meta["minirf_sar_window"]],
                }
                if meta.get("minirf_sar_source_crs_wkt"):
                    spec["crs"] = meta["minirf_sar_source_crs_wkt"]
                connection.execute(
                    "INSERT INTO joins(ce2_key,minirf_sar_json) VALUES(?,?) "
                    "ON CONFLICT(ce2_key) DO UPDATE SET "
                    "minirf_sar_json=excluded.minirf_sar_json",
                    (row[0], json.dumps(spec, separators=(",", ":"))),
                )
                counts["minirf_sar"] += 1
                if counts["minirf_sar"] % args.commit_every == 0:
                    connection.commit()
        connection.commit()
        connection.execute("DROP TABLE quad_map")
        connection.commit()
    finally:
        connection.close()
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ce2-pair-meta")
    parser.add_argument("--quad-meta")
    parser.add_argument("--sar-sidecar-meta", action="append", default=[])
    parser.add_argument("--commit-every", type=int, default=10000)
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke-test limit applied independently to each input.",
    )
    args = parser.parse_args()
    if not args.ce2_pair_meta and not args.quad_meta:
        parser.error("at least one of --ce2-pair-meta/--quad-meta is required")
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
