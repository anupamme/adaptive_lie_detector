#!/usr/bin/env python3
"""fetch_liars_bench.py

EXP-XL step 1 (PREREG_EXP_XL.md §2): fetch and pin the Liars' Bench test
parquet files.

The corpus is **gated** (`gated: auto`) and is not ours to redistribute, so
`data/external/liars_bench/` is gitignored. What gets committed instead is
`data/external/liars_bench_manifest.json`: per-file **sha256, byte size and the
API's `lastModified`**, plus the repo revision SHA. That verifies a reader who
obtains access has the same bytes we did -- it is not the pinned-submodule
guarantee the Apollo corpus gets, and the paper says so rather than implying
parity (PREREG §2).

If the gate has not been accepted the resolve endpoint returns 403. That is
reported as branch F, not worked around: no mirror, no scraped copy, no derived
redistribution (PREREG §0, §9.5).

Usage:
    cd code/adaptive_lie_detector
    ../.venv/bin/python3 experiments/fetch_liars_bench.py --check
    ../.venv/bin/python3 experiments/fetch_liars_bench.py --all-configs
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(BASE, "data", "external", "liars_bench")
MANIFEST = os.path.join(BASE, "data", "external", "liars_bench_manifest.json")

REPO = "Cadenza-Labs/liars-bench"
API = f"https://huggingface.co/api/datasets/{REPO}"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

# Config names and test-split row counts read from the dataset card's
# machine-readable metadata while writing the pre-registration, BEFORE any row
# existed on disk (PREREG §0). Row counts are an integrity record of what the
# API said then; the fetched files are re-counted, never assumed.
CONFIGS = {
    "instructed-deception": 26426,
    "soft-trigger": 24000,
    "alpaca": 8000,
    "harm-pressure-choice": 5400,
    "harm-pressure-knowledge-report": 7068,
    "insider-trading": 6873,
    "convincing-game": 888,
    "gender-secret": 762,
}

PROBE_FILE = "gender-secret/test-00000-of-00001.parquet"


def token():
    """The local HF token. Never printed, never written to the manifest."""
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env.strip()
    path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


def request(url, tok, method="GET", headers=None):
    req = urllib.request.Request(url, method=method)
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    return req


def check_access(tok):
    """Ranged GET on one small parquet. 200/206 = the gate is accepted."""
    url = f"{RESOLVE}/{PROBE_FILE}"
    req = request(url, tok, headers={"Range": "bytes=0-99"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, ""
    except urllib.error.HTTPError as e:
        return e.code, (e.read()[:300].decode("utf-8", "replace"))
    except Exception as e:  # network, DNS, TLS
        return -1, str(e)


def api_tree(tok):
    """The repo's file tree and revision SHA. Works even while files are 403."""
    out = {}
    with urllib.request.urlopen(request(API, tok), timeout=60) as r:
        info = json.load(r)
    out["sha"] = info.get("sha")
    out["lastModified"] = info.get("lastModified")
    out["gated"] = info.get("gated")
    out["siblings"] = [s.get("rfilename") for s in (info.get("siblings") or [])]
    return out


def parquet_paths(tree, config):
    """Every test-split parquet under `config/`. The shard count is read from
    the tree, not assumed to be one."""
    pre = f"{config}/"
    return sorted(p for p in tree["siblings"]
                  if p.startswith(pre) and p.endswith(".parquet")
                  and "/test-" in p)


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def download(rel, tok):
    dest = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{RESOLVE}/{rel}"
    with urllib.request.urlopen(request(url, tok), timeout=600) as r:
        last_modified = r.headers.get("Last-Modified")
        etag = (r.headers.get("ETag") or "").strip('"')
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    os.replace(tmp, dest)
    digest, size = sha256_of(dest)
    return {"path": rel, "sha256": digest, "bytes": size,
            "last_modified": last_modified, "etag": etag}


def n_rows(path):
    """Row count from the parquet footer only -- no column is materialised, so
    this does not read any example (PREREG §0's integrity boundary)."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="probe access only; fetch nothing")
    ap.add_argument("--all-configs", action="store_true",
                    help="fetch every config's test split (PREREG §2: the "
                         "eligibility survey runs on all of them)")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="fetch only these configs")
    args = ap.parse_args()

    tok = token()
    if not tok:
        raise SystemExit("no HF token found (~/.cache/huggingface/token or $HF_TOKEN)")

    status, body = check_access(tok)
    print(f"access probe on {PROBE_FILE}: HTTP {status}")
    if status not in (200, 206):
        print(f"  {body.strip()[:280]}")
        print("\nPREREG_EXP_XL.md §0 / §9.5: access not obtained -> BRANCH F.")
        print("Accept the gate at https://huggingface.co/datasets/"
              f"{REPO} and re-run. Nothing is inferred about the corpus's")
        print("contents, and no mirror or scraped copy will be used.")
        rec = {"repo": REPO, "access": "denied", "http_status": status,
               "branch": "F", "files": []}
        os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
        with open(MANIFEST, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"\nAccess record -> {MANIFEST}")
        return 1

    tree = api_tree(tok)
    want = (sorted(CONFIGS) if args.all_configs
            else (args.configs or sorted(CONFIGS)))
    if args.check:
        for c in want:
            print(f"  {c}: {len(parquet_paths(tree, c))} test parquet shard(s)")
        return 0

    os.makedirs(DEST, exist_ok=True)
    files, rows, missing = [], {}, []
    for c in want:
        paths = parquet_paths(tree, c)
        if not paths:
            missing.append(c)
            print(f"  {c}: NO test parquet in the tree -- recorded, not repaired")
            continue
        got = 0
        for rel in paths:
            rec = download(rel, tok)
            rec["config"] = c
            files.append(rec)
            got += n_rows(os.path.join(DEST, rel))
            print(f"  {rel}: {rec['bytes']:,} B  sha256 {rec['sha256'][:16]}...")
        rows[c] = got
        card = CONFIGS.get(c)
        flag = "" if card is None or card == got else f"  <-- card said {card:,}"
        print(f"  {c}: {got:,} rows{flag}")

    manifest = {
        "repo": REPO, "access": "granted", "revision": tree["sha"],
        "repo_last_modified": tree["lastModified"], "gated": tree["gated"],
        "prereg": "docs/PREREG_EXP_XL.md",
        "note": ("Gated corpus: the parquet files are NOT redistributed. These "
                 "sha256s let a reader who obtains access verify identical "
                 "bytes; that is weaker than the Apollo submodule pin, and the "
                 "paper states the asymmetry (PREREG §2)."),
        "rows_from_card_at_prereg_time": CONFIGS,
        "rows_measured": rows,
        "configs_with_no_test_parquet": missing,
        "files": files,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{len(files)} file(s), {sum(rows.values()):,} rows")
    print(f"Manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
