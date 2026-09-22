#!/usr/bin/env python3
"""fetch_pacchiardi_release.py

EXP-XP step 1 (PREREG_EXP_XP.md §2): fetch and pin the six files of the audit
target's own public release.

`pacchiardi2024catch` released `github.com/LoryPack/LLM-LieDetector`. The
repository is public, so unlike Liars' Bench there is no access gate -- but it
is still not ours to redistribute, so `data/external/pacchiardi_release/` is
gitignored and what gets committed is
`data/external/pacchiardi_release_manifest.json`: the pinned commit, and per
file its upstream path, byte size and sha256. A reader who fetches the same
commit gets the same bytes, verifiably.

One network call per file. Nothing after this script touches the network:
`analyze_pacchiardi_census.py` reads only what lands here.

Local filenames flatten the upstream directory structure, because two of the six
files share the basename `finetuning_dataset_validation_prepared.jsonl` under
different parent directories. The manifest records both names, so the mapping is
never inferred.

Usage:
    cd code/adaptive_lie_detector
    ../.venv/bin/python3 experiments/fetch_pacchiardi_release.py --check
    ../.venv/bin/python3 experiments/fetch_pacchiardi_release.py
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(BASE, "data", "external", "pacchiardi_release")
MANIFEST = os.path.join(BASE, "data", "external",
                        "pacchiardi_release_manifest.json")

REPO = "LoryPack/LLM-LieDetector"
COMMIT = "c5689fa2615368cd7c3f3c15dbb60ea2126006c7"  # 2024-06-19
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"

# The six files of PREREG_EXP_XP.md §0, grouped by which of the release's three
# designs they back. `local` flattens the path; see the module docstring.
FILES = [
    {"design": "instrumental_roleplay",
     "path": "results/instrumental_lying_df_original.json",
     "local": "instrumental_lying_df_original.json"},
    {"design": "instrumental_roleplay",
     "path": "results/instrumental_lying_df_all_scenarios_jb_resampling.json",
     "local": "instrumental_lying_df_all_scenarios_jb_resampling.json"},
    {"design": "instrumental_roleplay",
     "path": ("results/instrumental_lying_df_all_42_settings_lorenzos_"
              "hardcoded_answers_correct_prefixes.json"),
     "local": ("instrumental_lying_df_all_42_settings_lorenzos_hardcoded_"
               "answers_correct_prefixes.json")},
    {"design": "prompted_instructed",
     "path": "results/lying_rate.csv",
     "local": "lying_rate.csv"},
    {"design": "finetuned_liar",
     "path": "finetuning/v2_lie/finetuning_dataset_validation_prepared.jsonl",
     "local": ("finetuning_v2_lie_finetuning_dataset_validation_"
               "prepared.jsonl")},
    {"design": "finetuned_liar",
     "path": ("finetuning/v2_truthful/finetuning_dataset_validation_"
              "prepared.jsonl"),
     "local": ("finetuning_v2_truthful_finetuning_dataset_validation_"
               "prepared.jsonl")},
]


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


def head(rel):
    """Size and reachability at the pinned commit, without downloading."""
    req = urllib.request.Request(f"{RAW}/{rel}", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, int(r.headers.get("Content-Length") or -1), ""
    except urllib.error.HTTPError as e:
        return e.code, -1, e.read()[:200].decode("utf-8", "replace")
    except Exception as e:  # network, DNS, TLS
        return -1, -1, str(e)


def download(rel, local):
    dest = os.path.join(DEST, local)
    os.makedirs(DEST, exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(f"{RAW}/{rel}", timeout=600) as r:
        with open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    os.replace(tmp, dest)
    digest, size = sha256_of(dest)
    return digest, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="probe reachability at the pinned commit; fetch nothing")
    args = ap.parse_args()

    if args.check:
        bad = 0
        for f in FILES:
            status, size, body = head(f["path"])
            ok = "ok" if status == 200 else f"FAIL {body.strip()[:80]}"
            bad += status != 200
            print(f"  HTTP {status}  {size:>10,}  {f['path']}  {ok}")
        if bad:
            print(f"\n{bad} of {len(FILES)} unreachable at {COMMIT[:12]} "
                  "-> PREREG_EXP_XP.md §6 branch 4 (print decline).")
        return 1 if bad else 0

    files, failed = [], []
    for f in FILES:
        try:
            digest, size = download(f["path"], f["local"])
        except Exception as e:
            failed.append({"path": f["path"], "error": str(e)})
            print(f"  FAILED  {f['path']}: {e}")
            continue
        files.append({"design": f["design"], "path": f["path"],
                      "local": f["local"], "sha256": digest, "bytes": size})
        print(f"  {f['local']}: {size:,} B  sha256 {digest[:16]}...")

    manifest = {
        "repo": REPO,
        "url": f"https://github.com/{REPO}",
        "commit": COMMIT,
        "commit_date": "2024-06-19",
        "gated": False,
        "prereg": "docs/PREREG_EXP_XP.md",
        "note": ("Public repository, pinned by commit. The files are NOT "
                 "redistributed: data/external/pacchiardi_release/ is "
                 "gitignored and these sha256s let a reader who fetches the "
                 "same commit verify identical bytes (PREREG §2)."),
        "designs": ["prompted_instructed", "instrumental_roleplay",
                    "finetuned_liar"],
        "files": files,
        "failed": failed,
    }
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"\n{len(files)} of {len(FILES)} file(s) fetched at {COMMIT[:12]}")
    print(f"Manifest -> {MANIFEST}")
    if failed:
        print("PREREG_EXP_XP.md §6 branch 4 applies to the missing file(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
