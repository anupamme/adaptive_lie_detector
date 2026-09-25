#!/usr/bin/env python3
"""
record_model_digests.py

PREREG_EXP_C4B.md §4 requires the **manifest digest of each target to be
recorded before its weights are deleted**, and §12 DEVIATION 11 says why: "A
reader reproducing this must re-pull; if a tag has moved, the digest is what
identifies the weights actually used." An Ollama tag is mutable. `mistral:7b`
resolves to whatever the library points at on the day you pull it, so the tag
alone does not identify weights and a reproduction attempt years later cannot
tell a mismatch from a bug.

Nothing in the collection scripts wrote these digests. Family R's weights are
still resident, so they are captured here retrospectively rather than lost --
the record is only obtainable while the blobs are on disk, and the whole point
of §4's ordering is that deletion comes after it.

WHAT IS RECORDED, AND WHY EACH FIELD
------------------------------------
Read directly out of ~/.ollama/models/manifests/.../<name>/<tag>, not from
`ollama list`, whose short ID is a truncation:

  manifest_sha256   sha256 of the manifest file's exact bytes. Identifies the
                    whole (weights + template + params + license) bundle. If a
                    re-pull gives a different value, something about the model
                    changed even if the weights did not.
  config_digest     the manifest's own config digest, as the registry states it.
  model_layer       digest and byte size of the application/vnd.ollama.image.model
                    layer -- THE WEIGHTS. This is the field that answers "are
                    these the same parameters?".
  layers            every layer, so a template or parameter change is visible
                    too. A moved template changes generations at temperature 0
                    just as surely as moved weights.

A model that is absent is recorded as absent with its tag, not skipped
silently: "digest unavailable" is itself a fact about reproducibility, and a
missing row would read as an unexamined target.

Usage:
    cd code/adaptive_lie_detector
    ../.venv/bin/python3 experiments/record_model_digests.py                # all known targets
    ../.venv/bin/python3 experiments/record_model_digests.py gemma2:9b      # one, at pull time
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_crit4b import FAMILY_E, FAMILY_R  # noqa: E402

RESULTS = "data/results"
OUT_PATH = os.path.join(RESULTS, "crit4b_model_digests.json")
MANIFEST_ROOT = os.path.expanduser(
    "~/.ollama/models/manifests/registry.ollama.ai/library")
WEIGHTS_MEDIA = "application/vnd.ollama.image.model"


def manifest_path(tag):
    name, _, ver = tag.partition(":")
    return os.path.join(MANIFEST_ROOT, name, ver or "latest")


def digest_one(tag):
    path = manifest_path(tag)
    if not os.path.exists(path):
        return {"model": tag, "resident": False, "manifest_path": path,
                "note": "not resident; digest unavailable. If this target was "
                        "collected and then deleted without a digest, that gap "
                        "is a fact about its reproducibility and is reported "
                        "as one."}
    with open(path, "rb") as f:
        raw = f.read()
    m = json.loads(raw)
    layers = [{"media_type": l.get("mediaType"), "digest": l.get("digest"),
               "size": l.get("size")} for l in m.get("layers", [])]
    weights = [l for l in layers if l["media_type"] == WEIGHTS_MEDIA]
    return {
        "model": tag,
        "resident": True,
        "manifest_path": path,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": m.get("schemaVersion"),
        "config_digest": m.get("config", {}).get("digest"),
        "model_layer": weights[0] if len(weights) == 1 else None,
        "n_model_layers": len(weights),
        "layers": layers,
    }


def main():
    tags = sys.argv[1:] or list(FAMILY_R) + list(FAMILY_E)
    rows = [digest_one(t) for t in tags]

    # Merge, never overwrite: this script is run once per target at pull time,
    # and a later run for one target may not erase an earlier target's record.
    out = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            out = json.load(f)
    out.setdefault("experiment", "EXP-C4B")
    out.setdefault("prereg", "docs/PREREG_EXP_C4B.md §4, §12 DEVIATION 11")
    out.setdefault("note",
                   "Manifest digest per target, read from Ollama's manifest "
                   "file. An Ollama tag is mutable; the digest is what "
                   "identifies the weights actually used. §4 requires this "
                   "recorded BEFORE the weights are deleted.")
    targets = out.setdefault("targets", {})
    for r in rows:
        prior = targets.get(r["model"])
        if prior and prior.get("resident") and r.get("resident"):
            if prior.get("manifest_sha256") != r.get("manifest_sha256"):
                print(f"  !! {r['model']}: manifest CHANGED since it was "
                      f"first recorded\n     was {prior.get('manifest_sha256')}"
                      f"\n     now {r.get('manifest_sha256')}")
                r["previous_manifest_sha256"] = prior.get("manifest_sha256")
                r["manifest_changed_after_first_record"] = True
        if prior and prior.get("resident") and not r.get("resident"):
            # Deleted after being recorded: that is §4 working as intended.
            r = dict(prior)
            r["deleted_after_record"] = True
        targets[r["model"]] = r
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)

    print("=" * 78)
    print("EXP-C4B target manifest digests (PREREG §4)")
    print("=" * 78)
    fam = {t: "R" for t in FAMILY_R}
    fam.update({t: "E" for t in FAMILY_E})
    print(f"  {'target':<22}{'fam':>4}  {'weights layer digest':<24}"
          f"{'GB':>7}  manifest sha256")
    for tag in sorted(targets):
        r = targets[tag]
        if not r.get("resident") and not r.get("deleted_after_record"):
            print(f"  {tag:<22}{fam.get(tag,'?'):>4}  "
                  f"{'NOT RESIDENT, no digest':<24}{'':>7}")
            continue
        w = r.get("model_layer") or {}
        d = (w.get("digest") or "").replace("sha256:", "")
        gb = (w.get("size") or 0) / 1e9
        flag = "  [deleted, record kept]" if r.get("deleted_after_record") else ""
        print(f"  {tag:<22}{fam.get(tag,'?'):>4}  {d[:24]:<24}{gb:>7.1f}  "
              f"{r.get('manifest_sha256','')[:16]}…{flag}")
    n_res = sum(1 for r in targets.values() if r.get("resident"))
    print(f"\n  {n_res}/{len(targets)} recorded with a digest")
    print(f"  wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
