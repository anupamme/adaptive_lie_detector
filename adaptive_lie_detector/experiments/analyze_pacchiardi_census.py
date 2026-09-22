#!/usr/bin/env python3
"""analyze_pacchiardi_census.py

EXP-XP (PREREG_EXP_XP.md §§3--5): apply requirements (i)--(v) to each of the
three designs in the audit target's own public release, and write the verdicts.

Structural only. **No model call, no network call, no detector fit.** Every
quantity is a hash, a count, a field name or a set operation over the six files
pinned by `fetch_pacchiardi_release.py`; the sha256 of each is re-verified
against the committed manifest before anything is read, so the census is a
function of exactly the bytes the manifest names.

Per §5 no accuracy is computed for a design that is not criterion-4-expressible,
and `UNDECIDABLE_FROM_RELEASE` is a first-class predicate value, never rounded
to False.

Usage:
    cd code/adaptive_lie_detector
    ../.venv/bin/python3 experiments/analyze_pacchiardi_census.py
"""
import csv
import collections
import hashlib
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data", "external", "pacchiardi_release")
MANIFEST = os.path.join(BASE, "data", "external",
                        "pacchiardi_release_manifest.json")
OUT = os.path.join(BASE, "data", "results", "pacchiardi_census.json")

UNDEC = "UNDECIDABLE_FROM_RELEASE"

ROLEPLAY_FILES = [
    "instrumental_lying_df_original.json",
    "instrumental_lying_df_all_scenarios_jb_resampling.json",
    ("instrumental_lying_df_all_42_settings_lorenzos_hardcoded_answers_"
     "correct_prefixes.json"),
]
FT_LIE = "finetuning_v2_lie_finetuning_dataset_validation_prepared.jsonl"
FT_TRUE = "finetuning_v2_truthful_finetuning_dataset_validation_prepared.jsonl"
INSTRUCTED = "lying_rate.csv"


# ----------------------------------------------------------------- integrity

def verify_manifest():
    """Re-hash every fetched file against the committed manifest. A mismatch is
    fatal: the census must not run on bytes the manifest does not name."""
    with open(MANIFEST) as f:
        man = json.load(f)
    checked = []
    for rec in man["files"]:
        path = os.path.join(SRC, rec["local"])
        if not os.path.exists(path):
            raise SystemExit(
                f"missing {rec['local']}; run fetch_pacchiardi_release.py "
                "(PREREG_EXP_XP.md §6 branch 4 if it cannot be fetched)")
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != rec["sha256"]:
            raise SystemExit(f"sha256 mismatch on {rec['local']}")
        checked.append(rec["local"])
    return man, checked


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------- design 1: instructed lying

def census_instructed():
    """`lying_rate.csv`: the design the paper audits (EXP-R1/R1b/R1c).

    The release ships this design as a **dataset x instruction rate table**, not
    as rollouts, so several predicates are decided by the absence of a
    rollout-level record rather than by its contents. That absence is reported,
    not resolved by assumption (§4)."""
    with open(os.path.join(SRC, INSTRUCTED)) as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    instructions = header[1:]
    datasets = [r[0] for r in body]
    cells = sum(1 for r in body for v in r[1:] if v != "")

    return {
        "design": "prompted_instructed",
        "label": "prompted instructed lying (the design this paper audits)",
        "files": [INSTRUCTED],
        "n_rollouts": 0,
        "n_rollouts_note": ("the file is aggregated to one lying rate per "
                            "dataset x instruction; no rollout, prompt or "
                            "response text is released in it"),
        "n_items": len(datasets),
        "item_key": ("dataset name (the only key in the file); no question-, "
                     "prompt- or rollout-level key exists"),
        "n_instructions": len(instructions),
        "datasets": datasets,
        "n_rate_cells": cells,
        "n_distinct_prompt_hashes": UNDEC,
        "n_items_multi_prompt_hash": UNDEC,
        "deception_value": {
            "exists_per_rollout": False,
            "determined_by": "the instruction column the rate is tabulated under",
        },
        "channel": ["dataset name", "instruction text", "lying rate"],
        "predicates": {
            "i": [False, f"the {len(instructions)} instruction texts are the "
                         "column keys: the contrast *is* the prompt change, so "
                         "no two contrasted cells share a prompt hash"],
            "ii": [False, "the tabulated value is the rate of lying under a "
                          "named lie instruction; D is the generating "
                          "condition, which fails (ii) by definition"],
            "iii": [UNDEC, "the file is aggregated to rates; with no "
                           "question-level key released, whether any one item "
                           "was realized under both outcomes cannot be read "
                           "off it either way"],
            "iv": [UNDEC, "no prompt or response text is released in this "
                          "file, so what a black-box detector would read "
                          "cannot be checked here"],
            "v": [False, "D is definitionally the instruction that produced "
                         "the response"],
        },
    }


# ------------------------------------------------- design 2: instrumental RP

def census_roleplay():
    """The `instrumental_lying_df_*` files: goal-directed, **un-instructed**
    deception. This is the design a reader would check first, because it is the
    one thing the instructed paradigm lacks.

    Two item keys are reported. The **released** key is the whole-prompt sha256
    (§3: hashing a system message alone is forbidden). A *hand-supplied* pair
    key is reported as a supplementary diagnostic only, and is flagged as such,
    because the release carries no scenario-id field."""
    per_file = []
    for fn in ROLEPLAY_FILES:
        with open(os.path.join(SRC, fn)) as f:
            d = json.load(f)
        keys = sorted(d["prompt"], key=int)
        prompts = [d["prompt"][k] for k in keys]
        labels = [d["label"][k] for k in keys]
        hashes = [sha(p) for p in prompts]

        by_hash = collections.defaultdict(collections.Counter)
        for h, l in zip(hashes, labels):
            by_hash[h][l] += 1
        both = sum(1 for c in by_hash.values() if len(c) > 1)

        # Is the label the row's generating condition? If it equals index
        # parity on every row, it is recoverable with no text read at all.
        parity = sum(1 for i, l in enumerate(labels) if int(l) == i % 2)

        # Supplementary only: adjacent rows as a scenario pair.
        n_pairs = len(keys) // 2
        pairs_both = sum(1 for i in range(n_pairs)
                         if labels[2 * i] != labels[2 * i + 1])
        pairs_same_prompt = sum(1 for i in range(n_pairs)
                                if prompts[2 * i] == prompts[2 * i + 1])
        pairs_same_first_sentence = sum(
            1 for i in range(n_pairs)
            if prompts[2 * i].split(".")[0] == prompts[2 * i + 1].split(".")[0])

        probe = {c: len(d[c][keys[0]]) for c in
                 ("logprob_differences_lie", "logprob_differences_no_lie",
                  "logprob_differences_knowable") if c in d}
        per_file.append({
            "file": fn,
            "columns": sorted(d.keys()),
            "n_rollouts": len(keys),
            "label_multiset": {str(k): v for k, v in
                               sorted(collections.Counter(labels).items())},
            "n_distinct_prompt_hashes": len(set(hashes)),
            "n_items_with_both_outcomes": both,
            "rows_where_label_equals_index_parity": parity,
            "supplementary_adjacent_pair_key": {
                "note": ("the release has no scenario-id field; this key is "
                         "ours, and every number under it is supplementary"),
                "n_pairs": n_pairs,
                "pairs_carrying_both_outcomes": pairs_both,
                "pairs_with_byte_identical_prompt": pairs_same_prompt,
                "pairs_sharing_first_sentence": pairs_same_first_sentence,
            },
            "probe_battery_widths": probe,
            "probe_battery_total": sum(probe.values()) if probe else None,
        })

    tot = sum(f["n_rollouts"] for f in per_file)
    both_tot = sum(f["n_items_with_both_outcomes"] for f in per_file)
    parity_all = all(f["rows_where_label_equals_index_parity"]
                     == f["n_rollouts"] for f in per_file)
    same_prompt_pairs = sum(
        f["supplementary_adjacent_pair_key"]["pairs_with_byte_identical_prompt"]
        for f in per_file)

    return {
        "design": "instrumental_roleplay",
        "label": ("instrumental, goal-directed deception, un-instructed "
                  "(the `instrumental_lying_df_*` files)"),
        "files": ROLEPLAY_FILES,
        "n_rollouts": tot,
        "n_items": tot,
        "item_key": "sha256 of the whole released prompt",
        "n_distinct_prompt_hashes": sum(f["n_distinct_prompt_hashes"]
                                        for f in per_file),
        "n_items_with_both_outcomes": both_tot,
        "deception_value": {
            "exists_per_rollout": True,
            "field": "label",
            "determined_by": ("the generating condition: `label` equals the "
                              "row's index parity in every row of every file, "
                              "so it is recoverable without reading any text"),
        },
        "channel": ["transcript", "prompt",
                    "logprob_differences_{lie,no_lie,knowable}"],
        "per_file": per_file,
        "predicates": {
            "i": [False,
                  f"{sum(f['n_distinct_prompt_hashes'] for f in per_file)} "
                  "distinct whole-prompt hashes across the contrasted cells "
                  "where (i) requires one; the deceptive and honest cells "
                  "differ in the prompt by construction, and "
                  f"{same_prompt_pairs} of the supplementary scenario pairs "
                  "have a byte-identical prompt"],
            "ii": [False,
                   "`label` equals the row's index parity in "
                   + ("every row of all three files" if parity_all
                      else "most rows")
                   + ", so it is a deterministic function of the generating "
                     "condition, not an off-label grade"],
            "iii": [False,
                    "under the only released item key (whole-prompt sha256) "
                    f"{both_tot} items carry both outcomes; the supplementary "
                    "pair key does reach both outcomes, but only across two "
                    "different prompts, which is what (i) forbids"],
            "iv": [True,
                   "the released channel is the transcript plus elicitation "
                   "logprob differences: no scratchpad, no training metadata, "
                   "and no generating-condition field inside the text a "
                   "detector would read"],
            "v": [False,
                  "`label` is a property of which prompt variant produced the "
                  "rollout, so it is definitionally equivalent to the process "
                  "that produced the detector's input"],
        },
    }


# --------------------------------------------------- design 3: fine-tuned liar

def census_finetuned():
    """`finetuning/v2_{lie,truthful}/...jsonl`: models fine-tuned to lie.

    This is the one design in the release whose prompts are byte-identical
    across the two deception outcomes, so it is the census's strongest
    near-miss (§6 branch 3)."""
    def load(fn):
        with open(os.path.join(SRC, fn)) as f:
            return [json.loads(line) for line in f if line.strip()]

    lie, true = load(FT_LIE), load(FT_TRUE)
    lset = {r["prompt"] for r in lie}
    tset = {r["prompt"] for r in true}
    shared = lset & tset

    lcomp = collections.defaultdict(set)
    tcomp = collections.defaultdict(set)
    for r in lie:
        lcomp[r["prompt"]].add(r["completion"])
    for r in true:
        tcomp[r["prompt"]].add(r["completion"])
    differing = sum(1 for p in shared if lcomp[p] != tcomp[p])

    # A prompt appearing in only one file is a follow-up turn: it embeds the
    # previous answer, which differs between the two conditions by construction.
    def followup_share(rows, only):
        n = sum(1 for r in rows if r["prompt"] in only)
        f = sum(1 for r in rows if r["prompt"] in only
                and r["prompt"].count("Assistant:") > 1)
        return n, f

    lonly, tonly = lset - tset, tset - lset
    l_n, l_f = followup_share(lie, lonly)
    t_n, t_f = followup_share(true, tonly)

    return {
        "design": "finetuned_liar",
        "label": "fine-tuned liars (v2 lie vs. truthful validation splits)",
        "files": [FT_LIE, FT_TRUE],
        "n_rollouts": len(lie) + len(true),
        "keys": sorted(lie[0].keys()),
        "n_rollouts_per_condition": {"lie": len(lie), "truthful": len(true)},
        "item_key": "sha256 of the whole released prompt",
        "n_items": len(lset | tset),
        "n_distinct_prompt_hashes": {"lie": len(lset), "truthful": len(tset)},
        "n_items_with_both_outcomes": len(shared),
        "shared_prompts_whose_completions_differ": differing,
        "single_condition_prompts": {
            "lie_only": len(lonly), "truthful_only": len(tonly),
            "lie_only_multi_turn": l_f, "truthful_only_multi_turn": t_f,
            "n_rows_lie_only": l_n, "n_rows_truthful_only": t_n,
            "note": ("prompts appearing in one file only are follow-up turns "
                     "that embed the previous answer, so they cannot be "
                     "byte-identical across conditions; the first-turn prompts "
                     "are the ones that are"),
        },
        "deception_value": {
            "exists_per_rollout": True,
            "field": "which file the row came from",
            "determined_by": "the fine-tuning target",
        },
        "channel": ["prompt", "completion"],
        "min_cell_size_gate": {
            "threshold": 10, "cells": [len(shared), len(shared)],
            "passes": len(shared) >= 10,
            "note": ("PREREG_EXP_XA.md §6's floor, adopted by reference; "
                     "reported because this design clears it, not because a "
                     "contrast is run -- §5 forbids that while any predicate "
                     "is False"),
        },
        "predicates": {
            "i": [True,
                  f"{len(shared)} prompts are byte-identical across the two "
                  "conditions, hashed over the whole prompt and not over a "
                  "system message"],
            "ii": [False,
                   "D is which file the row came from, i.e. the fine-tuning "
                   "target: a deterministic function of the generating "
                   "condition, which (ii) excludes however accurate it is"],
            "iii": [True,
                    f"{len(shared)} items are realized under both values of D, "
                    f"and {differing} of them have differing completion sets, "
                    "so the contrast can be taken within an item"],
            "iv": [True,
                   "the released fields are a bare `User: ... Assistant:` "
                   "prompt and the completion: no instruction, no scratchpad, "
                   "no training metadata in the text"],
            "v": [False,
                  "D is definitionally the objective the model was fine-tuned "
                  "on, so it is equivalent to the process that produced the "
                  "detector's input"],
        },
    }


# ----------------------------------------------------------------- verdicts

def verdict(design):
    p = design["predicates"]
    cleared = [k for k in ("i", "ii", "iii", "iv", "v") if p[k][0] is True]
    failed = [k for k in ("i", "ii", "iii", "iv", "v") if p[k][0] is False]
    undec = [k for k in ("i", "ii", "iii", "iv", "v") if p[k][0] == UNDEC]
    design["requirements_cleared"] = cleared
    design["requirements_failed"] = failed
    design["requirements_undecidable"] = undec
    design["criterion4_expressible"] = len(cleared) == 5
    design["verdict"] = ("criterion-4-expressible" if len(cleared) == 5
                         else "inapplicable")
    design["detector_contrast_run"] = False
    design["detector_contrast_note"] = (
        "PREREG_EXP_XP.md §5: a detector contrast is run only when all five "
        "predicates are True, so no accuracy is computed or reported here.")
    return design


def main():
    man, checked = verify_manifest()
    print("=" * 72)
    print("EXP-XP -- requirements (i)-(v) over the audit target's own release")
    print(f"  pin: {man['repo']} @ {man['commit'][:12]} ({man['commit_date']})")
    print(f"  {len(checked)} file(s), sha256 re-verified against the manifest")
    print("  structural census only: no model call, no network, no detector fit")
    print("=" * 72)

    designs = [verdict(census_instructed()),
               verdict(census_roleplay()),
               verdict(census_finetuned())]

    for d in designs:
        print(f"\n{d['design']}  --  {d['label']}")
        print(f"  rollouts={d['n_rollouts']}  items={d['n_items']}  "
              f"key={d['item_key']}")
        for k in ("i", "ii", "iii", "iv", "v"):
            val, why = d["predicates"][k]
            mark = {True: "PASS", False: "FAIL"}.get(val, "UNDEC")
            print(f"    ({k:<3}) {mark:<5} {why[:150]}")
        print(f"  cleared={d['requirements_cleared'] or 'none'}  "
              f"failed={d['requirements_failed']}  "
              f"undecidable={d['requirements_undecidable'] or 'none'}")
        print(f"  VERDICT: {d['verdict']}")

    any_ok = [d["design"] for d in designs if d["criterion4_expressible"]]
    verdicts = sorted({d["verdict"] for d in designs})
    strongest = max(designs, key=lambda d: len(d["requirements_cleared"]))
    branch = ("1" if any_ok else
              ("2+3" if set(strongest["requirements_cleared"]) >=
               {"i", "iii", "iv"} else "2"))

    summary = {
        "experiment": "EXP-XP",
        "prereg": "docs/PREREG_EXP_XP.md",
        "exploratory": True,
        "exploratory_note": ("PREREG §0 / DEVIATION 1: a feasibility gate "
                             "preceded the pre-registration, so no number here "
                             "is confirmatory."),
        "pin": {"repo": man["repo"], "commit": man["commit"],
                "commit_date": man["commit_date"]},
        "files_verified": checked,
        "n_designs": len(designs),
        "designs_criterion4_expressible": any_ok,
        "distinct_verdicts": verdicts,
        "verdicts_differ": len(verdicts) > 1,
        "strongest_near_miss": {
            "design": strongest["design"],
            "cleared": strongest["requirements_cleared"],
            "failed": strongest["requirements_failed"],
        },
        "reporting_branch": branch,
        "claim4_survives": not any_ok,
        "split_prior_work_row": len(verdicts) > 1,
        "split_note": ("PREREG §6 branch 2 splits tab:prior_work_criteria's "
                       "single row if and only if the three designs' verdicts "
                       "differ; they do not, so the row is not split and the "
                       "per-design requirement patterns are reported in the "
                       "appendix instead."),
        "per_design": designs,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print("\n" + "=" * 72)
    print(f"designs supplying all five: "
          f"{any_ok if any_ok else 'none'}  ->  claim 4 "
          f"{'must be narrowed' if any_ok else 'survives'}")
    print(f"strongest near-miss: {strongest['design']} clears "
          f"{strongest['requirements_cleared']}, fails "
          f"{strongest['requirements_failed']}")
    print(f"PREREG §6 reporting branch: {branch}; "
          f"verdicts differ: {len(verdicts) > 1} (row "
          f"{'split' if len(verdicts) > 1 else 'not split'})")
    print(f"wrote {os.path.relpath(OUT, BASE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
