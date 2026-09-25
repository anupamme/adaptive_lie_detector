#!/usr/bin/env python3
"""EXP-C4B: the seal. PREREG §3.4 step 4, docs/PREREG_EXP_C4B.md.

Role C (custodian) only. This script is the ONLY thing that ever holds both the
salt and the true labels, and it is run exactly once, after grading and human
coding are complete (PREREG §11 steps 5-7) and before any analysis exists.

WHAT IT PRODUCES

    data/results/crit4b_salt.txt      32 bytes from os.urandom, hex. GIT-IGNORED
                                      AND UNCOMMITTED at this step; committed
                                      only at §3.4 step 6, after the blind
                                      results are committed.
    data/results/crit4b_seal.json     COMMITTED NOW. sha256(salt), the salted
                                      manifest commitments, K, the frozen
                                      analyze_crit4b.py commit hash, and the
                                      custodian-side gates (1, 2, 5, 7).
    data/results/crit4b_blind/        200 candidate files, T{nn}_cand{kk}.json.
                                      Each holds the probe vectors, re-indexed
                                      claim groups, the two precomputed L-doc
                                      fire flags, the belief flag, and ONE
                                      candidate D vector. Exactly one candidate
                                      per target is real; the other nineteen are
                                      independent within-claim permutations of
                                      it, i.e. draws from H1's own null, so they
                                      cost nothing beyond CPU.

WHY THE CANDIDATE FILES CONTAIN NO TEXT

PREREG §3.2 gives Role A the probe vectors and the candidate labels and nothing
else. EXP-C4's H2/H3 applied `marker_fires` to the answer text at analysis time,
which would have handed the analyst the graded channel. So the L-doc rule is
applied HERE, by the imported EXP-C4 `marker_fires`, and only the two resulting
binary flags cross the boundary. The rule is unchanged (PREREG §2.7); what
changes is that gate 4 stops being an assertion about what the analyst read and
becomes a checked property of what the analyst was given: the text is absent, not
merely unread. `analyze_crit4b._check_row_fields` verifies it on the way in.

WHAT THE SEAL CANNOT DO, stated here because it is easy to overstate

It cannot stop a single-analyst team from de-blinding itself: the cells are
committed, so anyone with the repository can recompute the true labels and
compare. PREREG §3.1(b) says so already. The protection is that the ANALYSIS is
frozen at a published commit before the data exist, and that all twenty
candidates' results are committed before the salt is; the blind makes deviation
auditable, and the freeze makes it unnecessary. Nothing here claims an
independent-team blind (DEVIATION 1).

Usage:
    seal_crit4b.py --k 20
"""
import argparse
import glob
import hashlib
import json
import os
import stat
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The frozen analysis owns every salt derivation, so the seal and the unseal
# cannot disagree about what "T03" or "the real candidate" means.
from analyze_crit4b import (                                  # noqa: E402
    SELECTION as A_SELECTION,
    BLIND_DIR,
    FAMILY_E,
    FAMILY_R,
    K_CANDIDATES,
    MIN_CELL,
    MIN_PAIRED_CLAIMS,
    PREREG,
    RAW_GRADES,
    RESULTS,
    ROW_FIELDS,
    SCREEN_MIN_KNOWN,
    SEAL_PATH,
    build_rows,
    cell_counts,
    _candidate_invariants,
    claim_reindex_rng,
    commit_digest,
    commit_model,
    decoy_rng,
    git_hash,
    known_stratum,
    load_cells,
    load_grades,
    pseudonym_order,
    real_index,
    row_order_rng,
    sha256_file,
)
# PREREG §2.6 / §8 gate 7: the rubric must be byte-identical to EXP-C4's, or
# Family R is not a replication. The hash comes from the EXP-C4 grader itself.
from grade_crit4_deception import RUBRIC_MD5 as RUBRIC_MD5_C4   # noqa: E402

SALT_PATH = os.path.join(RESULTS, "crit4b_salt.txt")

# PREREG §8 gate 2: fields whose mere presence in a trial record would mean the
# label is a condition label. Checked on the raw cell, not on the built rows.
BANNED_FIELDS = ("label", "ground_truth", "condition", "is_lying", "grade", "D")


def gate1_elicitation_fixed(rows):
    """§8.1, requirement (i). One distinct system prompt hash PER CLAIM, and the
    wording constant across claims. The reference text differs by claim by
    design, so the invariant is one hash per claim, not one hash per cell.
    """
    by_claim = {}
    for r in rows:
        by_claim.setdefault(r["claim_index"], set()).add(r["system_prompt_md5"])
    multi = {str(k): sorted(v) for k, v in by_claim.items() if len(v) > 1}
    return {"claims": len(by_claim),
            "distinct_hashes": len({h for v in by_claim.values() for h in v}),
            "claims_with_multiple_hashes": multi,
            "pass": not multi}


def gate2_no_condition_label(cell):
    """§8.2, requirement (ii). Verified by construction: the record has no
    condition, label or ground-truth field at all.
    """
    found = sorted({k for r in cell["records"] for k in r if k in BANNED_FIELDS})
    return {"banned_fields_found": found, "pass": not found}


def gate7_rubric_identity(rows):
    """§8.7. Every EXP-C4B judgement's RUBRIC_MD5 equals EXP-C4's. A mismatch
    means PREREG §2.6 was violated and Family R is not a replication -- so this
    is a hard failure, not a warning.
    """
    seen = sorted({r.get("rubric_md5") for r in rows})
    return {"expected_md5_from_EXP_C4": RUBRIC_MD5_C4,
            "observed": seen,
            "pass": seen == [RUBRIC_MD5_C4]}


def gate5_screen_sufficiency(model):
    """§8.5. KNOWN stratum >= 30 pairs over BELIEF_POOL's 120, from the
    committed screen. A target below the threshold should have been substituted
    per §4 BEFORE any trial, so reaching the seal below it is a disclosure.
    """
    n = known_stratum(model)
    return {"known_stratum": n, "threshold": SCREEN_MIN_KNOWN,
            "pass": bool(n is not None and n >= SCREEN_MIN_KNOWN)}


def protocol_wording(model, cell, selection):
    """§7a, checked here because the seal is the last point at which it can be
    checked cheaply and the first at which every cell exists.

    Family R is FIXED at P3 with no pilot -- that is what makes it a
    replication. Family E runs at the wording `--phase select` committed for
    that target, on base rate alone. A confirmatory cell at any other wording
    means the selection rule was not the thing that chose it, so this blocks the
    seal rather than being recorded as a note.
    """
    w = cell["wording_key"]
    if model in FAMILY_R:
        return {"wording": w, "expected": "P3",
                "source": "PREREG §7a: Family R is fixed at P3, no pilot",
                "pass": w == "P3"}
    sel = (selection.get("targets", {}).get(model) or {}).get("selected")
    return {"wording": w, "expected": sel,
            "source": f"the committed §7a selection in {A_SELECTION}",
            "pass": bool(sel) and w == sel}


def within_claim_permute_labels(claims, labels, rng):
    """A decoy: D permuted WITHIN each claim group.

    Deliberately the same operation as `analyze_crit4.within_claim_permute`,
    which is H1's null, and that identity is the point -- a decoy is a draw from
    the null H1 tests against, so the nineteen extra label sets are not overhead
    invented for the blind, they are nineteen samples of the null the experiment
    was always going to need.

    Written over python lists rather than reusing the numpy version because the
    seal writes JSON row-by-row and the claim groups here are the RE-INDEXED
    ones; the operation is identical.
    """
    out = list(labels)
    idx = {}
    for i, c in enumerate(claims):
        idx.setdefault(c, []).append(i)
    for c in sorted(idx):
        pos = idx[c]
        vals = [out[i] for i in pos]
        for i, v in zip(pos, rng.permutation(vals)):
            out[i] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K_CANDIDATES,
                    help=f"candidates per target (PREREG §3.4: {K_CANDIDATES})")
    ap.add_argument("--force-salt", action="store_true",
                    help="overwrite an existing salt file. Refuses by default: "
                         "resealing after the blind analysis has run would let "
                         "the real index be redrawn, which voids the blind.")
    ap.add_argument("--family", choices=["R", "E"], default=None,
                    help="seal ONE family's cells only (PREREG DEVIATION 16). "
                         "Omitted = every confirm cell on disk, which is "
                         "§3.4's pre-registered behaviour and what reproduces "
                         "Family R's seal. Required for Family E, because by "
                         "then Family R's cells are also on disk and sealing "
                         "them again under a new salt is the one thing that "
                         "must not happen.")
    args = ap.parse_args()
    k_total = args.k

    if os.path.exists(SEAL_PATH) and not args.force_salt:
        raise SystemExit(
            f"{SEAL_PATH} already exists. The seal is drawn ONCE (PREREG §3.4 "
            f"step 4). Resealing would redraw the real candidate index after "
            f"results exist. Use --force-salt only to abandon the run.")
    if os.path.exists(SALT_PATH) and not args.force_salt:
        raise SystemExit(f"{SALT_PATH} already exists; refusing to overwrite.")

    grades = load_grades("confirm")
    cells = load_cells("confirm")

    # PREREG DEVIATION (16). §3.4 was written for one seal over all ten
    # pseudonyms, which assumes all ten cells exist when it runs. They do not:
    # §4's storage loop forces sequential collection, so Family R was sealed
    # and unsealed as a complete chain before a single Family E cell existed.
    # Family E therefore gets its own seal, and this filter is the whole of the
    # amendment -- the paths, K, the permutation and the commitment scheme are
    # untouched, and `analyze_crit4b.py` is not modified at all, so the frozen
    # commit identity that §8 gate 6 checks still holds.
    #
    # Membership follows the roster rule already used below: anything not in
    # FAMILY_R is Family E, so a §4 substitution joins Family E automatically
    # rather than needing to be named here.
    if args.family:
        keep = [c for c in cells
                if (c["model"] in FAMILY_R) == (args.family == "R")]
        dropped = sorted({c["model"] for c in cells} - {c["model"] for c in keep})
        if not keep:
            raise SystemExit(
                f"--family {args.family}: no confirm cell belongs to that "
                f"family. Cells on disk: {sorted(c['model'] for c in cells)}")
        print(f"  --family {args.family}: sealing {len(keep)} cell(s); "
              f"excluded {len(dropped)} from the other family: {dropped}")
        cells = keep

    selection = {}
    if os.path.exists(A_SELECTION):
        with open(A_SELECTION) as f:
            selection = json.load(f)

    print("=" * 72)
    print("EXP-C4B: SEAL (PREREG §3.4 step 4). Role C, run once.")
    print("=" * 72)
    print(f"  cells={len(cells)}  K={k_total}  "
          f"gate3 thresholds: paired>={MIN_PAIRED_CLAIMS}, "
          f"per-D>={MIN_CELL}")

    # ---- custodian-side rows and gates, per target.
    built, gates, counts = {}, {}, {}
    for cell in cells:
        m = cell["model"]
        if m in built:
            raise SystemExit(f"two confirm cells for {m}: {cell['_path']}. "
                             f"Top-ups must extend one cell, not add another.")
        rows, n_ev, n_un = build_rows(cell, grades)
        if not rows:
            raise SystemExit(f"{m}: no graded rows.")
        built[m] = (cell, rows)
        counts[m] = cell_counts(cell, rows, n_ev, n_un)
        gates[m] = {
            "gate1_elicitation_fixed": gate1_elicitation_fixed(rows),
            "gate2_no_condition_label": gate2_no_condition_label(cell),
            "gate5_screen_sufficiency": gate5_screen_sufficiency(m),
            "gate7_rubric_identity": gate7_rubric_identity(rows),
            "protocol_wording": protocol_wording(m, cell, selection),
        }

    print()
    for m in sorted(built):
        c, g = counts[m], gates[m]
        flags = " ".join(
            f"{name.split('_')[0]}:{'PASS' if v['pass'] else 'FAIL'}"
            for name, v in sorted(g.items()))
        print(f"  {m:<22} w={c['wording']:<3} graded={c['n_graded']:>4} "
              f"paired={c['paired_claims']:>3} D1={c['n_D1']:>4} "
              f"D0={c['n_D0']:>4} evas={(c['evasive_rate'] or 0):>5.1%}  "
              f"{flags}")

    hard = [(m, n) for m in sorted(gates) for n, v in sorted(gates[m].items())
            if not v["pass"] and n in ("gate1_elicitation_fixed",
                                       "gate2_no_condition_label",
                                       "gate7_rubric_identity",
                                       "protocol_wording")]
    if hard:
        # Gate 1 discards a cell rather than repairing it (§8.1); gate 2, gate 7
        # and the §7a wording protocol are structural. None of the four is a
        # judgement call, so none is overridable from the command line.
        for m, n in hard:
            print(f"  {m}: {n} -> {gates[m][n]}")
        raise SystemExit(f"HARD GATE FAILURE, refusing to seal: {hard}. "
                         f"PREREG §8: a cell violating gate 1 is discarded, not "
                         f"repaired; a gate 7 mismatch means Family R is not a "
                         f"replication; a wording that is not §7a's means the "
                         f"selection rule did not choose it.")
    soft = [(m, n) for m in sorted(gates) for n, v in sorted(gates[m].items())
            if not v["pass"]]
    if soft:
        print(f"\n  NOTE: non-blocking gate failures, recorded in the seal and "
              f"reported: {soft}")

    roster = sorted(set(FAMILY_R) | set(FAMILY_E) | set(built))
    off_roster = sorted(set(built) - set(FAMILY_R) - set(FAMILY_E))
    if off_roster:
        print(f"  §4 substitutions present, joining Family E: {off_roster}")

    # ---- the salt. Drawn here, once, and NOT committed at this step.
    salt = os.urandom(32)
    os.makedirs(RESULTS, exist_ok=True)
    with open(SALT_PATH, "w") as f:
        f.write(salt.hex() + "\n")
    os.chmod(SALT_PATH, stat.S_IRUSR | stat.S_IWUSR)
    print(f"\n  salt: 32 bytes -> {SALT_PATH} (mode 0600)")
    print(f"  sha256(salt) = {hashlib.sha256(salt).hexdigest()}")
    print(f"  DO NOT COMMIT {SALT_PATH} YET (PREREG §3.4 steps 5-6).")

    pseudo_of = pseudonym_order(salt, sorted(built))
    print(f"  pseudonyms assigned from the salt, not from roster or disk order")

    # ---- candidate files.
    if os.path.isdir(BLIND_DIR) and glob.glob(os.path.join(BLIND_DIR, "*.json")):
        raise SystemExit(f"{BLIND_DIR} is not empty; refusing to mix two seals.")
    os.makedirs(BLIND_DIR, exist_ok=True)

    manifest, candidate_digests = {}, {}
    n_identical_total = 0
    for model in sorted(built):
        cell, rows = built[model]
        pseudo = pseudo_of[model]

        # Claim re-indexing: a per-target permutation of the DESIGN's claim
        # indices, so a candidate cannot be cross-referenced against EXP-C4's
        # committed cells by claim number (§3.4 step 4). Built over the design's
        # claims, not the graded ones, so the map does not itself leak which
        # claims survived grading.
        design = sorted({r["claim_index"] for r in cell["records"]})
        perm = claim_reindex_rng(salt, pseudo).permutation(len(design))
        remap = {ci: int(perm[i]) for i, ci in enumerate(design)}

        order = row_order_rng(salt, pseudo).permutation(len(rows))
        shuffled = [rows[i] for i in order]
        claims = [remap[r["claim_index"]] for r in shuffled]
        real = [int(r["D"]) for r in shuffled]

        base = [{"claim": claims[i],
                 "vector": shuffled[i]["vector"],
                 "fires_probe": int(shuffled[i]["fires_probe"]),
                 "fires_answer": int(shuffled[i]["fires_answer"]),
                 "known_preserved": bool(shuffled[i]["known_preserved"])}
                for i in range(len(shuffled))]

        kreal = real_index(salt, pseudo, k_total)
        n_identical = 0
        for k in range(k_total):
            if k == kreal:
                labels = list(real)
            else:
                rng = decoy_rng(salt, pseudo, k)
                labels = within_claim_permute_labels(claims, real, rng)
                # A decoy that happens to equal the real vector is a 2-in-K
                # leak, so redraw. Recorded rather than silently retried: if
                # every claim group is homogeneous in D the redraw cannot help,
                # and that fact belongs in the seal.
                tries = 0
                while labels == real and tries < 100:
                    labels = within_claim_permute_labels(claims, real, rng)
                    tries += 1
                if labels == real:
                    n_identical += 1
            out_rows = [dict(r, D=labels[i]) for i, r in enumerate(base)]
            for r in out_rows:
                assert frozenset(r) == ROW_FIELDS, sorted(r)
            path = os.path.join(BLIND_DIR, f"{pseudo}_cand{k:02d}.json")
            with open(path, "w") as f:
                json.dump({"pseudonym": pseudo, "candidate_index": k,
                           "n_rows": len(out_rows), "rows": out_rows}, f)
            candidate_digests[f"{pseudo}_cand{k:02d}"] = sha256_file(path)
        n_identical_total += n_identical

        inv = _candidate_invariants(counts[model])
        manifest[pseudo] = dict(
            inv,
            wording=counts[model]["wording"],
            n_design_claims=counts[model]["n_design_claims"],
            evasive_rate=counts[model]["evasive_rate"],
            # SALTED commitments: binding now, readable only after the salt is
            # committed (CORRECTION 1(a)).
            model_commitment=commit_model(salt, model),
            cell_commitment=commit_digest(salt, sha256_file(cell["_path"])),
            cell_path=cell["_path"],
            n_decoys_identical_to_real=n_identical,
            gates=gates[model],
        )
        print(f"  {pseudo} <- {model:<22} rows={len(base)} "
              f"claims_reindexed={len(design)} real_k={kreal:>2} "
              f"decoys_identical={n_identical}")

    # ---- the seal itself.
    seal = {
        "experiment": "EXP-C4B",
        "phase": "seal",
        "prereg": PREREG,
        "k": k_total,
        "n_targets": len(manifest),
        "salt_sha256": hashlib.sha256(salt).hexdigest(),
        "salt_bytes": 32,
        "salt_path_uncommitted_at_this_step": SALT_PATH,
        "analyze_commit": git_hash("experiments/analyze_crit4b.py"),
        "seal_commit": git_hash("experiments/seal_crit4b.py"),
        "rubric_md5": RUBRIC_MD5_C4,
        "roster": roster,
        "family_R": FAMILY_R,
        "family_E": FAMILY_E,
        "off_roster_substitutions": off_roster,
        "grades_path": RAW_GRADES,
        "grades_commitment": commit_digest(salt, sha256_file(RAW_GRADES)),
        "candidate_dir": BLIND_DIR,
        "candidate_file_sha256": candidate_digests,
        "n_candidate_files": len(candidate_digests),
        "n_decoys_identical_to_real_total": n_identical_total,
        "row_fields": sorted(ROW_FIELDS),
        "gate1_elicitation_fixed": {m: gates[m]["gate1_elicitation_fixed"]
                                    for m in sorted(gates)},
        "gate2_no_condition_label": {m: gates[m]["gate2_no_condition_label"]
                                     for m in sorted(gates)},
        "gate5_screen_sufficiency": {m: gates[m]["gate5_screen_sufficiency"]
                                     for m in sorted(gates)},
        "gate7_rubric_identity": {m: gates[m]["gate7_rubric_identity"]
                                  for m in sorted(gates)},
        "protocol_wording": {m: gates[m]["protocol_wording"]
                             for m in sorted(gates)},
        "pseudonyms": manifest,
        "published_counts_are_candidate_invariant": (
            "paired_claims, n_D1, n_D0, minority_D_n and the graded/evasive/"
            "ungraded counts are invariant to within-claim permutation of D. "
            "Every candidate for a pseudonym therefore shares them, including "
            "the majority-class baseline, so publishing them cannot help the "
            "analyst identify the real candidate. Nothing base-rate-like that "
            "is NOT invariant appears here."),
        "commitments_are_salted": (
            "PREREG §3.4 step 4 says the manifest carries the sha256 of each "
            "cell file. Published unsalted beside a pseudonym, that digest "
            "would invert the pseudonym map for anyone holding the committed "
            "cells. The seal therefore publishes sha256(salt || tag || value); "
            "binding is unchanged, hiding is restored. CORRECTION 1(a)."),
        "blind_limits": (
            "Single-analyst, machine-enforced (DEVIATION 1). A one-person team "
            "can always de-blind by recomputing labels from the committed "
            "cells (§3.1(b)); the protection is the frozen pre-data commit of "
            "analyze_crit4b.py plus the commit of all K candidates' results "
            "before the salt."),
    }
    with open(SEAL_PATH, "w") as f:
        json.dump(seal, f, indent=2)

    print()
    print(f"  wrote {len(candidate_digests)} candidate files to {BLIND_DIR}/")
    print(f"  wrote {SEAL_PATH}")
    if n_identical_total:
        print(f"  WARNING: {n_identical_total} decoy(s) equal the real label "
              f"vector after 100 redraws (homogeneous claim groups). Recorded "
              f"in the seal; the effective K is smaller for those targets.")
    if seal["analyze_commit"] is None:
        print("  WARNING: analyze_crit4b.py has no commit hash. PREREG §3.4 "
              "step 2 requires it to be COMMITTED before the data exist; "
              "gate 6 will report this as unverified.")
    print()
    print("  NEXT, IN THIS ORDER (PREREG §3.4; any reordering voids the blind):")
    print(f"    1. git add {SEAL_PATH} {BLIND_DIR} && git commit")
    print(f"       ({SALT_PATH} stays UNCOMMITTED)")
    print( "    2. analyze_crit4b.py --phase blind ; commit "
           "crit4b_blind_results.json")
    print(f"    3. git add {SALT_PATH} && git commit ; then --phase unseal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
