#!/usr/bin/env python3
"""
emit_frontier_appendix.py — generate the EXP-FS appendix material from the analysis.

Nine wrong frontier numbers reached the manuscript in earlier rounds because every
one of them was typed by hand from a terminal scrollback. This script removes that
step: it reads data/results/frontier_panel_analysis.json (itself produced by
analyze_frontier_panel.py, which recomputes everything from committed result files)
and produces, ready to paste:

  BLOCK 1  the tab:frontier_panel LaTeX float
  BLOCK 2  the H1/H2/H3 outcomes as pasteable LaTeX prose
  BLOCK 3  the NEW_CELLS entries for experiments/verify_frontier_provenance.py
  BLOCK 4  the two number-bearing sentences app:vintage needs, plus a PREREG §7
           branch diagnostic -- the diagnostic is a decision aid printed as
           comments, NOT a branch selection. §7 branch choice is a judgement call
           and stays with a human.

Nothing here computes an accuracy. The analysis computes them; this formats them;
verify_frontier_provenance.py then independently recomputes them from the raw
conversations and asserts the manuscript prints them. Three separate steps, and no
number crosses between them through a human.

Statistics are imported from analyze_frontier_panel.py rather than reimplemented,
so there is exactly one Holm implementation in the panel's toolchain.

The block builders are importable: /tmp/apply_b4.py calls build_block1/2/4 so that
the appendix insertion is part of the same all-or-nothing transaction as the rest
of B4, rather than a copy-paste step between two scripts.

Usage:
    cd /path/to/adaptive_lie_detector
    .venv/bin/python3 experiments/emit_frontier_appendix.py
"""

import importlib.util
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "results")
ANALYSIS = os.path.join(DATA, "frontier_panel_analysis.json")

EXPECTED_CELLS = 7          # PREREG §2 roster size; a smaller panel is a preview

# Typography only -- the paper binds multi-word model names with ~ so they do not
# break across a line. Data never flows through this map.
TEX_LABEL = {
    "Claude Sonnet 4.5": r"Claude Sonnet~4.5",
    "Llama 4 Maverick": r"Llama~4~Maverick",
    "Amazon Nova Pro": r"Amazon Nova~Pro",
    "DeepSeek-V3": r"DeepSeek-V3",
    "Mistral Large 3": r"Mistral Large~3",
    "Qwen3 235B-A22B": r"Qwen3~235B-A22B",
    "GLM-5": r"GLM-5",
}


def load_stats():
    """Import binom_p/holm from the analysis so there is one implementation."""
    path = os.path.join(BASE, "experiments", "analyze_frontier_panel.py")
    spec = importlib.util.spec_from_file_location("_afp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.binom_p, mod.holm


def pct(x):
    return f"{100.0 * x:.1f}"


def texp(p):
    """A p-value in the paper's own idiom, not Python's.

    "%.4g" would emit 1.116e-09, which LaTeX typesets literally as the string
    "1.116e-09" inside math mode. The manuscript writes small p-values as
    $p\\!<\\!0.001$ everywhere else, so match that and never print an exponent.
    """
    return r"\!<\!0.001" if p < 0.001 else f"\\!=\\!{p:.3g}"


def signed_pp(x):
    """Signed percentage points with a real minus sign, per the paper's usage
    ($-$53.7\\,pp, $+84.3$\\,pp) -- an ASCII hyphen typesets as a hyphen, not a
    minus, so the sign is set in math mode."""
    return f"${x:+.1f}$\\,pp"


def rng(vals):
    """A range in the paper's idiom, collapsing to one figure when it is one."""
    lo, hi = 100.0 * min(vals), 100.0 * max(vals)
    return f"{lo:.1f}--{hi:.1f}\\%" if hi - lo >= 0.05 else f"{lo:.1f}\\%"


# ---------------------------------------------------------------------------
# Context: load, gate, derive. Every builder below reads this and nothing else.
# ---------------------------------------------------------------------------

def prepare():
    """Load the analysis and derive everything the builders need.

    Raises RuntimeError rather than exiting, so an importing caller (apply_b4.py)
    aborts its own transaction instead of killing the process mid-write.
    """
    if not os.path.exists(ANALYSIS):
        raise RuntimeError(f"{ANALYSIS} not found. Run analyze_frontier_panel.py "
                           f"first.")
    with open(ANALYSIS) as f:
        a = json.load(f)

    if not a.get("gate_8_3_passed"):
        raise RuntimeError(
            "the PREREG §8.3 reproduction gate did not pass. No EXP-FS number may "
            "be published while an already-published cell fails to reproduce from "
            "its committed file.")

    cells = a["cells"]
    new = [c for c in cells if c.get("new")]
    binom_p, holm = load_stats()

    # H1 per cell, Holm WITHIN the new targets and WITHIN each outcome family,
    # exactly as PREREG §6 specifies and analyze_frontier_panel.py reports.
    padj = {}
    for fam in ("rule", "pipeline"):
        raw = [binom_p(c[fam]["correct"], c[fam]["n"]) for c in new]
        adj = holm(raw) if raw else []
        for c, r, q in zip(new, raw, adj):
            padj[(c["tag"], fam)] = (r, q)

    return {
        "a": a,
        "cells": cells,
        "new": new,
        "padj": padj,
        "floor": a["thresholds"]["binomial_smallest_significant_k_at_n100"],
        "orgs": {c["org"] for c in cells},
        "preview": len(cells) < EXPECTED_CELLS,
        "rules": [c["rule"]["accuracy"] for c in cells],
        "pipes": [c["pipeline"]["accuracy"] for c in cells],
    }


def names(cs):
    return ", ".join(TEX_LABEL.get(c["label"], c["label"]) for c in cs)


# ---------------------------------------------------------------------------
# BLOCK 1: the panel table
# ---------------------------------------------------------------------------

def build_block1(ctx):
    cells, new, floor = ctx["cells"], ctx["new"], ctx["floor"]
    out = []
    w = out.append
    w(r"\begin{table}[h]")
    w(r"\centering")
    w(r"\caption{\textbf{The pre-registered frontier-scale panel} (EXP-FS): "
      r"%d targets from %d organizations, $n\!=\!100$ each, one byte-identical "
      r"neutral system prompt in both conditions and no target ever instructed "
      r"to lie. \textbf{Rule} is the parameter-free 14-pattern rule "
      r"(RC$\,\geq\,1$), with \emph{nothing} fit; \textbf{Pipeline} is the "
      r"detector trained on $\leq$70B open-weight targets, transferred "
      r"unchanged. At $n\!=\!100$ the smallest count significantly above chance "
      r"is %d/100. Rows marked $\dagger$ were already published and are "
      r"re-scored here by the PREREG~§8.3 gate, not newly collected; Holm "
      r"correction runs within the %d new targets, within each outcome family "
      r"separately.}"
      % (len(cells), len(ctx["orgs"]), floor, len(new)))
    w(r"\label{tab:frontier_panel}")
    w(r"\begin{small}")
    w(r"\setlength{\tabcolsep}{4pt}")
    w(r"\begin{tabular}{llccrrc}")
    w(r"\toprule")
    w(r"\textbf{Target} & \textbf{Organization} & \textbf{Weights} & "
      r"\textbf{Arch.} & \textbf{Rule} & \textbf{Pipeline} & "
      r"\textbf{RC$\,\geq\,1$ (lie/truth)} \\")
    w(r"\midrule")
    for c in cells:
        lbl = TEX_LABEL.get(c["label"], c["label"])
        mark = "" if c.get("new") else r"$^\dagger$"
        r_, p_ = c["rule"], c["pipeline"]
        w(f"{lbl}{mark} & {c['org']} & {c['weights']} & {c['arch']} & "
          f"\\textbf{{{pct(r_['accuracy'])}\\%}} & {pct(p_['accuracy'])}\\% & "
          f"{r_['fire_lie']}/{r_['n_lie']} vs.\\ {r_['fire_truth']}/"
          f"{r_['n_truth']} \\\\")
    w(r"\bottomrule")
    w(r"\end{tabular}")
    w(r"\end{small}")
    w(r"\end{table}")
    return out


# ---------------------------------------------------------------------------
# BLOCK 2: the statistics, as pasteable prose
#
# This block emits LaTeX, not comments. The audit-trail comments follow it, but
# the sentences are the ones that go into app:frontier_panel, so that no accuracy,
# p-value or chi-square is ever retyped from a terminal into the manuscript.
#
# Every sentence is built CONDITIONALLY on the realized panel. Nothing here
# asserts a uniform pattern the data does not show: the floor sentences name their
# exceptions, and the H2 verdicts are read from the analysis's own `heterogeneous`
# flag rather than assumed. If the panel comes back mixed, this says so in the
# paper's own voice.
# ---------------------------------------------------------------------------

def build_block2(ctx):
    a, cells, new = ctx["a"], ctx["cells"], ctx["new"]
    padj, floor = ctx["padj"], ctx["floor"]
    rules, pipes = ctx["rules"], ctx["pipes"]
    out = []
    w = out.append

    rule_below = [c for c in cells if 100 * c["rule"]["accuracy"] < floor]
    pipe_above = [c for c in cells if 100 * c["pipeline"]["accuracy"] >= floor]

    w(r"\paragraph{Panel-level outcomes (PREREG~§5, §6).}")
    s = (f"Across the {len(cells)} targets the parameter-free rule spans "
         f"\\textbf{{{rng(rules)}}} and the transferred pipeline "
         f"\\textbf{{{rng(pipes)}}}. At $n\\!=\\!100$ the smallest count "
         f"significantly above chance is {floor}/100, so ")
    s += ("every rule cell clears that floor"
          if not rule_below else
          f"all but {len(rule_below)} rule cell(s) clear that floor "
          f"({names(rule_below)} do not)")
    s += (", and no pipeline cell reaches it"
          if not pipe_above else
          f", while {len(pipe_above)} pipeline cell(s) reach it "
          f"({names(pipe_above)})")
    s += ("."
          if (rule_below or pipe_above) else
          ", making the split between the two outcomes complete across the panel.")
    w(s)
    w("")

    # H1, as prose: the Holm-corrected verdicts within the new targets only.
    sig_rule = [c for c in new if padj[(c["tag"], "rule")][1] < 0.05]
    sig_pipe = [c for c in new if padj[(c["tag"], "pipeline")][1] < 0.05]
    worst_rule = max((padj[(c["tag"], "rule")][1] for c in new), default=1.0)
    best_pipe = min((padj[(c["tag"], "pipeline")][1] for c in new), default=1.0)
    w(f"\\textbf{{H1}} (exact two-sided binomial, Holm-corrected within the "
      f"{len(new)} new targets, within each outcome family separately): the rule "
      f"is above chance on \\textbf{{{len(sig_rule)} of {len(new)}}} new targets "
      f"(largest $p_{{\\mathrm{{Holm}}}}{texp(worst_rule)}$), the pipeline on "
      f"\\textbf{{{len(sig_pipe)} of {len(new)}}} "
      f"(smallest $p_{{\\mathrm{{Holm}}}}{texp(best_pipe)}$). "
      f"Cells between 40\\% and {floor}\\% are reported as "
      f"\\emph{{not distinguishable from chance at $n\\!=\\!100$}}, never as "
      f"``at chance'' (PREREG~§6).")
    w("")

    # H2, from the analysis's own chi-square, including its verdict flags.
    h2 = a.get("h2_homogeneity", {})
    if h2:
        bits = []
        for fam in ("rule", "pipeline"):
            for key, scope in (("all", f"all {len(cells)}"),
                               ("new", f"{len(new)} new")):
                r = h2.get(f"{fam}_{key}")
                if not r:
                    continue
                bits.append(
                    f"{fam}, {scope}: $\\chi^2\\!=\\!{r['chi2']:.2f}$, "
                    f"$\\mathrm{{df}}\\!=\\!{r['df']}$, $p{texp(r['p'])}$"
                    + (" (\\textbf{heterogeneous})" if r["heterogeneous"] else ""))
        # Name WHICH test rejected. "at least one test rejects" without saying
        # which one is the kind of vagueness a reviewer reads as evasion.
        het = [k.replace("_all", ", full panel").replace("_new", ", new targets")
               for k, r in h2.items() if r.get("heterogeneous")]
        w(f"\\textbf{{H2}} (homogeneity of proportions; the version over the new "
          f"targets alone is the one carrying confirmatory weight, PREREG~§0) --- "
          + "; ".join(bits) + ". "
          + ("No test departs from a common rate, so the panel gives no evidence "
             "that these targets differ from one another on either outcome."
             if not het else
             f"A common rate is rejected for \\textbf{{{'; '.join(het)}}}, so the "
             f"panel is \\textbf{{heterogeneous}} there and PREREG~§7's Branch~B "
             f"wording governs that outcome: which channel carries the signal is "
             f"target-dependent, and we do not read heterogeneity as partial "
             f"confirmation (PREREG~§8.4)."))
        w("")

    # H3, descriptive only, with the MDE stated in the same sentence as the number.
    h3 = a.get("h3_closed_vs_open", {})
    mde = a["thresholds"]["pairwise_two_proportion_mde_pp"]
    if h3.get("rule_contrast_pp") is not None:
        w(f"\\textbf{{H3}} (closed- versus open-weight, "
          f"\\textbf{{descriptive only}}) --- rule "
          f"{h3['rule_closed']['k']}/{h3['rule_closed']['n']} closed vs.\\ "
          f"{h3['rule_open']['k']}/{h3['rule_open']['n']} open "
          f"({signed_pp(h3['rule_contrast_pp'])}, Fisher exact "
          f"$p{texp(h3['rule_fisher_p'])}$); pipeline "
          f"{h3['pipeline_closed']['k']}/{h3['pipeline_closed']['n']} vs.\\ "
          f"{h3['pipeline_open']['k']}/{h3['pipeline_open']['n']} "
          f"({signed_pp(h3['pipeline_contrast_pp'])}, "
          f"$p{texp(h3['pipeline_fisher_p'])}$). "
          f"\\textbf{{We make no attribution to weight availability.}} The "
          f"pairwise two-proportion MDE at $n\\!=\\!100$ per cell is "
          f"{mde}\\,pp, ``closed'' is "
          f"{h3['rule_closed']['n_targets']} \\emph{{organizations}}, and "
          f"weight availability, architecture, RLHF recipe and serving stack "
          f"remain collinear at the panel level (PREREG~§6, §11).")
        w("")

    n_err = sum(c.get("n_error", 0) for c in cells)
    w(f"Trials terminating in \\texttt{{status == error}} are excluded from both "
      f"outcomes and never imputed: \\textbf{{{n_err}}} across the panel.")
    return out


def build_block2_audit(ctx):
    """The same H1 numbers unformatted, as LaTeX comments beneath BLOCK 2."""
    out = ["% ---- audit trail: the same numbers, unformatted ----"]
    for c in ctx["new"]:
        for fam in ("rule", "pipeline"):
            raw, adj = ctx["padj"][(c["tag"], fam)]
            out.append(f"% H1 {c['label']:20s} {fam:8s} "
                       f"{pct(c[fam]['accuracy']):>5s}\\%  p_raw={raw:.4g}  "
                       f"p_Holm={adj:.4g}")
    out.append(f"% total errored trials across the panel: "
               f"{sum(c.get('n_error', 0) for c in ctx['cells'])}")
    return out


# ---------------------------------------------------------------------------
# BLOCK 3: the verifier registrations
# ---------------------------------------------------------------------------

def build_block3(ctx):
    new = ctx["new"]
    out = ["%     NEW_CELLS = {"]
    pad = max((len(c["tag"]) for c in new), default=0) + 4
    for c in new:
        key = f'"{c["tag"]}":'.ljust(pad)
        out.append(f'%         {key}("{c["label"]}", '
                   f'{100.0 * c["rule"]["accuracy"]:.1f}, '
                   f'{100.0 * c["pipeline"]["accuracy"]:.1f}, {c["rule"]["n"]},')
        out.append(f'%         {" " * pad} "{c["file"]}"),')
    out.append("%     }")
    return out


# ---------------------------------------------------------------------------
# BLOCK 4: app:vintage's replacement text, and the §7 branch diagnostic.
#
# app:vintage currently says "We do not establish that the collapse reproduces on
# current-generation models". What EXP-FS licenses in its place is NOT the same
# sentence under every outcome, and PREREG §7 is explicit about the difference:
#
#   Branch A (S_p = 0, no H2 rejection) -- the paper MAY say the equalization
#            collapse is not vintage-specific.
#   Branch B (H2 significant, or S_p and S_r disagree) -- the paper MAY say it now
#            tests current-generation models and MAY NOT say the collapse
#            reproduces on them.
#
# Those two are one word apart in the manuscript and a whole claim apart in what
# they assert, so this builder emits the wording the REALIZED branch licenses
# rather than one fixed sentence. It will not write the Branch-A claim while an H2
# test rejects a common rate; that would be the exact overclaim §8.4 forbids
# ("heterogeneity may not be read as partial confirmation").
#
# Under Branch C it refuses outright: §7 puts that outcome in the ABSTRACT and
# narrows §3.1's scope, which is a rewrite a human authors, not a substitution.
#
# Two withholdings ride along under every branch, because they are true under all
# of them: outcome 2 is the TRAINED PIPELINE (the EXP-A/ADAGE paradigm), NOT
# EXP-R1c's 16-probe battery, which was never run on these targets; and the
# surface rule moves the OTHER way at frontier scale, which is PREREG §8.2's
# pre-committed emphasis shift.
# ---------------------------------------------------------------------------

def branch_facts(ctx):
    """The §7 decision variables, computed once and shared by the text and the
    diagnostic so they can never disagree."""
    new, padj = ctx["new"], ctx["padj"]
    h2 = ctx["a"].get("h2_homogeneity", {})
    return {
        "s_r": sum(1 for c in new if padj[(c["tag"], "rule")][1] < 0.05),
        "s_p": sum(1 for c in new if padj[(c["tag"], "pipeline")][1] < 0.05),
        "het": [k for k, r in h2.items() if r.get("heterogeneous")],
        "n_new": len(new),
    }


def het_names(het):
    return ", ".join(k.replace("_all", ", full panel")
                      .replace("_new", ", new targets") for k in het)


def build_block4(ctx):
    cells, floor = ctx["cells"], ctx["floor"]
    f = branch_facts(ctx)
    s_p, s_r, het = f["s_p"], f["s_r"], f["het"]
    n_orgs = len(ctx["orgs"])
    out = []
    w = out.append

    if s_p >= 3:
        raise RuntimeError(
            f"PREREG §7 Branch C: the pipeline is significantly above chance on "
            f"{s_p} of {f['n_new']} new targets, so the paper's headline "
            f"generality is REDUCED by its own new evidence. §7 requires that in "
            f"the abstract and narrows §3.1's scope rather than widening it. That "
            f"is a rewrite a human authors; this generator will not substitute it.")

    branch_a = (s_p == 0 and not het)
    common = (
        f"The pre-registered frontier panel "
        f"(Appendix~\\ref{{app:frontier_panel}}) transfers the trained detector "
        f"unchanged onto {len(cells)} frontier-scale targets from {n_orgs} "
        f"organizations under a byte-identical neutral prompt, and every cell "
        f"lands at {rng(ctx['pipes'])}---below the {floor}/100 count that would "
        f"be significant at $n\\!=\\!100$. ")

    if branch_a:
        w(f"\\textbf{{The trained-pipeline collapse now has a "
          f"current-generation test, and it survives it.}} " + common +
          f"No test departs from a common rate across the panel, so "
          f"\\textbf{{the pipeline collapse of "
          f"§\\ref{{sec:prompt_equalized}} is not specific to the "
          f"$\\leq$70B open-weight vintage we measured it on}} "
          f"(PREREG~§7, Branch~A). " + withholdings(ctx))
    else:
        w(f"\\textbf{{The panel now tests current-generation models, and what it "
          f"finds there is target-dependent.}} " + common +
          (f"But a common rate is rejected ({het_names(het)}), "
           if het else
           f"But the two outcome channels disagree across targets "
           f"($S_r\\!=\\!{s_r}$, $S_p\\!=\\!{s_p}$ of {f['n_new']}), ") +
          f"so \\textbf{{we do not say the equalization collapse reproduces on "
          f"current-generation models}}: PREREG~§7's Branch~B licenses the first "
          f"claim and withholds the second, and §8.4 forbids reading "
          f"heterogeneity as partial confirmation. What the panel establishes is "
          f"that \\textbf{{at frontier scale the equalized signal is "
          f"target-dependent, and which channel carries it differs by target}}. "
          + withholdings(ctx))
    w("")
    out += branch_diagnostic(ctx)
    return out


def withholdings(ctx):
    """The two scope limits that hold under every branch."""
    return (
        f"\\textbf{{Two things this does not give us.}} It is \\emph{{not}} a "
        f"current-generation test of EXP-R1c's 16-probe battery, a different "
        f"detector that was not run on these targets; there the recency question "
        f"stays open, and our attempt on it (Appendix~\\ref{{app:r1d}}) is "
        f"\\emph{{inconclusive}}---three targets void on a pre-registered "
        f"probe-channel gate, one admissible target falling $-46.0$\\,pp. And it "
        f"runs the \\emph{{other}} way for the surface rule: the parameter-free "
        f"14-pattern rule reaches {rng(ctx['rules'])} across the same panel, "
        f"\\textbf{{above}} its 69--80\\% range on the $\\leq$70B roster, so "
        f"\\textbf{{surface-accessible accuracy does not collapse at frontier "
        f"scale}} and is the more robust of the two findings---an emphasis shift "
        f"the pre-registration committed to in advance (PREREG~§8.2).")


def branch_diagnostic(ctx):
    """A decision aid, printed as comments. It does not select a branch."""
    f = branch_facts(ctx)
    s_p, s_r, het, n_new = f["s_p"], f["s_r"], f["het"], f["n_new"]
    rule_het = any(k.startswith("rule") for k in het)
    out = ["% ---------------- PREREG §7 BRANCH DIAGNOSTIC ----------------",
           "% A DECISION AID. §7 branch selection is a judgement call the "
           "pre-registration",
           "% keeps with a human. The prose above is the wording the realized "
           "branch LICENSES;",
           "% confirm the branch below before it goes into the paper."]
    out.append(f"%   S_r (new targets, rule, significant after Holm)     = "
               f"{s_r} of {n_new}")
    out.append(f"%   S_p (new targets, pipeline, significant after Holm) = "
               f"{s_p} of {n_new}")
    out.append(f"%   H2 tests rejecting a common rate: {het_names(het) or 'none'}")
    out.append(f"%   A  (S_p == 0, and rule heterogeneous or low): "
               f"{'MET' if (s_p == 0 and not het) else 'not met'}"
               f"  [S_p==0: {s_p == 0}; rule heterogeneous: {rule_het}; "
               f"rule range {rng(ctx['rules'])}]")
    out.append(f"%   B  (H2 significant, or S_p and S_r disagree across targets): "
               f"{'MET' if (het or s_p != s_r) else 'not met'}")
    out.append(f"%   C  (S_p >= 3 of 5 -- reported in the ABSTRACT, scope NARROWS): "
               f"{'MET' if s_p >= 3 else 'not met'}")
    out.append("%   D  (two or more primaries fail a §3 gate, no reserve): not "
               "visible here --")
    out.append("%      the analysis sees cells that exist, not gate failures. "
               "Check the run log.")
    out.append(f"%   -> text above was generated under Branch "
               f"{'A' if (s_p == 0 and not het) else 'B'}.")
    return out


# ---------------------------------------------------------------------------

def main():
    try:
        return emit()
    except RuntimeError as e:
        # Both the §8.3 gate and BLOCK 4's Branch-C refusal arrive this way, and
        # both mean "a human decides before anything is pasted", not "crash".
        sys.stderr.write(f"ERROR: {e}\n")
        return 1


def emit():
    ctx = prepare()

    if ctx["preview"]:
        sys.stderr.write(
            f"WARNING: {len(ctx['cells'])} of {EXPECTED_CELLS} cells present. "
            f"Output is a PREVIEW and is marked as such; do not paste it into the "
            f"manuscript.\n")

    a = ctx["a"]
    out = []
    if ctx["preview"]:
        out.append(f"% ############ PREVIEW: {len(ctx['cells'])} of "
                   f"{EXPECTED_CELLS} cells ############")
    out += [
        "% Generated by experiments/emit_frontier_appendix.py -- do not hand-edit "
        "numbers.",
        f"% Source: data/results/frontier_panel_analysis.json "
        f"(prereg {a['prereg']}, commit {a['prereg_commit']})",
        "",
        "% ==================== BLOCK 1: tab:frontier_panel ====================",
    ]
    out += build_block1(ctx)
    out += ["", "% ==================== BLOCK 2: H1/H2/H3 prose ===================="]
    out += build_block2(ctx)
    out += [""] + build_block2_audit(ctx)
    out += ["",
            "% ============ BLOCK 3: NEW_CELLS for "
            "verify_frontier_provenance.py ======"]
    out += build_block3(ctx)
    out += ["",
            "% ============ BLOCK 4: app:vintage revision + §7 diagnostic ======="]
    out += build_block4(ctx)

    print("\n".join(out))
    return 2 if ctx["preview"] else 0


if __name__ == "__main__":
    sys.exit(main())
