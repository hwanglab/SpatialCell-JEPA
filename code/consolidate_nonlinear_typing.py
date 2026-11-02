#!/usr/bin/env python
"""
consolidate_nonlinear_typing.py — rebuild the P16 T2 nonlinear-typing grid table from whatever
cells are on disk, and run the paired contrasts.

Runs under `afterany`, so it EXISTS even if a cell died — and its first duty is to name what is
missing rather than quietly omit it. CPU only, seconds.

Emits, next to the per-cell dirs:
  grid_lopo.tsv / grid_random30.tsv / grid_gap.tsv   the 5x5 tables
  contrasts.tsv                                      every paired test, with CI and fold count
  RESULTS.md                                         the readable summary + a MISSING CELLS block
"""
import argparse, csv, glob, json, os, time
import collections
import numpy as np

EST = ["logistic", "nystroem", "mlp", "histgb", "knn"]
ARMS = ["m", "mu", "mu_bar", "m+mu", "m+mu_bar"]
# the 2026-08-13 logistic row — a REGRESSION GATE, not a result
GATE = {"m": 0.5963, "mu": 0.6108, "mu_bar": 0.6110, "m+mu": 0.5936, "m+mu_bar": 0.5954}


def load(root):
    F, RN = collections.defaultdict(dict), collections.defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(root, "*__*"))):
        est, arm = os.path.basename(d).split("__")
        arm = arm.replace("_plus_", "+")
        t = glob.glob(os.path.join(d, "typing_*.tsv"))
        if not t:
            continue
        rows = list(csv.DictReader(open(t[0]), delimiter="\t"))
        if len(rows) < 12:          # partial write from a killed task -> not readable
            continue
        F[est][arm] = {r["held"]: float(r["macro_f1"]) for r in rows}
        rn = glob.glob(os.path.join(d, "random30_*.tsv"))
        if rn:
            rr = list(csv.DictReader(open(rn[0]), delimiter="\t"))
            RN[est][arm] = float(np.mean([float(r["random30_macro_f1"]) for r in rr]))
    return F, RN


def paired(d1, d2):
    s = sorted(set(d1) & set(d2))
    d = np.array([d1[x] - d2[x] for x in s])
    se = d.std(ddof=1) / np.sqrt(len(d))
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(d).pvalue)
    except Exception:
        p = float("nan")
    return dict(delta=round(float(d.mean()), 4), folds_won=int((d > 0).sum()), n=len(d),
                ci_lo=round(float(d.mean() - 2.201 * se), 4),
                ci_hi=round(float(d.mean() + 2.201 * se), 4), wilcoxon_p=round(p, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/reports/20260814_t2_nonlinear_typing_M1_vs_M12")
    a = ap.parse_args()
    F, RN = load(a.root)
    mean = {e: {arm: float(np.mean(list(v.values()))) for arm, v in F[e].items()} for e in F}

    missing = [f"{e}__{arm}" for e in EST for arm in ARMS if arm not in F.get(e, {})]

    # ---- regression gate
    gate_lines, gate_ok = [], True
    for arm, ref in GATE.items():
        got = mean.get("logistic", {}).get(arm)
        if got is None:
            gate_lines.append(f"| `{arm}` | (missing) | {ref:.4f} | ⚠ NOT CHECKED |"); gate_ok = False
        else:
            ok = abs(got - ref) < 5e-4
            gate_ok &= ok
            gate_lines.append(f"| `{arm}` | {got:.4f} | {ref:.4f} | {'PASS' if ok else '⚠ FAIL'} |")

    # ---- tables
    def table(dd, fmt="{:.4f}"):
        out = ["| estimator | " + " | ".join(f"`{x}`" for x in ARMS) + " |",
               "|---|" + "---:|" * len(ARMS)]
        for e in EST:
            cells = [fmt.format(dd[e][arm]) if e in dd and arm in dd[e] else "—" for arm in ARMS]
            out.append(f"| {e} | " + " | ".join(cells) + " |")
        return "\n".join(out)

    gap = {e: {arm: RN[e][arm] - mean[e][arm] for arm in RN[e] if arm in mean[e]} for e in RN}

    # ---- contrasts
    con = []
    for e in EST:
        for a1, a2, lab in [("mu_bar", "mu", "M=12 - M=1"), ("m+mu_bar", "m+mu", "M=12 - M=1 (concat)"),
                            ("mu", "m", "mu - m")]:
            if e in F and a1 in F[e] and a2 in F[e]:
                con.append(dict(kind=lab, estimator=e, arm=f"{a1} vs {a2}", **paired(F[e][a1], F[e][a2])))
    for e in EST:
        if e == "logistic":
            continue
        for arm in ARMS:
            if e in F and arm in F[e] and arm in F.get("logistic", {}):
                con.append(dict(kind="nonlinear - logistic", estimator=e, arm=arm,
                                **paired(F[e][arm], F["logistic"][arm])))

    os.makedirs(a.root, exist_ok=True)
    if con:
        with open(f"{a.root}/contrasts.tsv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(con[0]), delimiter="\t"); w.writeheader(); w.writerows(con)
    for name, dd in [("grid_lopo", mean), ("grid_random30", RN), ("grid_gap", gap)]:
        with open(f"{a.root}/{name}.tsv", "w") as fh:
            fh.write("estimator\t" + "\t".join(ARMS) + "\n")
            for e in EST:
                fh.write(e + "\t" + "\t".join(f"{dd[e][arm]:.4f}" if e in dd and arm in dd[e] else "NA"
                                              for arm in ARMS) + "\n")

    n_done = sum(len(v) for v in F.values())
    tag = f"[FINAL n=25/25 cells]" if n_done == 25 else f"[PARTIAL n={n_done}/25 cells]"
    md = [f"# P16 T2 — nonlinear cell typing × {{M=1, M=12}} — consolidated",
          "", f"**`{tag}`** · regenerated {time.strftime('%Y-%m-%d %H:%M')} · "
          "12 held-out slides, within-fold LOSO (trains on 5), per_slide=12000, fov112, seed 0, "
          "block_tt checkpoints. Metric = macro-F1 over 3 L1 classes, mean over the 12 slides.", ""]
    if missing:
        md += ["## ⚠ MISSING CELLS", "", "These did not produce a readable 12-fold table:", ""] + \
              [f"- `{x}`" for x in missing] + [""]
    else:
        md += ["## ✅ All 25 cells present", ""]
    md += ["## Regression gate — logistic must reproduce 2026-08-13", "",
           "| arm | this run | 2026-08-13 | |", "|---|---:|---:|---|"] + gate_lines + \
          ["", f"**Gate: {'PASSED' if gate_ok else '⚠ FAILED — no number below is readable'}**", "",
           "## LOPO macro-F1", "", table(mean), "",
           "## random-30 ⚠ WITHIN-DISTRIBUTION DIAGNOSTIC, not a competitor arm", "", table(RN), "",
           "## GAP = random-30 − LOPO", "", table(gap, "{:+.4f}"), "",
           "## Paired contrasts", "", "See `contrasts.tsv` — delta, folds won, 95% CI, Wilcoxon p.",
           "⚠ Multiplicity: ~25 comparisons, uncorrected. Bonferroni threshold ≈0.002.", "",
           "⚠ Balancing differs by estimator (logistic/nystroem `class_weight`; histgb "
           "`sample_weight`; mlp UPSAMPLES; knn none) — cross-estimator deltas carry that, "
           "within-estimator M contrasts do not.", "",
           "★ Narrative reading of the full grid: `REPORT.md` in this directory."]
    open(f"{a.root}/RESULTS.md", "w").write("\n".join(md) + "\n")
    json.dump(dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), root=os.path.abspath(a.root),
                   cells_done=n_done, cells_expected=25, missing=missing,
                   regression_gate_passed=bool(gate_ok), gate_reference="20260813_p3_typing_agg"),
              open(f"{a.root}/PROVENANCE_consolidated.json", "w"), indent=1)
    print(f"{tag}  gate={'PASS' if gate_ok else 'FAIL'}  missing={missing}")
    print(f"wrote {a.root}/{{RESULTS.md, grid_*.tsv, contrasts.tsv, PROVENANCE_consolidated.json}}")


if __name__ == "__main__":
    main()
