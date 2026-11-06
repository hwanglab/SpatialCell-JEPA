"""Camera-ready (Reviewer Hdrh W2): confidence intervals + equivalence test for Table B's paired delta.

Re-reads the per-fold typing files behind ms.md Table B (tab:typing), pairs mu vs m per held-out
slide at C=0.01, and reports mean delta, t-based 90/95 % CIs, TOST at the paper's own +-0.02
threshold, and the smallest symmetric margin the 90 % CI excludes. GATE: every arm mean must
reproduce Table B to 4 decimals, else the script exits non-zero.

No training, no new features: arithmetic over existing per-fold macro-F1 values.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REP = f"{ROOT}/results/reports"
OUT = f"{REP}/20261020_camera_ready_delta_ci"
MARGIN = 0.02  # paper's stated threshold (ms.md sec 3.1, "our +0.02 threshold")

# (crop, readout) -> (per-fold file, Table B m, Table B mu)  -- ms.md:409-415
ENTRIES = {
    ("56", "logistic/pca0"): ("20260818_t2_d2_pe_typing/perel_logistic_pca0/typing_d2_perel_fov56_logistic_pca0.tsv", 0.6159, 0.6191),
    ("56", "logistic/pca64"): ("20260818_t2_d2_pe_typing/perel_logistic_pca64/typing_d2_perel_fov56_logistic_pca64.tsv", 0.6201, 0.6220),
    ("56", "nystroem/pca0"): ("20260818_t2_d2_pe_typing/perel_nystroem_pca0/typing_d2_perel_fov56_nystroem_pca0.tsv", 0.6174, 0.6242),
    ("56", "nystroem/pca64"): ("20260818_t2_d2_pe_typing/perel_nystroem_pca64/typing_d2_perel_fov56_nystroem_pca64.tsv", 0.6182, 0.6228),
    ("112", "logistic/pca0"): ("20260815_corpus31_typing_fov112/typing_corpus31_fov112.tsv", 0.6190, 0.6148),
    ("112", "logistic/pca64"): ("20260816_pe_absolute_control/fov112_logistic_pca64/typing_corpus31_abs_fov112_logistic_pca64.tsv", 0.6236, 0.6258),
    ("112", "nystroem/pca0"): ("20260816_pe_absolute_control/fov112_nystroem_pca0/typing_corpus31_abs_fov112_nystroem_pca0.tsv", 0.6263, 0.6286),
    ("112", "nystroem/pca64"): ("20260816_pe_absolute_control/fov112_nystroem_pca64/typing_corpus31_abs_fov112_nystroem_pca64.tsv", 0.6264, 0.6280),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, gate_fail, inputs = [], [], []
    for (crop, readout), (rel, tab_m, tab_mu) in ENTRIES.items():
        path = f"{REP}/{rel}"
        st = os.stat(path)
        inputs.append({"path": path, "mtime": time.ctime(st.st_mtime), "size": st.st_size})
        df = pd.read_csv(path, sep="\t")
        df = df[np.isclose(df["C"], 0.01)]
        piv = df[df["arm"].isin(["m", "mu"])].pivot_table(index="held", columns="arm", values="macro_f1")
        assert len(piv) == 12 and piv.notna().all().all(), f"{path}: expected 12 paired folds"
        m_mean, mu_mean = piv["m"].mean(), piv["mu"].mean()
        if round(m_mean, 4) != tab_m or round(mu_mean, 4) != tab_mu:
            gate_fail.append(f"{crop} {readout}: m {m_mean:.4f} vs {tab_m}, mu {mu_mean:.4f} vs {tab_mu}")
        d = (piv["mu"] - piv["m"]).to_numpy()
        n, mean, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t90, t95 = stats.t.ppf(0.95, n - 1), stats.t.ppf(0.975, n - 1)
        p_lo = 1 - stats.t.cdf((mean + MARGIN) / se, n - 1)  # H0: delta <= -margin
        p_hi = stats.t.cdf((mean - MARGIN) / se, n - 1)       # H0: delta >= +margin
        rows.append({
            "crop_um": crop, "readout": readout, "n_folds": n,
            "m": round(m_mean, 4), "mu": round(mu_mean, 4), "delta": round(mean, 4),
            "folds_up": int((d > 0).sum()), "sd_delta": round(d.std(ddof=1), 4),
            "ci90_lo": round(mean - t90 * se, 4), "ci90_hi": round(mean + t90 * se, 4),
            "ci95_lo": round(mean - t95 * se, 4), "ci95_hi": round(mean + t95 * se, 4),
            "p_ttest": round(stats.ttest_1samp(d, 0).pvalue, 4),
            "p_wilcoxon": round(stats.wilcoxon(d).pvalue, 4),
            "tost_p_margin0.02": round(max(p_lo, p_hi), 6),
            "smallest_equiv_margin_90": round(max(abs(mean - t90 * se), abs(mean + t90 * se)), 4),
        })
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/delta_ci.tsv", sep="\t", index=False)
    json.dump({
        "script": os.path.abspath(__file__), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Reviewer Hdrh W2: CI + TOST of paired delta(mu-m), Table B",
        "filter": "C=0.01; arms m, mu; paired on held-out slide", "margin": MARGIN,
        "gate": "PASS" if not gate_fail else gate_fail, "deterministic": True, "inputs": inputs,
    }, open(f"{OUT}/PROVENANCE.json", "w"), indent=2)
    print(res.to_string(index=False))
    if gate_fail:
        print("GATE FAIL:", *gate_fail, sep="\n  ")
        sys.exit(1)
    print("GATE PASS: all 16 Table B means reproduced to 4 decimals")


if __name__ == "__main__":
    main()
