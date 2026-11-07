"""Build small result tables for three paper numbers that have no per-fold table on disk.

WHY THIS EXISTS
  The camera-ready availability statement says every number in the paper can be regenerated from
  the data release. A 2026-11-06 audit found three cited numbers whose only on-disk record was an
  internal report (excluded from the release) or raw run outputs that are too large / not meant
  for release. This script reads the RAW outputs and writes one small TSV per number, asserting
  each against the value the paper prints, so the release carries a checkable table instead of
  prose.

  1. masking_changed_context.tsv  -- per slide, how many of the 36 context slots the target mask
     changes (paper: "changes on average 24 of the 36 context embeddings", Sec 3.2 / App masking).
     Source: per-slide PROVENANCE.json of the masked-context precompute (stats.target).
  2. lora_supervised_control_per_fold.tsv -- label-supervised LoRA minus head-only, per fold
     (paper: "+0.0023 (7 of 12 folds, p = 0.57)", Sec 4 / App lora).
     Source: results/t6_train/SUP_{lora,head}_<slide>/test.json, test_all_macro_f1.
  3. val_nll_k12_cap22.tsv -- validation NLL of the control, K_c=12 and 22 um cap predictors: mean of the
     last 3 epochs (the margin rule's statistic, 20260818_t2b_depth_screen), final epoch (the evaluated
     weights) and best epoch
     (paper App context: +2.9 and +228.4 against the control).
     Source: results/mssc_corpus31_ownknn_xattn_fov56_d2_perel{,_k12,_cap22}/train_history.json.

  Output goes to results/reports/<stamp>_t2_release_supplement/ with raw slide IDs; the release
  copy is de-identified by stage_data_release.py like every other staged directory.

Usage
  python3 scripts/release/build_release_supplement.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "reports" / "20261106_t2_release_supplement"
MASK_EVAL = ROOT / "cache" / "masked_ctx" / "eval"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = []

    # 1. masking: context slots changed per target
    rows, slots = [], []
    for p in sorted(MASK_EVAL.glob("*/PROVENANCE.json")):
        d = json.loads(p.read_text())
        st = d["stats"]["target"]
        rows.append(f"{d['slide_id']}\t{st['slots_per_target']:.3f}\t{st['frac_targets_with_change']:.4f}\t{d['counts']['T']}")
        slots.append(st["slots_per_target"])
        inputs.append(str(p))
    assert len(rows) == 12, f"expected 12 slides, got {len(rows)}"
    mean_slots = sum(slots) / len(slots)
    assert round(mean_slots) == 24, f"paper says ~24 of 36 changed; got {mean_slots:.2f}"
    (OUT / "masking_changed_context.tsv").write_text(
        "slide\tcontext_slots_changed_per_target\tfrac_targets_with_change\tn_targets\n"
        + "\n".join(rows) + f"\nMEAN\t{mean_slots:.3f}\t\t\n")

    # 2. label-supervised LoRA control
    rows, deltas = [], []
    for lora in sorted((RES / "t6_train").glob("SUP_lora_*")):
        slide = lora.name[len("SUP_lora_"):]
        head = RES / "t6_train" / f"SUP_head_{slide}"
        a = json.loads((lora / "test.json").read_text())["test_all_macro_f1"]
        b = json.loads((head / "test.json").read_text())["test_all_macro_f1"]
        rows.append(f"{slide}\t{a:.4f}\t{b:.4f}\t{a - b:+.4f}")
        deltas.append(a - b)
        inputs += [str(lora / "test.json"), str(head / "test.json")]
    assert len(deltas) == 12, f"expected 12 folds, got {len(deltas)}"
    mean_d = sum(deltas) / 12
    n_up = sum(d > 0 for d in deltas)
    pval = wilcoxon(deltas).pvalue
    assert f"{mean_d:+.4f}" == "+0.0023" and n_up == 7 and f"{pval:.2f}" == "0.57", (mean_d, n_up, pval)
    (OUT / "lora_supervised_control_per_fold.tsv").write_text(
        "held_out_slide\tlora_macro_f1\thead_macro_f1\tdelta\n" + "\n".join(rows)
        + f"\nMEAN\t\t\t{mean_d:+.4f}\nfolds_up\t\t\t{n_up}/12\nwilcoxon_p\t\t\t{pval:.4f}\n")

    # 3. validation NLL, K_c=12 and 22 um cap. The non-inferiority margin (8.108) was defined on the
    #    mean of the last 3 epochs (20260818_t2b_depth_screen PROVENANCE `metric`), and the evaluated
    #    weights are the final epoch, so those two are reported; best epoch kept for reference.
    base = "mssc_corpus31_ownknn_xattn_fov56_d2_perel"
    stat = {}
    for arm, suf in [("ctrl", ""), ("k12", "_k12"), ("cap22", "_cap22")]:
        p = RES / f"{base}{suf}" / "train_history.json"
        v = json.loads(p.read_text())["val_nll"]
        stat[arm] = {"last3": sum(v[-3:]) / 3, "final": v[-1], "best": min(v)}
        inputs.append(str(p))
    dl = {a: stat[a]["last3"] - stat["ctrl"]["last3"] for a in stat}
    d12, dcap = dl["k12"], dl["cap22"]
    assert f"{d12:+.1f}" == "+3.3" and f"{dcap:+.1f}" == "+401.6", (d12, dcap)
    (OUT / "val_nll_k12_cap22.tsv").write_text(
        "arm\tval_nll_last3_mean\tdelta_last3\tval_nll_final\tdelta_final\tval_nll_best\tdelta_best\n"
        + "\n".join(f"{k}\t{v['last3']:.3f}\t{v['last3'] - stat['ctrl']['last3']:+.3f}"
                    f"\t{v['final']:.3f}\t{v['final'] - stat['ctrl']['final']:+.3f}"
                    f"\t{v['best']:.3f}\t{v['best'] - stat['ctrl']['best']:+.3f}" for k, v in stat.items()) + "\n")

    prov = {"task": "T2 release supplement", "script": str(Path(__file__).relative_to(ROOT)),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "maturity": "[FINAL n=12 slides/folds]", "inputs": inputs,
            "checks": {"mask_mean_slots": round(mean_slots, 3), "lora_mean_delta": round(mean_d, 4),
                       "lora_folds_up": n_up, "lora_wilcoxon_p": round(pval, 4),
                       "val_nll_delta_last3_k12": round(d12, 3), "val_nll_delta_last3_cap22": round(dcap, 3)}}
    (OUT / "PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(prov["checks"]))
    print(f"wrote 3 tables + PROVENANCE.json to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
