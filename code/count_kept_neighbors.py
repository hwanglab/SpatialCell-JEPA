#!/usr/bin/env python3
"""count_kept_neighbors.py — how many neighbour cells does the isolated-cell embedding m_iso(r) still see? (P16 T2)

[user 2026-11-05] "Keeping a band of 14 microns around the cell ... can be comparable to the ~7 most attended
neighbour cells in the attention module" -> count, per typing cell and r, the neighbours inside the KEPT patches.

Geometry is IMPORTED from scripts/precompute/cache_isolated_cell.py (tile_assignment, keep_set_tile), so the kept
patch set is the one the run actually used. No model, no image pixels: polygons + centroids only.

Per typing cell t (12,000 / slide) and r in --rings-um, plus r = "full" (all 256 patches = the paper's m):
  n_partly   = other cells whose convex-hull patch set (keep_set_tile, margin 0) intersects the kept patches
  n_centroid = other cells whose centroid lies in a kept patch (mostly visible)
  vis_k1..k8 = 1 if the k-th nearest neighbour (centroid distance) is partly visible
Neighbours are searched within --search-um of t's centroid (default 45 um > tile half-diagonal 40 um + nucleus).

Modes: per-slide (writes <out>/<slide>.tsv.gz + <slide>.PROVENANCE.json); --summarize (reads all, writes
SUMMARY.md + summary.tsv + PROVENANCE.json in <out>).
"""
import os, sys, json, glob, argparse, time
from datetime import datetime, timezone
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "precompute"))


def _stat(p):
    try:
        st = os.stat(p)
        return {"path": os.path.abspath(p), "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
    except OSError:
        return {"path": p, "size": None, "mtime": None}


def run_slide(a):
    import openslide, pandas as pd
    from scipy.spatial import cKDTree
    from cache_isolated_cell import tile_assignment, keep_set_tile
    from io_polygons_npz import load_polygons_npz

    t0 = time.time()
    rings = [float(x) for x in a.rings_um.split(",")]
    W, H = openslide.OpenSlide(a.slide).dimensions
    meta = json.load(open(os.path.join(a.cache_dir, "meta.json")))
    mpp = float(meta["mpp"])
    cen_cache = np.load(os.path.join(a.cache_dir, "cells.npz"))["centroid_xy"].astype(np.float64)
    cells = load_polygons_npz(a.polygons_npz)
    if a.max_cells > 0:
        cells = cells[:a.max_cells]
    cen = np.array([[float(c[1][0]), float(c[1][1])] for c in cells], np.float64)
    assert len(cells) == len(cen_cache) and np.allclose(cen, cen_cache, atol=1e-2), "G0 FAIL: polygon/cache order"
    PATCH = 14
    tx, ty, tile_px, n_side, stride = tile_assignment(W, H, cen, mpp, PATCH, 56.0, 0.5)
    assert int(meta["tile_px"]) == tile_px
    NP = n_side * n_side

    df = pd.read_parquet(a.targets)
    tg = np.sort(df[df.slide == a.slide_id].row_idx.values.astype(np.int64))
    if a.limit:
        tg = tg[:a.limit]
    tree = cKDTree(cen)
    rad_px = a.search_um / mpp
    keys = [f"r{r:g}" for r in rings] + ["full"]
    rows = []
    for t in tg:
        org = np.array([tx[t], ty[t]], np.float64)
        kp = {f"r{r:g}": set(keep_set_tile(cells[t][2] - org, max(0, int(round(r / mpp))), tile_px, n_side, PATCH).tolist())
              for r in rings}
        kp["full"] = set(range(NP))
        nb = [j for j in tree.query_ball_point(cen[t], rad_px) if j != t]
        d = np.hypot(*(cen[nb] - cen[t]).T) if nb else np.zeros(0)
        order = np.argsort(d)
        nb = [nb[i] for i in order]
        hull_sets, cpatch = [], []
        for j in nb:
            hull_sets.append(set(keep_set_tile(cells[j][2] - org, 0, tile_px, n_side, PATCH).tolist()))
            lx, ly = cen[j] - org
            cpatch.append(int(ly // PATCH) * n_side + int(lx // PATCH) if 0 <= lx < tile_px and 0 <= ly < tile_px else -1)
        rec = {"row_idx": int(t), "d_nn1_um": float(d[order[0]] * mpp) if len(nb) else np.nan}
        for k in keys:
            K = kp[k]
            part = [bool(hs & K) for hs in hull_sets]
            rec[f"{k}_kept_patches"] = len(K)
            rec[f"{k}_n_partly"] = int(sum(part))
            rec[f"{k}_n_centroid"] = int(sum(c in K for c in cpatch))
            for r_ in range(8):
                rec[f"{k}_vis_k{r_+1}"] = int(part[r_]) if r_ < len(part) else 0
        rows.append(rec)
    os.makedirs(a.out, exist_ok=True)
    out = pd.DataFrame(rows)
    out.insert(0, "slide", a.slide_id)
    out.to_csv(os.path.join(a.out, f"{a.slide_id}.tsv.gz"), sep="\t", index=False)
    json.dump({"task": "P16 T2 kept-neighbour counts for m_iso(r)", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "script": os.path.abspath(__file__), "args": vars(a), "mpp": mpp, "tile_px": tile_px, "n_targets": len(tg),
               "inputs": {"slide": _stat(a.slide), "cache_cells": _stat(os.path.join(a.cache_dir, "cells.npz")),
                          "polygons_npz": _stat(a.polygons_npz), "targets": _stat(a.targets)},
               "gates": {"G0_order": "PASS"}, "wall_s": time.time() - t0},
              open(os.path.join(a.out, f"{a.slide_id}.PROVENANCE.json"), "w"), indent=2)
    print(f"[{a.slide_id}] {len(tg)} targets, {time.time()-t0:.0f}s -> {a.out}", flush=True)


def summarize(a):
    import pandas as pd
    fs = sorted(glob.glob(os.path.join(a.out, "*.tsv.gz")))
    df = pd.concat([pd.read_csv(f, sep="\t") for f in fs])
    keys = [f"r{float(r):g}" for r in a.rings_um.split(",")] + ["full"]
    lines, tab = [], []
    for k in keys:
        rec = {"region": k, "n_cells": len(df), "kept_patches_median": df[f"{k}_kept_patches"].median()}
        for q in ("n_partly", "n_centroid"):
            v = df[f"{k}_{q}"]
            rec[f"{q}_median"] = v.median(); rec[f"{q}_q25"] = v.quantile(.25); rec[f"{q}_q75"] = v.quantile(.75)
            rec[f"{q}_mean"] = v.mean()
        for r_ in range(8):
            rec[f"vis_k{r_+1}"] = df[f"{k}_vis_k{r_+1}"].mean()
        per_slide = df.groupby("slide")[f"{k}_n_partly"].median()
        rec["n_partly_slide_median_range"] = f"{per_slide.min():g}-{per_slide.max():g}"
        tab.append(rec)
    T = pd.DataFrame(tab)
    T.to_csv(os.path.join(a.out, "summary.tsv"), sep="\t", index=False)
    md = [f"# Kept-neighbour counts for m_iso(r)  [FINAL n={df.slide.nunique()}/12 slides, {len(df):,} typing cells]",
          "", "Per typing cell: neighbours (other segmented cells) inside the patches the isolated-cell run KEPT.",
          "partly = the neighbour's convex hull touches a kept 14 px (3.5 um) patch; centroid = its centre lies in a kept patch.",
          "vis_kN = share of cells whose N-th nearest neighbour (centre distance) is partly visible. full = all 256 patches (the paper's m).",
          f"Median nearest-neighbour distance: {df.d_nn1_um.median():.2f} um.", "",
          "| region | kept patches | neighbours partly visible, median [IQR] | centroid inside, median [IQR] | per-slide median range (partly) |",
          "|---|---|---|---|---|"]
    for _, r in T.iterrows():
        md.append(f"| {r.region} | {r.kept_patches_median:g} | {r.n_partly_median:g} [{r.n_partly_q25:g}-{r.n_partly_q75:g}] | "
                  f"{r.n_centroid_median:g} [{r.n_centroid_q25:g}-{r.n_centroid_q75:g}] | {r.n_partly_slide_median_range} |")
    md += ["", "| region | " + " | ".join(f"k={i}" for i in range(1, 9)) + " |", "|---|" + "---|" * 8]
    for _, r in T.iterrows():
        md.append(f"| {r.region} | " + " | ".join(f"{r[f'vis_k{i}']:.2f}" for i in range(1, 9)) + " |")
    open(os.path.join(a.out, "SUMMARY.md"), "w").write("\n".join(md) + "\n")
    json.dump({"task": "P16 T2 kept-neighbour counts — summary", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "script": os.path.abspath(__file__), "inputs": [_stat(f) for f in fs]},
              open(os.path.join(a.out, "PROVENANCE.json"), "w"), indent=2)
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide"); ap.add_argument("--slide-id"); ap.add_argument("--polygons-npz")
    ap.add_argument("--cache-dir"); ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--targets", default="results/masked_ctx/eval12_typing_cells.parquet")
    ap.add_argument("--rings-um", default="4,8,14"); ap.add_argument("--search-um", type=float, default=45.0)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize(a) if a.summarize else run_slide(a)
