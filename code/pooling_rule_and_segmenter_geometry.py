#!/usr/bin/env python
"""P16 T2 — pooling-rule x segmenter geometry audit (CPU only, read-only).

WHY
---
Every number in this project (J1-J5, raw112=0.5845) is pooled with ONE rule:
    region = convex_hull(nucleus polygon), margin_um = 0, weight = area fraction
This script measures what that choice costs, against 8 alternatives, and how much
each rule's weights move when the *segmenter* changes (HoverFast <-> CellViT-Virchow-x40).

Selection criterion (user 2026-08-02): segmentation is imperfect prediction, so the
defensible pooling rule is the one whose weights are LEAST sensitive to which
segmenter produced the polygon. Weight-vector agreement is a hard bound on m_cell
agreement -- identical weights over identical patches => identical m_cell -- so this
is provable WITHOUT touching the encoder.

THE 9 ARMS  (all at margin_um = 0; settled: m3=0.573 vs m0=0.579)
    region   in {poly (raw contour), hull (convex hull), bbox (axis-aligned)}
    weight   in {frac (area overlap), bin (any overlap -> 1), center (patch centre inside)}
    hull_frac = current production.  bbox_bin = exact CellViT++
    (cellvit/inference/postprocessing_cupy.py:787-794, floor/ceil + torch.mean).

AXIS CONVENTIONS -- verified 2026-08-02 against real files, do not "fix":
    CellViT _cells.json : centroid = [X, Y]   contour = [[X, Y], ...]
                          bbox     = [[Y_min, X_min], [Y_max, X_max]]   <-- OPPOSITE
      The canonical loader's docstring claims bbox is [X_min, Y_min]; that is WRONG.
      We therefore DERIVE bbox from contour min/max and never read the bbox field.
    HoverFast .json.gz  : coordinates = [[X, Y], ...]  centroid = [X, Y]
    Both are level-0 px, base_mpp 0.2522 == our cache mpp -> same frame, no rescale.

R1 (cellvit skill): CellViT centroids come from the canonical `_cells_centroids.parquet`
sidecar (cx_l0_px, cy_l0_px). We assert it row-aligns with _cells.json before using
contours from the json. Raw c["centroid"][i] is never trusted on its own.

SAMPLING: ROI-based, never uniform-random -- the contamination metric needs a cell's
NEIGHBOURS to be present. We draw N_ROI square windows and take every cell inside.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from skimage.draw import polygon as sk_polygon

PATCH = 14          # L0 px per Virchow2 patch (448 tile / 32 side) = 3.5308 um @ mpp .2522
MPP = 0.2522
REGIONS = ("poly", "hull", "bbox")
WEIGHTS = ("frac", "bin", "center")
ARMS = [f"{r}_{w}" for r in REGIONS for w in WEIGHTS]


# ----------------------------------------------------------------------------- readers
def load_hoverfast(path, rois):
    """-> list of (cx, cy, contour_xy float32 (K,2)). HoverFast: Polygon + centroid, [X,Y]."""
    out = []
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    for feat in data:
        g = feat.get("geometry", {})
        if g.get("type") != "Polygon":
            continue
        ring = np.asarray(g["coordinates"][0], dtype=np.float32)   # (K,2) [X,Y]
        c = g.get("centroid")
        cx, cy = (float(c[0]), float(c[1])) if c else (float(ring[:, 0].mean()),
                                                       float(ring[:, 1].mean()))
        if in_rois(cx, cy, rois):
            out.append((cx, cy, ring))
    return out


def load_cellvit(json_path, parquet_path, rois, assert_n=2000):
    """-> list of (cx, cy, contour_xy, type). Centroids from the canonical parquet (R1)."""
    dfc = pd.read_parquet(parquet_path)
    for col in ("cx_l0_px", "cy_l0_px", "cell_type"):
        assert col in dfc.columns, f"sidecar missing {col}: {parquet_path}"
    with open(json_path, "r") as f:
        data = json.load(f)
    tmap = data.get("type_map", {})
    expected = {"0": "Tumor cell", "1": "Immune cell", "2": "Stroma",
                "3": "Necrotic cell", "4": "Normal Epithelial cell", "5": "Other"}
    if tmap != expected:
        sys.exit(f"STOP (cellvit skill R3): type_map drift in {json_path}\n  {tmap}")
    cells = data["cells"]
    assert len(cells) == len(dfc), (
        f"sidecar/json row mismatch: parquet {len(dfc)} vs json {len(cells)}")
    # R1: prove the parquet row-aligns with the json before we trust json contours.
    k = min(assert_n, len(cells))
    jx = np.array([cells[i]["centroid"][0] for i in range(k)], dtype=np.float64)
    jy = np.array([cells[i]["centroid"][1] for i in range(k)], dtype=np.float64)
    px = dfc["cx_l0_px"].values[:k].astype(np.float64)
    py = dfc["cy_l0_px"].values[:k].astype(np.float64)
    assert np.allclose(jx, px) and np.allclose(jy, py), (
        "canonical parquet does NOT row-align with _cells.json centroids -- STOP, "
        "do not fall back to raw centroid reads (cellvit skill R1).")
    cx_all = dfc["cx_l0_px"].values
    cy_all = dfc["cy_l0_px"].values
    ty_all = dfc["cell_type"].values
    out = []
    for i, c in enumerate(cells):
        cx, cy = float(cx_all[i]), float(cy_all[i])
        if not in_rois(cx, cy, rois):
            continue
        ring = np.asarray(c["contour"], dtype=np.float32)          # (K,2) [X,Y]
        if ring.ndim != 2 or ring.shape[0] < 3:
            continue
        out.append((cx, cy, ring, int(ty_all[i])))
    return out


def in_rois(x, y, rois):
    for (x0, y0, x1, y1) in rois:
        if x0 <= x < x1 and y0 <= y < y1:
            return True
    return False


# ------------------------------------------------------------------- per-cell geometry
def region_polygon(ring, region):
    """Return the polygon vertices (K,2)[X,Y] defining the pooling region."""
    if region == "poly":
        return ring
    if region == "hull":
        try:
            h = ConvexHull(ring)
        except Exception:
            return ring
        return ring[h.vertices]
    if region == "bbox":
        # derived from contour -- never from the [Y,X] bbox field (see module docstring)
        x0, y0 = ring[:, 0].min(), ring[:, 1].min()
        x1, y1 = ring[:, 0].max(), ring[:, 1].max()
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    raise ValueError(region)


def patch_weights(poly, patch=PATCH):
    """Rasterise a region on the GLOBAL patch lattice.

    Returns (pidx (M,2) int64 [py,px] global patch coords, frac (M,), binv (M,), cen (M,)).
    Grid is anchored globally at (0,0); production anchors per-tile, which differs only in
    sub-patch phase (documented simplification -- phase varies across cells either way).
    """
    x0 = int(np.floor(poly[:, 0].min() / patch)) * patch
    x1 = int(np.ceil(poly[:, 0].max() / patch)) * patch
    y0 = int(np.floor(poly[:, 1].min() / patch)) * patch
    y1 = int(np.ceil(poly[:, 1].max() / patch)) * patch
    nx, ny = max(1, (x1 - x0) // patch), max(1, (y1 - y0) // patch)
    m = np.zeros((ny * patch, nx * patch), dtype=bool)
    rr, cc = sk_polygon(poly[:, 1] - y0, poly[:, 0] - x0, shape=m.shape)
    if len(rr) == 0:
        return (np.zeros((0, 2), np.int64),) + (np.zeros(0, np.float32),) * 3
    m[rr, cc] = True
    blk = m.reshape(ny, patch, nx, patch)
    frac = blk.mean(axis=(1, 3)).astype(np.float32)
    binv = blk.any(axis=(1, 3)).astype(np.float32)
    cen = m[patch // 2::patch, patch // 2::patch].astype(np.float32)
    cen = cen[:ny, :nx]
    py, px = np.meshgrid(np.arange(ny) + y0 // patch,
                         np.arange(nx) + x0 // patch, indexing="ij")
    keep = (frac > 0) | (binv > 0) | (cen > 0)
    pidx = np.stack([py[keep], px[keep]], axis=1).astype(np.int64)
    return pidx, frac[keep], binv[keep], cen[keep]


def cell_arms(ring):
    """-> {arm: (pidx (M,2), w (M,))} for all 9 arms, plus areas."""
    out, areas = {}, {}
    for region in REGIONS:
        poly = region_polygon(ring, region)
        pidx, frac, binv, cen = patch_weights(poly)
        out[f"{region}_frac"] = (pidx, frac)
        out[f"{region}_bin"] = (pidx, binv)
        out[f"{region}_center"] = (pidx, cen)
        areas[region] = float(polygon_area(poly))
    return out, areas


def polygon_area(poly):
    x, y = poly[:, 0].astype(np.float64), poly[:, 1].astype(np.float64)
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# ------------------------------------------------------------------------- comparisons
def weight_cosine(a, b):
    """Cosine between two (pidx, w) weight vectors defined on global patch coords."""
    (ia, wa), (ib, wb) = a, b
    ma = {(int(p), int(q)): float(v) for (p, q), v in zip(ia, wa) if v > 0}
    mb = {(int(p), int(q)): float(v) for (p, q), v in zip(ib, wb) if v > 0}
    if not ma or not mb:
        return np.nan
    keys = set(ma) | set(mb)
    va = np.array([ma.get(k, 0.0) for k in keys])
    vb = np.array([mb.get(k, 0.0) for k in keys])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return np.nan
    return float(va @ vb / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", nargs="+", required=True)
    ap.add_argument("--hoverfast-root",
                    default="data/eval_cohort/cell_seg_hoverfast")
    ap.add_argument("--cellvit-root",
                    default="data/eval_cohort/CellViT-Virchow-x40")
    ap.add_argument("--gt-root", default="results/eval/gt_labels")
    ap.add_argument("--n-roi", type=int, default=12)
    ap.add_argument("--roi-px", type=int, default=2048)
    ap.add_argument("--match-tol-um", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    report = {"config": vars(a), "slides": {}}

    for sid in a.slides:
        print(f"\n===== {sid} =====", flush=True)
        hf_path = os.path.join(a.hoverfast_root, sid, f"{sid}.json.gz")
        cv_json = os.path.join(a.cellvit_root, f"{sid}_cells.json")
        cv_pq = os.path.join(a.cellvit_root, f"{sid}_cells_centroids.parquet")
        for p in (hf_path, cv_json, cv_pq):
            assert os.path.exists(p), f"missing input: {p}"

        # ROIs from the CellViT centroid extent (cheap, no slide open)
        dfc = pd.read_parquet(cv_pq, columns=["cx_l0_px", "cy_l0_px"])
        xlo, xhi = dfc.cx_l0_px.min(), dfc.cx_l0_px.max()
        ylo, yhi = dfc.cy_l0_px.min(), dfc.cy_l0_px.max()
        rois = []
        for _ in range(a.n_roi):
            x0 = float(rng.uniform(xlo, max(xlo + 1, xhi - a.roi_px)))
            y0 = float(rng.uniform(ylo, max(ylo + 1, yhi - a.roi_px)))
            rois.append((x0, y0, x0 + a.roi_px, y0 + a.roi_px))
        del dfc

        print("  loading HoverFast ...", flush=True)
        hf = load_hoverfast(hf_path, rois)
        print(f"    {len(hf):,} cells in ROIs", flush=True)
        print("  loading CellViT ...", flush=True)
        cv = load_cellvit(cv_json, cv_pq, rois)
        print(f"    {len(cv):,} cells in ROIs", flush=True)
        if not hf or not cv:
            print("  !! empty ROI intersection, skipping slide", flush=True)
            continue

        # ---- detection agreement: mutual nearest neighbour within tol
        tol_px = a.match_tol_um / MPP
        hxy = np.array([[c[0], c[1]] for c in hf])
        cxy = np.array([[c[0], c[1]] for c in cv])
        d_hc, i_hc = cKDTree(cxy).query(hxy, k=1)
        d_ch, i_ch = cKDTree(hxy).query(cxy, k=1)
        mutual = [(h, int(i_hc[h])) for h in range(len(hf))
                  if d_hc[h] <= tol_px and int(i_ch[int(i_hc[h])]) == h]
        det = {"n_hoverfast": len(hf), "n_cellvit": len(cv), "n_matched": len(mutual),
               "hoverfast_only": len(hf) - len(mutual), "cellvit_only": len(cv) - len(mutual),
               "match_tol_um": a.match_tol_um,
               "median_match_dist_um": float(np.median([d_hc[h] for h, _ in mutual]) * MPP)
               if mutual else None}
        print(f"  matched {det['n_matched']:,} | HF-only {det['hoverfast_only']:,} | "
              f"CV-only {det['cellvit_only']:,} | med dist "
              f"{det['median_match_dist_um']:.2f} um", flush=True)

        # ---- per-arm geometry, both segmenters
        occ = {(r, w): defaultdict(int) for r in REGIONS for w in ("frac", "center")}
        per_seg = {}
        for tag, cells in (("hoverfast", hf), ("cellvit", cv)):
            armstore, arearec = [], []
            for rec in cells:
                ring = rec[2]
                arms, areas = cell_arms(ring)
                armstore.append(arms)
                arearec.append(areas)
                if tag == "hoverfast":       # occupancy from the production segmenter
                    for r in REGIONS:
                        for pidx, wv, key in ((arms[f"{r}_frac"][0], arms[f"{r}_frac"][1], "frac"),
                                              (arms[f"{r}_center"][0], arms[f"{r}_center"][1], "center")):
                            for (p, q), v in zip(pidx, wv):
                                if v > 0:
                                    occ[(r, key)][(int(p), int(q))] += 1
            per_seg[tag] = (armstore, arearec)

        # ---- per-arm summary on the production segmenter
        arms_summary = {}
        hf_arms, hf_areas = per_seg["hoverfast"]
        for arm in ARMS:
            region = arm.split("_")[0]
            okey = "center" if arm.endswith("center") else "frac"
            nnz, sw, contam = [], [], []
            for arms in hf_arms:
                pidx, wv = arms[arm]
                sel = wv > 0
                if not sel.any():
                    continue
                w = wv[sel]
                nnz.append(int(sel.sum()))
                sw.append(float(w.sum()))
                shared = np.array([occ[(region, okey)][(int(p), int(q))] > 1
                                   for (p, q) in pidx[sel]])
                contam.append(float(w[shared].sum() / w.sum()) if w.sum() > 0 else np.nan)
            arms_summary[arm] = {
                "n_cells": len(nnz),
                "tokens_per_cell_median": float(np.median(nnz)) if nnz else None,
                "sum_w_median": float(np.median(sw)) if sw else None,
                "contamination_mean": float(np.nanmean(contam)) if contam else None,
                "contamination_median": float(np.nanmedian(contam)) if contam else None,
                "empty_rate": 1.0 - len(nnz) / max(1, len(hf_arms)),
            }

        # ---- segmenter sensitivity on matched cells
        cv_arms, cv_areas = per_seg["cellvit"]
        sens = {}
        for arm in ARMS:
            cos = [weight_cosine(hf_arms[h][arm], cv_arms[c][arm]) for h, c in mutual]
            cos = np.array([c for c in cos if not np.isnan(c)])
            sens[arm] = {"n": int(cos.size),
                         "cos_mean": float(cos.mean()) if cos.size else None,
                         "cos_median": float(np.median(cos)) if cos.size else None,
                         "frac_identical": float((cos > 0.9999).mean()) if cos.size else None,
                         "frac_below_0p9": float((cos < 0.9).mean()) if cos.size else None}

        # ---- cross-arm cosine (early exit) on the production segmenter
        ref = "hull_frac"
        cross = {}
        for arm in ARMS:
            if arm == ref:
                continue
            cos = np.array([weight_cosine(arms[ref], arms[arm]) for arms in hf_arms[:20000]])
            cos = cos[~np.isnan(cos)]
            cross[arm] = {"cos_median_vs_hull_frac": float(np.median(cos)) if cos.size else None}

        # ---- convex-hull inflation, by GT class where available
        gt_path = os.path.join(a.gt_root, f"{sid}.parquet")
        infl = {"hull_over_poly_median": float(np.median(
            [ar["hull"] / ar["poly"] for ar in hf_areas if ar["poly"] > 0])),
            "bbox_over_poly_median": float(np.median(
                [ar["bbox"] / ar["poly"] for ar in hf_areas if ar["poly"] > 0])),
            "by_class": None, "gt_note": None}
        if os.path.exists(gt_path):
            infl["gt_note"] = (f"GT at {gt_path}; per-class join needs cache_idx alignment "
                               "(ROI-subsampled cells are not cache-ordered) -- deferred to "
                               "the full run, reported unjoined here.")
        report["slides"][sid] = {"detection": det, "arms": arms_summary,
                                 "segmenter_sensitivity": sens, "cross_arm": cross,
                                 "hull_inflation": infl, "n_rois": len(rois)}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report["generated_utc"] = stamp
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(report, f, indent=1)
    write_markdown(report, os.path.join(a.out, "report.md"))
    print(f"\nwrote {a.out}/results.json + report.md", flush=True)


def write_markdown(rep, path):
    L = ["# P16 T2 — pooling rule x segmenter geometry", "",
         f"_generated {rep.get('generated_utc')} · CPU-only, read-only_", "",
         "margin_um = 0 for all arms. `hull_frac` = current production. "
         "`bbox_bin` = exact CellViT++.", ""]
    for sid, d in rep["slides"].items():
        det = d["detection"]
        L += [f"## {sid}", "",
              f"ROIs: {d['n_rois']} · HoverFast {det['n_hoverfast']:,} · "
              f"CellViT {det['n_cellvit']:,} · matched {det['n_matched']:,} "
              f"(HF-only {det['hoverfast_only']:,}, CV-only {det['cellvit_only']:,}, "
              f"median match dist {det['median_match_dist_um']} um)", "",
              "| arm | tokens/cell | sum w | contamination (mean) | empty | "
              "segmenter cos (median) | frac cos<0.9 | cos vs hull_frac |",
              "|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            s, g, c = d["arms"][arm], d["segmenter_sensitivity"][arm], d["cross_arm"].get(arm, {})
            L.append(
                f"| `{arm}` | {s['tokens_per_cell_median']} | "
                f"{None if s['sum_w_median'] is None else round(s['sum_w_median'],3)} | "
                f"{None if s['contamination_mean'] is None else round(s['contamination_mean'],4)} | "
                f"{round(s['empty_rate'],4)} | "
                f"{None if g['cos_median'] is None else round(g['cos_median'],4)} | "
                f"{None if g['frac_below_0p9'] is None else round(g['frac_below_0p9'],4)} | "
                f"{c.get('cos_median_vs_hull_frac')} |")
        h = d["hull_inflation"]
        L += ["", f"convex-hull inflation: hull/poly median **{h['hull_over_poly_median']:.4f}**, "
                  f"bbox/poly median **{h['bbox_over_poly_median']:.4f}**", ""]
        if h.get("gt_note"):
            L += [f"_{h['gt_note']}_", ""]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
