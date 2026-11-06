#!/usr/bin/env python3
"""cache_masked_context.py — target-masked CONTEXT embeddings for the predictor (P16 T2, W1 test).

[user 2026-11-03] "we test this (masking target cells and train the predictor and evaluate the cell
typing) tonight" — HD-12 eval + corpus31 training pool. Plan:
results/reports/20261103_masked_context_retrain_plan/plan.md (decisions: hull + 4 um ring; control
arm included; 300k training pool).

WHAT IT COMPUTES, per TARGET t (one row):
  ctx(t)  = t's Kc nearest OTHER cells (self excluded) — the deployment / own_knn regime.
  Each context cell j is pooled from ITS OWN grid tile T(j) (pooler_grid_fm geometry, unchanged).
  If T(j) contains any pixel of t's REGION (convex hull + --ring-um dilation), then j's cached
  embedding is LEAKY: the ViT attended over t. For every such tile we re-run the encoder with t's
  patch tokens MASKED AS KEYS in every block (attention mask, so no token can read them — exactly
  equivalent to deleting them), and re-pool j over its hull restricted to the KEPT patches.
  Neighbours whose tile does not touch t keep the cached m_j bit-for-bit.

  variant "target"  : the drop set D = patches with coverage(t's region) > 0.
  variant "control" : the SAME-SIZE patch set moved off the target (point reflection through the
                      tile centre, then flips, then half-tile cyclic shifts; first candidate with ZERO
                      overlap with D wins). Removes as many tokens at a comparable position but leaves
                      the target visible ⇒ target-drop − control-drop isolates TARGET information from
                      the masking perturbation itself.
  The target's own m_t is NEVER changed (it is the predictor's regression target and the readout's m).

STORAGE (sparse): per variant, chg (T,Kc) bool + val (nnz,D) fp16 in row-major order of chg.
  ctx_emb(variant) = cache_m[ctx]; ctx_emb[chg] = val.

GATES (written to PROVENANCE, fatal where noted):
  G0  cell order: polygons centroids == cache centroid_xy (fatal).
  G1  no-drop reproduction: re-pooling the first --gate-n affected neighbours with an EMPTY drop set
      through the attention-mask path must reproduce the cached m_j (median cos >= 0.9999, fatal).
  G3  non-vacuity: target-masked neighbours must move (median rel L2 > 1e-3, fatal).
"""
import os, sys, json, time, argparse, math
from collections import defaultdict, OrderedDict
from datetime import datetime, timezone
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from pooler_grid import hull_weights_tile
from fm_loaders import FM_SPECS, load_fm


def _stat(p):
    try:
        st = os.stat(p)
        return {"path": os.path.abspath(p), "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
    except OSError:
        return {"path": p, "size": None, "mtime": None}


def tile_assignment(W, H, cen, mpp, patch, fov_um, stride_frac):
    """VERBATIM pooler_grid_fm._tile_assignment / pool_cells_grid_fm geometry (native mpp)."""
    tile_px = max(patch, int(round(fov_um / mpp / patch)) * patch)
    n_side = tile_px // patch
    stride = max(patch, int(round(tile_px * stride_frac)))
    half = tile_px / 2.0
    ci = np.clip(np.round((cen[:, 0] - half) / stride).astype(int), 0, None)
    cj = np.clip(np.round((cen[:, 1] - half) / stride).astype(int), 0, None)
    tx = np.clip((half + ci * stride - half).astype(int), 0, max(0, W - tile_px))
    ty = np.clip((half + cj * stride - half).astype(int), 0, max(0, H - tile_px))
    return tx, ty, tile_px, n_side, stride


def control_set(D, n_side):
    """Same-size patch set disjoint from D. Returns (idx array, rule name) or (None, 'none')."""
    r, c = np.divmod(D, n_side)
    h = n_side // 2
    cands = [("point_reflect", (n_side - 1 - r, n_side - 1 - c)),
             ("flip_x", (r, n_side - 1 - c)),
             ("flip_y", (n_side - 1 - r, c)),
             ("shift_x", (r, (c + h) % n_side)),
             ("shift_y", ((r + h) % n_side, c)),
             ("shift_xy", ((r + h) % n_side, (c + h) % n_side))]
    Ds = set(D.tolist())
    for name, (rr, cc) in cands:
        idx = rr * n_side + cc
        if not (set(idx.tolist()) & Ds):
            return np.unique(idx), name
    return None, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True, help="WSI path")
    ap.add_argument("--slide-id", required=True)
    ap.add_argument("--polygons-npz", default=None)
    ap.add_argument("--geojson", default=None)
    ap.add_argument("--seg", default="cellvit")
    ap.add_argument("--max-cells", type=int, default=0, help="cells[:N] — MUST match the cache build")
    ap.add_argument("--cache-dir", required=True, help=".../<fm>/<slide>/fov56 holding cells.npz + meta.json")
    ap.add_argument("--targets", default=None, help="parquet (slide,row_idx): EVAL mode target list")
    ap.add_argument("--n-random", type=int, default=0, help="TRAIN mode: sample this many targets uniformly")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Kc", type=int, default=36)
    ap.add_argument("--ring-um", type=float, default=4.0, help="target region = hull dilated by this")
    ap.add_argument("--fov-um", type=float, default=56.0)
    ap.add_argument("--stride-frac", type=float, default=0.5)
    ap.add_argument("--margin-um", type=float, default=0.0, help="m_cell pooling margin (cache = 0)")
    ap.add_argument("--fm", default="hoptimus0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gate-n", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="smoke: keep only the first N targets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true", help="random tokens, no model (plumbing test)")
    a = ap.parse_args()
    t_start = time.time()
    os.makedirs(a.out, exist_ok=True)

    import openslide
    sl = openslide.OpenSlide(a.slide)
    W, H = sl.dimensions
    meta = json.load(open(os.path.join(a.cache_dir, "meta.json")))
    mpp = float(meta["mpp"])
    z = np.load(os.path.join(a.cache_dir, "cells.npz"))
    m_cache = z["m_cell"]                                   # fp16 (N, D)
    cen_cache = z["centroid_xy"].astype(np.float64)
    N, D = m_cache.shape

    if a.polygons_npz:
        from io_polygons_npz import load_polygons_npz
        cells = load_polygons_npz(a.polygons_npz)
    else:
        from io_polygons import load_polygons
        cells = load_polygons(a.geojson, a.seg)
    if a.max_cells > 0:
        cells = cells[:a.max_cells]
    cen = np.array([[float(c[1][0]), float(c[1][1])] for c in cells], np.float64)
    # ---- G0 cell order -------------------------------------------------------------------------
    assert len(cells) == N, f"G0 FAIL: {len(cells)} polygons vs {N} cache rows"
    assert np.allclose(cen, cen_cache, atol=1e-2), "G0 FAIL: centroid order differs from the cache"
    print(f"[{a.slide_id}] N={N:,} D={D} mpp={mpp} G0 order PASS", flush=True)

    spec = FM_SPECS[a.fm]
    PATCH, NPRE = spec.patch, spec.n_prefix
    assert int(meta["tile_px"]) == max(PATCH, int(round(a.fov_um / mpp / PATCH)) * PATCH)
    tx, ty, tile_px, n_side, stride = tile_assignment(W, H, cen, mpp, PATCH, a.fov_um, a.stride_frac)
    NP = n_side * n_side
    ring_px = max(0, int(round(a.ring_um / mpp)))
    marg_px = max(0, int(round(a.margin_um / mpp)))

    # ---- targets ------------------------------------------------------------------------------
    if a.targets:
        import pandas as pd
        df = pd.read_parquet(a.targets)
        tg = np.sort(df[df.slide == a.slide_id].row_idx.values.astype(np.int64))
        mode = "eval"
    else:
        assert a.n_random > 0, "need --targets or --n-random"
        tg = np.sort(np.random.default_rng(a.seed).choice(N, min(a.n_random, N), replace=False))
        mode = "train"
    if a.limit:
        tg = tg[:a.limit]
    T = len(tg)
    assert T > 0 and tg.min() >= 0 and tg.max() < N
    tree = cKDTree(cen)
    _, nn = tree.query(cen[tg], k=a.Kc + 3, workers=-1)
    ctx = np.zeros((T, a.Kc), np.int64)
    for i, (t, row) in enumerate(zip(tg, nn)):
        r = row[row != t][:a.Kc]
        assert len(r) == a.Kc, f"target {t}: only {len(r)} neighbours"
        ctx[i] = r

    # ---- job list: (tile, target row i) -> context slots k whose tile is that tile and is touched --
    jobs = OrderedDict()                      # (x0,y0) -> list of (i, drop_t, drop_c, slots)
    n_aff_slots = 0
    ctrl_rule = defaultdict(int)
    for i in range(T):
        t = int(tg[i])
        by_tile = defaultdict(list)
        for k, j in enumerate(ctx[i]):
            by_tile[(int(tx[j]), int(ty[j]))].append(k)
        for (x0, y0), slots in by_tile.items():
            wt = hull_weights_tile(cells[t][2] - np.array([x0, y0]), ring_px, tile_px, n_side, patch=PATCH)
            Dt = np.nonzero(wt > 0)[0]
            if Dt.size == 0:
                continue                      # tile never saw the target: cached m_j is leak-free
            Dc, rule = control_set(Dt, n_side)
            ctrl_rule[rule] += 1
            jobs.setdefault((x0, y0), []).append((i, Dt, Dc, slots))
            n_aff_slots += len(slots)
    n_jobs = sum(len(v) for v in jobs.values())
    print(f"[{a.slide_id}] mode={mode} T={T:,} tiles={len(jobs):,} jobs={n_jobs:,} "
          f"({n_jobs/max(T,1):.2f}/target) affected slots={n_aff_slots:,} "
          f"({n_aff_slots/max(T,1):.1f}/36 per target) ctrl_rules={dict(ctrl_rule)}", flush=True)

    # ---- encoder --------------------------------------------------------------------------------
    if a.mock:
        model, dev = None, "cpu"
    else:
        import torch
        model, spec = load_fm(a.fm)
        dev = "cuda"
    mean, std = spec.mean, spec.std

    def forward(imgs, drops):
        """imgs: list of (tile_px,tile_px,3) uint8; drops: list of patch-index arrays (may be empty).
        Returns patch tokens (B,NP,D) fp32 and CLS (B,D)."""
        if a.mock:
            rng = np.random.default_rng(len(imgs))
            return (rng.standard_normal((len(imgs), NP, D)).astype(np.float32),
                    rng.standard_normal((len(imgs), D)).astype(np.float32))
        import torch
        x = np.stack([((im.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1) for im in imgs])
        x = torch.from_numpy(x).to(dev)
        keep = torch.ones((len(imgs), NPRE + NP), dtype=torch.bool, device=dev)
        for b, d in enumerate(drops):
            if d is not None and len(d):
                keep[b, NPRE + torch.as_tensor(d, device=dev)] = False
        mask = torch.zeros((len(imgs), 1, 1, NPRE + NP), dtype=torch.float16, device=dev)
        mask.masked_fill_(~keep[:, None, None, :], float("-inf"))
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            f = model.forward_features(x, attn_mask=mask)
        f = f.float().cpu().numpy()
        return f[:, NPRE:, :], f[:, 0, :]

    tile_cache = OrderedDict()

    def read(x0, y0):
        key = (x0, y0)
        if key in tile_cache:
            tile_cache.move_to_end(key)
            return tile_cache[key]
        im = np.asarray(sl.read_region((x0, y0), 0, (tile_px, tile_px)).convert("RGB"), np.uint8)
        tile_cache[key] = im
        if len(tile_cache) > 512:
            tile_cache.popitem(last=False)
        return im

    wcache = {}

    def w_of(j, x0, y0):
        key = (j, x0, y0)
        if key not in wcache:
            wcache[key] = hull_weights_tile(cells[j][2] - np.array([x0, y0]), marg_px, tile_px, n_side, patch=PATCH)
        return wcache[key]

    def repool(Pt, cls, j, x0, y0, dropped):
        w = w_of(j, x0, y0).copy()
        if dropped is not None and len(dropped):
            w[dropped] = 0.0
        s = w.sum()
        if s > 0:
            return (Pt * w[:, None]).sum(0) / s, False
        return cls, True

    # ---- G1 no-drop reproduction (through the SAME attention-mask path, empty drop set) ------------
    g1 = {"n": 0}
    if not a.mock:
        sample = []
        for (x0, y0), lst in jobs.items():
            for (i, Dt, Dc, slots) in lst:
                for k in slots:
                    sample.append((x0, y0, int(ctx[i, k])))
            if len(sample) >= a.gate_n:
                break
        sample = sample[:a.gate_n]
        cos = []
        for b0 in range(0, len(sample), a.batch):
            bb = sample[b0:b0 + a.batch]
            P, C = forward([read(x0, y0) for x0, y0, _ in bb], [None] * len(bb))
            for q, (x0, y0, j) in enumerate(bb):
                v, _ = repool(P[q], C[q], j, x0, y0, None)
                ref = m_cache[j].astype(np.float32)
                cos.append(float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-12)))
        cos = np.array(cos)
        g1 = {"n": int(len(cos)), "median_cos": float(np.median(cos)), "min_cos": float(cos.min())}
        print(f"[{a.slide_id}] G1 no-drop reproduction: n={g1['n']} median cos={g1['median_cos']:.6f} "
              f"min={g1['min_cos']:.6f}", flush=True)
        assert g1["median_cos"] >= 0.9999, "G1 FAIL: attention-mask path does not reproduce the cache"

    # ---- main loop ------------------------------------------------------------------------------
    out = {v: {} for v in ("target", "control")}           # (i,k) -> vector
    n_fallback = {"target": 0, "control": 0}
    n_ctrl_none = 0
    work = []
    for (x0, y0), lst in jobs.items():
        for (i, Dt, Dc, slots) in lst:
            work.append((x0, y0, i, "target", Dt, slots))
            if Dc is None:
                n_ctrl_none += 1
                work.append((x0, y0, i, "control", np.empty(0, np.int64), slots))   # no valid move: unmasked
            else:
                work.append((x0, y0, i, "control", Dc, slots))
    t0 = time.time()
    for b0 in range(0, len(work), a.batch):
        bb = work[b0:b0 + a.batch]
        P, C = forward([read(x0, y0) for x0, y0, *_ in bb], [w[4] for w in bb])
        for q, (x0, y0, i, var, drop, slots) in enumerate(bb):
            for k in slots:
                v, fb = repool(P[q], C[q], int(ctx[i, k]), x0, y0, drop)
                out[var][(i, k)] = np.asarray(v, np.float16)
                n_fallback[var] += int(fb)
        if (b0 // a.batch) % 200 == 0:
            el = time.time() - t0
            print(f"  {b0+len(bb):,}/{len(work):,} forwards | {(b0+len(bb))/max(el,1e-9):.1f}/s", flush=True)
    fw_rate = len(work) / max(time.time() - t0, 1e-9)

    # ---- G3 non-vacuity + save ------------------------------------------------------------------
    save = {"tgt": tg, "ctx": ctx}
    stats = {}
    for var in ("target", "control"):
        chg = np.zeros((T, a.Kc), bool)
        for (i, k) in out[var]:
            chg[i, k] = True
        ii, kk = np.nonzero(chg)
        val = np.stack([out[var][(i, k)] for i, k in zip(ii, kk)]).astype(np.float16) if len(ii) else \
            np.zeros((0, D), np.float16)
        ref = m_cache[ctx[ii, kk]].astype(np.float32)
        rel = np.linalg.norm(val.astype(np.float32) - ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-12)
        save[f"chg_{var}"] = chg
        save[f"val_{var}"] = val
        stats[var] = {"nnz": int(len(ii)), "slots_per_target": float(chg.sum(1).mean()),
                      "frac_targets_with_change": float(chg.any(1).mean()),
                      "rel_l2_median": float(np.median(rel)) if len(rel) else None,
                      "rel_l2_p90": float(np.percentile(rel, 90)) if len(rel) else None,
                      "cls_fallback": int(n_fallback[var])}
    print(f"[{a.slide_id}] stats {json.dumps(stats)}", flush=True)
    if not a.mock:
        assert stats["target"]["rel_l2_median"] and stats["target"]["rel_l2_median"] > 1e-3, \
            "G3 FAIL: target-masked neighbours did not move"
    np.savez(os.path.join(a.out, "masked_ctx.npz"), **save)
    prov = {"task": "P16 T2 — target-masked context cache (W1 test)",
            "generated_utc": datetime.now(timezone.utc).isoformat(), "script": os.path.abspath(__file__),
            "slide_id": a.slide_id, "mode": mode, "args": vars(a),
            "inputs": {"slide": _stat(a.slide), "cache": _stat(os.path.join(a.cache_dir, "cells.npz")),
                       "polygons_npz": _stat(a.polygons_npz) if a.polygons_npz else None,
                       "geojson": _stat(a.geojson) if a.geojson else None,
                       "targets": _stat(a.targets) if a.targets else None},
            "geometry": {"mpp": mpp, "tile_px": tile_px, "n_side": n_side, "stride_px": stride,
                         "ring_px": ring_px, "margin_px": marg_px},
            "counts": {"N": int(N), "T": int(T), "tiles": len(jobs), "jobs": int(n_jobs),
                       "affected_slots": int(n_aff_slots), "control_rules": dict(ctrl_rule),
                       "control_no_valid_move": int(n_ctrl_none)},
            "gates": {"G0_order": "PASS", "G1_nodrop": g1, "G3_nonvacuity": stats["target"]["rel_l2_median"]},
            "stats": stats, "forwards_per_s": fw_rate, "wall_s": time.time() - t_start,
            "semantics": "ctx_emb = cache_m[ctx]; ctx_emb[chg_<variant>] = val_<variant> (row-major). "
                         "The target's own m is never modified."}
    json.dump(prov, open(os.path.join(a.out, "PROVENANCE.json"), "w"), indent=2)
    print(f"[{a.slide_id}] done in {time.time()-t_start:.0f}s ({fw_rate:.1f} forwards/s) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
