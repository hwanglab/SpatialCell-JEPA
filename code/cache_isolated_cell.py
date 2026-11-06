#!/usr/bin/env python3
"""cache_isolated_cell.py — ISOLATED-CELL embedding m_iso(r) for the typing cells (P16 T2, camera-ready).

[user 2026-11-05] "to fully answer the question in the title, we also extract the embedding from an isolated
target cell and compare this to the current setting." Plan: results/reports/20261105_camera_ready_finalize_plan/plan.md
(decisions [confirmed]: ring ladder r = 4 / 8 / 14 um, primary r = 4; token drop, same mechanism as the masking run).

WHAT IT COMPUTES, per typing cell t and radius r:
  T(t) = the grid tile m_t was pooled from (pooler_grid_fm geometry, VERBATIM from cache_masked_context.py).
  KEEP = patches with coverage(t's convex hull dilated by r) > 0; every other patch token is MASKED AS A KEY in
  every block (attention mask: no token can read it == deleting it; prefix tokens stay). m_iso(r) = t's hull
  (margin 0, the cache's pooling rule) pooled over the kept patches. hull ⊂ hull+r, so every patch carrying hull
  weight is kept: the pooling region is unchanged, only what those tokens could attend to is removed.
  The complement of cache_masked_context.py: there the TARGET was removed from the neighbours' tiles; here the
  NEIGHBOURHOOD is removed from the target's own tile.

OUTPUT: <out>/r<r>/<slide>/fov56/cells.npz = the cache with m_cell[t] := m_iso(r) at the typing rows ONLY (all
  other rows and centroid_xy unchanged), plus meta.json copy and PROVENANCE.json. Pointing
  p2_typing_and_slideid.py --cache at <out>/r<r> makes its "m" arm read m_iso(r) on exactly the typing cells.

GATES (fatal): G0 cell order (polygons vs cache centroids); G1 empty-drop reproduction through the same mask path
  (median cos >= 0.9999 vs cached m); G2 every typing row present in the target list; G3 non-vacuity (median rel L2 > 1e-3).
"""
import os, sys, json, time, argparse, shutil
from collections import OrderedDict
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pooler_grid import hull_weights_tile
from fm_loaders import FM_SPECS, load_fm
from cache_masked_context import tile_assignment, _stat


def keep_set_tile(ring_local, margin_px, tile_px, n_side, patch):
    """Patch indices with hull_weights_tile(...) > 0, computed EXACTLY the same way (same skimage hull raster, same
    bbox) but with "pixel within margin_px of the hull raster" via a Euclidean distance transform instead of
    binary_dilation(disk(margin_px)). Same pixel set (disk(r) = integer offsets with dx^2+dy^2 <= r^2 = EDT <= r);
    cost no longer grows with r^2 (hull_weights_tile: 968 ms/call at 56 px; 2026-11-05 8346693 TIMEOUT 12/12).
    Verified identical on real polygons before use (see T2 log 2026-11-05)."""
    from scipy.spatial import ConvexHull
    from scipy.ndimage import distance_transform_edt
    from skimage.draw import polygon as sk_polygon
    if len(ring_local) < 3:
        return np.zeros(0, np.int64)
    try:
        hull = ConvexHull(ring_local)
    except Exception:
        return np.zeros(0, np.int64)
    hp = ring_local[hull.vertices]
    bx0 = int(np.floor(hp[:, 0].min() - margin_px)); bx1 = int(np.ceil(hp[:, 0].max() + margin_px))
    by0 = int(np.floor(hp[:, 1].min() - margin_px)); by1 = int(np.ceil(hp[:, 1].max() + margin_px))
    px0 = max(0, bx0 // patch); py0 = max(0, by0 // patch)
    px1 = min(n_side - 1, bx1 // patch); py1 = min(n_side - 1, by1 // patch)
    if px1 < px0 or py1 < py0:
        return np.zeros(0, np.int64)
    rx0, ry0 = px0 * patch, py0 * patch
    rw, rh = (px1 - px0 + 1) * patch, (py1 - py0 + 1) * patch
    m = np.zeros((rh, rw), bool)
    rr, cc = sk_polygon(hp[:, 1] - ry0, hp[:, 0] - rx0, shape=m.shape)
    m[rr, cc] = True
    if margin_px > 0 and m.any():
        m = distance_transform_edt(~m) <= margin_px
    hit = m.reshape(py1 - py0 + 1, patch, px1 - px0 + 1, patch).any(axis=(1, 3))
    jj, ii = np.nonzero(hit)
    return np.sort((py0 + jj) * n_side + px0 + ii).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--slide-id", required=True)
    ap.add_argument("--polygons-npz", default=None)
    ap.add_argument("--geojson", default=None)
    ap.add_argument("--seg", default="cellvit")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--cache-dir", required=True, help=".../<fm>/<slide>/fov56")
    ap.add_argument("--targets", required=True, help="parquet (slide,row_idx) = the typing cells")
    ap.add_argument("--rings-um", default="4,8,14")
    ap.add_argument("--fov-um", type=float, default=56.0)
    ap.add_argument("--stride-frac", type=float, default=0.5)
    ap.add_argument("--fm", default="hoptimus0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gate-n", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N targets, no cells.npz written")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu = fp32 plumbing smoke only (2026-11-05 [confirmed]); production = cuda fp16")
    a = ap.parse_args()
    t_start = time.time()
    rings = [float(x) for x in a.rings_um.split(",")]

    import openslide, torch, pandas as pd
    sl = openslide.OpenSlide(a.slide)
    W, H = sl.dimensions
    meta = json.load(open(os.path.join(a.cache_dir, "meta.json")))
    mpp = float(meta["mpp"])
    z = np.load(os.path.join(a.cache_dir, "cells.npz"))
    zfiles = {k: z[k] for k in z.files}
    m_cache = zfiles["m_cell"]
    cen_cache = zfiles["centroid_xy"].astype(np.float64)
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
    assert len(cells) == N, f"G0 FAIL: {len(cells)} polygons vs {N} cache rows"
    assert np.allclose(cen, cen_cache, atol=1e-2), "G0 FAIL: centroid order differs from the cache"

    spec = FM_SPECS[a.fm]
    PATCH, NPRE = spec.patch, spec.n_prefix
    assert int(meta["tile_px"]) == max(PATCH, int(round(a.fov_um / mpp / PATCH)) * PATCH)
    tx, ty, tile_px, n_side, stride = tile_assignment(W, H, cen, mpp, PATCH, a.fov_um, a.stride_frac)
    NP = n_side * n_side

    df = pd.read_parquet(a.targets)
    tg = np.sort(df[df.slide == a.slide_id].row_idx.values.astype(np.int64))
    n_full = len(tg)
    if a.limit:
        tg = tg[:a.limit]
    T = len(tg)
    assert T > 0 and tg.min() >= 0 and tg.max() < N
    print(f"[{a.slide_id}] N={N:,} D={D} T={T:,} rings={rings} tile_px={tile_px} G0 PASS", flush=True)

    dev = a.device
    model, spec = load_fm(a.fm, device=dev)
    mean, std = spec.mean, spec.std

    def forward(imgs, keeps):
        x = np.stack([((im.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1) for im in imgs])
        x = torch.from_numpy(x).to(dev)
        keep = torch.ones((len(imgs), NPRE + NP), dtype=torch.bool, device=dev)
        for b, kp in enumerate(keeps):
            if kp is not None:
                keep[b, NPRE:] = False
                keep[b, NPRE + torch.as_tensor(kp, device=dev)] = True
        mdt = torch.float16 if dev == "cuda" else torch.float32
        mask = torch.zeros((len(imgs), 1, 1, NPRE + NP), dtype=mdt, device=dev)
        mask.masked_fill_(~keep[:, None, None, :], float("-inf"))
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            f = model.forward_features(x, attn_mask=mask)
        return f.float().cpu().numpy()[:, NPRE:, :]

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

    def pool(Pt, w, t):
        if w is None:
            return m_cache[t].astype(np.float32)
        return (Pt * w[:, None]).sum(0) / w.sum()

    # per target: pooling weights (margin 0) + keep sets per ring
    order = np.lexsort((ty[tg], tx[tg]))          # tile-major for read-cache hits
    tg = tg[order]
    w0, keeps, nkeep = [], {r: [] for r in rings}, {r: [] for r in rings}
    for t in tg:
        poly = cells[t][2] - np.array([tx[t], ty[t]])
        w = hull_weights_tile(poly, 0, tile_px, n_side, patch=PATCH)
        w0.append(w if w.sum() > 0 else None)          # degenerate polygon: keep cached m (counted)
        for r in rings:
            kp = keep_set_tile(poly, max(0, int(round(r / mpp))), tile_px, n_side, PATCH)
            assert w.sum() == 0 or set(np.nonzero(w > 0)[0]) <= set(kp.tolist()), "hull patch dropped"
            keeps[r].append(kp); nkeep[r].append(len(kp))

    # G1: empty drop reproduces the cache
    gi = list(range(min(a.gate_n, T)))
    cos = []
    for b0 in range(0, len(gi), a.batch):
        bb = gi[b0:b0 + a.batch]
        P = forward([read(int(tx[tg[i]]), int(ty[tg[i]])) for i in bb], [None] * len(bb))
        for q, i in enumerate(bb):
            v = pool(P[q], w0[i], tg[i]); ref = m_cache[tg[i]].astype(np.float32)
            cos.append(float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-12)))
    cos = np.array(cos)
    g1 = {"n": int(len(cos)), "median_cos": float(np.median(cos)), "min_cos": float(cos.min())}
    n_degen = int(sum(w is None for w in w0))
    print(f"[{a.slide_id}] G1 {g1} degenerate_polygons={n_degen}", flush=True)
    if dev == "cuda":
        assert g1["median_cos"] >= 0.9999, "G1 FAIL"
    elif g1["median_cos"] < 0.9999:
        print(f"[{a.slide_id}] G1 WARN (cpu fp32 vs cached gpu fp16): {g1}", flush=True)

    res, stats = {}, {}
    t0 = time.time()
    for r in rings:
        out = np.zeros((T, D), np.float16)
        for b0 in range(0, T, a.batch):
            bb = range(b0, min(T, b0 + a.batch))
            P = forward([read(int(tx[tg[i]]), int(ty[tg[i]])) for i in bb], [keeps[r][i] for i in bb])
            for q, i in enumerate(bb):
                out[i] = pool(P[q], w0[i], tg[i])
        ref = m_cache[tg].astype(np.float32)
        rel = np.linalg.norm(out.astype(np.float32) - ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-12)
        cs = (out.astype(np.float32) * ref).sum(1) / (np.linalg.norm(out.astype(np.float32), axis=1) * np.linalg.norm(ref, axis=1) + 1e-12)
        stats[f"r{r:g}"] = {"kept_patches_median": float(np.median(nkeep[r])), "kept_frac_median": float(np.median(nkeep[r]) / NP),
                            "rel_l2_median": float(np.median(rel)), "rel_l2_p90": float(np.percentile(rel, 90)),
                            "cos_median": float(np.median(cs))}
        assert stats[f"r{r:g}"]["rel_l2_median"] > 1e-3, f"G3 FAIL r={r}"
        res[r] = out
        print(f"[{a.slide_id}] r={r:g} {stats[f'r{r:g}']} ({time.time()-t0:.0f}s)", flush=True)
    fw = T * len(rings) / max(time.time() - t0, 1e-9)

    for r in rings:
        od = os.path.join(a.out, f"r{r:g}", a.slide_id, "fov56")
        os.makedirs(od, exist_ok=True)
        if not a.limit:
            assert T == n_full, "G2 FAIL"
            zz = dict(zfiles); m2 = m_cache.copy(); m2[tg] = res[r]; zz["m_cell"] = m2
            np.savez(os.path.join(od, "cells.npz"), **zz)
            shutil.copy(os.path.join(a.cache_dir, "meta.json"), od)
        else:
            np.savez(os.path.join(od, "smoke_m_iso.npz"), row_idx=tg, m_iso=res[r])
        json.dump({"task": "P16 T2 isolated-cell embedding m_iso (camera-ready)", "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "script": os.path.abspath(__file__), "slide_id": a.slide_id, "ring_um": r, "args": vars(a),
                   "inputs": {"slide": _stat(a.slide), "cache": _stat(os.path.join(a.cache_dir, "cells.npz")),
                              "polygons_npz": _stat(a.polygons_npz) if a.polygons_npz else None, "targets": _stat(a.targets)},
                   "geometry": {"mpp": mpp, "tile_px": tile_px, "n_side": n_side, "stride_px": stride},
                   "gates": {"G0_order": "PASS", "G1_nodrop": g1, "G2_rows": f"{T}/{n_full}", "G3": stats[f"r{r:g}"]["rel_l2_median"], "degenerate_kept_cached": n_degen},
                   "stats": stats[f"r{r:g}"], "forwards_per_s": fw, "wall_s": time.time() - t_start,
                   "semantics": "cells.npz == source cache except m_cell at the typing rows := m_iso(r)"},
                  open(os.path.join(od, "PROVENANCE.json"), "w"), indent=2)
    print(f"[{a.slide_id}] done {time.time()-t_start:.0f}s ({fw:.1f} fwd/s) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
