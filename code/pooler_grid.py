"""Grid-tiling per-cell pooler (CellViT++-style: run Virchow2 once per tile, index tokens per cell).

Replaces the per-cell read+forward in `pooler.py` (one WSI read + one Virchow2 forward PER CELL,
~28 ms/cell) with: tile the slide on a grid, forward each tile ONCE, and pool the patch tokens under
each cell's convex-hull+margin from the tile the cell is most central in. Cost scales with tissue
AREA (n tiles), not cell count -> ~10x fewer forwards (a 224 px tile holds ~13 cells).

Design (matches the verified `pooler.py` feature definition, only the *scheduling* changes):
  - 40x NATIVE extraction, tile_px = fov_um / mpp (snapped to a multiple of PATCH); NO resize.
  - FOV is a free knob (the tile size == the local receptive field of m_cell). Sweep via --fov_um.
  - Each cell is assigned to the nearest tile-grid node (most-central) so its ~symmetric context
    matches the old centered extraction; overlap controlled by --stride_frac (<1 = overlap).
  - Pooling region = convex_hull(nucleus) dilated by margin_um, fraction-weighted over patch tokens
    (identical to pooler.hull_margin_weights, generalized to arbitrary tile_px via a bbox-local raster).
  - CLS fallback when no patch overlaps.

Validate against `pooler.pool_cells` with scripts/precompute/validate_grid_vs_percell.py before use.
"""
import numpy as np
from collections import defaultdict
from scipy.spatial import ConvexHull
from skimage.draw import polygon as sk_polygon
from skimage.morphology import binary_dilation, disk

from pooler import PATCH, EMBED, IMAGENET_MEAN, IMAGENET_STD, load_virchow2  # shared with verified pooler


def hull_weights_tile(ring_local, margin_px, tile_px, n_side, patch=PATCH):
    """(n_side*n_side,) per-patch fraction inside convex_hull(nucleus)+margin, row-major (idx = yp*n_side+xp).

    Rasterizes only the hull+margin bounding box (constant cost, independent of tile_px) then scatters
    the per-patch fractions into the full token grid. Token order matches timm forward_features:
    patch idx = row(y)*n_side + col(x)."""
    w = np.zeros(n_side * n_side, np.float32)
    if len(ring_local) < 3:
        return w
    try:
        hull = ConvexHull(ring_local)
    except Exception:
        return w
    hp = ring_local[hull.vertices]
    bx0 = int(np.floor(hp[:, 0].min() - margin_px)); bx1 = int(np.ceil(hp[:, 0].max() + margin_px))
    by0 = int(np.floor(hp[:, 1].min() - margin_px)); by1 = int(np.ceil(hp[:, 1].max() + margin_px))
    px0 = max(0, bx0 // patch); py0 = max(0, by0 // patch)
    px1 = min(n_side - 1, bx1 // patch); py1 = min(n_side - 1, by1 // patch)
    if px1 < px0 or py1 < py0:
        return w
    rx0, ry0 = px0 * patch, py0 * patch
    rw, rh = (px1 - px0 + 1) * patch, (py1 - py0 + 1) * patch
    m = np.zeros((rh, rw), bool)
    rr, cc = sk_polygon(hp[:, 1] - ry0, hp[:, 0] - rx0, shape=m.shape)   # row=y, col=x
    m[rr, cc] = True
    if margin_px > 0:
        m = binary_dilation(m, disk(margin_px))
    frac = m.reshape(py1 - py0 + 1, patch, px1 - px0 + 1, patch).mean(axis=(1, 3))  # (npy, npx)
    for jj in range(frac.shape[0]):
        row = (py0 + jj) * n_side
        for ii in range(frac.shape[1]):
            w[row + px0 + ii] = frac[jj, ii]
    return w


def pool_cells_grid(slide, cells, mpp, margin_um, fov_um=56.0, stride_frac=0.75,
                    model=None, batch=16, device="cuda", mock=False):
    """Grid-tiling pool. cells = list of (cid, centroid_xy_L0px, ring_L0px, cls). Returns m_cell (N,1280) fp32.

    Also returns per-cell offset-from-tile-center (px) so the validation gate can bin the edge effect."""
    import math
    W, H = slide.dimensions
    margin_px = max(0, int(round(margin_um / mpp)))
    tile_px = max(PATCH, int(round(fov_um / mpp / PATCH)) * PATCH)
    n_side = tile_px // PATCH
    stride = max(PATCH, int(round(tile_px * stride_frac)))
    N = len(cells)
    out = np.zeros((N, EMBED), np.float32)
    off = np.zeros(N, np.float32)   # cell offset from its tile center (px), for edge-effect QC

    cen = np.array([[c[1][0], c[1][1]] for c in cells], np.float64)
    half = tile_px / 2.0
    ci = np.clip(np.round((cen[:, 0] - half) / stride).astype(int), 0, None)
    cj = np.clip(np.round((cen[:, 1] - half) / stride).astype(int), 0, None)
    tx = np.clip((half + ci * stride - half).astype(int), 0, max(0, W - tile_px))
    ty = np.clip((half + cj * stride - half).astype(int), 0, max(0, H - tile_px))

    groups = defaultdict(list)
    for gi in range(N):
        groups[(int(tx[gi]), int(ty[gi]))].append(gi)
        off[gi] = math.hypot(cen[gi, 0] - (tx[gi] + half), cen[gi, 1] - (ty[gi] + half))
    tiles = list(groups.keys())

    for b in range(0, len(tiles), batch):
        btiles = tiles[b:b + batch]
        if mock:
            for key in btiles:
                for gi in groups[key]:
                    out[gi] = np.random.randn(EMBED).astype(np.float32)
            continue
        import torch
        imgs = []
        for (x0, y0) in btiles:
            tile = slide.read_region((x0, y0), 0, (tile_px, tile_px)).convert("RGB")
            arr = (np.asarray(tile, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            imgs.append(arr.transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(imgs)).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            feats = model.forward_features(x)                 # (B, 1+4+n_side^2, 1280)
        P = feats[:, 5:, :].float().cpu().numpy()             # (B, n_side^2, 1280)
        CLS = feats[:, 0, :].float().cpu().numpy()            # (B, 1280)
        for t, (x0, y0) in enumerate(btiles):
            for gi in groups[(x0, y0)]:
                ring = cells[gi][2] - np.array([x0, y0])
                w = hull_weights_tile(ring, margin_px, tile_px, n_side)
                s = w.sum()
                out[gi] = (P[t] * w[:, None]).sum(0) / s if s > 0 else CLS[t]
    return out, off, {"tile_px": tile_px, "n_side": n_side, "stride": stride, "n_tiles": len(tiles)}
