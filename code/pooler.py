"""Frozen-Virchow2 per-cell pooler (40x native, hull+margin region) + regional-context grid.

Adapted from P2 cell_seg_classification `forward_lib.py` (Virchow2 load + fraction-weighted
patch pool). Re-targeted here to:
  - 40x NATIVE extraction: 224 L0 px tile centered on the cell, NO resize (mpp~0.25 => 3.5um patch);
  - pooling region = convex_hull(nucleus) dilated by margin_um (D1), fraction-weighted;
  - CLS fallback when no patch overlaps the region.
Regional context c: a coarse anchor grid; per anchor a crop downsampled to Virchow2 input (default
5x=2.0 um/px => 448 um FOV, the COARSEST in-distribution scale) -> CLS. NOTE: a 1024 um crop
downsampled to 224 = 4.57 um/px is COARSER than Virchow2's 2.0 um/px floor (OOD) -> see caveat in
cache_slide / card; default here is 448 um @ 5x (in-distribution).
"""
import os
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from skimage.draw import polygon as sk_polygon
from skimage.morphology import binary_dilation, disk

RESIZE = 224
PATCH = 14
NP = RESIZE // PATCH          # 16x16 = 256 patch tokens
EMBED = 1280
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Maintained Virchow2 weights (P1 canonical; used by GCICI_outcome_prediction scripts). Offline, no HF download.
# Env-overridable (added 2026-08-05) so this module runs off-OSC, e.g. on the desktop where the
# byte-identical weights live on the NAS at
#   /mnt/hwangt_share/lab_members/parks/references/models/Virchow2/model.safetensors
# (verified same size 2,525,001,112 B = same paige-ai/Virchow2). Default is UNCHANGED.
VIRCHOW2_WEIGHTS = os.environ.get(
    "P16_VIRCHOW2_WEIGHTS",
    "wsi_foundation_models/Virchow2/model.safetensors")


def load_virchow2(device="cuda", weights=VIRCHOW2_WEIGHTS):
    """Offline build matching P1 `script_classic_tissue_inference.py`: vit_huge_patch14_224 +
    maintained safetensors (NO hf-hub download)."""
    import torch, timm
    from safetensors.torch import load_file
    cfg = dict(img_size=224, init_values=1e-5, num_classes=0, reg_tokens=4,
               mlp_ratio=5.3375, global_pool="", dynamic_img_size=True)
    m = timm.create_model("vit_huge_patch14_224", pretrained=False,
                          mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, **cfg)
    m.load_state_dict(load_file(weights), strict=True)
    return m.eval().to(device)


def hull_margin_weights(ring_local, margin_px):
    """(256,) per-patch fraction inside convex_hull(nucleus) dilated by margin, on the 224 native tile."""
    hull = ConvexHull(ring_local)
    hp = ring_local[hull.vertices]
    m = np.zeros((RESIZE, RESIZE), bool)
    rr, cc = sk_polygon(hp[:, 1], hp[:, 0], shape=m.shape)   # (row=y, col=x)
    m[rr, cc] = True
    if margin_px > 0:
        m = binary_dilation(m, disk(margin_px))
    return m.reshape(NP, PATCH, NP, PATCH).mean(axis=(1, 3)).ravel().astype(np.float32)


def pool_cells(slide, cells, mpp, margin_um, model=None, batch=64, device="cuda", mock=False):
    """Return m_cell (N,1280) fp32. cells = list of (cid, centroid_xy_L0px, ring_L0px, cls)."""
    import math
    margin_px = max(0, int(round(margin_um / mpp)))
    W, H = slide.dimensions
    N = len(cells)
    out = np.zeros((N, EMBED), np.float32)
    imgs, ws, idxs = [], [], []

    def flush():
        if not idxs:
            return
        Wm = np.stack(ws)                                   # (B,256)
        if mock:
            emb = np.random.randn(len(idxs), EMBED).astype(np.float32)
        else:
            import torch
            x = torch.from_numpy(np.stack(imgs)).to(device)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                feats = model.forward_features(x)           # (B,261,1280)
            patch = feats[:, 5:, :].float()                 # (B,256,1280)
            w = torch.from_numpy(Wm).to(device).float().unsqueeze(-1)
            s = w.sum(1)                                     # (B,1)
            pooled = (patch * w).sum(1) / s.clamp(min=1e-6)
            pooled = torch.where(s > 0, pooled, feats[:, 0, :].float())   # CLS fallback
            emb = pooled.cpu().numpy()
        for a, gi in enumerate(idxs):
            out[gi] = emb[a]
        imgs.clear(); ws.clear(); idxs.clear()

    for gi, (cid, cen, ring, cls) in enumerate(cells):
        cx, cy = float(cen[0]), float(cen[1])
        x0 = int(np.clip(cx - RESIZE / 2, 0, max(0, W - RESIZE)))
        y0 = int(np.clip(cy - RESIZE / 2, 0, max(0, H - RESIZE)))
        try:
            w = hull_margin_weights(ring - np.array([x0, y0]), margin_px)
        except Exception:
            w = np.zeros(NP * NP, np.float32)
        ws.append(w)
        if not mock:
            tile = slide.read_region((x0, y0), 0, (RESIZE, RESIZE)).convert("RGB")
            arr = (np.asarray(tile, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            imgs.append(arr.transpose(2, 0, 1))
        idxs.append(gi)
        if len(idxs) >= batch:
            flush()
    flush()
    return out


def context_grid(slide, cells, mpp, model=None, spacing_um=256.0, crop_um=448.0,
                 device="cuda", mock=False, near_um=200.0):
    """Coarse regional-context grid. Anchors on a spacing_um grid, kept only near tissue (<near_um of a
    cell). Per anchor: crop_um FOV centered, downsampled to 224 -> CLS = c. Returns anchors(G,2 L0px), c(G,1280)."""
    W, H = slide.dimensions
    cen = np.array([[c[1][0], c[1][1]] for c in cells], np.float64)
    tree = cKDTree(cen)
    sp = spacing_um / mpp
    gx = np.arange(sp / 2, W, sp); gy = np.arange(sp / 2, H, sp)
    anchors = np.array([(x, y) for y in gy for x in gx], np.float32)
    keep = tree.query_ball_point(anchors, r=near_um / mpp, return_length=True) > 0
    anchors = anchors[keep]
    crop_px = int(round(crop_um / mpp))
    c = np.zeros((len(anchors), EMBED), np.float32)
    if mock:
        c[:] = np.random.randn(*c.shape).astype(np.float32)
        return anchors, c
    import torch
    from PIL import Image
    for b in range(0, len(anchors), 32):
        batch = anchors[b:b + 32]; imgs = []
        for (ax, ay) in batch:
            x0 = int(np.clip(ax - crop_px / 2, 0, max(0, W - crop_px)))
            y0 = int(np.clip(ay - crop_px / 2, 0, max(0, H - crop_px)))
            crop = slide.read_region((x0, y0), 0, (crop_px, crop_px)).convert("RGB").resize((RESIZE, RESIZE), Image.BILINEAR)
            arr = (np.asarray(crop, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            imgs.append(arr.transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(imgs)).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            feats = model.forward_features(x)
        c[b:b + len(batch)] = feats[:, 0, :].float().cpu().numpy()
    return anchors, c
