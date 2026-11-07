"""Multi-FM loader registry for the P16 Stage-1 frozen-ensemble cell probe.

Each FM is a frozen tile encoder consumed by `pooler_grid_fm.pool_cells_grid_fm`, which pools its
patch tokens under each cell's convex hull. The load-bearing per-FM facts are:
  - `n_prefix` = how many leading tokens (CLS + register) to SKIP before the patch tokens
    (wrong value silently misaligns the pool -> garbage). Virchow2/H-optimus=5, UNI2-h=9.
  - `mean/std` = input normalization. H-optimus-0 uses NON-ImageNet stats.
  - `patch`, `embed` = ViT patch size and feature dim.

Configs verified against (a) on-disk config.json and (b) the Trident loader
(polymath/repos/mahmoodlab__trident/trident/patch_encoder_models/load.py), 2026-07-24.

gigapath was ABSENT until 2026-09-20; `[user]` lifted the exclusion for a second-assay encoder ablation.
⚠ The old reason ("patch-16 breaks the patch-14 hull raster") was WRONG: `hull_weights_tile` already
takes `patch` as an argument and rasterizes in `patch`-sized blocks, so it is block-size agnostic.
What patch-16 really changes is TILE QUANTIZATION, `tile_px = round(fov_um/mpp/patch)*patch`:
    mpp 0.3125  fov56 : patch14 -> 182 px 13x13 (56.875 um) | patch16 -> 176 px 11x11 (55.000 um)
    mpp 0.3125  fov280: patch14 -> 896 px 64x64 (280.00 um) | patch16 -> 896 px 56x56 (280.00 um)
=> fov280 aperture is IDENTICAL; fov56 shrinks 3.3 % and must be disclosed as a confound. Both
coarsen the hull raster 4.375 -> 5.000 um/token. NEITHER breaks the code.
⚠⚠ The REAL constraint is the timm pin: prov-gigapath hard-asserts timm==0.9.16 (Trident
`load.py:795`); valis_env has 1.0.26. `load_fm("gigapath")` is therefore UNPROVEN under 1.0.26 --
the shape probe decides whether a dedicated timm-0.9.16 env is required.
⚠ n_prefix=1 for gigapath: the checkpoint has `cls_token` and NO `reg_token` (measured 2026-09-20,
566 tensors / 1,134,953,984 params / pos_embed (1,197,1536)). H-optimus-0 has 4 registers => 5.
"""
from dataclasses import dataclass
import numpy as np

FM_ROOT = "wsi_foundation_models"   # set to your local model directory
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
HOPT_MEAN = np.array([0.707223, 0.578729, 0.703617], np.float32)
HOPT_STD = np.array([0.211883, 0.230117, 0.177517], np.float32)


@dataclass
class FMSpec:
    name: str
    embed: int
    patch: int
    n_prefix: int          # CLS(1) + register tokens; patch tokens = forward_features()[:, n_prefix:, :]
    mean: np.ndarray
    std: np.ndarray
    weights: str


FM_SPECS = {
    "virchow2":  FMSpec("virchow2", 1280, 14, 5, IMAGENET_MEAN, IMAGENET_STD,
                        f"{FM_ROOT}/Virchow2/model.safetensors"),
    "uni2h":     FMSpec("uni2h", 1536, 14, 9, IMAGENET_MEAN, IMAGENET_STD,
                        f"{FM_ROOT}/Uni2_h/pytorch_model.bin"),
    "hoptimus0": FMSpec("hoptimus0", 1536, 14, 5, HOPT_MEAN, HOPT_STD,
                        f"{FM_ROOT}/H-optimus-0/pytorch_model.bin"),
    # patch=16 and n_prefix=1 are MEASURED from the checkpoint, not inherited from a sibling FM.
    # mean/std = ImageNet per Trident load.py:829 (get_constants('imagenet')) -- the same authority
    # the three specs above were verified against. Added 2026-09-20.
    "gigapath":  FMSpec("gigapath", 1536, 16, 1, IMAGENET_MEAN, IMAGENET_STD,
                        f"{FM_ROOT}/gigapath/pytorch_model.bin"),
}


def _load_state_dict(path):
    import torch
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_fm(name, device="cuda"):
    """Return (model.eval().to(device), FMSpec). Offline (no HF download)."""
    import torch, timm
    spec = FM_SPECS[name]
    if name == "virchow2":
        # identical to pooler.load_virchow2 (P1 canonical)
        m = timm.create_model("vit_huge_patch14_224", pretrained=False, img_size=224,
                              init_values=1e-5, num_classes=0, reg_tokens=4, mlp_ratio=5.3375,
                              global_pool="", dynamic_img_size=True,
                              mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
    elif name == "uni2h":
        m = timm.create_model("vit_giant_patch14_224", pretrained=False, img_size=224, patch_size=14,
                              depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
                              mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
                              mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                              reg_tokens=8, dynamic_img_size=True)
    elif name == "hoptimus0":
        m = timm.create_model("vit_giant_patch14_reg4_dinov2", pretrained=False, img_size=224,
                              init_values=1e-5, num_classes=0, dynamic_img_size=True)
    elif name == "gigapath":
        # timm_kwargs transcribed from Trident load.py:803-813 (the authority this registry already
        # cites). ⚠ ONE DELIBERATE DEVIATION: dynamic_img_size=True, which Trident does NOT set
        # because it feeds a fixed 224 CenterCrop. We feed tile_px != 224 (176 at fov56, 896 at
        # fov280), so pos_embed MUST interpolate from its trained 14x14 grid. Every other FM in this
        # registry sets it for the same reason, so this is our established pattern, not a new risk.
        m = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, img_size=224,
                              in_chans=3, patch_size=16, embed_dim=1536, depth=40, num_heads=24,
                              mlp_ratio=5.33334, num_classes=0, dynamic_img_size=True)
    else:
        raise ValueError(f"unknown FM: {name} (have {list(FM_SPECS)})")
    m.load_state_dict(_load_state_dict(spec.weights), strict=True)
    return m.eval().to(device), spec
