#!/usr/bin/env python3
"""
apply_predictor_hd.py — run a trained MSSC-JEPA predictor over every cell of the 3 HD caches,
producing per-cell μ and σ² (the cell-JEPA predictability signal) for the Phase-5 typing probe.

Mirrors scripts/train/train_mssc.py neighborhood construction so the predictor sees the training regime:
each cell is a TARGET; its context = the Kc nearest OTHER cells (target-excluded), fine tokens = context
m_cell@56, coarse tokens = SAME context cells' m_cell@112 and @168 at their positions. Kc = K - round(K*tgt_frac)
(defaults 48,0.25 -> 36, matching the trained model). Output per slide:
  results/eval/predictor_out/<slide>.npz : mu[N,D] fp16, s2[N,D] fp16  (cache-cell order)
D is AUTO-DETECTED from the cache (1280 Virchow2 / 1536 H-optimus-0) as of 2026-08-09; it was
hardcoded 1280 before that, which made this script unrunnable on H-optimus-0.

GPU cluster. Deterministic given the checkpoint (no sampling; forward only).

SINGLE-SCALE + SPHERE (added 2026-08-01, all default OFF => byte-identical to the 3-scale behaviour)
----------------------------------------------------------------------------------------------
  * `--coarse-fovs ""` -> coarse_fovs=[] -> MSSCPredictor(coarse_scales=()) -> net(fe, fx, {}, tx),
    which is EXACTLY the call train_lora_celljepa.py makes (its line 513). No new code path was
    needed: the fine/coarse FOVs were already CLI-settable.
  * `--sphere-normalize` / `--sphere-radius`: train_lora_celljepa.py projects BOTH branches onto the
    sphere of radius R right after pooling (its lines 499-506), so a predictor trained that way has
    only ever seen inputs with ||m|| == R exactly. The caches store UNPROJECTED m_cell (norms ~35.6
    -38.2 and varying), so applying without this flag feeds the predictor out-of-distribution input
    -- mu drifts and sigma^2 especially, since the radius was chosen precisely to keep the
    heteroscedastic head off its logvar clamp. MUST match the training run's setting.
  * `--predictor-key`: train_lora_celljepa.py saves {"predictor": sd, "lora": [...]}, not a bare
    state_dict. Use `--predictor-key predictor` for those checkpoints; omit for train_mssc.py ones.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
HD_SLIDES = ["Slide 6", "Slide 2", "Slide 12"]   # anonymized; see the paper's appendix
# Env override so a subset can be run without editing the file (e.g. a smoke on the
# slides already staged locally). Unset => unchanged behaviour.
HD_SLIDES = [x for x in os.environ.get("P16_HD_SLIDES", "").split(",") if x] or HD_SLIDES
MPP = 0.2522        # DEFAULT ONLY -- see resolve_mpp(). Kept so every pre-2026-08-23 caller that
                    # passes no --mpp is bit-identical to what it produced before that date.

# Relative disagreement between the mpp in use and the cache's own meta.json.
MPP_WARN_REL = 0.001        # 0.1 % -> print a warning, keep going
MPP_FAIL_REL = 0.01         # 1 %   -> refuse to run


def resolve_mpp(cache_root, slide, fine_fov, mpp=None):
    """Microns-per-pixel for this cache, and a gate against silently using the wrong one.

    ⚠⚠ WHY THIS EXISTS (2026-08-23). This module hardcoded MPP = 0.2522 while `train_mssc.py:55-61`
    reads the per-slide value out of the cache's `meta.json`. On this cohort that is a 0.12 % disagreement
    (8 of 12 eval slides are 0.2525) -- negligible, and it is why nothing caught it for months.
    On the G4X GN-133 cache the true value is **0.3125**, so apply would have fed the predictor
    coordinates scaled by 0.2522/0.3125 = 0.8070 -- a 19.3 % shrink, with the positional encoding
    seeing every neighbour as ~19 % nearer than it did in training. `--pe-frame target` re-centres
    the coordinates but does NOT rescale them, so it does not save it. There is no error, no crash
    and no gate downstream: the run produces a complete, plausible, wrong result. The G4X applies
    were only spared because an unrelated `--own-knn` argparse error killed them first.

    CONTRACT
      * `mpp` given        -> use it, and gate it against meta.json.
      * `mpp` not given    -> use MPP (0.2522), and gate THAT against meta.json.
        Default rather than auto-adopt on purpose: auto-adopting meta.json would silently move the
        8 slides at 0.2525 and perturb the standing `m` = 0.6159 harness-drift gate six days
        before a submission. Correctness of NEW work is bought with an explicit flag; the numbers
        already on disk are left alone, and the mismatch is made VISIBLE rather than fixed silently.
      * meta.json absent   -> no gate possible; proceed on the value in hand (older caches predate it).

    Raises on a >1 % disagreement, which is exactly the G4X case.
    """
    used = float(MPP if mpp is None else mpp)
    meta_p = os.path.join(cache_root, slide, f"fov{fine_fov}", "meta.json")
    if not os.path.exists(meta_p):
        return used
    try:
        declared = json.load(open(meta_p)).get("mpp")
    except Exception:
        return used
    if declared is None:
        return used
    declared = float(declared)
    rel = abs(used - declared) / declared
    if rel > MPP_FAIL_REL:
        raise SystemExit(
            f"FATAL mpp mismatch on {slide}: using {used} but {meta_p} declares {declared} "
            f"({100*rel:.1f}% off, limit {100*MPP_FAIL_REL:.0f}%).\n"
            f"  The trainer reads meta.json, so this apply would NOT match its own training run.\n"
            f"  Pass --mpp {declared} to run this cache correctly.")
    if rel > MPP_WARN_REL:
        print(f"[mpp] WARNING {slide}: using {used}, meta.json declares {declared} "
              f"({100*rel:.2f}% off) -- below the {100*MPP_FAIL_REL:.0f}% hard limit, continuing.",
              flush=True)
    return used


def load_cache(cache_root, slide, fine_fov=56, coarse_fovs=(112, 168), mpp=None):
    """Load the fine/target FOV embedding + each coarse FOV embedding. Centroids are FOV-independent
    (same cells/order across fov{N}) so read them from the fine FOV. Returns xy (um), feat={fov:m_cell}.

    `mpp=None` keeps the historical 0.2522; see resolve_mpp() for why that is a default and not a
    lookup, and for the gate that stops a wrong value from passing silently."""
    d = os.path.join(cache_root, slide)
    mpp_used = resolve_mpp(cache_root, slide, fine_fov, mpp)
    xy = np.load(os.path.join(d, f"fov{fine_fov}", "cells.npz"))["centroid_xy"].astype(np.float32) * mpp_used
    feat = {int(fine_fov): np.load(os.path.join(d, f"fov{fine_fov}", "cells.npz"))["m_cell"]}
    for s in coarse_fovs:
        feat[int(s)] = np.load(os.path.join(d, f"fov{s}", "cells.npz"))["m_cell"]
    return xy, feat


def probe_dim(cache_root, slide, fine_fov):
    """Embedding dim of `m_cell` WITHOUT decompressing the array.

    Added 2026-08-09. The 1280 hardcodes this replaces were fixed in BOTH TRAINERS on 2026-08-08
    (plan_v4.md:34, train_mssc.py:263-265 auto-detects) but that fix never reached this APPLIER,
    so H-optimus-0 (d=1536) could not be applied at all. Same auto-detect contract as the trainer.

    meta.json is NOT usable as the source: `embed` is present in cache_fm_hd/hoptimus0 but ABSENT
    in cache_ms_frozen (verified 2026-08-09), so it would work on one backbone and not the other.
    The .npy header inside the npz is always there; reading it costs ~1 KB instead of the ~1.2 GB
    a full m_cell read would cost on the largest slide.
    """
    import zipfile
    p = os.path.join(cache_root, slide, f"fov{fine_fov}", "cells.npz")
    with zipfile.ZipFile(p) as zf:
        with zf.open("m_cell.npy") as fh:
            np.lib.format.read_magic(fh)
            shape, _, _ = np.lib.format._read_array_header(fh, (1, 0))
    assert len(shape) == 2, f"{p}: m_cell is not 2-D ({shape})"
    return int(shape[1])


def sphere_project(z, radius, eps=1e-8):
    """z -> radius * z/||z||, per ROW. Byte-identical to train_lora_celljepa.sphere_project.

    Applied to the CONTEXT features only: they are the predictor's input, and in training both
    m_ctx and m_tgt were projected before the predictor saw either. mu is therefore already in
    sphere coordinates on output and needs no post-hoc rescale.
    """
    return z * (radius / z.norm(dim=-1, keepdim=True).clamp_min(eps))


def apply_slide(net, torch, dev, cache_root, slide, Kc, batch, fine_fov=56, coarse_fovs=(112, 168),
                sphere_radius=None, max_ctx_dist=0.0, pe_frame="absolute", self_fine_arr=None,
                self_tiles_dir=None, self_tiles_block="fine_cls_masked", mpp=None):
    """
    max_ctx_dist > 0 applies the MAX-RADIUS CEILING at inference, mirroring train_mssc.py:164-171.
    ⚠ It MUST match the value the predictor was TRAINED with — train_mssc.py's own --max-ctx-dist
    help says "Must be applied IDENTICALLY at inference". Training with a cap and applying without
    one reproduces the Phase-2 regime mismatch (trained M=12, inferred M=1).

    max_ctx_dist == 0.0 (default) leaves the pre-2026-08-15 path BIT-IDENTICAL: same k=Kc+2 query,
    same target-exclusion, no mask passed to the net.

    ⚠ ONE DELIBERATE DIVERGENCE FROM TRAINING, and it cannot be avoided. Training SKIPS a centre
    whose qualifying neighbour count is < 4 ("sparse region: unusable, not silently padded to k").
    Inference cannot skip: every cell must receive a mu or the downstream join breaks. So short
    contexts are PADDED AND MASKED (never faked — the padded slots carry mask=False and are
    excluded via src_key_padding_mask), and the number of cells falling below training's own
    4-neighbour floor is COUNTED and written to PROVENANCE as `n_below_train_floor`. Those cells
    are predicted from a context the trainer would never have shown the model; treat a slide with
    a large count as out-of-regime rather than trusting its mu.
    ⚠⚠ RETRACTED CLAIM, kept visible. This docstring originally said "Numerically safe:
    predictor_mssc.forward never pads the target queries, so even an all-masked context leaves the
    target attending to itself — no NaN softmax." **That was WRONG and it cost a 58-minute job
    (13616540, "Input X contains NaN" after the m arm had already scored 0.6153).** The NaN does not
    arise in the target's attention row; it arises in a fully-padded CONTEXT token's own row, whose
    softmax is over an all -inf key set, and it then propagates because a masked key has weight
    exactly 0 and 0 * NaN = NaN in the weighted sum. Zero-context cells are now given their single
    nearest neighbour unmasked and counted as `n_zero_ctx_fellback`. See the k == 0 branch below.
    """
    xy, feat = load_cache(cache_root, slide, fine_fov, coarse_fovs, mpp=mpp)
    m_fine = feat[int(fine_fov)]
    n = len(xy)
    # ── 56 um self-tile token, loaded HERE and not at the call site ─────────────────────────────
    # FIXED 2026-08-19 (backup .bak_selftilesfix_20260819). The call site read
    #   _load_self_tiles(args.self_tiles, slide, args.self_tiles_block, len(xy))
    # where NEITHER `slide` NOR `xy` exists in that scope (the loop variable is `s`, and `xy` is
    # local to this function). It raised NameError on the first slide and killed the job AFTER
    # a successful 200-epoch train.
    # ⚠⚠ WHY THE NO-OP GATE (6881623) DID NOT CATCH IT, and the lesson: Python evaluates
    #   `f(...) if args.self_tiles else None`
    # lazily, so with --self-tiles ABSENT the undefined names are never looked up. The gate proved
    # the OFF path bit-identical (max|diff| 0.000e+00 on mu and s2 — still true, still valuable) and
    # BY CONSTRUCTION could not touch the ON path. A gate that proves the UNCHANGED path says
    # nothing about the CHANGED one. Third instance of this class in P16 (see also the --max-cells
    # 20000 caching smoke that hid the production path).
    # Loading here also makes the row-alignment assert compare against THE SAME `n` this function
    # uses, rather than a separately-loaded count that could in principle disagree.
    if self_tiles_dir is not None and self_fine_arr is None:
        self_fine_arr = _load_self_tiles(self_tiles_dir, slide, self_tiles_block, n)
    tree = cKDTree(xy)
    dim = int(m_fine.shape[1])                              # 1280 Virchow2 / 1536 H-optimus-0
    mu_out = np.empty((n, dim), dtype=np.float16)
    s2_out = np.empty((n, dim), dtype=np.float16)
    capped = max_ctx_dist and max_ctx_dist > 0
    kq = Kc + 2 if not capped else min(n, max(Kc * 4, 64))  # same pool rule as train_mssc.py:164
    n_below_floor = 0
    n_short = 0
    n_zero_ctx = 0
    net.eval()
    for b0 in range(0, n, batch):
        b1 = min(n, b0 + batch)
        tgt = np.arange(b0, b1)
        dd, nn = tree.query(xy[tgt], k=kq)                  # +2 headroom to drop self
        ctx = np.zeros((len(tgt), Kc), dtype=np.int64)
        fmask = np.ones((len(tgt), Kc), dtype=bool)
        for i, (t, row) in enumerate(zip(tgt, nn)):
            r = np.asarray(row)
            sel = r != t                                    # target-excluded
            if capped:
                sel = sel & (np.asarray(dd[i]) <= max_ctx_dist)   # ★ the ceiling
            keep = r[sel][:Kc]
            k = len(keep)
            if k == 0:
                # ⚠⚠ ZERO qualifying neighbours — MUST NOT be left fully masked.
                # A fully-padded context row makes its OWN attention softmax NaN (every key is
                # -inf), and that NaN then PROPAGATES: a masked key has weight exactly 0 and
                # 0 * NaN = NaN in the weighted sum, so it is NOT confined to the padded token.
                # ⚠ My 2026-08-15 patch note claimed this was "numerically safe because target
                # queries are never padded" — that reasoning was WRONG. It is the CONTEXT token's
                # own row that goes NaN, not the target's. Measured: 3-31 such cells per slide
                # produced NaN mu and killed 13616540 with "Input X contains NaN" AFTER the m arm
                # had already scored 0.6153.
                # Fall back to the single nearest non-self neighbour, UNMASKED — the minimum that
                # keeps the softmax defined. Counted separately so every such cell is auditable.
                keep = r[r != t][:1]
                k = len(keep)
                n_zero_ctx += 1
            if k < Kc:
                n_short += 1
                if k < 4:
                    n_below_floor += 1                      # below training's own usability floor
                fmask[i, k:] = False                        # pad slots are MASKED, not faked
            ctx[i, :k] = keep                               # remainder stays index 0, masked out
        cx = xy[ctx]                                         # (B,Kc,2) um
        fe = torch.from_numpy(np.asarray(m_fine[ctx], np.float32)).to(dev)
        fx = torch.from_numpy(cx).float().to(dev)
        tx = torch.from_numpy(xy[tgt][:, None, :]).float().to(dev)    # (B,1,2)
        # ── POSITIONAL-ENCODING FRAME — MUST match what the predictor was TRAINED with ────────────
        # Mirrors train_mssc.py sample_batch(). "absolute" (DEFAULT) leaves positions as um from the
        # SLIDE ORIGIN => bit-identical to every run up to 2026-08-16. "target" re-centres on the
        # target per MODEL_DESIGN_CARD 6.5. Here M==1 by construction (one target per row), so `org`
        # IS the target exactly and tx becomes identically 0.
        # ⚠ A mismatch between this and training silently produces plausible garbage — the predictor
        #   would be fed a coordinate frame it never saw. Recorded in PROVENANCE as `pe_frame`.
        if pe_frame == "target":
            org = tx.mean(dim=1, keepdim=True)                        # (B,1,2)
            fx = fx - org
            tx = tx - org
        elif pe_frame != "absolute":
            raise ValueError(f"pe_frame must be 'absolute' or 'target', got {pe_frame!r}")
        coarse = {int(s): (torch.from_numpy(np.asarray(feat[int(s)][ctx], np.float32)).to(dev), fx)
                  for s in coarse_fovs}                               # AFTER re-centring
        if sphere_radius is not None:
            fe = sphere_project(fe, sphere_radius)
            coarse = {s: (sphere_project(f, sphere_radius), x) for s, (f, x) in coarse.items()}
        # 56 um TARGET-CENTRED self-tile token, row-aligned to cells.npz (asserted at load).
        # ⚠ Must be the SAME block the predictor was TRAINED on; a mismatch is silent.
        sfn = (torch.from_numpy(np.asarray(self_fine_arr[tgt], np.float32)).to(dev)
               if self_fine_arr is not None else None)
        with torch.no_grad():
            if capped:
                fm = torch.from_numpy(fmask).to(dev)
                cm = {int(s): fm for s in coarse_fovs}      # coarse tokens ride the same context cells
                mu, lv = net(fe, fx, coarse, tx, fine_mask=fm, coarse_mask=cm, self_fine=sfn)
            elif sfn is not None:
                mu, lv = net(fe, fx, coarse, tx, self_fine=sfn)
            else:
                mu, lv = net(fe, fx, coarse, tx)            # ⚠ untouched legacy path — bit-identical
        mu_out[b0:b1] = mu[:, 0].float().cpu().numpy().astype(np.float16)
        s2_out[b0:b1] = torch.exp(lv[:, 0]).float().cpu().numpy().astype(np.float16)
        if (b0 // batch) % 20 == 0:
            print(f"  [{slide}] {b1}/{n}", flush=True)
    if capped:
        print(f"  [{slide}] cap={max_ctx_dist}um -> {n_short}/{n} cells short of Kc={Kc} "
              f"({100.0*n_short/max(n,1):.1f}%), {n_below_floor} below the training floor of 4 "
              f"({100.0*n_below_floor/max(n,1):.2f}%), {n_zero_ctx} with ZERO qualifying neighbours "
              f"-> fell back to the single nearest ({100.0*n_zero_ctx/max(n,1):.3f}%)", flush=True)
    return mu_out, s2_out, n, int(n_short), int(n_below_floor), int(n_zero_ctx)


def _load_self_tiles(root, slide, block, n_expect):
    """Load one slide's 56 um self-tile block and ASSERT it is row-aligned to cells.npz.

    Row misalignment between a token block and its cell table is the failure class this project
    keeps paying for, so it is checked rather than trusted: the cacher writes one row per cell of
    the SAME cells.npz this script reads, and stores `cell_ids` as the identity index into it.
    """
    p = os.path.join(root, slide, "self_tiles.npz")
    z = np.load(p)
    assert block in z.files, f"{p} has {z.files}, missing block {block!r}"
    a = z[block]
    assert len(a) == n_expect, \
        f"{p}: {len(a)} self-tile rows != {n_expect} cells in cells.npz -- NOT row-aligned"
    if "cell_ids" in z.files:
        ci = z["cell_ids"]
        assert len(ci) == n_expect and int(ci[0]) == 0 and int(ci[-1]) == n_expect - 1, \
            f"{p}: cell_ids is not the identity index into cells.npz"
    out = np.asarray(a)
    z.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="results/mssc_5slide/predictor_mssc.pt")
    ap.add_argument("--cache-root", default="results/cache_pilot")
    ap.add_argument("--out", default="results/eval/predictor_out")
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--tgt-frac", type=float, default=0.25)
    ap.add_argument("--fine-fov", type=int, default=56,
                    help="cached FOV (um) used as the fine/target per-cell scale; must match training. Default 56.")
    ap.add_argument("--coarse-fovs", default="112,168",
                    help="comma list of cached coarse FOVs (um); must match training. Default '112,168'.")
    ap.add_argument("--self-tiles", default=None,
                    help="dir of 56 um TARGET-CENTRED self-tile tokens for THESE slides "
                         "(results/cache_self_tiles_eval12). MUST match what the predictor was "
                         "trained with -- the flag changes n_slot/slot_tgt and is NOT in the "
                         "state_dict, so a mismatch fails the load loudly (good) or, if shapes "
                         "happened to line up, would silently mis-slot (bad). Recorded in PROVENANCE.")
    ap.add_argument("--self-tiles-block", default="fine_cls_masked",
                    choices=["fine_cls", "fine_cls_masked", "fine_cls_masked_dil"],
                    help="K6 [user 2026-08-18] = fine_cls_masked. MUST match training.")
    ap.add_argument("--dp", type=int, default=384)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-ctx-dist", type=float, default=0.0,
                    help="MAX context radius um (0 = OFF, default, legacy path bit-identical). "
                         "⚠ MUST equal the value the predictor was TRAINED with — see train_mssc.py "
                         "--max-ctx-dist. Short contexts are padded and MASKED, never faked. "
                         "Training SKIPS centres with <4 qualifying neighbours; inference cannot, so "
                         "those cells are predicted anyway and COUNTED as n_below_train_floor.")
    ap.add_argument("--pe-frame", default="absolute", choices=["absolute", "target"],
                    help="frame for the Fourier positional encoding; MUST match training. "
                         "'absolute' (DEFAULT) = um from the SLIDE ORIGIN, the legacy path, "
                         "bit-identical to pre-2026-08-16. 'target' = um offset TO THE TARGET per "
                         "MODEL_DESIGN_CARD 6.5. Read it off the checkpoint's "
                         "train_history.json['args']['pe_frame'] — a mismatch does not error, it "
                         "silently produces garbage.")
    ap.add_argument("--attn-mode", default="dense", choices=["dense","block_tt","crossattn"],
                    help="target-query attention. dense = original (targets see each other). "
                         "block_tt = Option B, blocks target->target so the held-out set's "
                         "POSITIONS cannot leak between targets. crossattn additionally blocks "
                         "context->target. Adds NO parameters: any checkpoint loads under any mode.")
    ap.add_argument("--pe-scale-um", type=float, default=64.0,
                    help="MUST equal the training run's --pe-scale-um. pe_scale_um is NOT stored in "
                         "the state_dict, so a mismatch does not error -- it silently produces garbage, "
                         "exactly like --pe-frame. Recorded in PROVENANCE.")
    ap.add_argument("--mpp", type=float, default=None,
                    help="Microns per pixel for centroid_xy. DEFAULT None => 0.2522, which keeps "
                         "every pre-2026-08-23 caller bit-identical. The value in use is ALWAYS "
                         "gated against the cache's own meta.json: >0.1%% prints a warning, >1%% is "
                         "fatal. Pass it explicitly for any cache that is not ~0.2522 -- the G4X "
                         "GN-133 cache is 0.3125, where the old hardcoded constant was 19.3%% off "
                         "and would have mis-scaled every coordinate the PE sees. Recorded in "
                         "PROVENANCE as `mpp_arg`.")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--predictor-key", default=None,
                    help="key to unwrap from the checkpoint dict. train_lora_celljepa.py saves "
                         "{'predictor': sd, 'lora': [...]} -> use 'predictor'. Omit for a bare "
                         "state_dict (train_mssc.py).")
    ap.add_argument("--sphere-normalize", action="store_true",
                    help="project context features onto the sphere of --sphere-radius before the "
                         "predictor, matching train_lora_celljepa.py --sphere-normalize. REQUIRED "
                         "for a sphere-trained predictor: the caches hold UNPROJECTED m_cell, so "
                         "omitting this feeds out-of-distribution input.")
    ap.add_argument("--sphere-radius", type=float, default=36.3812,
                    help="must equal the training run's --sphere-radius")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Kc = args.K - max(1, int(round(args.K * args.tgt_frac)))
    coarse_fovs = [int(x) for x in str(args.coarse_fovs).split(",") if x.strip()]

    import torch
    from predictor_mssc import MSSCPredictor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Auto-detect the embedding dim from the cache, exactly as train_mssc.py:263-265 does, and
    # assert it is uniform across slides -- a mixed-dim cache root would otherwise build the net
    # from slide 0 and then fail (or silently truncate) partway through the loop.
    emb_dim = probe_dim(args.cache_root, HD_SLIDES[0], args.fine_fov)
    for _s in HD_SLIDES[1:]:
        _d = probe_dim(args.cache_root, _s, args.fine_fov)
        assert _d == emb_dim, f"mixed embedding dims across slides: {_s} d={_d} vs {emb_dim}"
    net = MSSCPredictor(emb=emb_dim, coarse_dim=emb_dim, dp=args.dp, depth=args.depth,
                        pe_scale_um=args.pe_scale_um,
                        coarse_scales=tuple(coarse_fovs), attn_mode=args.attn_mode,
                        self_fine=bool(args.self_tiles)).to(dev)
    _ck = torch.load(args.predictor, map_location=dev)
    if args.predictor_key:
        if args.predictor_key not in _ck:
            sys.exit(f"--predictor-key '{args.predictor_key}' not in checkpoint "
                     f"{args.predictor} (keys: {list(_ck.keys())})")
        _ck = _ck[args.predictor_key]
    net.load_state_dict(_ck)
    sphere_R = args.sphere_radius if args.sphere_normalize else None
    print(f"device={dev} | predictor={args.predictor} | emb_dim={emb_dim} (auto) "
          f"| Kc(context)={Kc} | batch={args.batch} "
          f"| fine_fov={args.fine_fov} coarse_fovs={coarse_fovs}"
          f" | predictor_key={args.predictor_key}"
          f" | sphere={'ON R=%g' % sphere_R if sphere_R else 'OFF'}")
    if not coarse_fovs:
        print("  SINGLE-SCALE: coarse_scales=() -> net(fe, fx, {}, tx), matching "
              "train_lora_celljepa.py exactly.")
    if sphere_R is None:
        print("  ** WARNING: sphere OFF. If this predictor was trained with --sphere-normalize, "
              "mu/sigma^2 are being produced from out-of-distribution input. **")

    meta = []
    for s in HD_SLIDES:
        t0 = time.time()
        mu, s2, n, n_short, n_floor, n_zero = apply_slide(
            net, torch, dev, args.cache_root, s, Kc, args.batch,
            args.fine_fov, coarse_fovs, sphere_radius=sphere_R,
            max_ctx_dist=args.max_ctx_dist, pe_frame=args.pe_frame,
            self_tiles_dir=args.self_tiles, self_tiles_block=args.self_tiles_block,
            mpp=args.mpp)
        np.savez(os.path.join(args.out, f"{s}.npz"), mu=mu, s2=s2)
        dt = time.time() - t0
        meta.append(dict(slide=s, n_cells=int(n), seconds=round(dt, 1),
                         s2_median=float(np.median(s2.astype(np.float32))),
                         n_short_of_Kc=n_short, n_below_train_floor=n_floor,
                         n_zero_ctx_fellback=n_zero))
        print(f"[{s}] n={n:,} in {dt:.0f}s  s2_median={meta[-1]['s2_median']:.3f} -> {s}.npz")
    prov = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), script=os.path.abspath(__file__),
                predictor=os.path.abspath(args.predictor), Kc=Kc, K=args.K, tgt_frac=args.tgt_frac,
                fine_fov=args.fine_fov, coarse_fovs=coarse_fovs, emb_dim=int(emb_dim),
                predictor_key=args.predictor_key,
                # `mpp_arg` is what was PASSED (None => the 0.2522 default); `mpp_effective` is what
                # was USED, after resolve_mpp gated it against the cache meta.json. Both recorded so
                # a reader can tell an explicit choice from an inherited default.
                mpp_arg=args.mpp, mpp_effective=resolve_mpp(args.cache_root, HD_SLIDES[0],
                                                            args.fine_fov, args.mpp),
                pe_frame=str(args.pe_frame),
                self_tiles=(os.path.abspath(args.self_tiles) if args.self_tiles else None),
                self_tiles_block=(args.self_tiles_block if args.self_tiles else None),
                pe_frame_note=("absolute (um from SLIDE ORIGIN) — legacy path, bit-identical to "
                               "pre-2026-08-16" if args.pe_frame == "absolute" else
                               "target-relative (um offset TO THE TARGET) per MODEL_DESIGN_CARD "
                               "§6.5; tgt_xy is identically 0. MUST match training."),
                # ★ 2026-08-15: both were previously UNRECORDED. attn_mode absence forced a
                # dir-name inference to identify the crossattn predictors; max_ctx_dist is new.
                attn_mode=args.attn_mode,
                max_ctx_dist=float(args.max_ctx_dist),
                max_ctx_dist_note=("OFF — legacy path, bit-identical to pre-2026-08-15"
                                   if not args.max_ctx_dist else
                                   "MUST equal the value the predictor was TRAINED with"),
                sphere_normalize=bool(args.sphere_normalize),
                sphere_radius=(args.sphere_radius if args.sphere_normalize else None),
                cache_root=os.path.abspath(args.cache_root), dp=args.dp, depth=args.depth, slides=meta)
    json.dump(prov, open(os.path.join(args.out, "PROVENANCE.json"), "w"), indent=2)
    print(f"\nwrote {args.out}/{{<slide>.npz, PROVENANCE.json}}")


if __name__ == "__main__":
    main()
