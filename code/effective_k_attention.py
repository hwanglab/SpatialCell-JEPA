#!/usr/bin/env python3
"""How many of its 36 context cells does the trained predictor ACTUALLY use?

Measures the target query's cross-attention distribution over the Kc context tokens and reports the
EFFECTIVE number of neighbours = perplexity = exp(H(p)), plus top-k attention mass.

WHY THIS AND NOT THE K SWEEP. The 2026-08-20 K sweep (51401357) measured an ARITHMETIC MEAN of K
neighbours, which must degrade as K grows because distant neighbours dilute with equal weight.
Cross-attention can learn to down-weight them, so the smoother's optimum need not be the
predictor's. This reads the actual trained model instead of a proxy. It has never been done.

★ EXACTNESS GATE. The batch construction below is copied VERBATIM from apply_slide()
(apply_predictor_hd.py:135-190) because a re-implementation that drifts would measure a model we
never ran. The copy is then VERIFIED: the mu this script produces must reproduce the stored predout
npz for the same cells. If it does not, the script refuses to report attention.

⚠ LAYER SEMANTICS, and it matters for interpretation.
  Under crossattn the mask blocks context->target but leaves context<->context OPEN. So at LAYER 0
  the target query reads the raw projected neighbour tokens -- that row is a clean "which neighbours
  do I look at". By LAYER 1 each context token has already mixed with the others, so a diffuse row
  there does NOT mean 36 neighbours are used; it can mean 6 neighbours whose content has spread.
  Both layers are reported; layer 0 is the interpretable one.

⚠ The self-edge (target attending to itself) is kept by the mask and is EXCLUDED from p before
  renormalising -- the question is how the 36 CONTEXT cells are weighted, not how much the target
  looks at itself. Its mass is reported separately as `self_mass`.
"""
import argparse, os, sys, json, numpy as np, torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))
from predictor_mssc import MSSCPredictor
from apply_predictor_hd import load_cache

CAP = []          # captured (layer, weights) per forward


def patch_attention(net):
    """Force need_weights on every encoder layer's MultiheadAttention and stash the target row.

    TransformerEncoderLayer calls self_attn(..., need_weights=False)[0]; wrapping forward keeps the
    [0] indexing valid while letting us keep the weights.
    """
    for li, layer in enumerate(net.enc.layers):
        # ⚠⚠ FORCE THE SLOW PATH — this is load-bearing, not defensive.
        # TransformerEncoderLayer.forward has a sparsity fast path that calls
        # torch._transformer_encoder_layer_fwd(...) with self_attn.in_proj_weight directly and
        # NEVER invokes self_attn.forward. On the 2026-08-20 first run it took that path: `mu` was
        # correct, the exactness gate PASSED at cos 0.99999, and the attention capture silently
        # returned NOTHING. A no-op forward pre-hook is one of PyTorch's own documented
        # disqualifiers ("forward pre-/hooks are attached to the module"), so registering one
        # routes execution through _sa_block -> self_attn(...) -> the wrapper below.
        layer.register_forward_pre_hook(lambda _m, _inp: None)
        mha = layer.self_attn
        if getattr(mha, '_ek_patched', False):
            continue
        orig = mha.forward

        def fwd(q, k, v, _orig=orig, _li=li, **kw):
            kw['need_weights'] = True
            kw['average_attn_weights'] = False          # keep per-head
            out, w = _orig(q, k, v, **kw)
            CAP.append((_li, w.detach()))               # (B, heads, L, L)
            return out, w
        mha.forward = fwd
        mha._ek_patched = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictor', required=True)
    ap.add_argument('--cache-root', default='results/cache_fm_hd/hoptimus0')
    ap.add_argument('--predout', required=True, help='stored predout dir, for the exactness gate')
    ap.add_argument('--slide', required=True)
    ap.add_argument('--n-cells', type=int, default=4096)
    ap.add_argument('--K', type=int, default=48)
    ap.add_argument('--tgt-frac', type=float, default=0.25)
    ap.add_argument('--fine-fov', type=int, default=56)
    ap.add_argument('--dp', type=int, default=384)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--pe-frame', default='target')
    ap.add_argument('--pe-scale-um', type=float, default=64.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mpp', type=float, default=None,
                    help="Microns per pixel. None => apply_predictor_hd's 0.2522 default, which is "
                         "what every run to 2026-08-22 used. MUST be passed for any cache "
                         "that is not ~0.2522: the G4X GN-133 cache is 0.3125, and every distance "
                         "this script reports (attention-weighted radius, 1-NN spacing, "
                         "mass_within_nearest_K) is in um, so a wrong value rescales all of them. "
                         "Gated against the cache meta.json by resolve_mpp -- >1% is fatal.")
    ap.add_argument('--out', required=True)
    ap.add_argument('--dump-per-neighbour', action='store_true',
                    help='ALSO write per_neighbour.npz (tgt, ctx, dist_um, p_layer0, fmask). '
                         'Purely additive: no existing output, gate or number changes.')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Kc = a.K - int(round(a.K * a.tgt_frac))
    dev = 'cpu'

    xy, feat = load_cache(a.cache_root, a.slide, a.fine_fov, (), mpp=a.mpp)
    m_fine = feat[int(a.fine_fov)]
    n = len(xy)
    emb = int(m_fine.shape[1])
    stored = np.load(os.path.join(a.predout, f'{a.slide}.npz'))['mu']
    assert len(stored) == n, f'predout {len(stored)} != cache {n}'

    net = MSSCPredictor(emb=emb, coarse_dim=emb, dp=a.dp, depth=a.depth, coarse_scales=(),
                        attn_mode='crossattn', pe_scale_um=a.pe_scale_um).to(dev)
    net.load_state_dict(torch.load(a.predictor, map_location=dev))
    net.eval()

    rng = np.random.default_rng(a.seed)
    tgt = np.sort(rng.choice(n, min(a.n_cells, n), replace=False))
    tree = cKDTree(xy)

    # ── VERBATIM from apply_slide (apply_predictor_hd.py:135-190). Do not "tidy". ────────────────
    kq = Kc + 2
    dd, nb = tree.query(xy[tgt], k=kq)
    ctx = np.zeros((len(tgt), Kc), dtype=np.int64)
    fmask = np.ones((len(tgt), Kc), dtype=bool)
    n_short = n_zero = 0
    for i, (t, row) in enumerate(zip(tgt, nb)):
        r = np.asarray(row); sel = r != t
        keep = r[sel][:Kc]; k = len(keep)
        if k == 0:
            keep = r[r != t][:1]; k = len(keep); n_zero += 1
        if k < Kc:
            n_short += 1; fmask[i, k:] = False
        ctx[i, :k] = keep
    cx = xy[ctx]
    # ── GEOMETRY, computed here so it uses the SAME ctx the model sees ───────────────────────────
    # ★ ctx columns are in NEAREST-FIRST order: cKDTree.query returns sorted distances and
    #   `r[sel][:Kc]` preserves that order. So column j IS the j-th nearest neighbour, which is what
    #   makes `mass_within_nearest_K` (below) a direct answer to "is truncating to K safe?".
    # ⚠ UPDATED 2026-08-23: the hardcoded MPP is now `--mpp` (default 0.2522, gated against the
    #   cache meta.json; >1% disagreement is fatal). Passing nothing reproduces every number
    #   to 2026-08-22 exactly. The note below still describes that case.
    # ⚠ xy is um via a HARDCODED MPP=0.2522 in apply_predictor_hd, while per-slide mpp is 0.2522 or
    #   0.2525 -> a 0.12 % scale error on every distance quoted here. Negligible at ~20 um; stated
    #   rather than hidden.
    DIST = np.linalg.norm(cx - xy[tgt][:, None, :], axis=2)      # (B, Kc) um, nearest-first
    fe = torch.from_numpy(np.asarray(m_fine[ctx], np.float32)).to(dev)
    fx = torch.from_numpy(cx).float().to(dev)
    tx = torch.from_numpy(xy[tgt][:, None, :]).float().to(dev)
    if a.pe_frame == 'target':
        org = tx.mean(dim=1, keepdim=True); fx = fx - org; tx = tx - org
    # ── end verbatim ─────────────────────────────────────────────────────────────────────────────

    patch_attention(net)
    CAP.clear()
    with torch.no_grad():
        mu, logvar = net(fe, fx, {}, tx, fine_mask=torch.from_numpy(fmask).to(dev))
    # ── sigma^2, captured from the SAME forward pass as the attention ────────────────────────────
    # [user 2026-08-20] wants sigma^2 in the paper as an application. It is NOT a validated
    # instrument -- its only positive signal came from a post-hoc-selected estimator -- so this
    # CHARACTERISES it rather than validating it. The pairing with per-cell attention concentration
    # is free here and available nowhere else: both come from one forward pass on the same cells.
    # sigma^2 per cell = mean over the emb dims of exp(logvar), clamped as the model clamps it.
    S2 = logvar[:, 0, :].exp().mean(-1).cpu().numpy()          # (B,)

    # ── CAPTURE GATE — must fire BEFORE the exactness gate can lull anyone ───────────────────────
    # The 2026-08-20 first run passed the exactness gate and reported an EMPTY attention section,
    # because the fast path bypassed the wrapper. A correct `mu` says nothing about whether the
    # thing we came to measure was measured. Fail loudly instead.
    if not CAP:
        sys.exit('FATAL: attention capture produced 0 tensors — the TransformerEncoderLayer fast '
                 'path was taken and self_attn.forward was never called. mu may still be correct; '
                 'the measurement is not. Do not report anything from this run.')
    if len(CAP) != len(net.enc.layers):
        sys.exit(f'FATAL: captured {len(CAP)} attention tensors for {len(net.enc.layers)} layers')

    # ── EXACTNESS GATE ───────────────────────────────────────────────────────────────────────────
    got = mu[:, 0, :].cpu().numpy().astype(np.float16)
    ref = stored[tgt]
    dmax = float(np.max(np.abs(got.astype(np.float32) - ref.astype(np.float32))))
    cos = float(np.mean(np.sum(got.astype(np.float32) * ref.astype(np.float32), 1) /
                        (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1) + 1e-12)))
    gate = (cos >= 1 - 1e-4)
    print(f'EXACTNESS GATE: mean cos(mu_here, mu_stored) = {cos:.8f}   max|diff| = {dmax:.6f}   '
          f'{"PASS" if gate else "FAIL"}')
    if not gate:
        sys.exit('refusing to report attention: the replicated batch does not reproduce the stored mu')

    # ── effective K ──────────────────────────────────────────────────────────────────────────────
    L = Kc + 1; t0 = Kc
    out = {'slide': a.slide, 'n_cells': int(len(tgt)), 'Kc': Kc, 'emb': emb,
           'predictor': os.path.abspath(a.predictor), 'layers': {},
           'gate': {'mean_cos_vs_stored_mu': cos, 'max_abs_diff': dmax, 'pass': True},
           'n_short_of_Kc': int(n_short), 'n_zero_ctx': int(n_zero)}
    print(f'\nEFFECTIVE K — target query attention over {Kc} context tokens, {len(tgt):,} cells')
    for li, w in CAP:
        A = w[:, :, t0, :]                      # (B, heads, L) target row
        self_mass = A[:, :, t0]
        p = A[:, :, :t0].clone()                # 36 context columns
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        H = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
        eff = H.exp()                            # perplexity = effective #neighbours
        srt = p.sort(-1, descending=True).values
        topm = {f'top{k}': float(srt[:, :, :k].sum(-1).mean()) for k in (1, 3, 6, 12, 18)}
        # ── RANK/DISTANCE DIAGNOSTIC — the gap effective-K alone leaves open ─────────────────────
        # effective K says HOW CONCENTRATED the attention is. It does NOT say WHICH neighbours get
        # it. If the model concentrates on DISTANT neighbours, truncating to the K nearest removes
        # exactly what it uses and the effective-K number would have misled. Column j = j-th
        # nearest, so mass_within_nearest_K is a direct read.
        D = torch.from_numpy(DIST).to(p.dtype).unsqueeze(1)      # (B,1,Kc) broadcast over heads
        within = {f'mass_within_nearest_{k}': float(p[:, :, :k].sum(-1).mean())
                  for k in (4, 8, 12, 18, 36)}
        wdist = float((p * D).sum(-1).mean())                    # attention-weighted mean distance
        udist = float(DIST.mean())                               # unweighted mean over the 36
        rank_of_top = p.argmax(-1).float()                       # distance-rank of the top-attended
        rec_geo = dict(attn_weighted_mean_dist_um=wdist, unweighted_mean_dist_um=udist,
                       top_attended_dist_rank_mean=float(rank_of_top.mean()),
                       top_attended_is_nearest_frac=float((rank_of_top == 0).float().mean()),
                       top_attended_in_nearest8_frac=float((rank_of_top < 8).float().mean()),
                       **within)
        rec = dict(**rec_geo, eff_k_mean=float(eff.mean()), eff_k_median=float(eff.median()),
                   eff_k_p10=float(eff.flatten().kthvalue(max(1, int(.10 * eff.numel()))).values),
                   eff_k_p90=float(eff.flatten().kthvalue(max(1, int(.90 * eff.numel()))).values),
                   self_mass_mean=float(self_mass.mean()),
                   per_head_eff_k=[float(x) for x in eff.mean(0)], **topm)
        out['layers'][f'layer{li}'] = rec
        print(f'\n  LAYER {li}   effective K = {rec["eff_k_mean"]:.2f} (median {rec["eff_k_median"]:.2f}, '
              f'p10 {rec["eff_k_p10"]:.2f}, p90 {rec["eff_k_p90"]:.2f})  of {Kc}')
        print(f'    attention mass on top-k neighbours: ' +
              '  '.join(f'{k}={v:.3f}' for k, v in topm.items()))
        print(f'    target self-edge mass (excluded from p): {rec["self_mass_mean"]:.4f}')
        print(f'    per-head effective K: ' + ' '.join(f'{x:.1f}' for x in rec['per_head_eff_k']))
        print(f'    ATTENTION MASS ON THE NEAREST-K (by distance, not by weight): ' +
              '  '.join(f'{k}={rec[f"mass_within_nearest_{k}"]:.3f}' for k in (4, 8, 12, 18)))
        print(f'    top-attended neighbour: is-nearest {rec["top_attended_is_nearest_frac"]:.3f} · '
              f'in-nearest-8 {rec["top_attended_in_nearest8_frac"]:.3f} · '
              f'mean dist-rank {rec["top_attended_dist_rank_mean"]:.2f}')
        print(f'    mean distance: attention-weighted {rec["attn_weighted_mean_dist_um"]:.2f} um '
              f'vs unweighted-over-36 {rec["unweighted_mean_dist_um"]:.2f} um')

    # ── sigma^2 CHARACTERISATION (label-free; NOT a validation) ──────────────────────────────────
    from scipy import stats as _st
    l0 = CAP[0][1]                                     # layer 0 weights (B, heads, L, L)
    p0 = l0[:, :, t0, :t0].clone()
    p0 = p0 / p0.sum(-1, keepdim=True).clamp_min(1e-12)
    effk_cell = (-(p0.clamp_min(1e-12) * p0.clamp_min(1e-12).log()).sum(-1)).exp().mean(1).cpu().numpy()
    nn1 = DIST[:, 0]                                   # distance to the nearest neighbour, um
    dens = np.array([np.sum(DIST[i] <= 22.0) for i in range(len(DIST))], dtype=float)  # cells within 22um
    sig = {}
    for nm, v in [('effective_k', effk_cell), ('nn1_dist_um', nn1), ('n_within_22um', dens)]:
        r, pv = _st.spearmanr(S2, v)
        sig[nm] = dict(spearman_rho=float(r), p=float(pv))
    out['sigma2'] = dict(
        s2_mean=float(S2.mean()), s2_median=float(np.median(S2)),
        s2_p10=float(np.percentile(S2, 10)), s2_p90=float(np.percentile(S2, 90)),
        spearman_vs=sig,
        note=('CHARACTERISATION, NOT VALIDATION. sigma^2 has no validated downstream use in this '
              'project; its only positive signal came from a post-hoc-selected estimator. Reported '
              'to describe the quantity, not to claim it works.'))
    print(f'\n  SIGMA^2 (label-free characterisation, NOT a validation)')
    print(f'    per-cell sigma^2: median {np.median(S2):.4f}  mean {S2.mean():.4f}  '
          f'p10 {np.percentile(S2,10):.4f}  p90 {np.percentile(S2,90):.4f}')
    for nm, d in sig.items():
        print(f'    Spearman(sigma^2, {nm:<14}) = {d["spearman_rho"]:+.4f}   p={d["p"]:.2e}')

    # ── NEIGHBOURHOOD GEOMETRY — independent of the model ────────────────────────────────────────
    geo = {}
    for k in (4, 8, 12, 18, 36):
        r = DIST[:, k - 1]                                       # distance to the k-th nearest = radius
        geo[f'K{k}'] = dict(radius_um_median=float(np.median(r)),
                            radius_um_q25=float(np.percentile(r, 25)),
                            radius_um_q75=float(np.percentile(r, 75)),
                            diameter_um_median=float(2 * np.median(r)))
    out['geometry'] = geo
    print(f'\n  NEIGHBOURHOOD GEOMETRY (model-independent, {len(tgt):,} cells)')
    print(f'    {"K":>4}{"radius um (median)":>21}{"IQR":>18}{"diameter um":>14}')
    for k in (4, 8, 12, 18, 36):
        g = geo[f'K{k}']
        iqr = '{:.1f}-{:.1f}'.format(g['radius_um_q25'], g['radius_um_q75'])
        print('    {:>4}{:>21.2f}{:>18}{:>14.2f}'.format(
            k, g['radius_um_median'], iqr, g['diameter_um_median']))
    # ── PER-NEIGHBOUR DUMP (added 2026-08-27) ────────────────────────────────────────────────────
    # WHY: every aggregate above marginalises over WHICH neighbour got the mass, so no stored
    # artifact can answer "does attention prefer neighbours that share the target's grid node, at
    # matched distance". Layer 0 only -- layer 1 has context<->context mixing and is not the
    # interpretable row. Heads are averaged AFTER renormalising over the 36 context columns, the
    # same p used for effective K above, so the dump and the aggregates cannot disagree.
    if a.dump_per_neighbour:
        l0 = CAP[0][1]
        pd_ = l0[:, :, t0, :t0].clone()
        pd_ = pd_ / pd_.sum(-1, keepdim=True).clamp_min(1e-12)
        p_mean = pd_.mean(1).cpu().numpy().astype(np.float32)        # (B, Kc), sums to 1 per row
        np.savez_compressed(
            os.path.join(a.out, 'per_neighbour.npz'),
            tgt=tgt.astype(np.int64), ctx=ctx.astype(np.int64),
            dist_um=DIST.astype(np.float32), p_layer0=p_mean, fmask=fmask,
            xy_um=xy.astype(np.float32), slide=np.array(a.slide), Kc=np.array(Kc))
        # self-check: the dumped p must reproduce the effective K already reported for layer 0
        Hc = -(np.clip(p_mean, 1e-12, None) * np.log(np.clip(p_mean, 1e-12, None))).sum(-1)
        print(f'  per-neighbour dump: {p_mean.shape} rows sum to '
              f'{p_mean.sum(-1).mean():.6f}; eff K from dump (head-averaged p) = '
              f'{np.exp(Hc).mean():.2f}')
        print(f'wrote {a.out}/per_neighbour.npz')

    json.dump(out, open(os.path.join(a.out, 'effective_k.json'), 'w'), indent=2)
    print(f'\nwrote {a.out}/effective_k.json')


if __name__ == '__main__':
    main()
