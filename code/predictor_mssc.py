"""MSSC-JEPA multiscale predictor g_phi.

Predicts a target cell's FROZEN Virchow2 embedding m_cell (mean mu + heteroscedastic
diagonal log-variance) from its MULTISCALE spatial context:
  - fine 56um per-cell neighbor tokens (target cells EXCLUDED),
  - coarse per-scale CLS(+mean) region tokens (112/224/448 um; pilot: 112/168),
  - target query tokens (position + scale-emb only; NO target feature).

Design: MODEL_DESIGN_CARD.md §6.5.
  * per-scale input projection W_s (handles the 1280-d fine vs 2560-d coarse mismatch),
  * learned scale-embedding per token + continuous 2-D Fourier position (absolute),
  * ONE joint MSA over all scales (heterogeneity absorbed at the input layer),
  * heads map dp -> emb=1280 (pinned by the frozen target space).
Only this module trains; Virchow2 is frozen + cached.
"""
import math
import torch
import torch.nn as nn


class FourierPE(nn.Module):
    """Continuous 2-D positional encoding of centroids (um) via random Fourier features."""
    def __init__(self, out_dim, n_freq=32, scale_um=64.0):
        super().__init__()
        self.register_buffer("B", torch.randn(2, n_freq))
        self.proj = nn.Linear(2 * n_freq, out_dim)
        self.scale = scale_um

    def forward(self, xy):                       # xy: (..., 2) in um
        x = (xy / self.scale) @ self.B
        f = torch.cat([torch.sin(2 * math.pi * x), torch.cos(2 * math.pi * x)], dim=-1)
        return self.proj(f)


class MSSCPredictor(nn.Module):
    def __init__(self, emb=1280, coarse_dim=1280, dp=384, depth=4, heads=8,
                 coarse_scales=(112, 168), dropout=0.1, logvar_clamp=(-7.0, 7.0),
                 attn_mode="dense", pe_scale_um=64.0, self_fine=False):
        """
        attn_mode — which attention edges are ALLOWED among the M target queries.
          "dense"     every token attends to every token. THE ORIGINAL BEHAVIOUR and the default,
                      so every existing checkpoint keeps loading and scoring identically.
          "block_tt"  target -> target is BLOCKED (a target still attends to itself and to all
                      context). Target queries carry no feature, only a POSITION, and the targets are
                      removed from the context, so the target set encodes WHERE the held-out cells are
                      and local density is strongly predictive of cell type.
                      ⚠⚠ THIS ONLY BLOCKS THE DIRECT EDGE AND IS NOT SUFFICIENT AT depth > 1.
                      context -> target stays open, so after layer 1 the context carries every target
                      position and at layer 2 a target reads it back: target_j -> context -> target_i
                      in TWO HOPS. Measured 2026-08-14 by perturbing one target's position: block_tt
                      isolated at depth 1, LEAKS at depth 2 and 4 (we run depth 4). Use "crossattn"
                      for actual isolation.
          "crossattn" additionally blocks context -> target, so the context encoding is computed
                      independently of which cells were held out (the design card's original spec).
        Implemented as a bool attn_mask (True = NOT allowed), built once per forward from M.
        No parameters are added, so a checkpoint trained under one mode loads under any other --
        which is what makes the ablation a pure inference-time switch as well as a training arm.
        """
        super().__init__()
        assert attn_mode in ("dense", "block_tt", "crossattn"), attn_mode
        self.attn_mode = attn_mode
        self.emb = emb
        self.coarse_scales = list(coarse_scales)
        # (1) per-scale input projections
        self.proj_fine = nn.Linear(emb, dp)                          # 56um per-cell (1280->dp)
        self.proj_coarse = nn.ModuleDict({
            str(s): nn.Linear(coarse_dim, dp) for s in coarse_scales})  # coarse CLS(+mean)->dp
        # (1b) OPTIONAL 56 um TARGET-CENTRED self-tile token  [user 2026-08-18]
        # A SEPARATE projection, deliberately not shared with proj_fine: the fine tokens are
        # NEIGHBOUR cells' pooled m_cell, whereas this is a CLS over the TARGET's own 56 um crop
        # with the target's nucleus hull masked out. Different entity, different statistics
        # ([user 2026-08-14] "they are different entities").
        # ★ K6 = `fine_cls_masked`: the mask is the same nucleus convex hull at margin 0 that
        #   defines m_cell's pooling region, so the two channels PARTITION the hull with no
        #   double-counting (the margin-coupling rule [user] set on 2026-08-15).
        # ⚠⚠ DEFAULT OFF, and that is load-bearing: enabling it adds a slot, which shifts
        #   `slot_tgt` and grows `scale_emb` from (n,dp) to (n+1,dp). With self_fine=False the
        #   module is byte-identical to before, so EVERY existing checkpoint still loads and every
        #   existing number is undisturbed. A checkpoint trained with one setting CANNOT be loaded
        #   under the other -- `scale_emb` has a different row count -- so this flag must be passed
        #   IDENTICALLY at train and apply time, exactly like attn_mode / pe_frame / pe_scale_um.
        self.use_self_fine = bool(self_fine)
        self.proj_self_fine = nn.Linear(emb, dp) if self.use_self_fine else None
        # (2) shared position + scale tags. slots: 0=fine, 1..=coarse in order, last=target
        # pe_scale_um sets the LENGTH UNIT of the positional encoder: positions are divided by it
        # before the sinusoids, so it fixes the range over which position is legible (correlation
        # half-height = pe_scale_um * 0.187). 64.0 is the historical value -- it was FourierPE's
        # class default and was never set deliberately anywhere.
        # ⚠ pe_scale_um is NOT part of the state_dict (it is a plain float, not a buffer), so a
        #   checkpoint does not carry it. It MUST be passed identically at apply time, exactly like
        #   attn_mode and pe_frame. It is recorded in train_history.json['args'].
        self.pe = FourierPE(dp, scale_um=pe_scale_um)
        self.pe_scale_um = float(pe_scale_um)
        self.n_slot = 1 + len(coarse_scales) + (1 if self.use_self_fine else 0) + 1
        self.scale_emb = nn.Embedding(self.n_slot, dp)
        self.slot_fine = 0
        self.slot_coarse = {s: 1 + i for i, s in enumerate(coarse_scales)}
        # ★ the self-tile token, the fine tokens and the target query can all sit at the SAME xy,
        #   so the PE cannot separate them -- scale_emb is the ONLY disambiguator. Hence a slot.
        self.slot_self_fine = (1 + len(coarse_scales)) if self.use_self_fine else None
        self.slot_tgt = self.n_slot - 1
        self.mask_token = nn.Parameter(torch.randn(dp) * 0.02)
        # (3) joint MSA + heteroscedastic heads
        layer = nn.TransformerEncoderLayer(dp, heads, dp * 4, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.mu_head = nn.Linear(dp, emb)
        self.lv_head = nn.Linear(dp, emb)
        self.clamp = logvar_clamp

    def _attn_mask(self, L, M, device):
        """(L,L) bool, True = edge FORBIDDEN. Targets are the LAST M positions.

        The self-edge (i,i) is always kept: a fully-masked row makes softmax undefined and
        PyTorch emits NaNs rather than an error, which would be silent.
        """
        if self.attn_mode == "dense":
            return None
        import torch as _t
        m = _t.zeros(L, L, dtype=_t.bool, device=device)
        t0 = L - M
        m[t0:, t0:] = True                                   # target -> target
        idx = _t.arange(t0, L, device=device)
        m[idx, idx] = False                                  # ...but keep each target's self-edge
        if self.attn_mode == "crossattn":
            m[:t0, t0:] = True                               # context -> target also blocked
        return m

    def forward(self, fine_emb, fine_xy, coarse, tgt_xy,
                fine_mask=None, coarse_mask=None, self_fine=None):
        """
        fine_emb  (B,K,emb),  fine_xy (B,K,2 um)   — 56um neighbor cells (target-excluded)
        coarse    dict{scale:(feat (B,G_s,coarse_dim), xy (B,G_s,2 um))}
        tgt_xy    (B,M,2 um)                        — target query positions
        fine_mask (B,K) bool True=real (False=pad); coarse_mask dict{scale:(B,G_s) bool}
        returns   mu (B,M,emb), logvar (B,M,emb)
        """
        B = fine_emb.shape[0]
        toks, pads = [], []
        # coarse scales first
        for s in self.coarse_scales:
            f, xy = coarse[s]
            t = self.proj_coarse[str(s)](f) + self.pe(xy) + self.scale_emb.weight[self.slot_coarse[s]]
            toks.append(t)
            m = coarse_mask[s] if (coarse_mask is not None and s in coarse_mask) else \
                torch.ones(t.shape[:2], dtype=torch.bool, device=t.device)
            pads.append(~m)
        # fine
        tf = self.proj_fine(fine_emb) + self.pe(fine_xy) + self.scale_emb.weight[self.slot_fine]
        toks.append(tf)
        fm = fine_mask if fine_mask is not None else \
            torch.ones(tf.shape[:2], dtype=torch.bool, device=tf.device)
        pads.append(~fm)
        # 56 um TARGET-CENTRED self-tile token — ONE per sample, appended as CONTEXT.
        # ★ Position: the target's OWN xy, so under pe_frame="target" (where tgt_xy is already the
        #   origin) this is PE(0) -- zero offset achieved by COORDINATE EQUALITY, which is SPEC K3.
        #   Do not "fix" this by passing literal zeros; the equality is the mechanism.
        # ★ Placement: BEFORE the target queries. _attn_mask derives the target rows from
        #   t0 = L - M, so any context token appended here is automatically masked correctly and
        #   NO change to _attn_mask is needed. Under "crossattn" it can attend to the fine tokens
        #   and be attended BY the targets, while context->target stays blocked. Verified by the
        #   perturbation gate, not assumed.
        if self.use_self_fine:
            assert self_fine is not None, \
                "predictor was built with self_fine=True but forward() got self_fine=None"
            sf = self_fine if self_fine.dim() == 3 else self_fine.unsqueeze(1)   # (B,1,emb)
            sf = (self.proj_self_fine(sf)
                  + self.pe(tgt_xy[:, :1, :])
                  + self.scale_emb.weight[self.slot_self_fine])
            toks.append(sf)
            pads.append(torch.zeros(B, sf.shape[1], dtype=torch.bool, device=sf.device))
        elif self_fine is not None:
            raise ValueError("self_fine passed but the predictor was built with self_fine=False "
                             "-- the flag must match between construction and forward")

        # target queries (never padded)
        M = tgt_xy.shape[1]
        q = self.mask_token + self.pe(tgt_xy) + self.scale_emb.weight[self.slot_tgt]
        toks.append(q)
        pads.append(torch.zeros(B, M, dtype=torch.bool, device=q.device))

        seq = torch.cat(toks, dim=1)
        pad = torch.cat(pads, dim=1)                        # True = ignore
        am = self._attn_mask(seq.shape[1], M, seq.device)   # None under "dense"
        h = self.enc(seq, mask=am, src_key_padding_mask=pad)
        ht = h[:, -M:, :]                                   # read the M target queries
        mu = self.mu_head(ht)
        logvar = self.lv_head(ht).clamp(*self.clamp)
        return mu, logvar


def _smoke():
    """CPU forward+backward smoke: shapes + gradient flow (module-level trainability)."""
    torch.manual_seed(0)
    B, K, M, emb = 2, 40, 10, 1280
    net = MSSCPredictor(emb=emb, coarse_dim=1280, dp=384, depth=4, coarse_scales=(112, 168))
    fine_emb = torch.randn(B, K, emb)
    fine_xy = torch.randn(B, K, 2) * 60
    tgt_xy = torch.randn(B, M, 2) * 60
    coarse = {112: (torch.randn(B, 6, 1280), torch.randn(B, 6, 2) * 120),
              168: (torch.randn(B, 3, 1280), torch.randn(B, 3, 2) * 180)}
    target = torch.randn(B, M, emb)                        # frozen m_cell of target cells
    mu, logvar = net(fine_emb, fine_xy, coarse, tgt_xy)
    assert mu.shape == (B, M, emb) and logvar.shape == (B, M, emb), (mu.shape, logvar.shape)
    # Gaussian NLL (Kendall-Gal)
    nll = 0.5 * (((target - mu) ** 2) * torch.exp(-logvar) + logvar).sum(-1).mean()
    nll.backward()
    ngrad = sum(int(p.grad is not None and p.grad.abs().sum() > 0) for p in net.parameters())
    nparam = sum(1 for _ in net.parameters())
    print(f"forward OK: mu{tuple(mu.shape)} logvar{tuple(logvar.shape)}  NLL={nll.item():.3f}")
    print(f"params with nonzero grad: {ngrad}/{nparam}  total params={sum(p.numel() for p in net.parameters()):,}")
    print("SMOKE PASS" if ngrad == nparam else "SMOKE WARN: some params got no gradient")


if __name__ == "__main__":
    _smoke()
