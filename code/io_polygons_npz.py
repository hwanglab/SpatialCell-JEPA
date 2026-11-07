#!/usr/bin/env python3
"""Read `results/polygons_v2/<slide>/polygons.npz` back into the io_polygons contract.

WHY THIS IS A LOADER AND NOT A REWRITE. `polygons_from_mask.load_polygons_from_mask` is documented
(`polygons_from_mask.py:105`) as returning `(object_id, centroid_xy_L0px, ring_L0px, cls)` — exactly
what `io_polygons.load_polygons` returns and what `cache_slide_grid.py` consumes. The materialised
`.npz` is that same list, flattened to CSR by `run_polygons_n12.py` (an object array of ~400k tiny
rings pickles slowly and forces `allow_pickle` on every reader). So the whole job here is CSR ->
list-of-tuples.

★ WHY THERE IS NO CLASS LABEL IN THE FILE — this is deliberate, not a gap. `run_polygons_n12.py:12`:
"`gt_labels_v2/` cannot be built here — its `cache_idx` key is the row order of the cache's
`cells.npz`, which does not exist until the caching array runs. The exact `object_id` join happens
after that, against `cell_meta.json`." The ids ARE the reference label set's `object_id`s, so the label join downstream
is a merge, not the 92.6% nearest-neighbour match `attach_gt_hd.py` does. We therefore emit `cls=""`
and let the join supply labels later. Anything that needs a class label must do that join; it must
NOT infer one here.

FRAME: `ring_xy`/`cen` are already 40x SVS-L0 px (PROVENANCE `frame` field:
"40x SVS-L0 px (mask 0.884 um/px -> B1 InstanSeg affine / B2 scalar)"). No transform is applied
here — re-transforming coordinates that are already in the target frame is the T22 bug class.
"""
from __future__ import annotations

import json
import os

import numpy as np

# The class placeholder. Deliberately NOT a real class name: any downstream consumer that treats
# this as a label should produce an obviously-empty result rather than a plausible wrong one.
UNLABELLED = ""


def load_polygons_npz(path, min_vertices=3):
    """-> list of (object_id:str, centroid_xy_L0px:(2,), ring_L0px:(k,2), cls:str).

    `path` may be the .npz itself or the slide dir containing it.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "polygons.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no polygons.npz at {path}")

    z = np.load(path, allow_pickle=False)
    missing = {"ids", "cen", "ring_xy", "ring_ptr"} - set(z.keys())
    if missing:
        raise KeyError(f"{path}: missing keys {sorted(missing)}; got {sorted(z.keys())}")

    ids, cen, ring_xy, ring_ptr = z["ids"], z["cen"], z["ring_xy"], z["ring_ptr"]
    n = len(ids)
    if len(cen) != n or len(ring_ptr) != n + 1:
        raise ValueError(f"{path}: ragged arrays — ids={n} cen={len(cen)} ptr={len(ring_ptr)}")
    if ring_ptr[-1] != len(ring_xy):
        raise ValueError(f"{path}: CSR tail {ring_ptr[-1]} != len(ring_xy) {len(ring_xy)}")

    cells, dropped = [], 0
    for i in range(n):
        ring = ring_xy[ring_ptr[i]:ring_ptr[i + 1]]
        if ring.shape[0] < min_vertices:          # same guard io_polygons applies to geojson rings
            dropped += 1
            continue
        cells.append((str(ids[i]), cen[i].astype(np.float64),
                      ring.astype(np.float64), UNLABELLED))
    if dropped:
        print(f"  [io_polygons_npz] dropped {dropped}/{n} rings with <{min_vertices} vertices",
              flush=True)
    return cells


def provenance(path):
    """The sidecar PROVENANCE.json, so a consumer can record WHICH polygons it saw."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    p = os.path.join(d, "PROVENANCE.json")
    return json.load(open(p)) if os.path.exists(p) else None


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        c = load_polygons_npz(p)
        ext = [max(r[:, 0].ptp(), r[:, 1].ptp()) for _, _, r, _ in c[:5000]]
        print(f"{p}: {len(c)} cells | id[0]={c[0][0]} | median nucleus max-extent "
              f"{np.median(ext) * 0.2522:.2f} um (expect ~5-7; <3 or >12 => wrong frame)")
