"""Unified cell-polygon loaders → list of (cell_id, centroid_xy_L0px, nucleus_poly_L0px, cls).

Sources (per P16 canonical sources):
  - CellViT++  `<slide>_cells.geojson` : list of features BY CLASS, each a MultiPolygon of nuclei (L0 px).
  - HoverFast  `<slide>.json.gz`       : gzipped FeatureCollection, one Polygon + centroid per cell.
Centroids for CellViT are computed from the polygon (avoids the T22 centroid X/Y axis bug).
"""
import json, gzip
import numpy as np


def _cls_of(props):
    if isinstance(props, dict):
        c = props.get("classification")
        return c.get("name") if isinstance(c, dict) else props.get("name")
    return None


def load_cellvit_geojson(path):
    d = json.load(open(path))
    cells = []
    for feat in d:
        g = feat["geometry"]; cls = _cls_of(feat.get("properties", {}))
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for k, poly in enumerate(polys):
            ring = np.asarray(poly[0], dtype=np.float64)   # exterior ring
            if ring.shape[0] < 3:
                continue
            cells.append((f"{cls}__{k}", ring.mean(0), ring, cls))
    return cells


def load_hoverfast_geojson_gz(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    feats = d["features"] if isinstance(d, dict) and "features" in d else d
    cells = []
    for k, feat in enumerate(feats):
        g = feat["geometry"]; cls = _cls_of(feat.get("properties", {}))
        ring = np.asarray(g["coordinates"][0], dtype=np.float64)
        if ring.shape[0] < 3:
            continue
        cen = np.asarray(g["centroid"], dtype=np.float64) if "centroid" in g else ring.mean(0)
        cells.append((f"hf__{k}", cen, ring, cls))
    return cells


def load_polygons(path, kind="auto"):
    if kind == "auto":
        kind = "hoverfast_gz" if path.endswith(".json.gz") else "cellvit"
    if kind == "cellvit":
        return load_cellvit_geojson(path)
    if kind == "hoverfast_gz":
        return load_hoverfast_geojson_gz(path)
    raise ValueError(f"unknown polygon kind: {kind}")
