# SpatialCell-JEPA

### *Do pathology foundation-model cell embeddings already encode neighborhood context?*

Accompanies *SpatialCell-JEPA: do pathology foundation-model cell embeddings already encode neighborhood context?* Park et al., Interpretability for Discovery workshop, NeurIPS 2026.

**License:** code (`code/`) under the [MIT License](LICENSE); result tables under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This repository accompanies our recent work, **SpatialCell-JEPA**, which studies how much of a cell's
identity in tissue is carried by the cells around it. It borrows the prediction objective of
joint-embedding predictive architectures (JEPA) — predicting the *latent representation* of a hidden
target from its visible context, rather than reconstructing pixels or comparing augmented views
([Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive
Architecture*, CVPR 2023](https://arxiv.org/abs/2301.08243)).

Below is a short summary of the idea, then a description of what is in this repository.

---

## 1. The idea

### 1.1 A cell has to be understood together with its neighbors

The tumor microenvironment is a spatially organized ecosystem in which cancer cells interact with
immune cells, fibroblasts, endothelial cells and extracellular matrix, and those interactions bear on
tumor growth, immune evasion and response to treatment. A cell's functional state therefore cannot
always be read from its appearance alone: two morphologically similar cells may occupy different
functional states because they sit in different cellular neighborhoods.

### 1.2 Our input is the H&E slide

Hematoxylin and eosin (H&E) staining is performed on essentially every resected tumor as part of
diagnosis, so H&E sections exist at a scale no molecular assay matches, and they carry the cellular
morphology and tissue architecture that spatial analysis needs. What is missing is a *per-cell*
representation from H&E informative enough for downstream analysis such as cell typing.

Because the image crop a cell is embedded from also contains its neighbors, such a representation
may already carry the neighborhood without being trained to. Whether it does is what this work tests.

### 1.3 Why a JEPA objective

Histopathology foundation models produce transferable embeddings from image crops without
task-specific labels. Many are trained contrastively, by encouraging augmented views of an image to
agree, which makes the objective depend on the augmentations chosen: transformations meant to remove
nuisance variation can also suppress biologically meaningful morphology, and cells that differ only in
ways an augmentation preserves may still be treated as negatives.

JEPAs take a different route. Rather than defining similarity through augmented pairs or
reconstructing pixels, they predict the latent representation of a hidden target from its visible
context. What that objective rewards is structure at a scale larger than the target: to predict what
is hidden, the model has to represent how a region coheres, so the learned features carry global
layout and semantic context rather than local texture alone.

### 1.4 SpatialCell-JEPA in outline

We borrow that objective but **not** its training regime, and this inverts the usual roles: the
predictor, ordinarily scaffolding, is here the object of study.

Asking how much spatial context a frozen embedding already carries needs a fixed reference — a jointly
trained encoder would move the cell's own embedding too, confounding what the neighborhood adds with
what adaptation did. So the encoder stays **frozen** and we train **only the predictor**:

```
   neighbor cells' frozen embeddings  +  their positions relative to the target
                              |
                              v
                          predictor  ->  mu       estimate of the target cell's embedding
                                         sigma^2  heteroscedastic uncertainty
                              |
                              v
        the target cell's own embedding m is WITHHELD from the context set
```

We then evaluate on **patient-held-out cell typing** across twelve slides. The main findings:

- The neighbor-derived estimate `mu` types cells **as well as** the cell's own embedding `m`
  (macro-F1 0.619 against 0.616 in the main setting); in all eight readout and crop settings their
  difference stays within ±0.011 at 90 percent confidence.
- The predictor's attention is local: it uses about **seven of its thirty-six** context cells, at a
  weighted mean distance inside the half-width of the crop the embedding was computed from. The
  target's embedding is withheld from the context set, but not from the neighbors' image crops,
  which overlap it.
- A separately configured predictor at a wider crop, and a predictor trained on a second foundation
  model, show the same null.
- **Masking the target** out of its neighbors' crops lowers `mu` slightly but leaves its combination
  with `m` unchanged, so the null is not an artifact of that overlap.
- A **cell-only embedding**, computed from the same crop with the encoder allowed to see only the
  patches covering the cell itself, types worse and knows no more about its neighbors' types than
  the cell's own type implies; adding the masked `mu` back restores the typing score. The context is
  already in the cell's own embedding: for H-optimus-0, at the resolution of three-class typing, the
  answer to the title's question is yes.
- The null holds only **on average**. Grouping cells by how many of their eight nearest neighbors
  share their reference type (a diagnostic, since it reads labels), the estimate is worse than `m`
  where none does, under both readouts; where all do, it is better under a nonlinear readout and not
  detectably different under the linear one. The uncertainty the predictor reports is, on average,
  higher at interfaces than in uniform tissue.

---

## 2. What is in this repository

### 2.1 Result tables

Per-fold and per-slide statistics behind every table in the paper. 405 files, TSV and JSON only,
across 25 directories, one per run, named by date and analysis. The paper names the source directory
or file beneath each table, and those names are preserved here.

**These are aggregate statistics.** They carry no patient-level information — no per-cell records, no
coordinates, no image data, no clinical variables. Every number reported in the paper's tables can be
recomputed from them, as can the target-masking, cell-only-embedding and confidence-interval results
in the text. `20261106_t2_release_supplement/` holds three small tables extracted from raw run outputs
that have no other released form: context slots changed by the target mask, the per-fold
label-supervised encoder-adaptation control, and the validation loss of the reduced-context arms.

They do **not** permit rerunning the feature extraction, which needs the whole-slide images. The
images and their matched spatial transcriptomics are patient-derived tissue held at a single
institution and are not publicly available.

### 2.2 Analysis code — `code/`

The analysis and evaluation code that produced the tables above: 22 files, about 5,500 lines. The
set is not curated by hand — each released run names its producing script in its `PROVENANCE.json`,
and `code/` is exactly that set plus the modules those scripts import.

| file | role |
|---|---|
| `p2_typing_and_slideid.py` | the cell-typing probe and the slide-ID nuisance control |
| `probe_backbone_fov_grid.py` | backbone x crop x split grid |
| `consolidate_nonlinear_typing.py` | readout-ablation consolidation |
| `selftile56_paired_test.py` | paired 12-fold test for the self-tile arm |
| `effective_k_attention.py` | attention entropy, top-k mass, weighted distance |
| `mu_m_geometry_and_sigma_spec.py` | cosine / R-squared / MSE geometry of `mu` vs `m` |
| `neighborhood_homogeneity_strata.py` | the homogeneity strata and their per-class breakdown |
| `percell_class_and_composition_strata.py` | pooled per-class comparison |
| `apply_predictor_hd.py` | runs a trained predictor over every cell |
| `celltype_probe_hd.py` | patient-out typing probe |
| `predictor_mssc.py` | the predictor architecture |
| `cache_masked_context.py` | re-embeds each neighbor with the target's patch tokens removed from attention |
| `merge_maskctx_split.py` | merges the split typing outputs of the masking runs |
| `cache_isolated_cell.py` | the cell-only embedding: attention limited to the cell's hull plus `r` microns |
| `count_kept_neighbors.py` | neighbor cells inside the patches the cell-only embedding keeps |
| `probe_nbr_composition.py` | ridge probe for the types of the eight nearest neighbors |
| `pooling_rule_and_segmenter_geometry.py` | pooling-rule geometry and the two-segmenter agreement |
| `camera_ready_delta_ci.py` | 90 percent confidence intervals and equivalence tests of `mu - m` |
| `build_release_supplement.py` | builds `20261106_t2_release_supplement/` |
| `pooler.py`, `pooler_grid.py`, `fm_loaders.py` | hull-weighted pooling of patch tokens and the encoder loaders |

⚠ **This code is not runnable as released, and does not claim to be.** It consumes cached
foundation-model embeddings, label files and whole-slide images, none of which are in this release
and none of which can be — they are patient-derived tissue held at a single institution. The code is
here as methods transparency; the **tables** are what make the paper's numbers checkable.

Two de-identification changes were made to the code beyond the rules in section 4. Specimen
accession numbers appearing as literal constants were replaced by the `Slide N` labels, derived from
the same verified mapping used for the tables. And an import that existed only to obtain the
canonical twelve-slide list was replaced by that list inline, which removes a dependency on
slide-image tooling that is not otherwise needed. Design annotations marked `[user ...]` are kept:
they name nobody and they record why each choice was made.

The model, training and evaluation procedures are also specified in full in the paper and its
appendices, so the method is reimplementable from the paper alone.

---

## 3. Slide numbering

Slides are numbered exactly as in the paper's slide-composition appendix: **Slide 1 to Slide 12 in
order of decreasing tumor fraction**. The numbering carries no other meaning and is the only slide
identifier used anywhere in this release.

| slide | labeled cells | immune | stromal | tumor | stratum |
|---|---:|---:|---:|---:|---|
| Slide 1 | 352,991 | 0.170 | 0.108 | 0.722 | tumor-dominant |
| Slide 2 | 83,534 | 0.104 | 0.319 | 0.577 | tumor-dominant |
| Slide 3 | 181,080 | 0.129 | 0.300 | 0.515 | tumor-dominant |
| Slide 4 | 378,995 | 0.402 | 0.068 | 0.470 | tumor-dominant |
| Slide 5 | 259,861 | 0.103 | 0.457 | 0.441 | tumor-dominant |
| Slide 6 | 153,774 | 0.107 | 0.452 | 0.356 | tumor-dominant |
| Slide 7 | 174,060 | 0.213 | 0.270 | 0.260 | stroma-dominant |
| Slide 8 | 275,624 | 0.342 | 0.424 | 0.206 | stroma-dominant |
| Slide 9 | 245,649 | 0.172 | 0.450 | 0.185 | stroma-dominant |
| Slide 10 | 133,101 | 0.187 | 0.580 | 0.182 | stroma-dominant |
| Slide 11 | 265,407 | 0.286 | 0.487 | 0.172 | stroma-dominant |
| Slide 12 | 169,294 | 0.177 | 0.686 | 0.131 | stroma-dominant |

---

## 4. What was removed

To de-identify patient data (first for anonymous review), the working copies were processed as follows.
The redacted fields still read `[redacted for anonymous review]`.

1. **Specimen accession numbers replaced by the slide numbers above**, in file contents, in JSON
   object keys, and in directory names.
2. **Absolute filesystem paths made repository-relative**, removing the storage project code and the
   user directory.
3. **Compute-environment fields redacted** — job identifiers, cluster names, host, and tooling
   attribution keys now read `[redacted for anonymous review]`. Nothing that a number depends on was
   touched.
4. **Internal working notes excluded.** The per-run `report.md`, `PAIRED_REPORT.md` and other
   Markdown notes were written for the authors and are not needed to reproduce any table, so they
   are not included. Per-cell `.tsv.gz` dumps are also excluded; the `summary.tsv` beside them holds
   every number the paper cites.
5. **Cohort and site names** in relative paths were replaced by generic names.

Substantive fields are unchanged: every metric, count, p-value, minimum-detectable-effect,
hyperparameter and random seed is exactly as produced by the original run.

## 5. Reading the `typing_split` field in `PROVENANCE.json`

⚠ **`typing_split` is a fixed label emitted by the evaluation script and does not on its own tell you
which split a run used.** Several files read `WITHIN-FOLD leave-one-slide-out (trains on 5)` even
though the run was a plain twelve-fold leave-one-patient-out that trains on eleven. Use the
**`fold_lists`** field to tell them apart:

| `fold_lists` | actual split | used for |
|---|---|---|
| **empty** | plain 12-fold leave-one-patient-out, **trains on 11** | the main cell-typing tables |
| **populated** (two slide lists) | within-fold leave-one-slide-out, **trains on 5** | the readout-ablation table only |

This is why the readout-ablation table's absolute values are lower than the main typing table's, and
why the paper says the two are not interchangeable.

The `estimator_spec` string is a fixed label in the same way: logistic runs read `C=1.0`, but every
typing TSV carries a `C` column, which records the value actually used (`0.01` for every number in
the paper). Where a run records the split explicitly — for
example `20260816_193412_t2_specmodel_fov112_all_readouts/PROVENANCE.json`, which reads
`plain 12-fold LOPO, trains on 11` — that statement is authoritative.

## 6. Verification

After de-identification, the released files were re-checked against the paper. Every value of the
main cell-typing table was recomputed from these files — both crop widths, all four readouts, the
means and the paired per-fold differences — together with the readout ablation, the second-encoder
comparison, the attention profile, the decision-margin quintiles, the neighborhood-homogeneity
strata and their per-class breakdown, the pooled per-class comparison, the encoder-adaptation table,
the representation-geometry values and the slide-composition table. All reproduce their published
values. The camera-ready additions were checked the same way: the target-masking comparisons, the
cell-only embedding at all three radii, the neighbor-composition probe, the confidence intervals and
equivalence tests, the two-segmenter agreement and the three supplement tables.

Two conventions to note when checking. Paired differences are computed **per fold and then
averaged**, which is what the paper reports; differencing two rounded means from a `verdict_*.tsv`
can disagree in the fourth decimal. And where a `verdict_*.tsv` carries a `best_C` column, the paper
uses the fixed `C = 0.01` stated in its captions, not the best-C value.

Directory names use `Slide_1` rather than `Slide 1` to avoid spaces in paths; file contents use the
paper's form.
