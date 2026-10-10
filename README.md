# style-or-artist

Code, split files and results for

> **How Much of Painting Style Recognition Is Artist Recognition? A Controlled Evaluation of Frozen Visual Encoders**
> Yoorim Kim, Jungyeob Han, Daeho Um

The study evaluates thirteen frozen visual encoders for painting-style classification on WikiArt and
MultitaskPainting100k under four leakage-verified split protocols (IID, artist-disjoint, genre-disjoint,
and jointly artist- and genre-disjoint), an exposure-matched factorial design that compares training
conditions on fixed test images, a genre-agnostic artist partition, a neighbour intervention, a zero-shot
reference and leakage probes. No encoder is fine-tuned.

The full design, every protocol decision and a dated change log are in [`docs/PROTOCOL.md`](docs/PROTOCOL.md);
dataset requirements are in [`docs/DATASETS.md`](docs/DATASETS.md).

## Contents

```
configs/      datasets/  splits/  encoders/encoders.yaml  experiments/  cleaning.yaml
src/stylebench/
  data/       dataset adapters, hashing, cleaning, audit, split construction and leakage validation
  models/     encoder wrappers (OpenCLIP/CLIP/SigLIP, timm models, CSD, ALADIN-ViT)
  features/   embedding extraction and cache
  evaluation/ GPU kNN, logistic regression and nearest-centroid classifiers, metrics, probes, statistics
  analysis/   correlation analyses
scripts/      numbered pipeline (01-30) and the run_*.sh drivers used for the paper
tests/        leakage-injection, determinism and factorial-design tests
splits/       every split file of the paper in packed, lossless form (restore with scripts/unpack_splits.py)
artifacts/    reports/ (audits, split validation), final/ (result tables, factorial effects with intervals),
              results/ (evaluations, probes, intervention, audits), metadata/ (cleaning decisions, label maps)
```

Images, embeddings and per-image kNN/centroid predictions are not redistributed.

## Setup

```bash
pip install -r requirements.txt          # Python 3.10; tested with torch 2.11 + CUDA 12.8
python -m pytest -q tests
python scripts/unpack_splits.py           # restores artifacts/splits/ (1,080 files) and verifies their SHA-256
bash scripts/get_aladin.sh                # official ALADIN-ViT code and weights (only needed for that encoder)
```

Place the datasets under `data/wikiart` and `data/multitask_painting` (see `docs/DATASETS.md`), or edit
`root:` in `configs/datasets/*.yaml`. Run all scripts from the repository root.

## Reproducing the results

**Using the released splits.** After `unpack_splits.py`, embeddings can be extracted and every evaluation
rerun on exactly the splits of the paper:

```bash
python scripts/01_hash_images.py wikiart && python scripts/02_clean.py wikiart   # rebuilds image paths
python scripts/05_extract.py wikiart clip_b16 dinov2_b14 ...                   # cached embeddings
python scripts/06_style_eval.py wikiart clip_b16 ...                           # standard protocols
python scripts/07_probes.py wikiart clip_b16 ...                               # leakage probes
python scripts/13_eval_with_predictions.py wikiart clip_b16 ...                # factorial designs
python scripts/18_factorial_analysis.py                                        # factorial effects + intervals
```

`02_clean.py` is deterministic; `artifacts/metadata/D/cleaning_decisions.csv` can be used to check that the
same images are retained. The restored split files are byte-identical to the originals (with the pinned
pandas/pyarrow versions), so the split-file hashes stored with every result can be checked.

**From scratch.** The complete pipeline, per dataset `D` in `wikiart`, `multitask_painting`:

| step | script |
|---|---|
| hashing, cleaning, audit | `01_hash_images.py D`, `02_clean.py D`, `03_audit.py D` |
| standard protocols (10 seeds), validation reports | `04_make_splits.py D`, `04b_split_report.py D` |
| embeddings | `05_extract.py D <encoders>` |
| standard evaluation, probes, analysis | `06_style_eval.py`, `07_probes.py`, `08_analyze.py`, `09_summary.py` |
| exposure-matched and genre-agnostic factorial splits | `12_make_emf_splits.py D`, `21_make_ga_factorial.py D` |
| per-image predictions, factorial analysis | `13_eval_with_predictions.py`, `18_factorial_analysis.py` |
| neighbour intervention, zero-shot reference | `14_neighbor_intervention.py`, `24_zero_shot.py` |
| dataset overlap, cleaning variants, duplicate audits | `15_*`, `16_*`, `17_*`, `22_*` |
| selection bias, decomposition, leakage re-analysis, Ashton setting | `20_*`, `23_*`, `25_*`, `26_*` |
| add-on share, eligibility, statistical sensitivity | `27_*`, `28_*`, `29_*`, `30_*` |

The `run_*.sh` files are the drivers used to produce the paper's results. `11_paper_assets.py` and
`19_verify_claims.py` generate the manuscript tables and figures and check every number in the text; they
require the manuscript sources, which are not part of this repository.

Every result row stores the hash of its split file, `artifacts/metadata/D/cleaning_stats.json` the version
of the cleaned metadata, and `*_provenance.json` the encoder checkpoint and evaluation configuration.

## Licence

Code: MIT (see `LICENSE`). Split files and results contain labels derived from WikiArt and
MultitaskPainting100k and are provided for non-commercial research, subject to the terms of those
datasets. ALADIN-ViT code and weights are downloaded from their original repository and are subject to its
terms.

## Citation

The citation will be added upon publication.
