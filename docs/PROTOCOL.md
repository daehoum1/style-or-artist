# Experimental protocol and decision log

Research question: *Do frozen vision representations recognize artistic style itself, or do they exploit
artist and genre shortcuts?* The hypothesis is treated as falsifiable; every rule below was fixed from
metadata only, before any full model result was inspected. Changes made after that point are logged at the
end with their reason.

## Terminology
style = movement label; artist = creator; genre = the dataset's subject/category metadata (never called
"content" in code); representation = frozen embedding. No encoder is fine-tuned.

## Data cleaning (`src/stylebench/data/cleaning.py`, `configs/cleaning.yaml`)
* SHA-256 over file bytes (exact duplicates) and 64-bit DCT pHash (near duplicates, Hamming <= 4, exact
  search by pigeonhole chunking). Exact and near pairs are merged with union-find into duplicate groups.
* Within a group: conflicting artists -> whole group dropped; conflicting styles -> whole group dropped
  (in WikiArt this is the same painting filed under two style folders); conflicting genres -> one
  representative kept with genre set to missing; otherwise one representative (largest pixel count) kept.
  So no artwork can appear on both sides of any split, and no artwork appears twice inside a split.
* Labels are normalized deterministically (NFKC, `_`->space, whitespace collapse, lowercase). No fuzzy
  merging. Mappings raw -> normalized are written to `artifacts/metadata/<ds>/label_map_*.csv`.
* Known limitation: pHash also links some *different* dark portraits; these are dropped conservatively
  (WikiArt 130, MP100k 172 images in cross-artist groups, part of which are true attribution disputes,
  e.g. Rubens / van Dyck).

### WikiArt specifics
* Labels from `wclasses.csv` (ArtGAN/huggan integer ids). Style id = 140 + index of the sorted style folder
  (verified 100%). Genre id 139 = unlabeled; 129-138 follow the ArtGAN genre order, verified by filename
  keywords (e.g. 76% of files named "portrait" carry 135 = portrait, 82% "still-life" -> 138).
* Artist id 0 is an unlabeled bucket (41,914 images), so the artist is the crawl slug in the file name
  (1,119 artists; the same source prior artist-disjoint work uses).
* 769 file names in the label file use a different mojibake than the files on disk; they are resolved by a
  unique ASCII-only key inside the same folder (all 769 resolved, no collisions).

## Controlled pool and core styles (`configs/splits/default.yaml`)
Pool = cleaned images with style, artist and genre all known. Core style: >= 100 images, >= 10 artists and
top-genre share <= 0.90 (a genre-disjoint split is undefined for a style that lives in one genre: Color
Field, Minimalism, Abstract Expressionism and Action Painting are therefore outside the controlled
comparison — reported, not hidden). All four protocols use the same pool, so gaps are not confounded by
different image populations.

## Split protocols (`src/stylebench/data/splits.py`)
* IID: stratified random image-level split (20% test).
* Artist-D: artists assigned globally (an artist spanning several styles stays on one side) by a
  randomized greedy local search that keeps every style at ~20% test and enforces >= 40 train / 20 test
  images and >= 3 train / 3 test artists per style.
* Genre-D: a held-out genre set G_te is drawn uniformly (per seed) from all sets whose test share is
  15-40% and that leave every core style >= 2x the minimum support on both sides. Exhaustive enumeration for
  <= 12 genres (WikiArt: 10 genres, 153 feasible sets); seeded sample of 200k subsets otherwise (MP100k:
  41 genres). If fewer than 10 sets are feasible, the least-feasible style is removed (metadata only,
  recorded; MP100k: japonism, neoplasticism).
* Joint (Artist+Genre-D): same G_te as Genre-D of that seed; artists then optimized so that test =
  test artists x G_te, train = train artists x train genres; the two cross cells are discarded.
* A style failing support in ANY protocol of a seed is removed from all four protocols of that seed, so the
  four protocols of a seed evaluate the same style set (WikiArt/MP100k default: no style removed).
* Validation is carved from the train side and mirrors the shift: IID random 10%; Artist-D 15% of train
  artists; Genre-D one train genre; Joint = held-out train artists x one train genre (rows mixing a
  validation and a fitting factor are excluded during selection only). Final models are refit on every
  non-test row.

### Matched factorial (MF) control
Standard protocols change the test set too, so a gap mixes "shortcut removed" with "different, possibly
harder test images". MF holds ONE test set (random half of the test-artist x test-genre cell) and
size-matched training sets per style, varying only whether test artists / test genres are available for
training: mf_iid (both seen), mf_artist (artists unseen), mf_genre (genres unseen), mf_joint (both unseen).
Effects = mf_iid - condition, paired on identical test images.

## Leakage assertions (`src/stylebench/data/validate.py`)
Every split file is validated before it is written: unique images; zero artist overlap (artist protocols)
and zero genre overlap (genre protocols) between the whole train side and test; validation factors disjoint
from fitting factors; shared style set; per-style support; no exact duplicate; no near-duplicate pair
(pHash <= 4) across train/test (recomputed). MF: identical test set and identical per-style training sizes
across the four conditions. Tests in `tests/` inject leaks and check that validation fails.

## Evaluation (`configs/experiments/evaluation.yaml`)
Primary: cosine k-NN, k=5 (no tuning; comparable with prior artist-disjoint work). Secondary: k-NN with k
chosen on validation; multinomial logistic regression (L-BFGS, wd chosen on validation); nearest class
centroid. Metrics: accuracy, balanced accuracy, macro-F1 over the styles present; per-class metrics and
confusion matrices saved.

## Probes (`src/stylebench/evaluation/leakage.py`)
Global style / artist (artists with >= 20 pool images, capped at 200 per artist) / genre probes, and
within-style artist (styles with >= 5 artists having >= 15 images) and within-style genre (>= 3 genres with
>= 15 images) probes; 5 seeds of stratified 70/30 splits; logistic probe (wd on an inner validation split)
and kNN-5. Reported as chance-normalized balanced accuracy so that tasks with different class counts are
comparable.

## Analysis (`configs/experiments/analysis.yaml`)
Primary family (Holm across three Spearman tests over encoders): within-style artist leakage vs Artist gap,
within-style genre leakage vs Genre gap, combined within-style leakage vs Joint gap. Also reported: global
probes, relative gaps, partial Spearman controlling for IID accuracy (a stronger encoder may simply have
more to lose), MF effects, Kendall tau between IID and Joint rankings. Correlations are not causal; with
~12 encoders, effect sizes and bootstrap intervals are emphasized over p-values.

## Change log (after the protocol above was fixed)
* 2026-10-08: logistic regression max_iter 300 -> 200 for run time (timing test, test-accuracy change
  <= 0.3 pt); only a one-seed CLIP smoke test had been looked at.
* 2026-10-08: genre-candidate generator made scalable (needed for MP100k, 41 genres). Candidate order
  changed, so the WikiArt splits were regenerated with the final code before any full evaluation.
* 2026-10-08 15:55: logistic-regression grid [1e-6..1e-2] -> [1e-4, 1e-3, 1e-2, 1e-1], max_iter 500, because
  wd <= 1e-5 did not converge (validation predictions agreed only 77-95% with 2000-iteration fits), so model
  selection depended on optimizer details. Only the CLIP-B/16 WikiArt results existed at that point; all
  results were deleted and recomputed with the new setting.
* 2026-10-08 15:55: the minimum-support sensitivity runs use kNN-5 / kNN-sel / NCC only (no logistic
  regression) to fit the compute budget.
* 2026-10-08 16:25 (compute budget, before any full result was kept): logistic regression uses a
  warm-started regularization path (largest -> smallest wd) and a warm-started refit (validated: >= 99.8%
  test-prediction agreement with cold 2000-iteration fits and identical selected wd); it runs on split seeds
  0-4 only, while kNN-5 (primary), kNN-sel and NCC run on all 10 seeds. Probes use 3 seeds instead of 5.
* 2026-10-08 17:25: small logistic problems (n*d*K < 5e7, i.e. the within-style probes) are solved on the CPU instead of the GPU (identical model and minimizer; GPU kernel-launch overhead made them 10-50x slower). Probe results were recomputed for every encoder with this code.
* 2026-10-09 (author decision): labels come from the original releases (WikiArt `wclasses.csv`; MP100k
  original ground-truth CSVs). The earlier custom label files (WikiArt 15 merged styles; MP100k top-20
  styles used in previous projects) are deliberately not used; class sets follow only the metadata rules above.
* 2026-10-09 (author decision): WeART is excluded from the paper. Its public release has no per-image style
  labels (only 6 top-level categories + artists), so it cannot serve as the intended style/artist external
  validation. Runs are kept for provenance in artifacts/excluded_weart/; scripts/09_summary.py no longer
  includes it.
* 2026-10-09: WeART raw data (37 GB) and all derived artifacts (embeddings, metadata, splits, results, reports, logs) deleted at the author's request. The adapter code (src/stylebench/data/weart.py) and scripts/run_weart.sh remain.
* 2026-10-09 (author request): logistic regression restored to all 10 split seeds and probes to 5 seeds; WikiArt and MP100k style evaluation and probes recomputed (kNN/NCC results are unaffected by this change).
* 2026-10-09 (author decision): StyleDecoupler (Jia et al., ICASSP 2026) and ALADIN are not evaluated —
  no released code or weights. Mention in related work. If a reviewer asks, planned re-implementation:
  image-only variant from the paper's own ablation (CLIP-L/14 image feature; DINOv2 aligned to CLIP space by
  a projection head trained on COCO train2017 image-caption pairs; confidence-weighted orthogonal projection
  s = Norm(b - a*(b.c/|c|^2) c), a = max(0, 1 - cos(b, c))); GPT-4o style/content text branch omitted.
* 2026-10-09 (revision round 1, after the original results were known; see paper/RESPONSE_round1.md):
  added ALADIN-ViT (official release v1.0.0, vendored in third_party/aladin) as 13th encoder; exposure-matched
  factorial design (splits.split_exposure_matched, scripts/12); per-image predictions + hierarchical
  (seed x test-artist) bootstrap and a post hoc 5-pt equivalence margin (scripts/13, 18); same-artist
  neighbour intervention (scripts/14); WikiArt/MP100k overlap and the MP-only replication subset mp_novel
  (scripts/15); cleaning-policy variants pHash<=8 and keep-one-copy (scripts/16, 5 seeds, kNN/NCC);
  residual-duplicate audit with exact GPU Hamming distances and DINOv2 cosine (scripts/17); automated claim
  checks (scripts/19). Corrections: split table omitted Joint selection-excluded rows; "no style removed" was
  wrong for MP100k seed 1 (two styles); file count is 160, not 320.
* 2026-10-10 (revision round 2, all added after the results were known; see paper/REVISION2_REPORT.md):
  C1 logit/relative scales of factorial effects; C2 genre dependence of the Joint artist partition
  (scripts/20); C3 leakage-gap with logistic gaps and Joint/style-probe covariates (scripts/25); C4 BCa
  intervals, bootstrap bias, styles that lose all test images keep their point recall (scripts/18); C5
  decomposition of Artist-D->Joint and training-size control (scripts/23); C6 near-duplicate audit of the
  factorial artist-and-genre-seen training images (scripts/22) and effects without flagged test images; C7
  logistic regression for the exposure-matched design (scripts/13 --only_logreg); C8/C9 rank-matched control,
  same-style-only removal and k sweep of the neighbour intervention (scripts/14); C10 Ashton setting on
  MP100k (scripts/26); C11 non-specific artist identifiers (none found); D1 genre-agnostic partition
  (Artist-D artists x Genre-D genres) for both factorial designs (scripts/21); D2 zero-shot text-prompt
  reference (scripts/24). The exposure-matched design became the main factorial design (author decision).
* 2026-10-10 (revision round 3, all added after the results were known; see paper/RESPONSE_round3.md):
  R1-C2 add-on share sensitivity: shared base set subsampled to 50/25/10% (same subsample in all four
  conditions; test set and add-ons unchanged; validate_exposure_matched re-applied), add-on share 8% -> 46-48%
  (scripts/27, run_r3_addon.sh, scripts/18 --tag addon; kNN-5 and NCC only); R2-C2 eligibility of Joint test
  artists for the exposure-matched design and few-/many-genre split of evaluated artists (scripts/28; kNN-5
  recomputed on IID/Artist-D/Joint splits); R2-C4 artist-macro effects, global artist-cluster bootstrap across
  seeds with per-encoder Bonferroni intervals, seed overlap (scripts/29), artist-size terciles (scripts/30).
  No independent style+artist-labelled collection could be obtained (WeART: no style labels; Pandora:
  movement-only; Painting-91: no download link). Moayeri et al. updated to ICLR 2025.
