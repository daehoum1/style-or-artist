#!/bin/bash
# Revision round 1: remaining jobs, run sequentially with bounded threads (avoid CPU oversubscription).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
ENC="clip_b16 clip_l14 openclip_b16 siglip_b16 siglip2_l16 siglip_so400m dinov2_b14 dinov2_l14 dinov3_b16 mae_b16 vit_b16_in21k csd_l14 aladin_vit"
# 1) per-image predictions for MF / EMF (kNN-5, NCC)
for ds in wikiart multitask_painting mp_novel; do
  for e in $ENC; do python scripts/13_eval_with_predictions.py $ds $e --no_logreg >> artifacts/logs/preds_$ds.log 2>&1; done
done
echo PREDS_DONE >> artifacts/logs/revq.log
# 2) neighbour diagnostics / intervention (13 encoders)
for ds in wikiart multitask_painting; do
  python scripts/10_neighbor_diag.py $ds > artifacts/logs/analysis/nd_$ds.log 2>&1
  python scripts/14_neighbor_intervention.py $ds > artifacts/logs/nn_int_$ds.log 2>&1
done
echo NN_DONE >> artifacts/logs/revq.log
# 3) residual duplicate audit
for ds in wikiart multitask_painting; do python scripts/17_residual_duplicate_audit.py $ds > artifacts/logs/dup_audit_$ds.log 2>&1; done
echo DUP_DONE >> artifacts/logs/revq.log
# 4) cleaning-policy variants
python scripts/16_cleaning_variants.py > artifacts/logs/cleaning_variants.log 2>&1
for v in wikiart_phash8 wikiart_keepconf multitask_painting_phash8 multitask_painting_keepconf; do
  [ -f artifacts/splits/$v/split_summary.csv ] || python scripts/04_make_splits.py $v --config default5 >> artifacts/logs/splits_$v.log 2>&1
  for e in $ENC; do python scripts/06_style_eval.py $v $e --evaluators knn5,ncc >> artifacts/logs/eval_$v.log 2>&1; done
done
echo CLEANVAR_DONE >> artifacts/logs/revq.log
