#!/bin/bash
# Revision round 2, C7: logistic regression on the exposure-matched factorial design (same selection procedure).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
ENC="clip_b16 clip_l14 openclip_b16 siglip_b16 siglip2_l16 siglip_so400m dinov2_b14 dinov2_l14 dinov3_b16 mae_b16 vit_b16_in21k csd_l14 aladin_vit"
for ds in wikiart multitask_painting; do for e in $ENC; do
  python scripts/13_eval_with_predictions.py $ds $e --only_logreg --protocols emf_AG,emf_A,emf_G,emf_none >> artifacts/logs/c7_$ds.log 2>&1
done; done
echo C7_DONE >> artifacts/logs/rev2.log
