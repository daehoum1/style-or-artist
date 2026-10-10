#!/bin/bash
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
ENC="clip_b16 clip_l14 openclip_b16 siglip_b16 siglip2_l16 siglip_so400m dinov2_b14 dinov2_l14 dinov3_b16 mae_b16 vit_b16_in21k csd_l14 aladin_vit"
for ds in wikiart_ga multitask_painting_ga; do for e in $ENC; do
  python scripts/13_eval_with_predictions.py $ds $e --no_logreg >> artifacts/logs/d1_$ds.log 2>&1
done; done
echo D1_DONE >> artifacts/logs/rev2.log
