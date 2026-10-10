#!/bin/bash
cd "$(dirname "$0")/.."
ENC="clip_b16 clip_l14 openclip_b16 siglip_b16 siglip2_l16 siglip_so400m dinov2_b14 dinov2_l14 dinov3_b16 mae_b16 vit_b16_in21k csd_l14 aladin_vit"
for ds in wikiart multitask_painting; do
  for e in $ENC; do
    until [ -f artifacts/embeddings/$ds/$e/embeddings.npy ]; do sleep 60; done
    python scripts/13_eval_with_predictions.py $ds $e >> artifacts/logs/preds_$ds.log 2>&1
  done
done
echo PREDS_DONE >> artifacts/logs/preds_done.log
