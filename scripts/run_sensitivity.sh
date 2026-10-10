#!/bin/bash
# Minimum-support sensitivity: style evaluation on the lenient / strict split variants (5 seeds each).
cd "$(dirname "$0")/.."
ENC="clip_b16 dinov2_b14 mae_b16 siglip_b16 csd_l14 vit_b16_in21k openclip_b16 dinov3_b16 clip_l14 dinov2_l14 siglip2_l16 siglip_so400m"
for ds in wikiart multitask_painting; do
  for v in lenient strict; do
    for e in $ENC; do
      until [ -f artifacts/embeddings/$ds/$e/embeddings.npy ]; do sleep 60; done
      [ -f artifacts/results/style_eval/${ds}_$v/$e.parquet ] && continue
      python scripts/06_style_eval.py $ds $e --split_root ${ds}_$v --evaluators knn5,knn_sel,ncc >> artifacts/logs/sens_${ds}_$v.log 2>&1
    done
  done
done
echo SENS_DONE >> artifacts/logs/sens_done.log
