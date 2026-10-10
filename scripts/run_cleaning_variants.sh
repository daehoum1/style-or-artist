#!/bin/bash
cd "$(dirname "$0")/.."
until [ -f artifacts/embeddings/multitask_painting/aladin_vit/embeddings.npy ]; do sleep 60; done
python scripts/16_cleaning_variants.py > artifacts/logs/cleaning_variants.log 2>&1 || exit 1
ENC="clip_b16 clip_l14 openclip_b16 siglip_b16 siglip2_l16 siglip_so400m dinov2_b14 dinov2_l14 dinov3_b16 mae_b16 vit_b16_in21k csd_l14 aladin_vit"
for v in wikiart_phash8 wikiart_keepconf multitask_painting_phash8 multitask_painting_keepconf; do
  python scripts/04_make_splits.py $v --config default5 >> artifacts/logs/splits_$v.log 2>&1
  for e in $ENC; do
    python scripts/06_style_eval.py $v $e --evaluators knn5,knn_sel,ncc >> artifacts/logs/eval_$v.log 2>&1
  done
done
echo CLEANVAR_DONE >> artifacts/logs/cleanvar_done.log
