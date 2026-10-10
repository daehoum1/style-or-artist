#!/bin/bash
# Sequential extraction (single GPU). Logged per dataset/encoder. Re-running skips finished caches.
cd "$(dirname "$0")/.."
ENC="clip_b16 dinov2_b14 mae_b16 siglip_b16 csd_l14 vit_b16_in21k openclip_b16 dinov3_b16 clip_l14 dinov2_l14 siglip2_l16 siglip_so400m"
for ds in ${DATASETS:-wikiart multitask_painting}; do
  for e in $ENC; do
    python scripts/05_extract.py $ds $e --workers 20 >> artifacts/logs/extract_${ds}_${e}.log 2>&1 || echo "FAILED $ds $e" >> artifacts/logs/extract_failures.log
  done
done
echo ALLDONE >> artifacts/logs/extract_failures.log
