#!/bin/bash
# Revision round 1: add ALADIN-ViT (13th encoder): extraction, evaluation, probes, kNN sensitivity runs.
cd "$(dirname "$0")/.."
for ds in wikiart multitask_painting; do
  python scripts/05_extract.py $ds aladin_vit --workers 20 >> artifacts/logs/extract_${ds}_aladin_vit.log 2>&1
done
python scripts/run_eval_loop.py wikiart >> artifacts/logs/loop_aladin_wa.log 2>&1 &
python scripts/run_eval_loop.py multitask_painting >> artifacts/logs/loop_aladin_mp.log 2>&1 &
for ds in wikiart multitask_painting; do for v in lenient strict; do
  python scripts/06_style_eval.py $ds aladin_vit --split_root ${ds}_$v --evaluators knn5,knn_sel,ncc >> artifacts/logs/sens_${ds}_$v.log 2>&1
done; done
wait
echo ALADIN_DONE >> artifacts/logs/aladin_done.log
