#!/bin/bash
# Revision round 3 (R1-C2): predictions for the add-on share sensitivity (kNN-5, NCC)
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
ENC="clip_b16 clip_l14 openclip_b16 siglip_b16 siglip2_l16 siglip_so400m dinov2_b14 dinov2_l14 dinov3_b16 mae_b16 vit_b16_in21k csd_l14 aladin_vit"
run() { for e in $ENC; do python scripts/13_eval_with_predictions.py $1 $e --no_logreg --protocols emf_AG,emf_A,emf_G,emf_none >> artifacts/logs/r3_addon_$1.log 2>&1; done; }
for ds in wikiart multitask_painting; do for f in 50 25 10; do run ${ds}_ao$f & done; done
wait
echo R3_ADDON_DONE >> artifacts/logs/r3.log
