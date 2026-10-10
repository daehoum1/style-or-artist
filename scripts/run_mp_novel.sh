#!/bin/bash
# Revision: replication on MultitaskPainting100k paintings that have no WikiArt duplicate.
cd "$(dirname "$0")/.."
python scripts/03_audit.py mp_novel > artifacts/logs/audit_mp_novel.log 2>&1
python scripts/04_make_splits.py mp_novel > artifacts/logs/splits_mp_novel.log 2>&1 || exit 1
python scripts/04b_split_report.py mp_novel >> artifacts/logs/splits_mp_novel.log 2>&1
python scripts/12_make_emf_splits.py mp_novel >> artifacts/logs/splits_mp_novel.log 2>&1
ENC="clip_b16 clip_l14 openclip_b16 siglip_b16 siglip2_l16 siglip_so400m dinov2_b14 dinov2_l14 dinov3_b16 mae_b16 vit_b16_in21k csd_l14 aladin_vit"
for e in $ENC; do
  until [ -f artifacts/embeddings/multitask_painting/$e/embeddings.npy ]; do sleep 60; done
  python scripts/06_style_eval.py mp_novel $e --evaluators knn5,knn_sel,ncc \
     --protocols iid,artist_disjoint,genre_disjoint,joint,mf_iid,mf_artist,mf_genre,mf_joint,emf_AG,emf_A,emf_G,emf_none >> artifacts/logs/eval_mp_novel.log 2>&1
done
echo MPNOVEL_DONE >> artifacts/logs/mp_novel_done.log
