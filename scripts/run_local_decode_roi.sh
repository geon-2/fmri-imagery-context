#!/bin/bash
# 영역별 brain decoding(영역당 복셀 400개 x 5회), 피험자 4명, MPNet. 끊겨도 다시 실행하면 끝난 조합은 건너뛴다.
cd /Users/gunlee/Development/fmri-imagery-context/src/fmri || exit 1
export PYTHONPATH=../labeling
R=../../data
st() { echo "[$(date '+%F %T')] $*"; }
for S in subj01 subj02 subj05 subj07; do
  st "$S MPNet 영역별 decoding"
  python3 -u axes_decode.py --roi-mode --fmri-dir $R/stage/fmri_$S --emb-dir $R/embeddings_v2 --model MPNet --out-dir $R/results/axes_decode --tag $S --skip-existing
done
st "전체 완료"
