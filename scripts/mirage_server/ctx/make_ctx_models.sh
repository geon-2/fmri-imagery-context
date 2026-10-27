#!/usr/bin/bash
# 맥락 합치기 조건별 모델 폴더를 만든다. 각 폴더에는 원본(LLaVA) 릿지 가중치(image, text, blurry)를 심볼릭 링크로 넣는다(복사하지 않아 디스크 절약).
# 맥락 릿지는 CTX_WEIGHTS 환경변수로 따로 지정한다(run_nb.py). 사용: SUBJ=1 NUM_SESSIONS=40 bash make_ctx_models.sh
set -eo pipefail
source "$(dirname "$0")/../env.sh"
BASE=${BASE_MODEL:-subj0${SUBJ}_${NUM_SESSIONS}sess_server_ridge_sc2_ret16}
for k in image text blurry; do
  [ -s $ROOT/train_logs/$BASE/ridge_${k}_weights.pkl ] || { echo "❌ 없음: $ROOT/train_logs/$BASE/ridge_${k}_weights.pkl"; exit 1; }
done
for arm in ctxblend050 ctxonly100 shufblend050; do
  d=$ROOT/train_logs/subj0${SUBJ}_${NUM_SESSIONS}sess_$arm; mkdir -p $d
  for k in image text blurry; do ln -sfn $ROOT/train_logs/$BASE/ridge_${k}_weights.pkl $d/ridge_${k}_weights.pkl; done
  echo "모델 폴더: $d"; ls -la $d | tail -3
done
for f in ridge_ctx_weights.pkl ridge_ctxshuf_weights.pkl; do
  [ -s $ROOT/train_logs/ctx_T10_subj0${SUBJ}/$f ] || { echo "❌ 맥락 릿지가 없음: $f (ctxtrain 단계 확인)"; exit 1; }
done
[ -s $ROOT/dataset/preprocessed_data/subject${SUBJ}/ctxT10_textstats_${NUM_SESSIONS}sess.pt ] || { echo "❌ 맥락 통계 파일이 없음"; exit 1; }
echo "맥락 가중치와 통계 확인됨"
