#!/usr/bin/bash
# 맥락 합치기 조건별 모델 폴더를 만든다. 각 폴더에는 원본(LLaVA) 릿지 가중치(image, text, blurry)를 심볼릭 링크로 넣는다(복사하지 않아 디스크 절약).
# 맥락 릿지는 CTX_WEIGHTS 환경변수로 따로 지정한다(run_nb.py). 사용: SUBJ=1 bash make_ctx_models.sh   (조건 선택: CTX_ARMS="p1seqblend050" 등, 기본은 T10 세 조건)
set -eo pipefail
source "$(dirname "$0")/../env.sh"
export NUM_SESSIONS=40   # 이 실험은 40세션 고정(env.sh 기본값 10으로 잘못 찾는 일을 막는다)
BASE=${BASE_MODEL:-subj0${SUBJ}_${NUM_SESSIONS}sess_server_ridge_sc2}   # 기준선 no-retrieval 재구성을 만든 바로 그 가중치(비교가 같은 가중치 위에서 이뤄진다)
ARMS=${CTX_ARMS:-"ctxblend050 ctxonly100 shufblend050"}
PRE=$ROOT/dataset/preprocessed_data/subject${SUBJ}; W=$ROOT/train_logs/ctx_T10_subj0${SUBJ}
for k in image text blurry; do
  [ -s $ROOT/train_logs/$BASE/ridge_${k}_weights.pkl ] || { echo "❌ 없음: $ROOT/train_logs/$BASE/ridge_${k}_weights.pkl"; exit 1; }
done
for arm in $ARMS; do
  case $arm in   # 조건 -> 필요한 맥락 릿지 가중치, 정규화 통계
    ctxblend050|ctxonly100) need="$W/ridge_ctx_weights.pkl $PRE/ctxT10_textstats_${NUM_SESSIONS}sess.pt" ;;
    shufblend050)           need="$W/ridge_ctxshuf_weights.pkl $PRE/ctxT10_textstats_${NUM_SESSIONS}sess.pt" ;;
    p1seqblend050)          need="$W/ridge_ctxp1seq_weights.pkl $PRE/ctxP1seq_textstats_${NUM_SESSIONS}sess.pt" ;;
    *) echo "❌ 알 수 없는 조건: $arm"; exit 1 ;;
  esac
  for f in $need; do [ -s $f ] || { echo "❌ $arm 에 필요한 파일이 없음: $f"; exit 1; }; done
  d=$ROOT/train_logs/subj0${SUBJ}_${NUM_SESSIONS}sess_$arm; mkdir -p $d
  for k in image text blurry; do ln -sfn $ROOT/train_logs/$BASE/ridge_${k}_weights.pkl $d/ridge_${k}_weights.pkl; done
  echo "모델 폴더: $d (필요한 맥락 파일 확인됨)"; ls -la $d | tail -3
done
