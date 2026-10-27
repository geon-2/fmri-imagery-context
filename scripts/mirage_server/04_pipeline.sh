#!/usr/bin/bash
# 한 작업 안에서 단계를 이어서 실행(sbatch 작업은 STAGE=pipeline 으로 호출). 환경변수: SUBJ, NUM_SESSIONS, USE_RETRIEVAL, N_IMG, SEEDS(밑줄 구분 1_2_3), STEPS(밑줄 구분)
# STEPS: prepare(코드/데이터 다운로드와 패치) train(학습) recon(시드마다 생성) clean(이 피험자의 임베딩/가중치/베타 삭제로 디스크 확보)
#        ctxtrain(맥락 릿지 학습) ctxmodel(맥락 조건별 모델 폴더 구성)  — 맥락 합치기 실험, docs/ctx_blend_overnight_plan.md
set -eo pipefail
source "$(dirname "$0")/env.sh"
STEPS=${STEPS:-prepare_train_recon_clean}; SEEDS=${SEEDS:-1}
echo "== pipeline | subj $SUBJ | $NUM_SESSIONS sess | retrieval=$USE_RETRIEVAL N_IMG=$N_IMG | seeds=$SEEDS | steps=$STEPS | $(date)"
for s in ${STEPS//_/ }; do
  echo "---- step: $s | $(date)"
  case $s in
    prepare) bash $SCRIPTS/02_prepare.sh ;;
    train)   STAGE=train LEAN_RIDGE=1 bash $SCRIPTS/03_run.sh ;;
    recon)   for sd in ${SEEDS//_/ }; do STAGE=recon SEED=$sd bash $SCRIPTS/03_run.sh; done ;;
    ctxtrain) export CTX_JSON=${CTX_JSON:-$SCRIPTS/ctx/context_T10_by_nsdid.json} CTX_OUT=${CTX_OUT:-$ROOT/train_logs/ctx_T10_subj0${SUBJ}}
             PYTHONNOUSERSITE=1 $PY $SCRIPTS/ctx/train_ctx_ridge.py ;;     # 맥락 릿지(+셔플 대조) 학습: ctx/train_ctx_ridge.py
    ctxmodel) bash $SCRIPTS/ctx/make_ctx_models.sh ;;                       # 조건별 모델 폴더(LLaVA 릿지 심볼릭 링크) 만들기
    clean)   rm -rf $ROOT/train_logs/$MODEL_NAME $ROOT/dataset/preprocessed_data/subject$SUBJ/*_embeddings_* $ROOT/dataset/betas_all_subj0${SUBJ}_fp32_renorm.hdf5
             df -h /local_datasets | tail -1 ;;
    *) echo "알 수 없는 단계 $s"; exit 1 ;;
  esac
done
echo "== pipeline 끝 $(date)"
