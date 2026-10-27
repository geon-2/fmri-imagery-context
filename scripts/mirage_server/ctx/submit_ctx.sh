#!/usr/bin/bash
# 맥락 합치기 실험(피험자 1, 40세션, retrieval 없음) 야간 제출. 작업을 순서대로 잇는다. docs/ctx_blend_overnight_plan.md
# 사용(서버의 /data/$USER/mirage_server 에서): bash ctx/submit_ctx.sh [dry]   (dry면 제출하지 않고 명령만 출력)
# 순서: J0 맥락 릿지 학습(ctx, shuf)과 모델 폴더 구성 → J1 합치기 0.5 시드 1~3 → J2 셔플 대조(0.5) 시드 1~3 → J3 맥락만(1.0) 시드 1~3 → J4 합치기 0.5 시드 4~5
# J0가 실패하면 뒤 작업은 모두 취소된다(afterok). 생성 작업 하나가 실패해도 다음 작업은 이어서 돈다(afterany).
cd /data/$USER/mirage_server; mkdir -p logs; DRY=$1
CTXW=/local_datasets/$USER/mirage_run/MIRAGE/train_logs/ctx_T10_subj01
CTXS=/local_datasets/$USER/mirage_run/MIRAGE/dataset/preprocessed_data/subject1/ctxT10_textstats_40sess.pt
COMMON="ALL,STAGE=pipeline,SUBJ=1,NUM_SESSIONS=40,USE_RETRIEVAL=0,N_IMG=1,GEN_REP=1,STEPS=recon"
sub(){   # 이름, 의존성, export, 추가 인자...
  local name=$1 dep=$2 exp=$3; shift 3
  if [[ "$DRY" == "dry" ]]; then echo "sbatch --parsable ${dep:+--dependency=$dep} $* --export=$exp job.sbatch   # $name" >&2; echo "DRY"; return; fi
  local id=$(sbatch --parsable ${dep:+--dependency=$dep} "$@" --export=$exp job.sbatch | tr -d ' '); echo "$name -> 작업 $id" >&2; echo $id
}
J0=$(sub "J0 맥락 릿지 학습 + 모델 폴더" "" "ALL,STAGE=pipeline,SUBJ=1,NUM_SESSIONS=40,STEPS=ctxtrain_ctxmodel" --mem-per-gpu=80G -t 0-03:00:00)
J1=$(sub "J1 합치기 0.5 시드 1~3" "afterok:$J0" "$COMMON,MODEL_NAME=subj01_40sess_ctxblend050,CTX_WEIGHTS=$CTXW/ridge_ctx_weights.pkl,CTX_STATS=$CTXS,CTX_ALPHA=0.5,SEEDS=1_2_3" -t 0-04:00:00)
J2=$(sub "J2 셔플 대조(0.5) 시드 1~3" "afterok:$J0,afterany:$J1" "$COMMON,MODEL_NAME=subj01_40sess_shufblend050,CTX_WEIGHTS=$CTXW/ridge_ctxshuf_weights.pkl,CTX_STATS=$CTXS,CTX_ALPHA=0.5,SEEDS=1_2_3" -t 0-04:00:00)
J3=$(sub "J3 맥락만(1.0) 시드 1~3" "afterok:$J0,afterany:$J2" "$COMMON,MODEL_NAME=subj01_40sess_ctxonly100,CTX_WEIGHTS=$CTXW/ridge_ctx_weights.pkl,CTX_STATS=$CTXS,CTX_ALPHA=1.0,SEEDS=1_2_3" -t 0-04:00:00)
J4=$(sub "J4 합치기 0.5 시드 4~5" "afterok:$J0,afterany:$J3" "$COMMON,MODEL_NAME=subj01_40sess_ctxblend050,CTX_WEIGHTS=$CTXW/ridge_ctx_weights.pkl,CTX_STATS=$CTXS,CTX_ALPHA=0.5,SEEDS=4_5" -t 0-03:00:00)
echo "제출 끝: $J0 $J1 $J2 $J3 $J4   (확인: squeue -u \$USER)"
