#!/usr/bin/bash
# P1 평균 조건(p1seqblend050) 추가 제출: 구를 하나씩 bigG로 인코딩해 토큰 위치별로 평균한 77x1280을 타깃으로 하는 맥락 릿지(잘림 없음)를 학습하고,
# LLaVA 예측과 0.5로 합쳐 피험자 1에서 시드 1~3을 생성한다. docs/ctx_blend_overnight_plan.md
# 사용(서버의 /data/$USER/mirage_server 에서): bash ctx/submit_ctx_p1.sh [dry] [앞 작업 번호]
#   앞 작업 번호를 주면 그 작업이 끝난 뒤(성공 여부 무관) 시작한다(예: submit_ctx.sh가 낸 마지막 작업 J4). 안 주면 바로 큐에 들어간다(GPU 1장이라 알아서 순서를 기다린다).
cd /data/$USER/mirage_server; mkdir -p logs
DRY=""; AFTER=""; for a in "$@"; do [ "$a" = dry ] && DRY=dry || AFTER=$a; done
CTXW=/local_datasets/$USER/mirage_run/MIRAGE/train_logs/ctx_T10_subj01
CTXS=/local_datasets/$USER/mirage_run/MIRAGE/dataset/preprocessed_data/subject1/ctxP1seq_textstats_40sess.pt
sub(){   # 이름, 의존성, export, 추가 인자...
  local name=$1 dep=$2 exp=$3; shift 3
  if [[ "$DRY" == "dry" ]]; then echo "sbatch --parsable ${dep:+--dependency=$dep} $* --export=$exp job.sbatch   # $name" >&2; echo "DRY"; return; fi
  local id=$(sbatch --parsable ${dep:+--dependency=$dep} "$@" --export=$exp job.sbatch | tr -d ' '); echo "$name -> 작업 $id" >&2; echo $id
}
DEP0=${AFTER:+afterany:$AFTER}
J5=$(sub "J5 P1 평균 맥락 릿지 학습 + 모델 폴더" "$DEP0" "ALL,STAGE=pipeline,SUBJ=1,NUM_SESSIONS=40,STEPS=ctxtrain_ctxmodel,CTX_VARIANTS=p1seq,CTX_ARMS=p1seqblend050" --mem-per-gpu=80G -t 0-02:30:00)
J6=$(sub "J6 P1 평균 합치기 0.5 시드 1~3" "afterok:$J5" "ALL,STAGE=pipeline,SUBJ=1,NUM_SESSIONS=40,USE_RETRIEVAL=0,N_IMG=1,GEN_REP=1,STEPS=recon,MODEL_NAME=subj01_40sess_p1seqblend050,CTX_WEIGHTS=$CTXW/ridge_ctxp1seq_weights.pkl,CTX_STATS=$CTXS,CTX_ALPHA=0.5,SEEDS=1_2_3" -t 0-04:00:00)
echo "제출 끝: $J5 $J6   (확인: squeue -u \$USER)"
