#!/usr/bin/bash
# 재현 최종 실행: A(피험자 1, 40세션, retrieval 후보 16, 반복 3회) + B(피험자 2, 5, 7, 40세션, retrieval 없음, 반복 5회). 작업을 순서대로 이어서 제출한다.
# 사용: bash submit_AB.sh [dry]   (dry면 제출하지 않고 명령만 출력)
cd /data/$USER/mirage_server; mkdir -p logs; prev=""
sub(){   # 이름, export 문자열, sbatch 추가 인자...
  local name=$1 exp=$2; shift 2; local dep=""; [[ -n "$prev" ]] && dep="--dependency=afterany:$prev"
  if [[ "$DRY" == "dry" ]]; then echo "sbatch --parsable $dep $* --export=$exp job.sbatch   # $name"; prev=DRY; return; fi
  prev=$(sbatch --parsable $dep "$@" --export=$exp job.sbatch | tr -d ' '); echo "$name -> 작업 $prev"
}
DRY=$1
for s in 2 5 7; do
  sub "B 피험자 $s (준비, 학습, 생성 5회, 정리)" "ALL,STAGE=pipeline,SUBJ=$s,NUM_SESSIONS=40,SEEDS=1_2_3_4_5,STEPS=prepare_train_recon_clean" -t 0-10:00:00
done
sub "A 학습(피험자 1, 40세션, retrieval)" "ALL,STAGE=pipeline,SUBJ=1,NUM_SESSIONS=40,USE_RETRIEVAL=1,N_IMG=16,STEPS=prepare_train" --mem-per-gpu=100G -t 0-05:00:00
for sd in 1 2 3; do
  sub "A 생성 시드 $sd (후보 16)" "ALL,STAGE=pipeline,SUBJ=1,NUM_SESSIONS=40,USE_RETRIEVAL=1,N_IMG=16,SEEDS=$sd,STEPS=recon" -t 0-14:00:00
done
