#!/usr/bin/bash
# 같은 모델로 시드만 바꿔 생성을 반복(작업을 순서대로 이어서 제출). 사용: bash submit_seeds.sh <세션수> "<시드들>" [retrieval 후보수]
# 예: bash submit_seeds.sh 40 "1 2 3 4"        (retrieval 없음)   /  bash submit_seeds.sh 10 "1 2" 16   (retrieval 후보 16)
NS=$1; SEEDS=$2; RET=${3:-0}; cd /data/$USER/mirage_server; mkdir -p logs; prev=""
for sd in $SEEDS; do
  EXP="ALL,STAGE=recon,NUM_SESSIONS=$NS,SEED=$sd"
  [[ $RET != 0 ]] && EXP="$EXP,USE_RETRIEVAL=1,N_IMG=$RET"
  dep=""; [[ -n "$prev" ]] && dep="--dependency=afterany:$prev"
  prev=$(sbatch --parsable $dep --export=$EXP job.sbatch | tr -d ' '); echo "시드 $sd -> 작업 $prev"
done
