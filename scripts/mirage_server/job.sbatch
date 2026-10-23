#!/usr/bin/bash
#SBATCH -J mirage
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -w moana-u4
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out
# 제출: sbatch --export=ALL,STAGE=train job.sbatch   (STAGE: smoke|train|recon|viz|all). 제출 위치 아래에 logs/ 폴더가 있어야 함.
cd "$(dirname "$(readlink -f "$0")")"
bash 03_run.sh
