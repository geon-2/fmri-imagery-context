#!/usr/bin/bash
# 서버 디스크 점검(읽기 전용: du, ls, df만 쓴다. 아무것도 지우거나 바꾸지 않는다). moana-u4 노드 안에서 실행한다.
# 목적: 어떤 파일이 얼마나 차지하고, 다음 생성에 다시 쓰는지(유지) 재생성하거나 백업이 있는지(삭제 후보) 구분해서 보여 준다. 판단과 삭제는 사람이 한다.
# 사용: bash ctx/ctx_disk_inventory.sh
source "$(dirname "$0")/../env.sh"
sz(){ if [ -e "$1" ]; then du -shL "$1" 2>/dev/null | awk '{print $1}'; else echo "-"; fi; }
row(){ printf "%8s  %-12s %s\n" "$(sz "$1")" "$2" "$1"; }
echo "== 디스크(/local_datasets, 다른 사용자와 같은 디스크)"; df -h /local_datasets | tail -1
echo; echo "== 내 노드 로컬 사용량($USER)"; du -sh /local_datasets/$USER 2>/dev/null
echo "-- 최상위(큰 순)"; du -sh /local_datasets/$USER/* 2>/dev/null | sort -h | tail -12
echo; echo "== MIRAGE 작업 폴더 안(큰 순 15개)"; du -h --max-depth=2 $ROOT 2>/dev/null | sort -h | tail -15

echo; echo "== 모델 가중치 폴더 train_logs (파일별)"
for d in $ROOT/train_logs/*/; do echo "-- $d"; ls -l --block-size=M "$d" 2>/dev/null | awk 'NR>1{printf "   %8s  %s\n",$5,$9}'; done
echo; echo "== 데이터셋 폴더 dataset (최상위)"; for f in $ROOT/dataset/*; do row "$f" "-"; done
echo; echo "== 전처리 폴더 preprocessed_data (피험자별 큰 파일 8개)"
for d in $ROOT/dataset/preprocessed_data/subject*/; do echo "-- $d"; ls -l --block-size=M "$d" 2>/dev/null | sort -k5 -n -r | head -8 | awk '{printf "   %8s  %s\n",$5,$9}'; done
echo; echo "== 노트북 실행 결과와 생성 결과(서버 로컬 사본)"
for f in $ROOT/src/evals/*; do row "$f" "백업확인"; done
for f in $ROOT/src/*_out.ipynb; do row "$f" "-"; done
echo; echo "== 캐시(재다운로드 가능)"
for f in $HF_HOME $XDG_CACHE_HOME $TORCH_HOME $PIP_CACHE_DIR $UV_CACHE_DIR $WORK/MIRAGE_repo; do row "$f" "-"; done
echo; echo "== NAS 결과 백업($RESULTS): 조건 폴더별 크기와 파일 수"
for f in $RESULTS/*/; do printf "%8s  %3s개  %s\n" "$(sz "$f")" "$(ls "$f" 2>/dev/null | wc -l)" "$f"; done
echo; echo "== NAS(/data/$USER) 최상위"; du -sh /data/$USER/* 2>/dev/null | sort -h | tail -8
echo; echo "(참고: 이 스크립트는 읽기 전용이다. 지우는 명령은 이 결과를 보고 따로 정한다.)"
