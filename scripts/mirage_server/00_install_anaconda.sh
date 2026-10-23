#!/usr/bin/bash
# Private Anaconda 설치(튜토리얼 3, 16~21쪽): 반드시 NAS(/data/$USER/anaconda3)에. /home 금지. 노드에서 실행(srun)하세요.
# 사용: bash 00_install_anaconda.sh [설치파일 URL]   (기본 URL은 제가 확인하지 못했습니다. 안 되면 anaconda.com/download 에서 링크를 복사해 인자로 주세요)
set -eo pipefail
URL=${1:-https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh}
mkdir -p /data/$USER/downloads && cd /data/$USER/downloads
[ -f installer.sh ] || wget -q -O installer.sh "$URL"
[ -x /data/$USER/anaconda3/bin/conda ] || bash installer.sh -b -p /data/$USER/anaconda3   # -b: 질문 없이 설치, -p: NAS 경로
/data/$USER/anaconda3/bin/conda --version
echo "※ ~/.bashrc 의 conda init 은 노드마다 /home(노드 로컬)이라 따로 필요하지만, 이 스크립트들은 conda를 절대경로로 부르므로 필요 없습니다."
