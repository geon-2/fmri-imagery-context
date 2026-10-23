# 공통 변수. 각 스크립트가 source 한다. (학교 Seraph/moana 규칙: 코드·환경은 /data, 큰 데이터·학습 산출물은 노드 로컬 /local_datasets)
export ENV_DIR=/data/$USER/envs/mei-env                 # conda 환경(NAS, 모든 노드에서 보임)
export PY=$ENV_DIR/bin/python
export WORK=/local_datasets/$USER/mirage_run            # 노드 로컬(해당 노드에서만 보임: 항상 같은 노드 -w moana-u4 에서 실행)
export ROOT=$WORK/MIRAGE
export HF_HOME=/local_datasets/$USER/hf                 # 홈(~/.cache)에 큰 파일이 쌓이지 않게
export RESULTS=/data/$USER/mirage_results               # 작은 결과(그림, 표)만 NAS로
export SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 실험 설정 (환경변수로 덮어쓸 수 있음)
export SUBJ=${SUBJ:-1}
export NUM_SESSIONS=${NUM_SESSIONS:-10}                 # RAM 32GB 한도 때문에 10부터
export CAPTION_TYPE=${CAPTION_TYPE:-medium}
export USE_RETRIEVAL=${USE_RETRIEVAL:-0}                # 1이면 retrieval pool(논문 구성요소), N_IMG는 8의 배수여야 함
export N_IMG=${N_IMG:-1}
export GEN_REP=${GEN_REP:-1}
export MODEL_NAME=${MODEL_NAME:-subj0${SUBJ}_${NUM_SESSIONS}sess_server_ridge_sc2}
# 튜토리얼 5·6: /home(~/.cache 등)에 큰 파일 금지 -> 모든 캐시를 노드 로컬로
export XDG_CACHE_HOME=/local_datasets/$USER/xdg_cache
export TORCH_HOME=/local_datasets/$USER/torch
export PIP_CACHE_DIR=/local_datasets/$USER/pip_cache
export UV_CACHE_DIR=/local_datasets/$USER/uv_cache
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" 2>/dev/null || true
# Private Anaconda 경로(튜토리얼 3). Public(/data/opt/anaconda3)에는 절대 설치하지 않는다.
export CONDA=/data/$USER/anaconda3/bin/conda
