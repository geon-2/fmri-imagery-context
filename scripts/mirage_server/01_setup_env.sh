#!/usr/bin/bash
# conda 환경 설치. 마스터가 아니라 노드에서 실행: srun -p debug_ugrad_advisor_x -w moana-u4 --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G --pty $SHELL
set -eo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p /data/$USER/envs
# 네트워크 확인(노드에서 외부 접속이 막혀 있을 수 있음)
curl -sI https://pypi.org | head -1; curl -sI https://huggingface.co | head -1
if [ ! -x "$CONDA" ]; then
  echo "❌ Private Anaconda가 없습니다: $CONDA"
  echo "   이 서버(NAS)에는 설치가 안 되어 있습니다. 먼저 00_install_anaconda.sh 를 실행하세요(튜토리얼 3, 16~21쪽 절차)."
  exit 1
fi
case "$CONDA" in /data/opt/*) echo "❌ Public Anaconda에는 설치 금지(튜토리얼 3, 15쪽)"; exit 1;; esac
[ -x "$PY" ] || $CONDA create -y -p "$ENV_DIR" python=3.10
$PY -m pip -q install uv
CONS=/data/$USER/envs/constraints.txt
cat > $CONS <<'CEOF'
huggingface-hub>=0.23.2,<1.0
numpy==1.26.3
torch==2.1.0
torchvision==0.16.0
transformers==4.37.2
diffusers==0.30.3
protobuf<5
setuptools<81
CEOF
U="$PY -m uv pip install --python $PY -c $CONS"
# 핵심 패키지(Colab 노트북에서 실측으로 통과한 구성과 같음)
$U matplotlib==3.8.2 tqdm scikit-image==0.22.0 accelerate==0.24.1 webdataset==0.2.73 pandas==2.2.0 "einops>=0.7.0" ftfy regex \
   kornia==0.7.1 h5py==3.10.0 open_clip_torch torch==2.1.0 torchvision==0.16.0 transformers==4.37.2 xformers==0.0.22.post7 \
   torchmetrics==1.3.0.post0 diffusers==0.30.3 omegaconf==2.3.0 pytorch-lightning==2.0.1 sentence-transformers==2.5.1 \
   nibabel==5.2.1 munch==4.0.0 "setuptools<81" safetensors timm scipy pillow ipykernel jupyter nbconvert nbformat scikit-learn pyyaml psutil
$U git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git || echo "WARN gradual-warmup"
$U git+https://github.com/pabloppp/pytorch-tools || echo "WARN pytorch-tools"
$U --no-deps git+https://github.com/openai/CLIP.git || echo "WARN CLIP"
$U dalle2-pytorch || echo "WARN dalle2"
$U insightface==0.7.3 opencv-python==4.8.1.78 onnxruntime==1.16.3 onnx2torch==1.5.13 || echo "WARN insightface"   # StableCascade/modules/controlnet.py가 import함(Colab 노트북에서 설치했던 줄, 서버 스크립트에서 빠뜨렸음)
$U wandb==0.16.2 evaluate==0.4.1 nltk==3.8.1 rouge_score==0.1.2 opencv-python==4.8.1.78 || echo "WARN optional"
$U "huggingface-hub>=0.23.2,<1.0" "protobuf<5"
# 커널 스펙을 /home이 아니라 환경(NAS) 안에 설치: 노드의 홈에 의존하지 않음
$PY -m ipykernel install --prefix "$ENV_DIR" --name mei-env --display-name "mei-env (py3.10)"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
