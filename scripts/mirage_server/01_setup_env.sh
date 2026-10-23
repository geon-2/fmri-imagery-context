#!/usr/bin/bash
# conda 환경 설치. 마스터가 아니라 노드에서 실행: srun -p debug_ugrad_advisor_x -w moana-u4 --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G --pty $SHELL
set -eo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p /data/$USER/envs
[ -x "$PY" ] || /data/$USER/anaconda3/bin/conda create -y -p "$ENV_DIR" python=3.10
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
$U wandb==0.16.2 evaluate==0.4.1 nltk==3.8.1 rouge_score==0.1.2 opencv-python==4.8.1.78 || echo "WARN optional"
$U "huggingface-hub>=0.23.2,<1.0" "protobuf<5"
$PY -m ipykernel install --user --name mei-env --display-name "mei-env (py3.10)"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
