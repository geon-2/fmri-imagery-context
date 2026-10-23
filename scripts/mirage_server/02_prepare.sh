#!/usr/bin/bash
# 코드 클론 + 데이터 다운로드 + 12GB GPU용 메모리 패치. 반드시 u4 노드에서(데이터가 노드 로컬에 저장됨).
set -eo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p $ROOT/dataset/preprocessed_data/subject$SUBJ $ROOT/cache $HF_HOME $RESULTS
cd $WORK
[ -d MIRAGE_repo ] || git clone -q https://github.com/MedARC-AI/MIRAGE.git MIRAGE_repo
rsync -a MIRAGE_repo/ $ROOT/
[ -d $ROOT/src/StableCascade ] || git clone -q https://github.com/Stability-AI/StableCascade.git $ROOT/src/StableCascade
export ROOT SUBJ CAPTION_TYPE
$PY - <<'PYE'
import os
from huggingface_hub import hf_hub_download, snapshot_download
ROOT=os.environ["ROOT"]; SUBJ=int(os.environ["SUBJ"]); CAP=os.environ["CAPTION_TYPE"]
CACHE=f"{ROOT}/cache"; DATA=f"{ROOT}/dataset"
for fn in ["effnet_encoder.safetensors","previewer.safetensors","stage_a.safetensors","stage_b_bf16.safetensors","stage_c_bf16.safetensors","imagenet64-iter-1600000-model-ema.th"]:
    hf_hub_download("reesekneeland/MIRAGE", fn, repo_type="dataset", local_dir=CACHE)
snapshot_download("reesekneeland/umn-imagery", repo_type="dataset", local_dir=DATA, allow_patterns=[
    "nsddata_stimuli/stimuli/imagery_stimuli_18.pt","nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3","nsddata_stimuli/stimuli/imagery_images/*",
    f"preprocessed_data/subject{SUBJ}/nsd_imagery.pt","preprocessed_data/captions_18.npy","nsddata/experiments/nsd/nsd_stim_info_merged.csv",
    f"nsddata/ppdata/subj0{SUBJ}/func1pt8mm/roi/nsdgeneral.nii.gz"])
caps={"coco":"annots_73k.npy","short":"short_length_captions.npy","medium":"mid_length_captions_73K.npy"}
for k in {"coco",CAP}:
    hf_hub_download("reesekneeland/MIRAGE", caps[k], repo_type="dataset", local_dir=f"{DATA}/preprocessed_data")
hf_hub_download("pscotti/mindeyev2", f"betas_all_subj0{SUBJ}_fp32_renorm.hdf5", repo_type="dataset", local_dir=DATA)
hf_hub_download("pscotti/mindeyev2", "coco_images_224_float16.hdf5", repo_type="dataset", local_dir=DATA)
hf_hub_download("pscotti/mindeyev2", "shared1000.npy", repo_type="dataset", local_dir=DATA)
print("download done")
PYE
# 프롬프트 구성요소를 끈 설정에서 노트북이 요구할 수 있는 더미 파일(Colab 노트북과 동일)
$PY - <<PYE
import h5py, numpy as np, os
p="$ROOT/dataset/git_image_features.hdf5"
if not os.path.exists(p):
    with h5py.File(p,"w") as f: f.create_dataset("features", data=np.zeros((1,),np.float32))
PYE
$PY $SCRIPTS/patch_offload.py $ROOT/src/sc_reconstructor.py
df -h /local_datasets | tail -1
