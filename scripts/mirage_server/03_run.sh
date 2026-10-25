#!/usr/bin/bash
# 사용: STAGE=smoke|train|recon|viz|all bash 03_run.sh   (sbatch job.sbatch 가 호출)
set -eo pipefail
source "$(dirname "$0")/env.sh"
export PYTHONNOUSERSITE=1
cd $ROOT/src
STAGE=${STAGE:-all}
echo "== $STAGE | $MODEL_NAME | host $(hostname) | $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY $SCRIPTS/deps_probe.py   # 누락 모듈 자동 설치(최초 1회는 설치가 일어날 수 있음)
if [[ $STAGE == smoke ]]; then
  $PY - <<PYE
import sys, os, time, torch
sys.path.insert(0,"$ROOT/src"); sys.path.insert(0,"$ROOT/src/StableCascade"); os.chdir("$ROOT/src")
from PIL import Image
from sc_reconstructor import SC_Reconstructor
t0=time.time(); rec=SC_Reconstructor(device="cuda:0", cache_dir="$ROOT/cache", embedder_only=False, compile_models=False)
print("로드", round(time.time()-t0),"초, GPU GB", round(torch.cuda.max_memory_allocated()/1e9,1))
for k in (6,7):
    im=Image.open("$ROOT/dataset/nsddata_stimuli/stimuli/imagery_images/%d.png"%k).convert("RGB")
    c_i=rec.embed_image(im); c_t=rec.embed_text(["a photo"]); t1=time.time()
    out=rec.reconstruct(c_i=c_i,c_t=c_t,n_samples=1,strength=1.0,num_steps_c=10,num_steps_b=5)
    print("자극",k,"생성",round(time.time()-t1),"초 GPU 최대 GB",round(torch.cuda.max_memory_allocated()/1e9,1),"NaN:",bool(torch.isnan(out).any()))
    Image.fromarray((out[0].float().cpu().permute(1,2,0).numpy()*255).astype("uint8")).save("$RESULTS/smoke_%d.png"%k)
PYE
fi
[[ $STAGE == train || $STAGE == all ]] && { $PY $SCRIPTS/run_nb.py train; $PY $SCRIPTS/link_embeddings.py; ls $ROOT/train_logs/$MODEL_NAME; }
[[ $STAGE == recon || $STAGE == all ]] && { $PY $SCRIPTS/link_embeddings.py; $PY $SCRIPTS/run_nb.py recon
  if [[ -n "$SEED" ]]; then   # 시드를 바꿔 반복 생성: 결과 파일에 시드를 붙여 서로 덮어쓰지 않게 NAS로 복사
    E=$ROOT/src/evals/$MODEL_NAME; mkdir -p $RESULTS/$MODEL_NAME
    for f in $E/*_all_recons_imagery.pt $E/*_all_blurryrecons_imagery.pt; do cp $f $RESULTS/$MODEL_NAME/$(basename $f .pt)_seed$SEED.pt; done
    echo "시드 $SEED 결과 저장: $RESULTS/$MODEL_NAME"; exit 0
  fi; }
[[ $STAGE == viz || $STAGE == recon || $STAGE == all ]] && { cp -r $ROOT/src/evals/$MODEL_NAME $RESULTS/ 2>/dev/null || true; $PY $SCRIPTS/viz.py; }   # 결과 복사를 먼저(viz가 실패해도 결과는 NAS에 남김)
[[ $STAGE == pipeline ]] && bash $SCRIPTS/04_pipeline.sh
echo "== 끝 $(date)"
