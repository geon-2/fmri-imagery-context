#!/usr/bin/bash
# 맥락 합치기 실험(제출 전) 서버 사전 점검. moana-u4 에서 실행한다(데이터가 노드 로컬). 사용: bash ctx_preflight.sh
source "$(dirname "$0")/../env.sh"
SUBJ=${SUBJ:-1}; NUM_SESSIONS=${NUM_SESSIONS:-40}; BASE=${BASE_MODEL:-subj0${SUBJ}_${NUM_SESSIONS}sess_server_ridge_sc2_ret16}
echo "== host $(hostname) | ROOT=$ROOT | BASE=$BASE"
ok=1
chk(){ if [ -s "$1" ]; then echo "OK      $(ls -lh "$1" | awk '{print $5}')  $1"; else echo "없음    $1"; ok=0; fi; }
for k in image text blurry; do chk $ROOT/train_logs/$BASE/ridge_${k}_weights.pkl; done
echo "-- 임베딩 통계(정규화용). 40세션 stats 파일이 image, latent, text(medium) 각각 있어야 함"
ls $ROOT/dataset/preprocessed_data/subject$SUBJ/ 2>/dev/null | grep -i "stats_${NUM_SESSIONS}\(\.0\)\?sess" || { echo "없음: *_embeddings_stats_${NUM_SESSIONS}sess.pt"; ok=0; }
chk $ROOT/dataset/betas_all_subj0${SUBJ}_fp32_renorm.hdf5
chk $ROOT/dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv
chk $ROOT/dataset/preprocessed_data/subject$SUBJ/nsd_imagery.pt
chk $ROOT/dataset/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3
chk $ROOT/cache/stage_c_bf16.safetensors
chk $ROOT/cache/stage_b_bf16.safetensors
chk $SCRIPTS/ctx/context_T10_by_nsdid.json
$PY - <<PYE
import json, hashlib, os
p="$SCRIPTS/ctx/context_T10_by_nsdid.json"
if os.path.exists(p):
    b=open(p,"rb").read(); d=json.loads(b); print("문장 파일 md5", hashlib.md5(b).hexdigest(), "| 항목 수", len(d), "(맥에서 만든 값: md5 9486eabaee41f30b61d92fd23167705c, 36977개)")
PYE
echo "-- 디스크(/local_datasets). 맥락 릿지 2개(각 6.2GB)와 임시 파일에 여유 25GB 이상 필요"; df -h /local_datasets | tail -1
echo "-- RAM / GPU"; free -g | head -2; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "-- 이미 있는 결과 폴더(덮어쓰지 않는지 확인)"; ls $RESULTS 2>/dev/null | grep -i "ctx" || echo "(ctx 결과 폴더 없음)"
[ $ok = 1 ] && echo "== 전부 OK, 제출해도 됨" || echo "== ❌ 없는 항목이 있음(위 '없음' 확인)"
