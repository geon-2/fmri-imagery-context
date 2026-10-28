#!/usr/bin/bash
# 맥락 합치기 실험(제출 전) 서버 사전 점검. moana-u4 에서 실행한다(데이터가 노드 로컬). 사용: bash ctx_preflight.sh
source "$(dirname "$0")/../env.sh"
SUBJ=${SUBJ:-1}; NUM_SESSIONS=40   # 이 실험은 40세션 고정(env.sh의 기본값은 10이라 환경변수가 없으면 10으로 점검하게 된다)
BASE=${BASE_MODEL:-subj0${SUBJ}_${NUM_SESSIONS}sess_server_ridge_sc2}   # 기준선 no-retrieval 재구성을 만든 바로 그 가중치
echo "== host $(hostname) | ROOT=$ROOT | BASE=$BASE"
ok=1
chk(){ if [ -s "$1" ]; then echo "OK      $(ls -lh "$1" | awk '{print $5}')  $1"; else echo "없음    $1"; ok=0; fi; }
for k in image text blurry; do chk $ROOT/train_logs/$BASE/ridge_${k}_weights.pkl; done
echo "-- 정규화용 학습 임베딩(recon이 *_embeddings_train.pt 링크로 읽음). image, latent, text(medium) 각각 stats 또는 전체 파일이 있어야 함"
PRE=$ROOT/dataset/preprocessed_data/subject$SUBJ
for p in stable_cascade_image vdvae_latent stable_cascade_medium_text; do
  f=$(ls $PRE/${p}_embeddings_stats_${NUM_SESSIONS}*sess.pt $PRE/${p}_embeddings_train_${NUM_SESSIONS}*sess.pt 2>/dev/null | head -1)
  if [ -n "$f" ] && [ -s "$f" ]; then echo "OK      $(ls -lhL "$f" | awk '{print $5}')  $f"; else echo "없음    $PRE/${p}_embeddings_{stats,train}_${NUM_SESSIONS}*sess.pt"; ok=0; fi
done
chk $ROOT/dataset/betas_all_subj0${SUBJ}_fp32_renorm.hdf5
chk $ROOT/dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv
chk $ROOT/dataset/preprocessed_data/subject$SUBJ/nsd_imagery.pt
chk $ROOT/dataset/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3
chk $ROOT/cache/stage_c_bf16.safetensors
chk $ROOT/cache/stage_b_bf16.safetensors
chk $SCRIPTS/ctx/context_T10_by_nsdid.json
if [ -s $SCRIPTS/ctx/context_phrases_by_nsdid.json ]; then echo "OK      $(ls -lh $SCRIPTS/ctx/context_phrases_by_nsdid.json | awk '{print $5}')  $SCRIPTS/ctx/context_phrases_by_nsdid.json (P1 평균 조건용)"; else echo "참고    없음: context_phrases_by_nsdid.json (P1 평균 조건 submit_ctx_p1.sh에만 필요, T10 조건 제출에는 영향 없음)"; fi
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
