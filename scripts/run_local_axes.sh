#!/bin/bash
# RQ1 보강 분석을 로컬(Mac)에서 순서대로 돌린다. 끊겨도 다시 실행하면 끝난 조합은 건너뛰고 이어서 한다(--skip-existing, 영역별 체크포인트).
cd /Users/gunlee/Development/fmri-imagery-context/src/fmri || exit 1
export PYTHONPATH=../labeling
R=../../data
EMB=$R/embeddings_v2; BUN=$R/colab_bundle; OUT=$R/results
st() { echo "[$(date '+%F %T')] $*"; }
fm() { echo $R/stage/fmri_$1; }

st "D1·D2 (피험자 1)"
for M in MPNet CLIP; do python3 -u axes_object_check.py --fmri-dir $(fm subj01) --emb-dir $EMB --bundle-dir $BUN --model $M --out-dir $OUT/axes_object --tag subj01 --k 12 --skip-existing; done

st "영역별 분석: 피험자 1 MPNet 맥락 (시간 측정용 첫 조합)"
python3 -u axes_roi.py --fmri-dir $(fm subj01) --emb-dir $EMB --model MPNet --rep ctx_P1 --out-dir $OUT/axes_roi --tag subj01 --skip-existing
st "첫 조합 완료"

st "ctxres_P1 k 곡선"
for S in subj01 subj02 subj05 subj07; do
  for M in MPNet CLIP; do
    if [ "$M" = CLIP ] && [ "$S" != subj01 ]; then continue; fi
    python3 -u axes_k_curve.py --fmri-dir $(fm $S) --emb-dir $EMB --model $M --rep ctxres_P1 --out-dir $OUT/axes --tag $S --boot 200 --skip-existing
  done
done

st "영역별 분석: 나머지"
python3 -u axes_roi.py --fmri-dir $(fm subj01) --emb-dir $EMB --model MPNet --rep ctxres_P1 --out-dir $OUT/axes_roi --tag subj01 --skip-existing
for S in subj02 subj05 subj07; do python3 -u axes_roi.py --fmri-dir $(fm $S) --emb-dir $EMB --model MPNet --rep ctx_P1 --out-dir $OUT/axes_roi --tag $S --skip-existing; done
python3 -u axes_roi.py --fmri-dir $(fm subj01) --emb-dir $EMB --model CLIP --rep ctx_P1 --out-dir $OUT/axes_roi --tag subj01 --skip-existing
for REP in obj_P1 orig_P1; do python3 -u axes_roi.py --fmri-dir $(fm subj01) --emb-dir $EMB --model MPNet --rep $REP --out-dir $OUT/axes_roi --tag subj01 --skip-existing; done
st "전체 완료"
