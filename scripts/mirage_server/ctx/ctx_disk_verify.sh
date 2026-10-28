#!/usr/bin/bash
# 지우기 전 검증(읽기 전용: cmp, du, git rev-parse만 쓴다. 아무것도 지우거나 바꾸지 않는다). moana-u4 노드 안에서 실행한다.
# "정말 중복인지"를 바이트 단위로 확인해서, 안전한 삭제 후보만 남기려는 목적이다. 6GB 파일 비교가 있어 1~3분 걸릴 수 있다.
# 사용: bash ctx/ctx_disk_verify.sh
source "$(dirname "$0")/../env.sh"
A=$ROOT/train_logs/subj01_40sess_server_ridge_sc2          # retrieval 없음(기준선 재구성을 만든 가중치)
B=$ROOT/train_logs/subj01_40sess_server_ridge_sc2_ret16    # retrieval 있음
echo "== 1. 두 가중치 폴더의 image/text/blurry가 바이트 단위로 같은가"
echo "   (모두 '동일'이면 ret16 쪽을 심볼릭 링크로 대체해 약 11GB를 아낄 수 있다. '다름'이 있으면 그 파일은 대체하면 안 된다)"
for k in image text blurry; do
  if cmp -s $A/ridge_${k}_weights.pkl $B/ridge_${k}_weights.pkl; then echo "   동일   $k"; else echo "   다름   $k"; fi
done
echo; echo "== 2. 서버 로컬 evals 파일에 NAS($RESULTS)의 동일 사본이 있는가"
for f in $ROOT/src/evals/*/*.pt; do
  [ -e "$f" ] || continue
  n=$(basename $(dirname $f)); found=0
  for g in $RESULTS/$n/*.pt; do [ -e "$g" ] && cmp -s "$f" "$g" && { found=1; break; }; done
  if [ $found = 1 ]; then echo "   NAS에 동일 사본 있음   $n/$(basename $f)"; else echo "   ❌ NAS에 없음          $n/$(basename $f)"; fi
done
echo; echo "== 3. hf 캐시($HF_HOME) 내용"
du -sh $HF_HOME/* 2>/dev/null
du -sh $HF_HOME/*/* 2>/dev/null | sort -h | tail -8
echo; echo "== 4. MIRAGE 작업 폴더의 .git(1.8GB)이 MIRAGE_repo의 복사본인가(HEAD 비교)"
echo "   작업 폴더: $(git -C $ROOT rev-parse HEAD 2>&1 | head -1)"
echo "   MIRAGE_repo: $(git -C $WORK/MIRAGE_repo rev-parse HEAD 2>&1 | head -1)"
echo; echo "(읽기 전용 검증이다. 지우는 명령은 이 결과를 보고 따로 안내한다.)"
