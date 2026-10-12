#!/bin/zsh
# R1(코어 v10 재라벨링, pass 1)과 R2(무작위 2,000장 재현성, pass 2)를 이어서 실행한다. 실패 단위는 최대 3번까지 run_scale.py가 다시 시도한다.
# 기준: docs/label_crosscheck_criteria.md. 한도·일시 오류로 멈추면 30분 뒤 이어서 한다.
cd "$(dirname "$0")/../.." || exit 1
LOG=data/logs/run_crosscheck.log
for i in $(seq 1 24); do
  while pgrep -f "src/labeling/run_scale.py" >/dev/null; do sleep 120; done
  echo "$(date '+%F %T') 반복 $i R1 시작" >> $LOG
  python3 src/labeling/run_scale.py --pass 1 --units 0:20 --workers 4 --model gpt-5.5 --effort medium >> $LOG 2>&1
  echo "$(date '+%F %T') 반복 $i R2 시작" >> $LOG
  python3 src/labeling/run_scale.py --pass 2 --units 0:40 --workers 4 --model gpt-5.5 --effort medium --queue-file queue_repro2000.csv >> $LOG 2>&1
  r1=$(python3 - <<'E'
import glob,json
ids=set()
for f in glob.glob("data/labels/classify_scale/accepted/u*_p1_*.json"):
    ids|={str(i["image_id"]) for i in json.load(open(f))["items"]}
print(1000-len(ids & {str(x) for x in range(0)} ) if False else len(ids))
E
)
  r2=$(python3 - <<'E'
import glob,json
ids=set()
for f in glob.glob("data/labels/classify_scale/accepted/u*_p2_*.json"):
    ids|={str(i["image_id"]) for i in json.load(open(f))["items"]}
print(len(ids))
E
)
  echo "$(date '+%F %T') 반복 $i 종료: pass1 $r1 / pass2 $r2" >> $LOG
  [ "$r1" -ge 1000 ] && [ "$r2" -ge 2000 ] && { echo "$(date '+%F %T') 완료" >> $LOG; break; }
  sleep 1800
done
