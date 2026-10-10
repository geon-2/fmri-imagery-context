#!/bin/zsh
# run_scale.py를 남은 이미지가 없을 때까지 반복 실행한다. 사용량 한도나 일시 오류로 멈추면 30분 뒤 이어서 한다.
# 진전이 12번 연속(약 6시간) 없으면(체계적 문제) 멈춘다. 이미 통과한 이미지는 run_scale.py가 건너뛴다.
cd "$(dirname "$0")/../.." || exit 1
LOG=data/logs/run_scale_loop.log
remaining() {
  python3 - <<'E'
import csv, glob, json
q = [r["cocoId"] for r in csv.DictReader(open("data/labels/classify_scale/queue_order.csv", encoding="utf-8"))]
ids = set()
for f in glob.glob("data/labels/classify_scale/accepted/u*_p0*.json"):
    ids |= {str(i["image_id"]) for i in json.load(open(f, encoding="utf-8"))["items"]}
print(len([c for c in q if c not in ids]))
E
}
stall=0
for i in $(seq 1 60); do
  while pgrep -f "src/labeling/run_scale.py" >/dev/null; do sleep 120; done
  before=$(remaining)
  echo "$(date '+%F %T') 반복 $i 시작, 남은 이미지 $before" >> $LOG
  [ "$before" -eq 0 ] && { echo "$(date '+%F %T') 완료" >> $LOG; break; }
  python3 src/labeling/run_scale.py --pass 0 --units 0:760 --workers 4 --model gpt-5.5 --effort medium >> $LOG 2>&1
  after=$(remaining)
  if [ "$after" -ge "$before" ]; then stall=$((stall+1)); else stall=0; fi
  echo "$(date '+%F %T') 반복 $i 종료, 남은 이미지 $after, 진전없음 연속 $stall" >> $LOG
  [ "$stall" -ge 12 ] && { echo "$(date '+%F %T') 진전 없음 12회, 중단" >> $LOG; break; }
  sleep 1800
done
