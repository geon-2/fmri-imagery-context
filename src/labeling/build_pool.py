"""파일럿 라벨링 후보 풀을 만든다.

포함: 서브젝트 1/2/5/7 중 한 명 이상이 본 NSD 이미지.
제외: shared1000, NSD-Imagery 자연장면 5장(cue_pair_list.xlsx stim_set B), flagged(이미지 내용 플래그).

출력:
  data/processed/pilot_pool.csv        (git 추적 안 함)
  results/tables/pool_exclusions.md    (단계별 개수 요약)
"""
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STIM_INFO = REPO / "data" / "raw" / "nsd_stim_info_merged.csv"
OUT_POOL = REPO / "data" / "processed" / "pilot_pool.csv"
OUT_SUMMARY = REPO / "results" / "tables" / "pool_exclusions.md"

SUBJECTS = (1, 2, 5, 7)
# NSD-Imagery stim_set B (cue_pair_list.xlsx). nsd00000 은 NSD 이미지가 아니므로 제외 대상 아님.
IMAGERY_NSD_IDS = {28752, 30857, 53882, 61178, 65873}


def main():
    rows = list(csv.DictReader(open(STIM_INFO, encoding="utf-8")))
    steps = [("NSD 전체", len(rows))]

    pool = [r for r in rows if any(r[f"subject{s}"] != "0" for s in SUBJECTS)]
    steps.append((f"서브젝트 {'/'.join(map(str, SUBJECTS))} 중 한 명 이상이 본 이미지", len(pool)))

    pool = [r for r in pool if r["shared1000"] != "True"]
    steps.append(("shared1000 제외", len(pool)))

    pool = [r for r in pool if int(r["nsdId"]) not in IMAGERY_NSD_IDS]
    steps.append(("NSD-Imagery 자연장면 5장 제외", len(pool)))

    pool = [r for r in pool if r["flagged"] != "True"]
    steps.append(("flagged 제외", len(pool)))

    OUT_POOL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_POOL, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nsdId", "cocoId", "cocoSplit", "subjects_seen"])
        for r in pool:
            seen = ",".join(str(s) for s in SUBJECTS if r[f"subject{s}"] != "0")
            w.writerow([r["nsdId"], r["cocoId"], r["cocoSplit"], seen])

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 파일럿 후보 풀 제외 단계", "", "| 단계 | 남은 이미지 |", "|---|---|"]
    lines += [f"| {name} | {n} |" for name, n in steps]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
