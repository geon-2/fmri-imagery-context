"""사람 평가(2AFC 식별) 응답을 key.json과 대조해 정답 비율을 계산한다.
사용: python scripts/mirage_eval/analyze_human_2afc.py [--resp ...] [--key ...] [--out results/tables/human_2afc.md]
판정(합격/불합격)은 하지 않는다 — docs/human_2afc_protocol.md 의 사전 규칙대로 비율과 신뢰구간만 보고한다."""
import argparse, json, os
import numpy as np
from scipy.stats import binomtest

ap = argparse.ArgumentParser()
ap.add_argument("--key", default="data/results/human_eval/key.json")
ap.add_argument("--resp", default="data/results/human_eval/responses.jsonl")
ap.add_argument("--out", default="results/tables/human_2afc.md")
a = ap.parse_args()

PAPER = {"전체": 78.30, "simple": 73.93, "complex": 83.19}  # 논문 Table 2A, MIRAGE imagery(평가자 약 445명)
key = {t["id"]: t for t in json.load(open(a.key))["trials"]}
resp = {}
for line in open(a.resp):
    if line.strip():
        r = json.loads(line); resp[r["id"]] = r  # 같은 문항은 마지막 응답

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

missing = [i for i in key if i not in resp]
att = [(key[i], resp[i]) for i in key if key[i]["kind"] == "attention" and i in resp]
att_wrong = sum(r["choice"] != t["target_side"] for t, r in att)
valid = att_wrong < 2   # 논문: 주의력 점검 2개 이상 틀리면 제외
main = [(key[i], resp[i]) for i in key if key[i]["kind"] == "main" and i in resp]
corr = lambda rows: sum(r["choice"] == t["target_side"] for t, r in rows)

def line(name, rows):
    n, k = len(rows), corr(rows); lo, hi = wilson(k, n)
    p = binomtest(k, n, 0.5).pvalue if n else float("nan")
    ref = PAPER.get(name); refs = f"{ref:.2f}%" if ref else "-"
    return f"| {name} | {n} | {k} | {100*k/max(n,1):.1f}% | {100*lo:.1f}~{100*hi:.1f}% | {p:.2g} | {refs} |"

L = ["# 사람 평가(2AFC 식별) 결과 — 평가자 1명", "",
     f"- 응답 {len(resp)}/{len(key)}문항" + (f" (미응답 {len(missing)})" if missing else ""),
     f"- 주의력 점검: {len(att)}문항 중 오답 {att_wrong}개 → {'유효' if valid else '**무효(논문 기준 2개 이상 오답)**'}",
     f"- 중앙 응답시간(본 문항): {np.median([r['rt_ms'] for _, r in main if r.get('rt_ms')]):.0f} ms" if main else "",
     "", "| 구분 | 문항 | 정답 | 정답 비율 | 95% 신뢰구간(Wilson) | p(이항, 50% 대비) | 논문 Table 2A |", "|---|---|---|---|---|---|---|",
     line("전체", main), line("simple", [x for x in main if x[0]["type"] == "simple"]), line("complex", [x for x in main if x[0]["type"] == "complex"])]
L += ["", "## 피험자별", "", "| 피험자 | 문항 | 정답 비율 |", "|---|---|---|"]
for s in sorted({t["subj"] for t, _ in main}):
    rows = [x for x in main if x[0]["subj"] == s]; L.append(f"| {s} | {len(rows)} | {100*corr(rows)/len(rows):.1f}% |")
L += ["", "## 자극별", "", "| 자극 | 종류 | 문항 | 정답 비율 |", "|---|---|---|---|"]
for st in range(12):
    rows = [x for x in main if x[0]["stim"] == st]
    if rows: L.append(f"| {st} | {rows[0][0]['type']} | {len(rows)} | {100*corr(rows)/len(rows):.1f}% |")
L += ["", "주의: 평가자 1명, 문항 96개(피험자 4 × 자극 12 × 샘플 2)다. 논문 값은 약 445명의 평균이며 문항 구성이 달라 직접 비교는 거칠다. "
      "피험자 1만 retrieval 후보 16장이다. 합격/불합격 판정은 하지 않는다."]
os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, "w").write("\n".join(l for l in L if l is not None) + "\n")
print("\n".join(L))
