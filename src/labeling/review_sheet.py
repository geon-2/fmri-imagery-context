"""직접 확인 판정 시트 만들기와 집계.

  python3 src/labeling/review_sheet.py make    # review_sample_100.md → review_sheet_100.csv (라벨 한 줄씩, 판정 칸 비어 있음)
  python3 src/labeling/review_sheet.py score   # 채운 CSV를 집계해 review_result.md 생성

판정 칸에는 o(맞음), ?(애매), x(틀림) 중 하나를 쓴다(맞음/애매/틀림이라고 써도 된다). 비워 두면 집계에서 빠지고 개수를 알려 준다.
통과 기준은 Notion 「직접 확인 통과 기준 확정」(2026-10-01)과 같다: 틀림 5% 이하 아주 좋음, 10% 이하 통과, 20% 이하 경고, 초과 불합격.
암묵적 사건(implied_event)은 한 단계 완화(10/20/30%). 기준은 항목별 라벨 단위 틀림 비율에 적용하고, 무작위 80장에만 적용한다.
표적 20장(COCO-Stuff 불일치·누락)은 틀림이 더 많이 나오도록 뽑았으므로 기준에 쓰지 않고 오류 유형만 본다.
"""
import csv
import math
import re
import sys
from pathlib import Path

D = Path("data/labels/classify_scale")
MD, KEY, CSV, RES = D / "review_sample_100.md", D / "review_sample_100_key.md", D / "review_sheet_100.csv", D / "review_result.md"
DIMS = ("place_type", "environment", "activity", "implied_event", "scale", "notes")
CODE = {"o": "맞음", "맞음": "맞음", "?": "애매", "애매": "애매", "x": "틀림", "틀림": "틀림"}
STEPS = {  # (아주 좋음, 통과, 경고) 상한
    "default": (0.05, 0.10, 0.20),
    "implied_event": (0.10, 0.20, 0.30),
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def verdict(dim, rate):
    a, b, c = STEPS.get(dim, STEPS["default"])
    return "아주 좋음" if rate <= a else "통과" if rate <= b else "경고" if rate <= c else "불합격"


def make():
    rows, cur = [], None
    for line in open(MD, encoding="utf-8"):
        line = line.rstrip("\n")
        m = re.match(r"^## (\d+)\s+\(data/raw/images/\d+\.jpg, (v\d+)\)", line)
        if m:
            cur = {"id": m.group(1), "ver": m.group(2), "cap": ""}
            continue
        if cur is None:
            continue
        if line.startswith("- 캡션:"):
            cur["cap"] = line[len("- 캡션:"):].strip()
            continue
        m = re.match(r"^- \*\*(\w+)\*\*: (.*)$", line)
        if m and m.group(2) != "(없음)":
            for item in m.group(2).split(" | "):
                mm = re.match(r"^(.*) \(([\d.]+)\)$", item)
                rows.append([cur["id"], cur["ver"], m.group(1), mm.group(1) if mm else item, mm.group(2) if mm else "", "", ""])
    with open(CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "version", "dimension", "label", "confidence", "판정(o/?/x)", "메모"])
        w.writerows(rows)
    print(f"저장: {CSV} ({len(rows)}줄, 이미지 {len({r[0] for r in rows})}장). 이미지는 data/raw/images/{{id}}.jpg, 캡션은 review_sample_100.md")


def layer_map():
    lay = {}
    for line in open(KEY, encoding="utf-8"):
        m = re.match(r"^\| (\d+) \| ([^|]+) \| (v\d+) \|", line)
        if m:
            lay[m.group(1)] = m.group(2).strip()
    return lay


def score():
    lay = layer_map()
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    done, skipped, unknown = [], 0, 0
    for r in rows:
        v = (r["판정(o/?/x)"] or "").strip().lower()
        if not v:
            skipped += 1
        elif v in CODE:
            done.append((r, CODE[v]))
        else:
            unknown += 1
    out = ["# 직접 확인 결과", "", f"판정한 라벨 {len(done)}줄, 비어 있음 {skipped}줄" + (f", 인식 못한 값 {unknown}줄(o/?/x만 인정)" if unknown else ""), ""]

    rand = [(r, j) for r, j in done if lay.get(r["image_id"]) == "무작위"]
    out += [f"## 무작위 80장 (통과 기준 적용 대상, 판정한 이미지 {len({r['image_id'] for r, _ in rand})}장)", "",
            "| 항목 | 라벨 수 | 맞음 | 애매 | 틀림 | 틀림 비율 (95% 구간) | 애매 비율 | 판정 |", "|---|---|---|---|---|---|---|---|"]
    for d in DIMS:
        sub = [j for r, j in rand if r["dimension"] == d]
        if not sub:
            continue
        n, x, q = len(sub), sub.count("틀림"), sub.count("애매")
        lo, hi = wilson(x, n)
        vd = "기준 없음(참고)" if d == "notes" else verdict(d, x / n)
        out.append(f"| {d} | {n} | {sub.count('맞음')} | {q} | {x} | {x / n:.1%} ({lo:.1%}~{hi:.1%}) | {q / n:.1%} | **{vd}** |")
    imgs = {}
    for r, j in rand:
        imgs.setdefault(r["image_id"], []).append(j)
    bad = sum(1 for v in imgs.values() if "틀림" in v)
    lo, hi = wilson(bad, len(imgs))
    out += ["", f"- 이미지 단위: 틀린 라벨이 하나라도 있는 이미지 {bad}/{len(imgs)}장 = {bad / max(len(imgs), 1):.1%} ({lo:.1%}~{hi:.1%})",
            "- 기준은 항목별 라벨 단위 틀림 비율에 적용한다(5/10/20%, 암묵적 사건은 10/20/30%). 같은 이미지의 라벨은 서로 상관이 있어 실제 불확실성은 위 구간보다 크다.", ""]
    for ver in ("v10", "v9"):
        sub = [j for r, j in rand if r["version"] == ver]
        if sub:
            out.append(f"- {ver}: 라벨 {len(sub)}줄, 틀림 {sub.count('틀림') / len(sub):.1%}, 애매 {sub.count('애매') / len(sub):.1%} (참고용, v9 단독 결론은 내지 않음)")
    out.append("")

    tgt = [(r, j) for r, j in done if lay.get(r["image_id"], "").startswith("표적")]
    if tgt:
        out += ["## 표적 20장 (기준에 쓰지 않음, 오류 유형 확인)", "", "| 층 | 환경 라벨 중 틀림 | 환경 라벨 줄 수 | 모든 항목 틀림 | 모든 항목 줄 수 |", "|---|---|---|---|---|"]
        for lname in ("표적 A(불일치)", "표적 B(누락)"):
            s = [(r, j) for r, j in tgt if lay.get(r["image_id"]) == lname]
            e = [j for r, j in s if r["dimension"] == "environment"]
            out.append(f"| {lname} | {e.count('틀림')} | {len(e)} | {sum(1 for _, j in s if j == '틀림')} | {len(s)} |")
        out += ["", "표적 B(누락)는 라벨이 말한 것이 틀렸는지가 아니라 빠뜨렸는지를 보는 층이라, 환경 라벨의 '틀림'이 거의 없는 게 정상이다.", ""]
    RES.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()
