"""라벨 차원 후보 검토용 점검 (읽기·집계만, 새 라벨링 호출 없음).

docs/orchestrator_prompt_dimension_review.md 의 할 일 2~4를 수행한다.
- 표현 목록은 프롬프트에서 받은 그대로 고정(결과를 보기 전에 정해졌다). 매칭 방식도 실행 전에 고정:
    주 기준   = 영어는 단어 단위 일치(앞뒤가 알파벳이 아닌 위치), close-up은 'close-up'/'close up',
                한국어는 부분 문자열
    민감도    = 영어도 부분 문자열(open이 openness, opening에도 걸림)
  두 기준을 나란히 보고한다. 어느 쪽도 채택 여부 판단이 아니라 "이미 기존 차원에 흡수돼 있는가"를 보는 것이다.
- 자료: (A) 파일럿 300장 adjudicate v3(값별 support·mean_confidence), (B) 1만 장 classify_scale(N=1, v9/v10, mean_confidence),
        (C) 사람 라벨 200장(영어/한국어 열).
- COCO 사람/동물 존재는 라벨이 아니라 공변량 분포(비율)로만 저장한다.

출력: results/tables/{dimension_expression_check.md, dimension_expression_check.csv, coco_presence_covariates.md, six_dim_distribution.md}
"""
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
DIMS6 = ("place_type", "environment", "activity", "implied_event", "scale", "notes")

FAMILIES = {
    "개방/폐쇄": {"en": ["open", "closed", "enclosed"], "ko": ["개방", "폐쇄", "탁 트인", "밀폐"]},
    "깊이/규모": {"en": ["deep", "shallow", "close-up", "distant"], "ko": ["원경", "근경", "넓은", "좁은"]},
    "이동 가능성": {"en": ["navigable", "walkable"], "ko": ["이동 가능", "통행"]},
}


def en_pat(tok, strict):
    if tok == "close-up":
        core = r"close[- ]?up"
    else:
        core = re.escape(tok)
    return re.compile((rf"(?<![a-z]){core}(?![a-z])" if strict else core), re.I)


PATS = {}
for fam, d in FAMILIES.items():
    PATS[fam] = {
        "strict": [(t, en_pat(t, True)) for t in d["en"]] + [(t, re.compile(re.escape(t))) for t in d["ko"]],
        "loose": [(t, en_pat(t, False)) for t in d["en"]] + [(t, re.compile(re.escape(t))) for t in d["ko"]],
    }


def match(fam, mode, text):
    return [t for t, p in PATS[fam][mode] if p.search(text)]


# ---------- 자료 적재 ----------
def load_pilot_v3():
    pilot = {l.split(",")[1] for l in open(REPO / "data/processed/pilot_300.csv", encoding="utf-8").read().splitlines()[1:]}
    recs = {}
    for line in open(REPO / "data/labels/agent_logs/adjudicate.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["prompt_version"] == "v3" and str(r["image_id"]) in pilot:
            recs[str(r["image_id"])] = r["parsed_label"]["labels"]
    data = {}
    for iid, labels in recs.items():
        data[iid] = {d: [(v["value"], v.get("support"), v.get("mean_confidence")) for v in vs] for d, vs in labels.items()}
    return data


def load_scale():
    data = {}
    for f in sorted(glob.glob(str(REPO / "data/labels/classify_scale/accepted/u*_p0_a*.json"))):
        a = json.load(open(f, encoding="utf-8"))
        for it in a["items"]:
            data[str(it["image_id"])] = {d: [(v["value"], None, v.get("confidence")) for v in vs if isinstance(v.get("value"), str)]
                                         for d, vs in it["labels"].items()}
    return data


def load_human():
    out = {}
    for lang, fn in (("en", "human_labels_200_en.csv"), ("ko", "human_labels_200.csv")):
        rows = list(csv.DictReader(open(REPO / "data/labels" / fn, encoding="utf-8-sig")))
        for r in rows:
            iid = re.search(r"(\d+)\.jpg", r["image_path"]).group(1) if r.get("image_path") else r["image_id"]
            d = out.setdefault(str(iid), {})
            for col, val in r.items():
                if col is None or not isinstance(val, str) or col in ("image_id", "image_path") or not val.strip():
                    continue
                col = {"space scale": "space_scale", "dropped (objects / appearance)": "dropped"}.get(col, col)
                d.setdefault(f"{col}", []).extend((x.strip(), None, None) for x in re.split(r"[;；]", val) if x.strip())
    return out


# ---------- 표현 점검 ----------
def expression_table(name, data, dims, rows_out):
    n_img = len(data)
    lines = [f"### {name} (이미지 {n_img:,}장)\n"]
    for mode, mlabel in (("strict", "주 기준: 단어 단위"), ("loose", "민감도: 부분 문자열")):
        lines.append(f"**{mlabel}**\n")
        lines.append("| 표현 계열 | 차원 | 일치 이미지 | 비율 | 일치 라벨 수 | 고유 값 | 평균 support | 평균 confidence | 상위 값(이미지 수) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for fam in FAMILIES:
            for dim in dims:
                imgs, vals, sup, conf, n_lab = set(), Counter(), [], [], 0
                for iid, labels in data.items():
                    for v, s, c in labels.get(dim, []):
                        if match(fam, mode, v.lower() if mode else v):
                            imgs.add(iid); vals[v.lower()] += 1; n_lab += 1
                            if s is not None: sup.append(s)
                            if c is not None: conf.append(c)
                top = ", ".join(f"{k}({n})" for k, n in vals.most_common(4))
                ms = f"{sum(sup) / len(sup):.2f}" if sup else "-"
                mc = f"{sum(conf) / len(conf):.2f}" if conf else "-"
                lines.append(f"| {fam} | {dim} | {len(imgs)} | {100 * len(imgs) / n_img:.1f}% | {n_lab} | {len(vals)} | {ms} | {mc} | {top or '-'} |")
                rows_out.append([name, mode, fam, dim, len(imgs), round(100 * len(imgs) / n_img, 2), n_lab, len(vals), ms, mc])
        lines.append("")
    return "\n".join(lines)


def six_dim_distribution(data):
    n_img = len(data)
    lines = [f"# 6차원 값 분포 (1만 장 pass 0, 이미지 {n_img:,}장, casefold 기준)\n",
             "값 하나가 얼마나 많은 이미지에 나오는지(이미지 비율)를 본다. 어휘 재정의 필요 여부의 임계값은 정하지 않았다.\n",
             "| 차원 | 라벨 수 | 고유 값 | 가장 흔한 값의 이미지 비율 | 상위 8개 값(이미지 수) |", "|---|---|---|---|---|"]
    for dim in DIMS6:
        per = Counter(); n_lab = 0
        for labels in data.values():
            seen = set()
            for v, _, _ in labels.get(dim, []):
                n_lab += 1; seen.add(v.lower())
            per.update(seen)
        top = per.most_common(8)
        share = f"{100 * top[0][1] / n_img:.1f}%" if top else "-"
        lines.append(f"| {dim} | {n_lab:,} | {len(per):,} | {share} | " + ", ".join(f"{k}({n})" for k, n in top) + " |")
    return "\n".join(lines)


def coco_presence(pilot_ids, scale_ids):
    cats_animal, cat_person = set(), 1
    presence = {}
    for split in ("val", "train"):
        d = json.load(open(REPO / f"data/raw/annotations/instances_{split}2017.json", encoding="utf-8"))
        animal = {c["id"] for c in d["categories"] if c["supercategory"] == "animal"}
        want = pilot_ids | scale_ids
        for a in d["annotations"]:
            iid = str(a["image_id"])
            if iid in want:
                p = presence.setdefault(iid, [False, False])
                if a["category_id"] == 1: p[0] = True
                if a["category_id"] in animal: p[1] = True
        del d
    lines = ["# COCO 주석 기반 사람/동물 존재 (공변량 분포, 비율만 — 라벨로 쓰지 않음)\n",
             "사람 = COCO category 'person', 동물 = supercategory 'animal'(bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe). "
             "인스턴스 주석이 하나도 없는 이미지는 '없음'으로 셌다.\n",
             "| 대상 | 이미지 | 사람 있음 | 동물 있음 | 사람+동물 | 사람 또는 동물 | 둘 다 없음 |", "|---|---|---|---|---|---|---|"]
    for name, ids in (("파일럿 300장", pilot_ids), ("1만 장 pass 0", scale_ids)):
        n = len(ids); p = a = both = 0
        for i in ids:
            pr = presence.get(i, [False, False])
            p += pr[0]; a += pr[1]; both += pr[0] and pr[1]
        either = sum(1 for i in ids if any(presence.get(i, [False, False])))
        f = lambda x: f"{100 * x / n:.1f}%"
        lines.append(f"| {name} | {n:,} | {f(p)} | {f(a)} | {f(both)} | {f(either)} | {f(n - either)} |")
    return "\n".join(lines)


def main():
    pilot, scale, human = load_pilot_v3(), load_scale(), load_human()
    rows = []
    parts = ["# 라벨 차원 후보 검토 — 표현 점검 (채택 여부 판단이 아님)\n",
             "표현 목록과 매칭 방식은 결과를 보기 전에 고정했다(스크립트 상단 주석). "
             "이 표는 개방/폐쇄, 깊이·규모, 이동 가능성 표현이 **이미 기존 차원에 얼마나 나오는지**만 보여 준다.\n"]
    parts.append(expression_table("A. 파일럿 300장 (adjudicate v3, 5차원, 값별 support 집계)", pilot,
                                  ("place_type", "environment", "activity", "implied_event", "scale"), rows))
    parts.append(expression_table("B. 1만 장 (v9/v10, N=1이라 support 없음, 6차원)", scale, DIMS6, rows))
    hum_dims = sorted({d for v in human.values() for d in v})
    parts.append(expression_table("C. 사람 라벨 200장 (영어/한국어 열을 함께 봄, 열 이름이 곧 차원)", human, hum_dims, rows))
    (OUT / "dimension_expression_check.md").write_text("\n".join(parts), encoding="utf-8")
    with open(OUT / "dimension_expression_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "mode", "family", "dimension", "n_images", "pct", "n_labels", "distinct_values", "mean_support", "mean_confidence"])
        w.writerows(rows)
    (OUT / "six_dim_distribution.md").write_text(six_dim_distribution(scale), encoding="utf-8")
    (OUT / "coco_presence_covariates.md").write_text(coco_presence(set(pilot), set(scale)), encoding="utf-8")
    print("저장:", [p.name for p in OUT.glob("dimension_expression_check.*")], "six_dim_distribution.md", "coco_presence_covariates.md")
    print("자료 규모:", len(pilot), len(scale), len(human), "| 사람 라벨 열:", hum_dims)


if __name__ == "__main__":
    main()
