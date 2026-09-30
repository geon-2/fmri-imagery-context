"""배치1(10장) 기준으로 지금까지 만든 모든 버전을 한 표에 모은다:
  - old: 4차원, Claude classify, critique-a v2 (원래 첫 파일럿)
  - v4: 5차원, Claude classify x3, critique-a v3(원본)/v4(person 캐비앗 추가) 둘 다,
        + critique-b로 Gemini v4/Codex v4도 같은 v4 classify 라벨을 판정한 결과
  - v5: 5차원, Gemini+Codex classify(개수 제한 없이 뽑기 지침), critique-a v4,
        + critique-b Codex v4

재라벨링 없이 이미 만든 로그/스크래치 파일만 읽는다.

사용:
  python3 src/labeling/compare_all_versions.py > results/tables/all_versions_comparison.csv
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "data" / "labels" / "agent_logs"
EXT_DIR = REPO / "data" / "labels" / "external_labeling"
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-gunlee-Development-fmri-imagery-context/"
    "3b79a60c-98b0-41c9-8335-84a9d8030348/scratchpad"
)

IMAGE_IDS = [
    "172088", "452622", "7095", "105452", "498449",
    "168329", "336245", "128553", "480961", "332078",
]
OLD_DIMS = ("place_type", "environment", "activity", "scale")
NEW_DIMS = ("place_type", "environment", "activity", "implied_event", "scale")


def load_jsonl(path, version):
    out = {}
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("prompt_version") == version:
            out[str(rec["image_id"])] = rec["parsed_label"]
    return out


def load_json_items(path):
    if not path.exists():
        return {}
    d = json.load(open(path, encoding="utf-8"))
    return {str(it["image_id"]): it for it in d.get("items", [])}


def top_values(labels, dim, n=3):
    rows = labels.get(dim, [])
    rows = sorted(rows, key=lambda r: -r.get("support", 0))[:n]
    return "; ".join(f"{r['value']}({r.get('support', 0):.2f})" for r in rows)


def overall_of(rec):
    if rec is None:
        return ""
    if "overall" in rec:
        return rec["overall"]
    if "status" in rec:
        flags = rec.get("flags", [])
        if rec["status"] == "review":
            return "review"
        return "leak" if "leakage_suspected" in flags else "ok"
    return ""


def main():
    old_adj = load_jsonl(LOG_DIR / "adjudicate.jsonl", "v2")

    # v4 classify aggregated labels + v3(원본) critique-a: 이미 만든 배치1 조합 파일에서 가져온다
    v4_combo = load_json_items(SCRATCH / "b01_adj_real_in.json")  # {aggregated, critique_a(v3)}
    v4_crit_v4 = load_json_items(SCRATCH / "b01_crita_v4_out.json")  # critique-a v4 재판정
    v4_gemini_v4 = load_json_items(EXT_DIR / "critique_b" / "b01_out_gemini_v4.json")
    v4_codex_v4 = load_json_items(EXT_DIR / "critique_b" / "b01_out_codex_v4.json")

    v5_agg = load_json_items(EXT_DIR / "classify" / "b01_agg.json")
    v5_crita = load_json_items(SCRATCH / "b01_crita_v5out.json")
    v5_codex = load_json_items(EXT_DIR / "critique_b" / "b01_crit_out_codex_v5.json")

    writer = csv.writer(sys.stdout)
    header = ["image_id", "old_leak"]
    for d in OLD_DIMS:
        header.append(f"old_{d}")
    header += ["v4_leak_crita_v3", "v4_leak_crita_v4", "v4_leak_gemini_v4", "v4_leak_codex_v4"]
    for d in NEW_DIMS:
        header.append(f"v4_{d}")
    header += ["v5_leak_crita_v4", "v5_leak_codex_v4"]
    for d in NEW_DIMS:
        header.append(f"v5_{d}")
    writer.writerow(header)

    for iid in IMAGE_IDS:
        row = [iid]

        old = old_adj.get(iid, {})
        row.append(overall_of(old))
        for d in OLD_DIMS:
            row.append(top_values(old.get("labels", {}), d))

        v4c = v4_combo.get(iid, {})
        v4_agg_labels = v4c.get("aggregated", {})
        row.append(overall_of(v4c.get("critique_a")))
        row.append(overall_of(v4_crit_v4.get(iid)))
        row.append(overall_of(v4_gemini_v4.get(iid)))
        row.append(overall_of(v4_codex_v4.get(iid)))
        for d in NEW_DIMS:
            row.append(top_values(v4_agg_labels, d))

        row.append(overall_of(v5_crita.get(iid)))
        row.append(overall_of(v5_codex.get(iid)))
        v5_labels = v5_agg.get(iid, {}).get("labels", {})
        for d in NEW_DIMS:
            row.append(top_values(v5_labels, d))

        writer.writerow(row)


if __name__ == "__main__":
    main()
