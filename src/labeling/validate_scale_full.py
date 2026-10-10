"""라벨 36,977장(pass 0)의 요약 통계와 검증. 판정 기준은 실행 전에 docs/label_validation_criteria_scale.md에 기록했다.

출력: data/results/label_summary_full.json, results/tables/label_summary_full.md, data/results/label_review_sample_100.csv(V7 표본)
사용: python3 src/labeling/validate_scale_full.py
"""
import collections
import csv
import glob
import json
import re
import statistics as st
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "data/labels/classify_scale"
DIMS = ("place_type", "environment", "activity", "implied_event", "scale", "notes")
CORE_DIMS = ("place_type", "environment", "activity", "implied_event")
LONG_WORDS, SHARE_N = 5, 10
MIN_COUNT = {"place_type": 2, "environment": 3, "activity": 2}
WORD = re.compile(r"[a-z0-9']+")


def load():
    items = {}
    for f in sorted(glob.glob(str(BASE / "accepted/u*_p0_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for it in d["items"]:
            items[str(it["image_id"])] = (it, d["meta"]["prompt_version"])
    return items


def values(it, dim):
    return [x["value"].strip() for x in it["labels"].get(dim, []) if x.get("value", "").strip()]


def norm(s):
    return " ".join(WORD.findall(s.lower()))


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def describe(ids, items):
    out = {"n_images": len(ids), "dims": {}}
    total_words = []
    for d in DIMS:
        cnt = [len(values(items[i][0], d)) for i in ids]
        w = [len(WORD.findall(v.lower())) for i in ids for v in values(items[i][0], d)]
        vals = collections.Counter(norm(v) for i in ids for v in values(items[i][0], d))
        tot = sum(vals.values())
        out["dims"][d] = {
            "labels_per_image": {"mean": float(np.mean(cnt)), "median": float(np.median(cnt)), "p5": pct(cnt, 5), "p95": pct(cnt, 95),
                                 "min": int(min(cnt)), "max": int(max(cnt))},
            "empty_rate": float(np.mean([c == 0 for c in cnt])),
            "words_per_label": {"mean": float(np.mean(w)) if w else None, "median": float(np.median(w)) if w else None, "p95": pct(w, 95)},
            "unique_values": len(vals), "n_labels": tot,
            "top1_share": vals.most_common(1)[0][1] / tot if tot else None,
            "top10_share": sum(c for _, c in vals.most_common(10)) / tot if tot else None,
            "top3": vals.most_common(3)}
    for i in ids:
        total_words.append(sum(len(WORD.findall(v.lower())) for d in DIMS for v in values(items[i][0], d)))
    out["total_words_per_image"] = {"mean": float(np.mean(total_words)), "median": float(np.median(total_words)),
                                    "p95": pct(total_words, 95)}
    out["total_labels_per_image"] = float(np.mean([sum(len(values(items[i][0], d)) for d in DIMS) for i in ids]))
    fl = collections.Counter(f for i in ids for f in items[i][0].get("flags", []))
    out["flags"] = {k: {"n": v, "rate": v / len(ids)} for k, v in fl.items()}
    out["flag_any_rate"] = float(np.mean([bool(items[i][0].get("flags")) for i in ids]))
    return out


def v1_structure(ids, items):
    ok = 0
    bad = []
    for i in ids:
        good = all(d in items[i][0]["labels"] for d in DIMS) and all(len(values(items[i][0], d)) >= m for d, m in MIN_COUNT.items())
        ok += good
        if not good:
            bad.append(i)
    return {"rate": ok / len(ids), "n_bad": len(bad), "bad_examples": bad[:10]}


def v2_copy(ids, items):
    share = collections.defaultdict(set)
    for i in ids:
        for d in ("implied_event", "notes", "activity"):
            for v in values(items[i][0], d):
                n = norm(v)
                if len(n.split()) >= LONG_WORDS:
                    share[n].add(i)
    shared = {p: s for p, s in share.items() if len(s) >= SHARE_N}
    imgs = set().union(*shared.values()) if shared else set()
    top = sorted(shared.items(), key=lambda kv: -len(kv[1]))[:10]
    return {"n_shared_phrases": len(shared), "image_rate": len(imgs) / len(ids), "n_images": len(imgs),
            "top": [(p, len(s)) for p, s in top]}


def rel(a, b):
    return abs(a - b) / ((a + b) / 2) if (a + b) else 0.0


def compare(ids_a, ids_b, items):
    a, b = describe(ids_a, items), describe(ids_b, items)
    la, lb = a["total_labels_per_image"], b["total_labels_per_image"]
    wa = np.mean([a["dims"][d]["words_per_label"]["mean"] for d in CORE_DIMS])
    wb = np.mean([b["dims"][d]["words_per_label"]["mean"] for d in CORE_DIMS])
    return {"labels_rel_diff": rel(la, lb), "words_rel_diff": rel(float(wa), float(wb)),
            "flag_rate_diff": abs(a["flag_any_rate"] - b["flag_any_rate"]), "a": {"labels": la, "words": float(wa), "flag": a["flag_any_rate"]},
            "b": {"labels": lb, "words": float(wb), "flag": b["flag_any_rate"]}}


def coco_primary(ids):
    """이미지마다 면적이 가장 큰 COCO 범주(동률이면 category id가 작은 쪽)와 80범주 multi-hot."""
    stim = {r["cocoId"]: r["cocoSplit"] for r in csv.DictReader(open(BASE / "queue_order.csv", encoding="utf-8"))}
    want = set(ids)
    area = collections.defaultdict(lambda: collections.defaultdict(float))
    names = {}
    for split in ("train2017", "val2017"):
        d = json.load(open(REPO / f"data/raw/annotations/instances_{split}.json", encoding="utf-8"))
        for c in d["categories"]:
            names[c["id"]] = c["name"]
        for a in d["annotations"]:
            k = str(a["image_id"])
            if k in want:
                area[k][a["category_id"]] += a["area"]
        del d
    prim = {k: names[min(v, key=lambda c: (-v[c], c))] for k, v in area.items() if v}
    cats = {k: set(names[c] for c in v) for k, v in area.items()}
    return prim, cats, sorted(names.values())


def v6_leak(ids, items):
    prim, cats, catnames = coco_primary(ids)
    pats = {}
    for c in catnames:
        w = c.lower()
        pats[c] = re.compile(r"\b" + re.escape(w) + r"(es|s)?\b")
    n_lab = n_hit = img_hit = 0
    per_cat = collections.Counter()
    for i in ids:
        hit = False
        for d in DIMS:
            for v in values(items[i][0], d):
                n_lab += 1
                vv = v.lower()
                h = [c for c, p in pats.items() if p.search(vv)]
                if h:
                    n_hit += 1; hit = True
                    per_cat.update(h)
        img_hit += hit
    # 선형 분류기: TF-IDF(라벨 전체) -> 주요 범주(상위 30개 + other), 5겹, 릿지 least squares
    keep = [i for i in ids if i in prim]
    top = [c for c, _ in collections.Counter(prim[i] for i in keep).most_common(30)]
    y = np.array([top.index(prim[i]) if prim[i] in top else len(top) for i in keep])
    docs = [norm(" . ".join(v for d in DIMS for v in values(items[i][0], d))) for i in keep]
    vocab = {}
    rows, cols, vals = [], [], []
    df = collections.Counter()
    toks = []
    for t in docs:
        ws = t.split()
        uni = ws + [a + "_" + b for a, b in zip(ws, ws[1:])]
        c = collections.Counter(uni)
        toks.append(c)
        df.update(c.keys())
    keepv = {w for w, n in df.items() if n >= 5}
    vocab = {w: j for j, w in enumerate(sorted(keepv))}
    N = len(docs)
    for r, c in enumerate(toks):
        for w, n in c.items():
            j = vocab.get(w)
            if j is not None:
                rows.append(r); cols.append(j); vals.append((1 + np.log(n)) * np.log(N / df[w]))
    X = sp.csr_matrix((vals, (rows, cols)), shape=(N, len(vocab)))
    nrm = np.sqrt(X.multiply(X).sum(1)).A1 + 1e-9
    X = sp.diags(1 / nrm) @ X
    classes = len(top) + 1
    Y = np.eye(classes)[y]
    rng = np.random.default_rng(0)
    folds = rng.permutation(N) % 5
    pred = np.zeros(N, dtype=int)
    for k in range(5):
        tr, te = folds != k, folds == k
        W = np.zeros((X.shape[1], classes))
        for c in range(classes):
            W[:, c] = lsqr(X[tr], Y[tr, c], damp=0.5, iter_lim=200)[0]
        pred[te] = np.argmax(X[te] @ W, axis=1)
    acc = float(np.mean(pred == y))
    maj = float(np.mean(y == np.bincount(y).argmax()))
    # 상한: 사물 목록 multi-hot
    H = np.zeros((N, len(catnames)))
    for r, i in enumerate(keep):
        for c in cats.get(i, ()):
            H[r, catnames.index(c)] = 1
    pred2 = np.zeros(N, dtype=int)
    for k in range(5):
        tr, te = folds != k, folds == k
        A = H[tr].T @ H[tr] + 1.0 * np.eye(H.shape[1])
        W = np.linalg.solve(A, H[tr].T @ Y[tr])
        pred2[te] = np.argmax(H[te] @ W, axis=1)
    acc2 = float(np.mean(pred2 == y))
    return {"label_rate_with_coco_name": n_hit / n_lab, "image_rate_with_coco_name": img_hit / len(ids), "n_labels": n_lab,
            "top_names": per_cat.most_common(8), "classifier": {"n_images": N, "n_classes": classes, "accuracy": acc,
                                                              "chance_majority": maj, "object_list_upper": acc2}}


def v7_sample(ids, items, n=100, seed=0):
    prim, _, _ = coco_primary(ids)
    rng = np.random.default_rng(seed)
    strata = collections.defaultdict(list)
    for i in ids:
        cnt = sum(len(values(items[i][0], d)) for d in DIMS)
        bucket = "few" if cnt < 8 else ("mid" if cnt < 12 else "many")
        strata[(items[i][1], bucket)].append(i)
    keys = sorted(strata)
    per = max(1, n // len(keys))
    pick = []
    for k in keys:
        pool = strata[k]
        pick += list(rng.choice(pool, min(per, len(pool)), replace=False))
    rest = [i for i in ids if i not in set(pick)]
    pick += list(rng.choice(rest, n - len(pick), replace=False)) if len(pick) < n else []
    pick = pick[:n]
    path = REPO / "data/results/label_review_sample_100.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "prompt_version", "image_path", "primary_coco"] + list(DIMS) + ["verdict(맞음/부분/틀림)", "memo"])
        for i in pick:
            w.writerow([i, items[i][1], f"data/raw/images/{i}.jpg", prim.get(i, "")] + [" | ".join(values(items[i][0], d)) for d in DIMS] + ["", ""])
    return {"n": len(pick), "path": str(path.relative_to(REPO))}


def verdict(ok):
    return "통과" if ok else "미통과"


def main():
    items = load()
    ids = sorted(items)
    q = [r["cocoId"] for r in csv.DictReader(open(BASE / "queue_order.csv", encoding="utf-8"))]
    pos = {c: k for k, c in enumerate(q)}
    v10 = sorted([i for i in ids if items[i][1] == "v10"], key=lambda i: pos[i])
    v9 = [i for i in ids if items[i][1] == "v9"]
    res = {"n_images": len(ids), "n_v9": len(v9), "n_v10": len(v10)}
    res["all"] = describe(ids, items)
    res["v9"] = describe(v9, items)
    res["v10"] = describe(v10, items)
    t = len(v10) // 3
    res["v10_first"], res["v10_last"] = describe(v10[:t], items), describe(v10[-t:], items)
    res["V1"] = v1_structure(ids, items)
    res["V2"] = v2_copy(ids, items)
    res["V3"] = {d: {"top1": res["all"]["dims"][d]["top1_share"], "top10": res["all"]["dims"][d]["top10_share"], "top3": res["all"]["dims"][d]["top3"]}
                 for d in DIMS}
    res["V4"] = compare(v9, v10, items)
    res["V5"] = compare(v10[:t], v10[-t:], items)
    res["V6"] = v6_leak(ids, items)
    res["V7"] = v7_sample(ids, items)
    out = REPO / "data/results/label_summary_full.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    # 판정
    v3_ok = all(res["V3"][d]["top1"] <= 0.05 and res["V3"][d]["top10"] <= 0.20 for d in CORE_DIMS)
    j = {"V1": res["V1"]["rate"] >= 0.99, "V2": res["V2"]["image_rate"] <= 0.02, "V3": v3_ok,
         "V4": res["V4"]["labels_rel_diff"] <= 0.10 and res["V4"]["words_rel_diff"] <= 0.10,
         "V5": res["V5"]["labels_rel_diff"] <= 0.10 and res["V5"]["words_rel_diff"] <= 0.10 and res["V5"]["flag_rate_diff"] <= 0.02}
    res["judgement"] = {k: verdict(v) for k, v in j.items()}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_images", "n_v9", "n_v10", "V1", "V2", "V4", "V5", "judgement")}, ensure_ascii=False, indent=1))
    print("V3", json.dumps(res["V3"], ensure_ascii=False))
    print("V6", json.dumps(res["V6"], ensure_ascii=False))
    print("V7", res["V7"])


if __name__ == "__main__":
    main()
