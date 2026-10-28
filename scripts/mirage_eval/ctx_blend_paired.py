"""맥락 합치기 실험의 조건 간 직접 비교(짝지은 부트스트랩)와 그룹별 평균 차이를 표로 저장한다.
nsd_imagery_eval이 저장한 results/tables/ctx_blend_eval/per_stimulus_*.csv 만 읽는다(GPU, torch 불필요).
구간: 자극 12개 복원추출 5000회, 시드는 자극별로 평균(nsd_imagery_eval과 같은 방식). 시드 1~3만 쓰는 blend3은 shuf, p1seq와 시드를 맞춘 것이다.
사용: python3 scripts/mirage_eval/ctx_blend_paired.py   (저장소 루트에서)
출력: results/tables/ctx_blend_eval/paired_comparisons.md"""
import csv
import numpy as np

D = "results/tables/ctx_blend_eval"
M = ["PixCorr", "SSIM", "Alex(2)", "Alex(5)", "Incep", "CLIP", "EffNet-B"]   # 공식 7지표(보정 안 한 값)
GROUPS = {"단순 0-5": range(0, 6), "자연 6-10": range(6, 11), "그림 11": [11]}
PAIRS = [("blend", "base"), ("shuf", "base"), ("ctxonly", "base"), ("p1seq", "base"),
         ("blend3", "base"), ("blend3", "shuf"), ("p1seq", "shuf"), ("p1seq", "blend3"), ("ctxonly", "shuf")]


def load(name, maxrep=None):
    d = {}
    for r in csv.DictReader(open(f"{D}/per_stimulus_{name}.csv")):
        if maxrep is not None and int(r["rep"]) >= maxrep: continue
        d.setdefault(int(r["stimulus"]), []).append([float(r[m]) for m in M])
    return np.array([np.mean(d[s], 0) for s in range(12)])   # (12 자극, 7 지표)


def main():
    A = {n: load(n) for n in ["base", "blend", "shuf", "ctxonly", "p1seq"]}
    A["blend3"] = load("blend", 3)                            # 시드 1~3
    idx = np.random.default_rng(0).integers(0, 12, (5000, 12))
    out = ["# 맥락 합치기 조건 간 직접 비교 (짝지은 부트스트랩)\n",
           "자극 12개 복원추출 5000회, 95% 구간. `*` = 구간이 0을 제외(28개 이상 비교 중 몇 개는 우연히 나올 수 있음). 탐색 결과이며 결론이 아님. "
           "blend3 = blend의 시드 1~3(shuf, p1seq와 시드를 맞춤). 피험자 1, 자극 12개(단순 6, 자연 5, 그림 1).\n"]
    for a, b in PAIRS:
        d = A[a] - A[b]; lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5], 0)
        out.append(f"\n## {a} − {b}\n\n| 지표 | 평균 차이 | 95% 구간 | |\n|---|---|---|---|")
        for i, k in enumerate(M):
            out.append(f"| {k} | {d[:, i].mean():+.3f} | [{lo[i]:+.3f}, {hi[i]:+.3f}] | {'*' if lo[i] > 0 or hi[i] < 0 else ''} |")
    out.append("\n## 그룹별 평균 차이 (기준선 대비, 참고용: 그림은 자극 1개)\n\n| 조건 | 그룹 | " + " | ".join(M) + " |\n|---|---|" + "---|" * len(M))
    for n in ["blend3", "shuf", "ctxonly", "p1seq"]:
        for g, s in GROUPS.items():
            d = (A[n] - A["base"])[list(s)].mean(0)
            out.append(f"| {n} | {g} | " + " | ".join(f"{x:+.3f}" for x in d) + " |")
    open(f"{D}/paired_comparisons.md", "w").write("\n".join(out) + "\n")
    print(f"저장: {D}/paired_comparisons.md")


if __name__ == "__main__":
    main()
