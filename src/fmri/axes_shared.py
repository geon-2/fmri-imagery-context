"""RQ1 보강 확인. 「저차원 하위공간이 있다」를 「공유되고 해석 가능하다」로 넓힐 수 있는지 본다.

(A) 대조 k 곡선: axes_k_curve.py --rep obj_P1 / orig_P1 로 같은 곡선을 사물·원본에 대해 그린다(이 파일에 코드 없음).
(B) 피험자 간 공유(맥락 임베딩 공간이 모든 피험자에게 같으므로 직접 비교한다):
    - 하위공간 겹침: 피험자 a, b의 RRR 입력 방향(전체 학습 표본으로 적합)의 평균 cos². 기준선은 무작위 k/d와 같은 피험자 내
      절반 분할(상한에 가까운 참조, 표본이 절반이라 보수적).
    - 기능적 전이: a의 방향 k개를 그대로 b에 적용(b의 학습 폴드에서 그 k차원 OLS만 다시 적합)해 b의 교차검증 P를 잰다.
      같은 k에서 b 자신의 RRR 방향, b의 PCA 방향, 무작위 방향과 비교한다. a의 방향은 b의 이미지와 겹치지 않는다(피험자별 고유 이미지).
    모든 입력은 라벨 있는 전체 이미지의 평균·표준편차로 공통 표준화한다(피험자별로 다르게 표준화하면 방향을 비교할 수 없다).
(C) 해석: 한 피험자의 RRR 상위 방향마다 점수가 가장 높은/낮은 이미지와, 차원(장소 유형, 환경, 활동, 암묵적 사건, 규모, 기타)별
    단어 연관(해당 방향 점수 평균의 Cohen's d)을 낸다. 방향의 부호는 임의다. 개별 축이 아니라 하위공간 단위로 읽는다.

사용(Colab): python3 axes_shared.py --subjects subj01:.../fmri/roi_betas,subj02:.../fmri_subj02,... --emb-dir ... --bundle-dir ... --out-dir ... --model MPNet
시험:        python3 axes_shared.py --selftest
"""
import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np

import embedding_compare as ec
from axes_k_curve import Scorer, overlap, predict_dir, ridge_dirs, save_json

KS = (5, 12, 20, 32)
STOP = set("a an the of in on at to and or with for from by is are was be as it its this that their there into near next than very some one two more most while being has have had".split())


def orth(D, k):
    return np.linalg.qr(D[:, :k])[0]


def fit_dirs(X, Y, seed=0):
    """공통 표준화된 X와 열 중심화된 Y 전체로 RRR·PCA 방향을 구한다."""
    ym = Y.mean(0)
    Yc = Y - ym
    ai = ec.choose_alpha(X, Y, np.random.default_rng(seed))
    return ridge_dirs(X, Yc, ai)[1]


def within_split_half(X, Y, ks, reps=3, seed=0):
    rng = np.random.default_rng(seed + 31)
    out = {k: [] for k in ks}
    for _ in range(reps):
        p = rng.permutation(len(X))
        a, b = p[: len(p) // 2], p[len(p) // 2:]
        Da, Db = fit_dirs(X[a], Y[a], seed)["RRR"], fit_dirs(X[b], Y[b], seed)["RRR"]
        for k in ks:
            out[k].append(overlap(orth(Da, k), orth(Db, k)))
    return {k: float(np.mean(v)) for k, v in out.items()}


def transfer_scores(Dsrc_by_subj, tgt, ks, masks, B=200, seed=0, log=print):
    """tgt = (X, Y). Dsrc_by_subj: {출처 이름: D(d×d, 중요도 순)}. 반환 {k: {이름: P, ...}}과 부트스트랩 차이."""
    X, Y = tgt
    n, d = X.shape
    folds = ec.outer_folds(n, seed)
    W = ec.boot_weights(n, B, seed + 7)
    sc = Scorer(Y, folds, masks, W)
    rnd = np.linalg.qr(np.random.default_rng(seed + 9).standard_normal((d, max(ks))))[0]
    fit = []
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        ym = Y[tr].mean(0)
        Yc = Y[tr] - ym
        D = ridge_dirs(X[tr], Yc, ec.choose_alpha(X[tr], Y[tr], np.random.default_rng(seed + 1)))[1]
        fit.append((f, tr, ym, Yc, D))
    log("  폴드 준비 완료")
    out = {}
    for k in ks:
        res = {}
        specs = {"own_RRR": lambda D: D["RRR"], "own_PCA": lambda D: D["PCA"], "RAND": lambda D: rnd}
        specs.update({f"from_{s}": (lambda D, s=s: Dsrc_by_subj[s]) for s in Dsrc_by_subj})
        for name, get in specs.items():
            oof = np.empty(Y.shape, np.float32)
            for f, tr, ym, Yc, D in fit:
                oof[f] = predict_dir(X[tr], X[f], Yc, ym, get(D), k)
            res[name] = sc.score(oof)
        out[k] = res
    return out


def summarize_transfer(out):
    rows = {}
    for k, res in out.items():
        r = {}
        for name, v in res.items():
            r[name] = {"P": v["P"], "ci": [float(np.percentile(v["boot"], 2.5)), float(np.percentile(v["boot"], 97.5))]}
        for name in res:
            if name.startswith("from_"):
                for ref in ("own_RRR", "own_PCA", "RAND"):
                    dd = res[name]["boot"] - res[ref]["boot"]
                    r[f"{name}_minus_{ref}"] = {"diff": res[name]["P"] - res[ref]["P"], "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
        rows[k] = r
    return rows


def words(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 3 and w not in STOP]


def interpret(S, ids, labels, caps, D, k=12, top=8, min_n=40, topw=10):
    """S: 공통 표준화된 N×d 임베딩, ids: 이미지 id, labels/caps: id → 라벨·캡션. D: d×d 방향(중요도 순)."""
    import text_variants as tv
    Z = S @ D[:, :k]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    by = [tv.context_by_dim(labels[i]["labels"]) for i in ids]
    out = []
    for j in range(k):
        z = Z[:, j]
        order = np.argsort(z)
        ex = lambda idx: [{"cocoId": ids[i], "z": float(z[i]), "caption": caps[ids[i]]["captions"][0],
                           "context": "; ".join(p for d in tv.DIMS for p in by[i][d][:2])} for i in idx]
        assoc = {}
        for d in tv.DIMS:
            cnt = {}
            for i, b in enumerate(by):
                for w in {w for ph in b[d] for w in words(ph)}:
                    cnt.setdefault(w, []).append(i)
            rows = []
            for w, idx in cnt.items():
                if len(idx) < min_n:
                    continue
                m = np.zeros(len(z), bool)
                m[idx] = True
                sd = np.sqrt((z[m].var() * m.sum() + z[~m].var() * (~m).sum()) / len(z)) + 1e-9
                rows.append((w, float((z[m].mean() - z[~m].mean()) / sd), int(m.sum())))
            rows.sort(key=lambda r: r[1])
            assoc[d] = {"low": rows[:topw], "high": rows[::-1][:topw],
                        "max_abs_d": float(max([abs(r[1]) for r in rows[:topw] + rows[-topw:]] or [0]))}
        out.append({"direction": j + 1, "top": ex(order[::-1][:top]), "bottom": ex(order[:top]), "assoc": assoc})
    return out


def selftest():
    rng = np.random.default_rng(0)
    d, V, n = 24, 40, 700
    A = np.linalg.qr(rng.standard_normal((d, d)))[0]
    shared = A[:, :3] * 3.0          # 공유되는 3방향
    Wsh = rng.standard_normal((3, V))

    def subject(own):
        X = rng.standard_normal((n, d)).astype(np.float32)
        Y = (X @ shared @ Wsh * 2 + (X @ own @ rng.standard_normal((own.shape[1], V)) if own is not None else 0) + rng.standard_normal((n, V)) * 2).astype(np.float32)
        return X, Y
    B_ = np.linalg.qr(rng.standard_normal((d, d)))[0][:, :3]
    (Xa, Ya), (Xb, Yb) = subject(None), subject(None)
    Da, Db = fit_dirs(Xa, Ya)["RRR"], fit_dirs(Xb, Yb)["RRR"]
    ov = overlap(orth(Da, 3), orth(Db, 3))
    assert ov > 0.7 and ov > 5 * 3 / d, ov                        # 같은 구조 → 겹침 높음
    Xc = rng.standard_normal((n, d)).astype(np.float32)
    Yc = (Xc @ (np.linalg.qr(rng.standard_normal((d, d)))[0][:, :3] * 3.0) @ rng.standard_normal((3, V)) * 2 + rng.standard_normal((n, V)) * 2).astype(np.float32)
    Dc = fit_dirs(Xc, Yc)["RRR"]
    assert overlap(orth(Da, 3), orth(Dc, 3)) < 0.35                # 다른 구조 → 겹침 낮음
    masks = {"R2": np.arange(V) < 30, "R3": np.arange(V) >= 10}
    out = summarize_transfer(transfer_scores({"a": Da}, (Xb, Yb), (3,), masks, B=30, log=lambda s: None))[3]
    assert out["from_a_minus_RAND"]["ci"][0] > 0.05 and abs(out["from_a_minus_own_RRR"]["diff"]) < 0.05, out
    out2 = summarize_transfer(transfer_scores({"c": Dc}, (Xb, Yb), (3,), masks, B=30, log=lambda s: None))[3]
    assert out2["from_c_minus_own_RRR"]["diff"] < -0.1, out2        # 다른 구조에서 온 방향은 전이하지 못한다
    print("selftest 통과", {"overlap_same": round(ov, 2)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", help="이름:fMRI 폴더, 쉼표로 구분(예: subj01:.../roi_betas,subj02:.../fmri_subj02)")
    ap.add_argument("--emb-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--model", default="MPNet")
    ap.add_argument("--rep", default="ctx_P1")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--interpret", default="subj01", help="해석할 피험자")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    from text_compare import load_brain
    emb = Path(a.emb_dir)
    ids_all = json.load(open(emb / "ctx_ids.json"))
    E = np.load(emb / f"{a.model}__{a.rep}.npy").astype(np.float32)
    mu, sd = E.mean(0), E.std(0) + 1e-6
    S = (E - mu) / sd
    row = {c: i for i, c in enumerate(ids_all)}
    subj = {}
    for item in a.subjects.split(","):
        name, path = item.split(":", 1)
        train, Ytr, test, Yte, masks, nk, nv = load_brain(path, ids_all)
        keep = masks["R2"] | masks["R3"]
        subj[name] = {"X": S[[row[c] for c in train]], "Y": Ytr[:, keep], "masks": {g: m[keep] for g, m in masks.items()}}
        print(f"{name}: {len(train)}장, 복셀 {int(keep.sum())}", flush=True)
    names = list(subj)
    D = {s: fit_dirs(subj[s]["X"], subj[s]["Y"]) for s in names}
    d = S.shape[1]
    res = {"model": a.model, "rep": a.rep, "ks": list(KS), "overlap": {}, "split_half": {}, "chance": {k: k / d for k in KS}, "transfer": {}}
    for k in KS:
        res["overlap"][k] = {f"{x}-{y}": overlap(orth(D[x]["RRR"], k), orth(D[y]["RRR"], k)) for i, x in enumerate(names) for y in names[i + 1:]}
        res["overlap"][k]["PCA_vs_RRR(same subject)"] = float(np.mean([overlap(orth(D[s]["RRR"], k), orth(D[s]["PCA"], k)) for s in names]))
    for s in names:
        res["split_half"][s] = within_split_half(subj[s]["X"], subj[s]["Y"], KS)
    print("겹침:", json.dumps(res["overlap"], default=float)[:1500], "\n반분할:", json.dumps(res["split_half"], default=float), flush=True)
    for t in names:
        src = {s: D[s]["RRR"] for s in names if s != t}
        print(f"전이 대상 {t} (출처 {list(src)})", flush=True)
        res["transfer"][t] = summarize_transfer(transfer_scores(src, (subj[t]["X"], subj[t]["Y"]), KS, subj[t]["masks"], B=a.boot, log=lambda s: print(s, flush=True)))
        for k in KS:
            r = res["transfer"][t][k]
            print(f"  k={k}: own_RRR={r['own_RRR']['P']:.3f} own_PCA={r['own_PCA']['P']:.3f} RAND={r['RAND']['P']:.3f} | " +
                  " ".join(f"{n}={v['P']:.3f}" for n, v in r.items() if n.startswith("from_") and "_minus_" not in n), flush=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(res, out / f"axes_shared_{a.model}_{a.rep}.json")
    # 해석
    cap = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "captions.jsonl.gz", "rt"))}
    lab = {r["cocoId"]: r for r in map(json.loads, gzip.open(Path(a.bundle_dir) / "labels_pass0.jsonl.gz", "rt"))}
    interp = interpret(S, ids_all, lab, cap, D[a.interpret]["RRR"], k=a.k)
    save_json({"subject": a.interpret, "k": a.k, "directions": interp}, out / f"axes_interpret_{a.interpret}_{a.model}_{a.rep}.json")
    for it in interp[:4]:
        print(f"\n== 방향 {it['direction']}: 높은 쪽 예시")
        for e in it["top"][:3]:
            print("  ", e["cocoId"], e["caption"][:70], "|", e["context"][:90])
        print("   낮은 쪽 예시")
        for e in it["bottom"][:3]:
            print("  ", e["cocoId"], e["caption"][:70], "|", e["context"][:90])
        for dname, v in it["assoc"].items():
            print(f"   [{dname}] 높음: {[w for w, _, _ in v['high'][:6]]}  낮음: {[w for w, _, _ in v['low'][:6]]}")
    print("저장:", out)


if __name__ == "__main__":
    main()
