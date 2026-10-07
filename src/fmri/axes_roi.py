"""RQ1 보강: 영역(ROI)별 저차원 하위공간. 영역을 합치지 않고 각각 따로 적합한다.

영역 합쳐서 구한 하위공간(axes_k_curve.py의 기본, R2∪R3)이 영역들의 혼합인지 확인한다.
  - 영역별 k 곡선과 k*(RRR/PCA), 점수는 그 영역 복셀의 median r
  - 영역 간 하위공간 겹침(평균 cos², 무작위 기대값 k/d, 같은 영역 반분할이 참조)
  - 영역 간 전이: 영역 a의 방향 k개를 영역 b의 복셀에 적용했을 때 b 자신의 방향과 비교
영역은 roi_groups.npz에 들어 있는 이름(make_roi_groups.py): places, faces, bodies, words, S_early, S_midventral,
S_midlateral, S_midparietal, S_ventral, S_lateral, S_parietal, EARLY, R1(nsdgeneral), R2, R3. 영역끼리 겹치는 복셀이 있다
(예: floc 복셀은 대개 streams의 ventral/lateral/parietal에도 속한다). 신뢰도 0.1 이상 복셀이 min-voxels보다 적은 영역은 건너뛴다.

사용(Colab): python3 axes_roi.py --fmri-dir .../roi_betas --emb-dir .../embeddings_v2 --model MPNet --out-dir .../axes_roi --tag subj01
"""
import argparse
import json
from pathlib import Path

import numpy as np

import embedding_compare as ec
from axes_k_curve import run, save_json
from axes_shared import fit_dirs, orth, overlap, summarize_transfer, transfer_scores, within_split_half
from text_compare import load_brain

ROIS = ["EARLY", "S_midventral", "S_midlateral", "S_midparietal", "S_ventral", "S_lateral", "S_parietal", "places", "faces", "bodies", "words", "R2", "R3"]
KS = (5, 12, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--model", default="MPNet")
    ap.add_argument("--rep", default="ctx_P1")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tag", default="subj01")
    ap.add_argument("--rois", default=",".join(ROIS))
    ap.add_argument("--min-voxels", type=int, default=150)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    rois = a.rois.split(",")
    emb = Path(a.emb_dir)
    ids_all = json.load(open(emb / "ctx_ids.json"))
    E = np.load(emb / f"{a.model}__{a.rep}.npy").astype(np.float32)
    S = (E - E.mean(0)) / (E.std(0) + 1e-6)
    row = {c: i for i, c in enumerate(ids_all)}
    train, Ytr, test, Yte, masks, nk, nv = load_brain(a.fmri_dir, ids_all, extra=tuple(r for r in rois if r not in ("R1", "R2", "R3", "EARLY")))
    X = S[[row[c] for c in train]]
    d = X.shape[1]
    log = lambda s: print(s, flush=True)
    data, res = {}, {"tag": a.tag, "model": a.model, "rep": a.rep, "n_images": len(train), "rois": {}, "chance": {k: k / d for k in KS}}
    for r in rois:
        n = int(masks[r].sum())
        if n < a.min_voxels:
            log(f"{r}: 신뢰도 통과 복셀 {n}개 < {a.min_voxels}, 건너뜀")
            continue
        Yr = Ytr[:, masks[r]]
        data[r] = Yr
        full_mask = {"R2": np.ones(n, bool), "R3": np.ones(n, bool)}
        log(f"===== {r} ({n}복셀)")
        k_res = run(X, Yr, full_mask, B=a.boot, log=log, n_rand=1)
        res["rois"][r] = {"n_voxels": n, "full_P": k_res["full"]["P"], "kstar": k_res["kstar"], "curves": k_res["curves"],
                          "rrr_minus_pca": k_res.get("rrr_minus_pca"), "split_half": within_split_half(X, Yr, KS)}
    names = list(data)
    D = {r: fit_dirs(X, data[r]) for r in names}
    res["overlap"] = {k: {f"{x}|{y}": overlap(orth(D[x]["RRR"], k), orth(D[y]["RRR"], k)) for i, x in enumerate(names) for y in names[i + 1:]} for k in KS}
    res["transfer"] = {}
    for b in names:
        src = {s: D[s]["RRR"] for s in names if s != b}
        log(f"전이 대상 {b}")
        out = transfer_scores(src, (X, data[b]), KS, {"R2": np.ones(data[b].shape[1], bool), "R3": np.ones(data[b].shape[1], bool)}, B=a.boot, log=lambda s: None)
        res["transfer"][b] = summarize_transfer(out)
        k = 12
        r = res["transfer"][b][k]
        log(f"  k={k}: own_RRR={r['own_RRR']['P']:.3f} own_PCA={r['own_PCA']['P']:.3f} RAND={r['RAND']['P']:.3f} | " +
            " ".join(f"{n[5:]}={v['P']:.3f}" for n, v in r.items() if n.startswith("from_") and "_minus_" not in n))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(res, out / f"axes_roi_{a.tag}_{a.model}_{a.rep}.json")
    log("\n영역별 요약(점수는 그 영역 복셀의 median r):")
    log(f"{'ROI':14s} {'복셀':>6s} {'full P':>7s} {'k*RRR':>6s} {'RRR k=5':>8s} {'RRR k=12':>9s} {'PCA k=12':>9s} {'반분할겹침 k=12':>15s}")
    for r in names:
        v = res["rois"][r]
        c = {m: {x["k"]: x["P"] for x in v["curves"][m]} for m in ("RRR", "PCA")}
        log(f"{r:14s} {v['n_voxels']:6d} {v['full_P']:7.3f} {v['kstar']['RRR']:6d} {c['RRR'][5]:8.3f} {c['RRR'][12]:9.3f} {c['PCA'][12]:9.3f} {v['split_half'][12]:15.2f}")
    log("\n영역 간 하위공간 겹침(k=12, 무작위 %.3f):" % res["chance"][12])
    for pair, val in sorted(res["overlap"][12].items(), key=lambda t: -t[1])[:12]:
        log(f"  {pair:28s} {val:.2f}")


if __name__ == "__main__":
    main()
