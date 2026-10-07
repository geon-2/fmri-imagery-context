"""NSD-Imagery: 영역별 imagery 활성과 재현성(영역 선택용). 지표와 규칙은 계산 전에 Notion에 기록했다(2026-10-08).

영역 선택에는 세트 A(도형)와 C(단어)의 imagery 시행만 쓴다. 평가 대상인 세트 B(자연 장면)의 imagery 반응은 이 스크립트가 읽지 않는다.
베타 순서: betas_nsdimagery는 run 순서(0~11)이고, 한 run 안에서는 설계 행렬(designmatrixGLMsingle.mat)의 조건(열)별로 묶여 있다(열마다 8회, 열 안에서는 시행 순).
  확인: 인접 시행 상관의 구조(run당 48 또는 96 시행 블록, 같은 조건 8개씩 연속)와 viewing 식별 정확도. 처음에는 조건(열) 순서 후 run 순서로 가정해 imagery 식별이 우연 이하로 나왔고, 위 구조를 확인해 바로잡았다.
  열 0~5 viewing(run 0, 3, 6 = 세트 A, B, C), 6~17 attention, 18~23 imagery A(run 2, 9), 24~29 imagery B(run 5, 10), 30~35 imagery C(run 8, 11).
지표: 영역별 imagery 평균 베타(세트 A∪C), 같은 세트 viewing 평균 베타, imagery run 간 식별 정확도(target 6개, 우연 1/6),
  viewing 식별(한 run의 8회를 홀짝으로 나눔, 양성 대조), target 라벨 순열 1,000회.
사용: python3 imagery_roi.py --out-dir data/results/imagery_roi
시험: python3 imagery_roi.py --selftest
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io as sio

from axes_k_curve import save_json

ROOT = Path("data/raw")
ROIS = ["EARLY", "S_midventral", "S_midlateral", "S_midparietal", "S_ventral", "S_lateral", "S_parietal", "places", "faces", "bodies", "words", "R2R3"]
SETS = {"A": {"vis_run": 0, "img_cols": range(18, 24), "img_runs": (2, 9)},
        "C": {"vis_run": 6, "img_cols": range(30, 36), "img_runs": (8, 11)}}   # B(자연 장면)는 읽지 않는다


def trial_table():
    """betas의 열 순서(조건, run, 시행 순)에 대한 (run, column) 목록."""
    st = sio.loadmat(str(ROOT / "nsd_imagery/exp/designmatrixGLMsingle.mat"))["stimulus"][0]
    runs, cols = [], []
    for r, x in enumerate(st):
        rows, c = np.nonzero(np.asarray(x))
        o = np.argsort(rows)
        runs += [r] * len(o)
        cols += c[o].tolist()
    runs, cols = np.array(runs), np.array(cols)
    order = np.lexsort((np.arange(len(cols)), cols, runs))   # (run, 열, 시행): run 순서, 한 run 안에서는 조건(열)별로 묶여 있음
    return runs[order], cols[order]


def ident_accuracy(P1, P2):
    """P1, P2: target × voxel 패턴. 각 패턴에서 target 평균을 뺀 뒤 같은 target끼리의 상관이 가장 큰 비율(양방향 평균)."""
    def z(P):
        P = P - P.mean(0, keepdims=True)
        return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    C = z(P1) @ z(P2).T
    return float(((C.argmax(1) == np.arange(len(C))).mean() + (C.argmax(0) == np.arange(len(C))).mean()) / 2), C


def perm_p(P1, P2, obs, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        a, _ = ident_accuracy(P1, P2[rng.permutation(len(P2))])
        cnt += a >= obs
    return (cnt + 1) / (n + 1)


def roi_stats(B, runs, cols, vox_idx):
    """B: 시행 × 복셀(선택된 복셀들, 이미 베타 단위). 세트 A, C의 지표를 낸다."""
    out = {"imagery_beta": {}, "viewing_beta": {}}
    P1s, P2s, V1s, V2s, imgs, vis = [], [], [], [], [], []
    for s, info in SETS.items():
        for k, cs in enumerate(info["img_cols"]):
            pass
        pats = []
        for r in info["img_runs"]:
            m = np.isin(cols, list(info["img_cols"])) & (runs == r)
            Xr, cr = B[m][:, vox_idx], cols[m]
            pats.append(np.stack([Xr[cr == c].mean(0) for c in info["img_cols"]]))
        P1s.append(pats[0]); P2s.append(pats[1])
        imgs.append(B[np.isin(cols, list(info["img_cols"]))][:, vox_idx])
        mv = (runs == info["vis_run"]) & (cols < 6)
        Xv, cv = B[mv][:, vox_idx], cols[mv]
        h1, h2 = [], []
        for c in range(6):
            tr = Xv[cv == c]
            h1.append(tr[0::2].mean(0)); h2.append(tr[1::2].mean(0))
        V1s.append(np.stack(h1)); V2s.append(np.stack(h2))
        vis.append(Xv)
    P1, P2 = np.concatenate(P1s), np.concatenate(P2s)      # 12 target
    V1, V2 = np.concatenate(V1s), np.concatenate(V2s)
    # 세트별로 식별(후보는 같은 세트의 6개)한 뒤 평균
    acc_img = float(np.mean([ident_accuracy(P1s[i], P2s[i])[0] for i in range(2)]))
    acc_vis = float(np.mean([ident_accuracy(V1s[i], V2s[i])[0] for i in range(2)]))
    # 순열: 세트 안에서 target을 섞는다
    rng = np.random.default_rng(0)
    cnt_i = cnt_v = 0
    N = 1000
    for _ in range(N):
        ai = np.mean([ident_accuracy(P1s[i], P2s[i][rng.permutation(6)])[0] for i in range(2)])
        av = np.mean([ident_accuracy(V1s[i], V2s[i][rng.permutation(6)])[0] for i in range(2)])
        cnt_i += ai >= acc_img
        cnt_v += av >= acc_vis
    out.update(imagery_ident=acc_img, imagery_p=(cnt_i + 1) / (N + 1), viewing_ident=acc_vis, viewing_p=(cnt_v + 1) / (N + 1),
               imagery_mean_beta=float(np.concatenate(imgs).mean()), viewing_mean_beta=float(np.concatenate(vis).mean()),
               imagery_mean_abs_beta=float(np.abs(np.concatenate(imgs)).mean()))
    return out


def load_subject(subj):
    R = ROOT / f"nsd_fmri/{subj}/roi_betas"
    v = np.load(R / "voxels.npz")
    g = np.load(R / "roi_groups.npz")
    img = nib.load(ROOT / f"nsd_imagery/{subj}/betas_nsdimagery.nii.gz")
    assert tuple(img.shape[:3]) == tuple(v["shape"]), (img.shape, v["shape"])
    arr = np.asanyarray(img.dataobj).reshape(-1, img.shape[-1])
    B = arr[v["flat_index"]].T.astype(np.float32) / 300.0        # 시행 × (core의 복셀 집합)
    masks = {r: g[r] for r in ROIS if r in g.files}
    masks["R2R3"] = g["R2"] | g["R3"]
    return B, masks


def selftest():
    rng = np.random.default_rng(0)
    # 신호 있는 영역(복셀 0~49)과 잡음 영역(50~99): 12 target 중 target별 고유 패턴 + 잡음, run 사이 같은 패턴
    V = 100
    pat = rng.standard_normal((12, V))
    runs = np.concatenate([[r] * 48 for r in range(12)])
    cols = np.zeros(576, int)
    B = rng.standard_normal((576, V)) * 2
    # 세트 A(imagery run 2, 9; 열 18~23), C(run 8, 11; 열 30~35), viewing run 0, 6(열 0~5)
    for run, cs in ((2, range(18, 24)), (9, range(18, 24)), (8, range(30, 36)), (11, range(30, 36))):
        m = runs == run
        cols[m] = np.tile(list(cs), 8)
        k = np.array([c - cs[0] for c in cols[m]]) + (0 if cs[0] == 18 else 6)
        B[np.where(m)[0], :50] += pat[k, :50] * 1.0
    for run in (0, 6):
        m = runs == run
        cols[m] = np.tile(list(range(6)), 8)
        k = np.array(cols[m]) + (0 if run == 0 else 6)
        B[np.where(m)[0], :50] += pat[k, :50] * 1.0
    a = roi_stats(B, runs, cols, np.arange(50))
    b = roi_stats(B, runs, cols, np.arange(50, 100))
    print("영역 시험: 신호 영역 식별 %.2f(p=%.3f), 잡음 영역 %.2f(p=%.3f)" % (a["imagery_ident"], a["imagery_p"], b["imagery_ident"], b["imagery_p"]))
    assert a["imagery_ident"] > 0.8 and a["imagery_p"] < 0.01
    assert b["imagery_ident"] < 0.5 and b["imagery_p"] > 0.05
    print("selftest 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/results/imagery_roi")
    ap.add_argument("--subjects", default="subj01,subj02,subj05,subj07")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    runs, cols = trial_table()
    allres = {}
    for s in a.subjects.split(","):
        B, masks = load_subject(s)
        res = {}
        for r, m in masks.items():
            res[r] = roi_stats(B, runs, cols, np.flatnonzero(m))
            x = res[r]
            print(f"{s} {r:14s} 복셀 {int(m.sum()):6d}  imagery 식별 {x['imagery_ident']:.2f} (p={x['imagery_p']:.3f})  viewing 식별 {x['viewing_ident']:.2f} (p={x['viewing_p']:.3f})  "
                  f"평균 베타 imagery {x['imagery_mean_beta']:+.3f} / viewing {x['viewing_mean_beta']:+.3f}", flush=True)
        allres[s] = res
    save_json(allres, Path(a.out_dir) / "imagery_roi_selection_sets_AC.json")


if __name__ == "__main__":
    main()
