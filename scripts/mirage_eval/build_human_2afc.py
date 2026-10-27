"""NSD-Imagery 사람 평가(논문 Table 2A, 2AFC 식별)를 혼자 수행하기 위한 문항 생성.
기준 이미지(원본)와 후보 둘(정답 재구성 / 같은 피험자·같은 종류의 다른 자극 재구성)을 만든다.
정답 위치 등 키는 key.json에만 두고 평가 페이지(site/)에는 넣지 않는다.
사용: python scripts/mirage_eval/build_human_2afc.py [--seed 0]   (이미 있으면 --force 없이는 덮어쓰지 않음)"""
import argparse, json, os, random, secrets, shutil
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

R = "data/results/mirage_server"
OUT = "data/results/human_eval"          # data/results/ 는 git 추적 안 함(NSD 파생 이미지)
SITE, IMG = f"{OUT}/site", f"{OUT}/site/img"
SIZE = 384
SUBJECTS = {  # 피험자: (폴더, 시드 목록) — 피험자 1만 retrieval 16
    "subj01": ("subj01_40sess_server_ridge_sc2_ret16", [1, 2, 3]),
    "subj02": ("subj02_40sess_server_ridge_sc2", [1, 2, 3, 4, 5]),
    "subj05": ("subj05_40sess_server_ridge_sc2", [1, 2, 3, 4, 5]),
    "subj07": ("subj07_40sess_server_ridge_sc2", [1, 2, 3, 4, 5]),
}
TYPE = {i: ("simple" if i < 6 else "complex") for i in range(12)}
SAMPLES_PER_CELL, N_ATTN = 2, 6

def small(t):  # (3,H,W) float 0-1 -> uint8 (SIZE,SIZE,3)
    t = F.interpolate(t[None].float(), size=(SIZE, SIZE), mode="area" if t.shape[-1] > SIZE else "bilinear")[0]
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if os.path.exists(f"{OUT}/key.json") and not a.force:
        raise SystemExit(f"{OUT}/key.json 이 이미 있다. 이미 응답을 받았다면 덮어쓰지 말 것(--force).")
    rng = random.Random(a.seed)
    gt = torch.load(f"{R}/imagery_stimuli_18.pt", map_location="cpu")[:12]
    gt_img = [small(g) for g in gt]
    rec = {}
    for s, (d, seeds) in SUBJECTS.items():
        for sd in seeds:
            x = torch.load(f"{R}/{d}/{d}_all_recons_imagery_seed{sd}.pt", map_location="cpu")[:12, 0]
            rec[(s, sd)] = [small(x[i]) for i in range(12)]
            del x

    trials = []  # 내부 정보 포함
    for s, (_, seeds) in SUBJECTS.items():
        for stim in range(12):
            tgt_seeds = rng.sample(seeds, SAMPLES_PER_CELL)
            for ts in tgt_seeds:
                ds = rng.choice([i for i in range(12) if TYPE[i] == TYPE[stim] and i != stim])
                trials.append(dict(kind="main", subj=s, stim=stim, type=TYPE[stim], target_seed=ts,
                                   distractor_stim=ds, distractor_seed=rng.choice(seeds)))
    cells = [(s, st) for s in SUBJECTS for st in range(12)]
    simple = [c for c in cells if TYPE[c[1]] == "simple"]; complex_ = [c for c in cells if TYPE[c[1]] == "complex"]
    for s, stim in rng.sample(simple, N_ATTN // 2) + rng.sample(complex_, N_ATTN - N_ATTN // 2):
        seeds = SUBJECTS[s][1]
        ds = rng.choice([i for i in range(12) if TYPE[i] == TYPE[stim] and i != stim])
        trials.append(dict(kind="attention", subj=s, stim=stim, type=TYPE[stim], target_seed=None,
                           distractor_stim=ds, distractor_seed=rng.choice(seeds)))
    rng.shuffle(trials)

    shutil.rmtree(SITE, ignore_errors=True); os.makedirs(IMG)
    key, page = [], []
    for n, t in enumerate(trials):
        tid = secrets.token_hex(4)
        tgt = gt_img[t["stim"]] if t["kind"] == "attention" else rec[(t["subj"], t["target_seed"])][t["stim"]]
        dis = rec[(t["subj"], t["distractor_seed"])][t["distractor_stim"]]
        t["target_side"] = rng.choice("LR")
        cand = {t["target_side"]: tgt, "L" if t["target_side"] == "R" else "R": dis}
        files = {"ref": f"{tid}_r.jpg", "L": f"{tid}_a.jpg", "R": f"{tid}_b.jpg"}
        Image.fromarray(gt_img[t["stim"]]).save(f"{IMG}/{files['ref']}", quality=92)
        Image.fromarray(cand["L"]).save(f"{IMG}/{files['L']}", quality=92)
        Image.fromarray(cand["R"]).save(f"{IMG}/{files['R']}", quality=92)
        key.append(dict(id=tid, order=n, **t)); page.append(dict(id=tid, **files))
    shutil.copy("scripts/mirage_eval/human_2afc_index.html", f"{SITE}/index.html")
    json.dump(dict(seed=a.seed, trials=key), open(f"{OUT}/key.json", "w"), ensure_ascii=False, indent=1)
    open(f"{SITE}/trials.js", "w").write("window.TRIALS = " + json.dumps(page) + ";\n")
    print(f"문항 {len(key)}개 (본 {sum(k['kind']=='main' for k in key)} + 주의력 {sum(k['kind']=='attention' for k in key)}) -> {OUT}")

main()
