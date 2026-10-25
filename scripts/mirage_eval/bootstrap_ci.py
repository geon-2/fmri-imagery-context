"""자극 12개를 복원추출(bootstrap)해서 각 지표 평균의 95% 구간과, 첫 번째 설정 대비 차이의 95% 구간을 구한다.
이 구간은 '자극을 다시 뽑았을 때'의 불확실성만 반영한다(샘플링 반복, 피험자 변동은 포함하지 않음). 사용: python bootstrap_ci.py <recons1.pt> <recons2.pt> ..."""
import sys, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from paper_metrics import *
gt = torch.load(GT_PATH, map_location="cpu")[:N_GT].float().clamp(0, 1)
ga = feats_alex(gt, ["l2", "l5"]); gi = feats_incep(gt); gc = feats_clip(gt)
rng = np.random.default_rng(0); B = 5000
per = {}
for p in sys.argv[1:]:
    name = p.split("/")[-1].replace("_all_recons_imagery.pt", "").replace("subj01_", "").replace("_server_ridge_sc2", "")
    rec = load_recons(p); ra = feats_alex(rec, ["l2", "l5"])
    per[name] = {"Alex(2)": per_stim_two_way(ga["l2"], ra["l2"]), "Alex(5)": per_stim_two_way(ga["l5"], ra["l5"]),
                 "Incep": per_stim_two_way(gi, feats_incep(rec)), "CLIP": per_stim_two_way(gc, feats_clip(rec))}
names = list(per); base = names[0]
idx = rng.integers(0, N_GT, size=(B, N_GT))
for m in ["Alex(2)", "Alex(5)", "Incep", "CLIP"]:
    print(f"== {m}")
    for n in names:
        x = per[n][m]; bs = x[idx].mean(1)
        line = f"  {n:12s} 평균 {x.mean():.3f}  95% [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]"
        if n != base:
            d = (per[n][m] - per[base][m])[idx].mean(1)
            line += f"   {base} 대비 차이 {(per[n][m]-per[base][m]).mean():+.3f} 95% [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]"
        print(line)
