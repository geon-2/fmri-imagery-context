"""NSD-Imagery 자극 원본과 피험자별 재구성을 나란히 그려 사람이 직접 보게 한다(사람 평가용 육안 확인).
사용: python scripts/mirage_eval/make_human_view.py [seed(기본 1)]
행: 원본, 피험자 1(retrieval 16), 2, 5, 7. 열: 자극. 자극 0-5 단순 도형, 6-10 자연 장면, 11 그림."""
import sys, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
R = "data/results/mirage_server"
files = {
    "subj01 (ret16)": f"{R}/subj01_40sess_server_ridge_sc2_ret16/subj01_40sess_server_ridge_sc2_ret16_all_recons_imagery_seed{seed}.pt",
    "subj02": f"{R}/subj02_40sess_server_ridge_sc2/subj02_40sess_server_ridge_sc2_all_recons_imagery_seed{seed}.pt",
    "subj05": f"{R}/subj05_40sess_server_ridge_sc2/subj05_40sess_server_ridge_sc2_all_recons_imagery_seed{seed}.pt",
    "subj07": f"{R}/subj07_40sess_server_ridge_sc2/subj07_40sess_server_ridge_sc2_all_recons_imagery_seed{seed}.pt",
}
gt = torch.load(f"{R}/imagery_stimuli_18.pt", map_location="cpu")
recs = {k: torch.load(v, map_location="cpu")[:, 0] for k, v in files.items()}  # (18,3,1024,1024)

def show(t):
    t = F.interpolate(t[None].float(), size=(320, 320), mode="area")[0]
    return t.permute(1, 2, 0).clamp(0, 1).numpy()

def fig(idx, name, title):
    rows = ["원본"] + list(recs)
    f, ax = plt.subplots(len(rows), len(idx), figsize=(2.6 * len(idx), 2.6 * len(rows)))
    for c, i in enumerate(idx):
        for r, row in enumerate(rows):
            a = ax[r, c]; a.axis("off")
            a.imshow(show(gt[i] if r == 0 else recs[row][i]))
            if r == 0: a.set_title(f"자극 {i}", fontsize=11)
            if c == 0: a.text(-0.04, 0.5, row, transform=a.transAxes, ha="right", va="center", fontsize=10)
    f.suptitle(title, fontsize=13); f.tight_layout()
    f.savefig(f"docs/report_figures/{name}", dpi=110); plt.close(f); print(name)

plt.rcParams["font.family"] = "AppleGothic"
fig(range(0, 6), f"human_view_simple_seed{seed}.png", f"단순 도형 6개 (시드 {seed})")
fig(range(6, 12), f"human_view_complex_seed{seed}.png", f"복합 자극 6개: 자연 장면 5 + 그림 1 (시드 {seed})")
