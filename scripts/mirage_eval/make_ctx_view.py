"""맥락 합치기 실험의 원본, 기준선, 각 조건 재구성을 나란히 그려 사람이 직접 본다(육안 확인, 지표와 따로 적는다).
사용: python3 scripts/mirage_eval/make_ctx_view.py [seed(기본 1, 1~3)]   (저장소 루트에서)
행: 원본, 기준선(LLaVA만), blend(T10 0.5), shuf(셔플 대조 0.5), p1seq(P1 평균 0.5), ctxonly(맥락만 1.0). 열: 자극. 같은 시드라 조건 간 차이는 텍스트 조건뿐이다.
자극 0-5 단순 도형, 6-10 자연 장면, 11 그림."""
import sys, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
R = "data/results/mirage_server"
ARMS = {"기준선": "server_ridge_sc2", "blend": "ctxblend050", "shuf": "shufblend050", "p1seq": "p1seqblend050", "ctxonly": "ctxonly100"}
gt = torch.load(f"{R}/imagery_stimuli_18.pt", map_location="cpu")
recs = {k: torch.load(f"{R}/subj01_40sess_{m}/subj01_40sess_{m}_all_recons_imagery_seed{seed}.pt", map_location="cpu")[:, 0] for k, m in ARMS.items()}   # (18,3,1024,1024)


def show(t):
    t = F.interpolate(t[None].float(), size=(320, 320), mode="area")[0]
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


def fig(idx, name, title):
    rows = ["원본"] + list(recs)
    f, ax = plt.subplots(len(rows), len(idx), figsize=(2.4 * len(idx), 2.4 * len(rows)))
    for c, i in enumerate(idx):
        for r, row in enumerate(rows):
            a = ax[r, c]; a.axis("off")
            a.imshow(show(gt[i] if r == 0 else recs[row][i]))
            if r == 0: a.set_title(f"자극 {i}", fontsize=11)
            if c == 0: a.text(-0.04, 0.5, row, transform=a.transAxes, ha="right", va="center", fontsize=10)
    f.suptitle(title, fontsize=13); f.tight_layout()
    f.savefig(f"docs/report_figures/{name}", dpi=100); plt.close(f); print(name)


plt.rcParams["font.family"] = "AppleGothic"
fig(range(0, 6), f"ctx_view_simple_seed{seed}.png", f"단순 도형 6개: 맥락 합치기 조건 비교 (피험자 1, 시드 {seed})")
fig(range(6, 12), f"ctx_view_complex_seed{seed}.png", f"복합 자극 6개: 자연 장면 5 + 그림 1 (피험자 1, 시드 {seed})")
