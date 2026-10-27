"""VDVAE 흐린 이미지(필터 후)가 최종 재구성에 어떻게 이어지는지 보여 주는 그림. 피험자 2, 시드 1.
추론 단계 그림이다(학습에서는 원본 이미지를 VDVAE 인코더에 넣어 타깃 잠재를 만든다). 원본은 비교용일 뿐 입력이 아니다.
행: 원본, 흐린 이미지(fMRI로 예측한 VDVAE 잠재를 디코드 + 선명도·대비 필터, Stable Cascade의 시작점), 최종 재구성.
사용: python3 scripts/mirage_eval/make_vdvae_view.py"""
import torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

R = "data/results/mirage_server"
d = "subj02_40sess_server_ridge_sc2"
gt = torch.load(f"{R}/imagery_stimuli_18.pt", map_location="cpu")
blur = torch.load(f"{R}/{d}/{d}_all_blurryrecons_imagery_seed1.pt", map_location="cpu").reshape(18, 3, 768, 768)
rec = torch.load(f"{R}/{d}/{d}_all_recons_imagery_seed1.pt", map_location="cpu")[:, 0]

def show(t):
    return F.interpolate(t[None].float(), size=(320, 320), mode="area")[0].permute(1, 2, 0).clamp(0, 1).numpy()

idx = [0, 4, 6, 7, 9, 10]
rows = [("원본\n(비교용, 입력 아님)", gt), ("흐린 이미지\n(fMRI로 예측한 VDVAE 잠재\n→ 디코드 + 필터)", blur), ("최종 재구성", rec)]
plt.rcParams["font.family"] = "AppleGothic"
f, ax = plt.subplots(3, len(idx), figsize=(2.6 * len(idx), 2.7 * 3))
for c, i in enumerate(idx):
    for r, (name, src) in enumerate(rows):
        a = ax[r, c]; a.axis("off"); a.imshow(show(src[i]))
        if r == 0: a.set_title(f"자극 {i}", fontsize=11)
        if c == 0: a.text(-0.04, 0.5, name, transform=a.transAxes, ha="right", va="center", fontsize=10)
f.suptitle("추론(테스트) 단계, 피험자 2 시드 1: 원본 이미지는 쓰지 않고 fMRI만으로 만든 결과", fontsize=12)
f.tight_layout(); f.savefig("docs/report_figures/vdvae_role_subj02_seed1.png", dpi=110)
print("ok")
