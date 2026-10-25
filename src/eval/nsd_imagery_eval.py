"""NSD-Imagery 벤치마크 평가(재사용용). MIRAGE 공식 평가 노트북(MedARC-AI/MIRAGE `src/final_evaluations_mi_multi.ipynb`)의 이미지 특징 지표를 같은 방식으로 계산한다.
어떤 방법(MIRAGE 재현, 이후 개선 모델 등)이든 같은 코드로 평가해 비교하려는 것이 목적이다.

입력: 재구성 텐서 (자극 N, 반복 R, 3, H, W) 또는 (N, 3, H, W). 한 방법의 여러 파일(시드별)을 쉼표로 이으면 반복 축으로 합친다.
출력: 지표별 평균(자극 → 반복 순으로 평균), 자극별 값 CSV, 기준 방법 대비 차이의 bootstrap 95% 구간.

공식 코드 확인 사항:
- 정답 이미지가 있는 앞 12개 자극(simple 6 + complex 6)만 평가. 개념 자극은 제외.
- 재구성은 512x512로 리사이즈. PixCorr/SSIM은 정답과 재구성 모두 512로 리사이즈.
- 2-way: 재구성 i를 고정하고 정답 i와의 상관이 다른 정답 j들과의 상관보다 큰 비율. (논문 부록 A.6은 distractor를 '다른 재구성'이라 쓰지만 코드는 다른 정답 이미지를 쓴다.)
- 공식 코드는 `np.sum(correlations < congruent) - 1`로 센다. 엄격한 부등호라 자기 자신은 이미 빠져 있어 1을 또 빼는 꼴이라, 값이 1/(N-1)(12개면 약 0.091) 낮게 나온다.
  그래서 `official`(공식 코드 그대로, 논문 표와 비교용)과 `corrected`(-1 제거)를 둘 다 낸다. 논문 값이 이 코드로 나왔는지는 확인하지 못했다.
- EffNet-B1(avgpool, resize 255), SwAV-ResNet50(avgpool, resize 224)은 상관 거리(낮을수록 좋음).
미구현: brain correlation(GNet 필요), 캡션 지표.

사용 예:
  python -m src.eval.nsd_imagery_eval --gt data/results/mirage_server/imagery_stimuli_18.pt \\
     --recons mirage10=a_seed1.pt,a_seed2.pt mirage40=b_seed1.pt --baseline mirage10 --out results/tables/imagery_eval
"""
import argparse, csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_fn
from scipy.spatial.distance import correlation as corr_dist

N_GT = 12
IMN = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP_NORM = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
# 논문 Table 1, MIRAGE, Mental Imagery Reconstructions (simple + complex). 2-way는 0~1 비율로 환산.
PAPER_TABLE1_IMAGERY = {"PixCorr": 0.104, "SSIM": 0.398, "Alex(2)": 0.6392, "Alex(5)": 0.6246, "Incep": 0.5225, "CLIP": 0.5746, "EffNet-B": 0.914, "SwAV": 0.575}
HIGHER_BETTER = ["PixCorr", "SSIM", "Alex(2)", "Alex(5)", "Incep", "CLIP"]
LOWER_BETTER = ["EffNet-B", "SwAV"]


ANTIALIAS = True   # 공식 노트북은 torchvision transforms.Resize의 기본값을 쓴다. 최신 torchvision은 텐서에서 antialias=True가 기본(0.17부터).
                   # 논문 저자 환경의 torchvision 버전은 알 수 없어, 값 차이를 보려면 --antialias false로 다시 계산해 볼 수 있다.


def resize(x, s):
    return transforms.functional.resize(x, [s, s], interpolation=transforms.InterpolationMode.BILINEAR, antialias=ANTIALIAS)


class Extractors:
    """모델은 한 번만 만들어 재사용한다."""
    def __init__(self, with_swav=True):
        fx = create_feature_extractor
        self.alex = fx(models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).eval(), {"features.4": "l2", "features.11": "l5"})
        self.inc = fx(models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).eval(), {"avgpool": "f"})
        self.eff = fx(models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT).eval(), {"avgpool": "f"})
        from transformers import CLIPModel
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
        self.swav = None
        if with_swav:
            try:
                self.swav = fx(torch.hub.load("facebookresearch/swav:main", "resnet50").eval(), {"avgpool": "f"})
            except Exception as e:
                print("SwAV 건너뜀:", repr(e)[:120], file=sys.stderr)

    @torch.no_grad()
    def features(self, x):
        o = self.alex(IMN(resize(x, 256)))
        f = {"alex2": o["l2"].flatten(1), "alex5": o["l5"].flatten(1),
             "inc": self.inc(IMN(resize(x, 342)))["f"].flatten(1),
             "eff": self.eff(IMN(resize(x, 255)))["f"].flatten(1)}
        c = self.clip.get_image_features(pixel_values=CLIP_NORM(resize(x, 224)))
        c = c.pooler_output if hasattr(c, "pooler_output") else c
        assert c.shape[-1] == 768, f"CLIP ViT-L/14 투영 임베딩(768)이 아님: {tuple(c.shape)}"
        f["clip"] = c
        if self.swav is not None:
            f["swav"] = self.swav(IMN(resize(x, 224)))["f"].flatten(1)
        return f


def two_way(f_gt, f_rec):
    """공식 코드와 같은 방향. (official, corrected) 자극별 배열 반환."""
    n = len(f_gt)
    r = np.corrcoef(f_gt.double().numpy(), f_rec.double().numpy())[:n, n:]      # r[j, i] = corr(정답 j, 재구성 i)
    off, cor = [], []
    for i in range(n):
        n_less = np.sum(r[:, i] < r[i, i])
        off.append((n_less - 1) / (n - 1)); cor.append(n_less / (n - 1))
    return np.array(off), np.array(cor)


def pixcorr(gt, rec):
    g, r = resize(gt, 512).flatten(1), resize(rec, 512).flatten(1)
    return np.array([np.corrcoef(g[i].numpy(), r[i].numpy())[0, 1] for i in range(len(g))])


def ssim(gt, rec):
    g = rgb2gray(resize(gt, 512).permute(0, 2, 3, 1).numpy()); r = rgb2gray(resize(rec, 512).permute(0, 2, 3, 1).numpy())
    return np.array([ssim_fn(r[i], g[i], gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0) for i in range(len(g))])


def dist(a, b):
    return np.array([corr_dist(a[i].numpy(), b[i].numpy()) for i in range(len(a))])


def load_gt(path):
    return torch.load(path, map_location="cpu")[:N_GT].float().clamp(0, 1)


def load_recons(paths):
    """파일 여러 개를 반복 축으로 합쳐 (12, R, 3, H, W)로 반환."""
    reps = []
    for p in paths:
        r = torch.load(p, map_location="cpu").float()
        if r.dim() == 4: r = r.unsqueeze(1)
        reps.append(r[:N_GT].clamp(0, 1))
    return torch.cat(reps, dim=1)


def per_rep_metrics(ex, gt, gt_f, rec):
    """재구성 한 반복(12,3,H,W)의 지표별 자극별 값."""
    rec = resize(rec, 512)                         # 공식: 재구성을 512로
    f = ex.features(rec); out = {}
    out["PixCorr"], out["SSIM"] = pixcorr(gt, rec), ssim(gt, rec)
    for key, nm in (("alex2", "Alex(2)"), ("alex5", "Alex(5)"), ("inc", "Incep"), ("clip", "CLIP")):
        out[nm], out[nm + "|corrected"] = two_way(gt_f[key], f[key])
    out["EffNet-B"] = dist(gt_f["eff"], f["eff"])
    if "swav" in f: out["SwAV"] = dist(gt_f["swav"], f["swav"])
    return out


def evaluate(ex, gt, recons):
    """반환: {지표: 배열(12 자극, R 반복)}."""
    gt_f = ex.features(gt)                          # 정답 특징(정답은 425 원본, 각 전처리가 리사이즈)
    reps = [per_rep_metrics(ex, gt, gt_f, recons[:, r]) for r in range(recons.shape[1])]
    return {k: np.stack([m[k] for m in reps], axis=1) for k in reps[0]}


def summarize(res):
    """자극 → 반복 순으로 평균한 지표."""
    return {k: float(v.mean()) for k, v in res.items()}


def paired_bootstrap(a, b, n_boot=5000, seed=0):
    """a, b: (12, R) 배열. 자극을 복원추출해 평균 차이(a-b)의 95% 구간. 자극 변동만 반영하고 반복/피험자 변동은 포함하지 않는다."""
    d = (a.mean(1) - b.mean(1)); idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="imagery_stimuli_18.pt")
    ap.add_argument("--recons", nargs="+", required=True, help="이름=파일1,파일2,... (파일마다 한 반복 이상)")
    ap.add_argument("--baseline", default=None, help="차이 구간의 기준 방법 이름")
    ap.add_argument("--out", default=None, help="결과 폴더(CSV, JSON, markdown)")
    ap.add_argument("--no-swav", action="store_true")
    ap.add_argument("--combine", nargs="*", default=[], help="이름=방법1,방법2,... : 방법별(예: 피험자별) 자극 평균을 다시 평균해 합친 항목을 추가")
    ap.add_argument("--paper", default=None, help="논문 Table 1(MIRAGE imagery)과 비교할 항목 이름. 95% 구간 안에 논문 값이 들어가는지 표시")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--antialias", choices=["true", "false"], default="true", help="리사이즈 antialias(공식 노트북은 torchvision 기본값)")
    a = ap.parse_args()
    global ANTIALIAS
    ANTIALIAS = a.antialias == "true"
    ex = Extractors(with_swav=not a.no_swav); gt = load_gt(a.gt)
    allres = {}
    for spec in a.recons:
        name, files = spec.split("=", 1)
        rec = load_recons(files.split(","))
        print(f"[{name}] 반복 {rec.shape[1]}개, 재구성 {tuple(rec.shape)}", flush=True)
        allres[name] = evaluate(ex, gt, rec)
    for spec in a.combine:
        name, members = spec.split("=", 1); members = members.split(",")
        allres[name] = {k: np.mean([allres[m][k].mean(1, keepdims=True) for m in members], axis=0) for k in allres[members[0]]}
    cols = [c for c in HIGHER_BETTER + LOWER_BETTER if all(c in r for r in allres.values())]
    lines = ["| 방법 | 반복 | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for n, r in allres.items():
        s = summarize(r); lines.append(f"| {n} (official) | {next(iter(r.values())).shape[1]} | " + " | ".join(f"{s[c]:.3f}" for c in cols) + " |")
    for n, r in allres.items():   # corrected 행
        s = summarize(r); lines.append(f"| {n} (2-way −1 제거) | | " + " | ".join(f"{s.get(c + '|corrected', s[c]):.3f}" for c in cols) + " |")
    if a.paper:
        r = allres[a.paper]; rng = np.random.default_rng(0); idx = rng.integers(0, N_GT, size=(a.bootstrap, N_GT)); n_in = 0; n_all = 0
        REL_TOL = 0.15   # docs/mirage_repro_criteria.md 판정 규칙 1(상대 차이 15% 이내)
        n_rel = 0
        lines += ["", f"논문 Table 1(MIRAGE imagery)과 비교: `{a.paper}`의 95% 구간(자극 {N_GT}개 복원추출 {a.bootstrap}회, 자극 변동만 반영)", "| 지표 | 우리 평균 | 95% 구간 | 논문 | 구간 안? | 상대 차이 | 15% 이내? |", "|---|---|---|---|---|---|---|"]
        for c in cols:
            if c not in PAPER_TABLE1_IMAGERY: continue
            v = r[c].mean(1); bs = v[idx].mean(1); lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5); pv = PAPER_TABLE1_IMAGERY[c]; ok = lo <= pv <= hi
            rel = (v.mean() - pv) / pv; within = abs(rel) <= REL_TOL; n_rel += within
            n_in += ok; n_all += 1; lines.append(f"| {c} | {v.mean():.3f} | [{lo:.3f}, {hi:.3f}] | {pv:.3f} | {'예' if ok else '아니오'} | {rel:+.1%} | {'예' if within else '아니오'} |")
        need = int(np.ceil(0.75 * n_all))
        lines.append(f"\n상대 차이 15% 이내: {n_rel}/{n_all} (필요 {need}개 이상, 주 규칙). 95% 구간 안: {n_in}/{n_all} (참고, 구분력이 약함)")
    if a.baseline:
        lines += ["", f"기준 `{a.baseline}` 대비 차이(official, 95% 구간, 자극 {N_GT}개 복원추출 {a.bootstrap}회):"]
        for n, r in allres.items():
            if n == a.baseline: continue
            parts = []
            for c in cols:
                m, lo, hi = paired_bootstrap(r[c], allres[a.baseline][c], a.bootstrap)
                parts.append(f"{c} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            lines.append(f"- {n}: " + "; ".join(parts))
    print("\n".join(lines))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(f"{a.out}/summary.md", "w").write("\n".join(lines) + "\n")
        json.dump({n: summarize(r) for n, r in allres.items()}, open(f"{a.out}/summary.json", "w"), indent=1)
        for n, r in allres.items():
            with open(f"{a.out}/per_stimulus_{n}.csv", "w", newline="") as f:
                w = csv.writer(f); keys = list(r); w.writerow(["stimulus", "rep"] + keys)
                for s in range(N_GT):
                    for k in range(r[keys[0]].shape[1]):
                        w.writerow([s, k] + [f"{r[c][s, k]:.6f}" for c in keys])
        print("저장:", a.out)


if __name__ == "__main__":
    main()
