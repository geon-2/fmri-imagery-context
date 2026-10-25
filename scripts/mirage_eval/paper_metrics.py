"""MIRAGE 논문 Table 1의 이미지 특징 지표 중 구현 가능한 것(PixCorr, SSIM, AlexNet(2), AlexNet(5), Inception, CLIP)을 계산한다.
정의는 논문 Appendix A.6: 2-way comparison = 정답 자극의 특징이 '대상 재구성'과 '다른 자극의 재구성(distractor)' 중 어느 쪽과 더 비슷한지.
distractor는 같은 방식으로 재구성한 다른 11개 자극(정답 이미지가 있는 simple 6 + complex 6 = 12개)이다. 개념 자극 6개는 정답 이미지가 없어 제외.
세부 전처리(리사이즈 크기, 특징 층)는 논문에 적혀 있지 않아 MindEye 계열 평가의 관례를 따랐다(확인한 것이 아니라 관례).
미구현: EffNet-B/SwAV 거리, brain correlation(GNet 필요). 사용: python paper_metrics.py <recons.pt> [<recons2.pt> ...]"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_fn

GT_PATH = "data/results/mirage_server/imagery_stimuli_18.pt"
N_GT = 12   # 정답 이미지가 있는 자극 수(simple 6 + complex 6)
dev = "cpu"

def load_recons(p):
    r = torch.load(p, map_location="cpu")
    if r.dim() == 5: r = r[:, 0]          # (자극, gen_rep, 3, H, W) -> 첫 반복
    return r[:N_GT].float().clamp(0, 1)

def resize(x, s): return F.interpolate(x, size=(s, s), mode="bilinear", antialias=True, align_corners=False)

@torch.no_grad()
def feats_alex(x, layers):
    m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).eval()
    fe = create_feature_extractor(m, return_nodes={"features.4": "l2", "features.11": "l5"})
    t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    out = fe(t(resize(x, 256)))
    return {k: out[k].flatten(1) for k in layers}

@torch.no_grad()
def feats_incep(x):
    m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).eval()
    fe = create_feature_extractor(m, return_nodes={"avgpool": "f"})
    t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return fe(t(resize(x, 342)))["f"].flatten(1)

@torch.no_grad()
def feats_clip(x):
    from transformers import CLIPModel
    m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    t = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    f = m.get_image_features(pixel_values=t(resize(x, 224)))
    f = f.pooler_output if hasattr(f, "pooler_output") else f   # 최신 transformers는 출력 객체를 돌려줌
    assert f.shape[-1] == 768, f"CLIP ViT-L/14의 투영 임베딩(768)이 아님: {tuple(f.shape)}"
    return f

def per_stim_two_way(f_gt, f_rec):
    n = len(f_gt)
    a = (f_gt - f_gt.mean(1, keepdim=True)); a = a / a.norm(dim=1, keepdim=True)
    b = (f_rec - f_rec.mean(1, keepdim=True)); b = b / b.norm(dim=1, keepdim=True)
    c = a @ b.T; d = c.diag()
    return np.array([(d[i] > torch.cat([c[i, :i], c[i, i + 1:]])).float().mean().item() for i in range(n)])

def two_way(f_gt, f_rec):
    """f_gt, f_rec: (n,d). 정답 i와 재구성 j의 피어슨 상관 행렬에서, 대각(i,i)이 같은 행의 다른 열(i,j)보다 큰 비율의 평균."""
    n = len(f_gt)
    a = (f_gt - f_gt.mean(1, keepdim=True)); a = a / a.norm(dim=1, keepdim=True)
    b = (f_rec - f_rec.mean(1, keepdim=True)); b = b / b.norm(dim=1, keepdim=True)
    c = a @ b.T
    d = c.diag()
    return float(np.mean([(d[i] > torch.cat([c[i, :i], c[i, i + 1:]])).float().mean().item() for i in range(n)]))

def pixcorr(gt, rec):
    g, r = resize(gt, 425).flatten(1), resize(rec, 425).flatten(1)
    return float(np.mean([np.corrcoef(g[i].numpy(), r[i].numpy())[0, 1] for i in range(len(g))]))

def ssim(gt, rec):
    g = rgb2gray(resize(gt, 425).permute(0, 2, 3, 1).numpy()); r = rgb2gray(resize(rec, 425).permute(0, 2, 3, 1).numpy())
    return float(np.mean([ssim_fn(r[i], g[i], gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0) for i in range(len(g))]))

if __name__ == "__main__":
    gt = torch.load(GT_PATH, map_location="cpu")[:N_GT].float().clamp(0, 1)
    ga = feats_alex(gt, ["l2", "l5"]); gi = feats_incep(gt); gc = feats_clip(gt)
    res = {}
    for p in sys.argv[1:]:
        rec = load_recons(p); ra = feats_alex(rec, ["l2", "l5"])
        res[p.split("/")[-1].replace("_all_recons_imagery.pt", "")] = {
            "PixCorr": pixcorr(gt, rec), "SSIM": ssim(gt, rec),
            "Alex(2)": two_way(ga["l2"], ra["l2"]), "Alex(5)": two_way(ga["l5"], ra["l5"]),
            "Incep": two_way(gi, feats_incep(rec)), "CLIP": two_way(gc, feats_clip(rec))}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n논문 Table 1(MIRAGE, imagery): PixCorr 0.104, SSIM 0.398, Alex(2) 63.92%, Alex(5) 62.46%, Incep 52.25%, CLIP 57.46%  (10장 평균, 5회 반복 평균)")
