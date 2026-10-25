"""공식 평가 노트북(MedARC-AI/MIRAGE final_evaluations_mi_multi.ipynb)의 함수를 *그대로 옮겨* 같은 재구성에 돌리고,
src/eval/nsd_imagery_eval.py의 값과 일치하는지 대조한다(맥 CPU). 노트북에서 바꾼 것은 device="cpu", SSIM의 `multichannel=True`(최신 skimage에서 제거됨, 회색조 2D라 의미 없음),
brain correlation/캡션 셀 제외뿐이다. 사용: python scripts/mirage_eval/crosscheck_official.py <recons.pt>"""
import sys, numpy as np, torch, scipy as sp
sys.path.insert(0, ".")
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import alexnet, AlexNet_Weights, inception_v3, Inception_V3_Weights, efficientnet_b1, EfficientNet_B1_Weights
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
import clip

device = "cpu"

# ---- 공식 노트북 셀(그대로) ----
@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, return_avg=False):
    preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()
    reals_T = reals.T
    preds_T = preds.T
    r = np.corrcoef(reals_T, preds_T, rowvar=False)
    N = len(all_images)
    r = r[:N, N:]
    congruents = np.diag(r)
    success_counts = []
    total_comparisons = N - 1
    for i in range(N):
        correlations = r[:, i]
        congruent = congruents[i]
        successes = np.sum(correlations < congruent) - 1
        success_rate = successes / total_comparisons
        success_counts.append(success_rate)
    if return_avg:
        return np.mean(success_counts)
    else:
        return success_counts

preprocess_pixcorr = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR)])
def get_pix_corr(all_images, all_recons, return_avg=False):
    all_images_flattened = preprocess_pixcorr(all_images).reshape(len(all_images), -1).cpu()
    all_recons_flattened = preprocess_pixcorr(all_recons).view(len(all_recons), -1).cpu()
    correlations = []
    for i in range(len(all_images)):
        correlations.append(np.corrcoef(all_images_flattened[i], all_recons_flattened[i])[0][1])
    return np.mean(correlations) if return_avg else correlations

preprocess_ssim = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR)])
def get_ssim(all_images, all_recons, return_avg=False):
    img_gray = rgb2gray(preprocess_ssim(all_images).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(preprocess_ssim(all_recons).permute((0, 2, 3, 1)).cpu())
    ssim_score = []
    for im, rec in zip(img_gray, recon_gray):
        ssim_score.append(structural_similarity(rec, im, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))
    return np.mean(ssim_score) if return_avg else ssim_score

alex_model = create_feature_extractor(alexnet(weights=AlexNet_Weights.IMAGENET1K_V1), return_nodes=['features.4', 'features.11']).to(device).eval().requires_grad_(False)
preprocess_alexnet = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
def get_alexnet(all_images, all_recons, return_avg=False):
    a2 = two_way_identification(all_recons.to(device).float(), all_images, alex_model, preprocess_alexnet, 'features.4', return_avg=return_avg)
    a5 = two_way_identification(all_recons.to(device).float(), all_images, alex_model, preprocess_alexnet, 'features.11', return_avg=return_avg)
    return a2, a5

inception_model = create_feature_extractor(inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes=['avgpool']).to(device).eval().requires_grad_(False)
preprocess_inception = transforms.Compose([transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
def get_inceptionv3(all_images, all_recons, return_avg=False):
    return two_way_identification(all_recons.float(), all_images.float(), inception_model, preprocess_inception, 'avgpool', return_avg=return_avg)

clip_model, _ = clip.load("ViT-L/14", device=device)
preprocess_clip = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                                      transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
def get_clip(all_images, all_recons, return_avg=False):
    return two_way_identification(all_recons, all_images, clip_model.encode_image, preprocess_clip, None, return_avg=return_avg)

eff_model = create_feature_extractor(efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT), return_nodes=['avgpool']).eval().requires_grad_(False)
preprocess_efficientnet = transforms.Compose([transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
def get_efficientnet(all_images, all_recons, return_avg=False):
    gt = eff_model(preprocess_efficientnet(all_images))['avgpool'].reshape(len(all_images), -1).cpu().numpy()
    fake = eff_model(preprocess_efficientnet(all_recons))['avgpool'].reshape(len(all_recons), -1).cpu().numpy()
    effnet = [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
    return np.mean(effnet) if return_avg else effnet
# ---- 공식 셀 끝 ----

if __name__ == "__main__":
    from src.eval import nsd_imagery_eval as mine
    gt = torch.load("data/results/mirage_server/imagery_stimuli_18.pt", map_location="cpu")[:12]          # 공식: all_images[:12], 425x425
    rec = torch.load(sys.argv[1], map_location="cpu").float()
    rec = rec[:12] if rec.dim() == 4 else rec[:12, 0]
    rec512 = torch.stack([transforms.Resize((512, 512))(r) for r in rec]).float()                       # 공식: Resize((512,512))
    off = {"PixCorr": get_pix_corr(gt, rec512, True), "SSIM": get_ssim(gt, rec512, True)}
    off["Alex(2)"], off["Alex(5)"] = get_alexnet(gt, rec512, True)
    off["Incep"] = get_inceptionv3(gt, rec512, True); off["CLIP"] = get_clip(gt, rec512, True); off["EffNet-B"] = get_efficientnet(gt, rec512, True)
    ex = mine.Extractors(with_swav=False); g = mine.load_gt("data/results/mirage_server/imagery_stimuli_18.pt")
    m = mine.summarize(mine.evaluate(ex, g, mine.load_recons([sys.argv[1]])))
    print(f"{'지표':10s} {'공식 코드(복사)':>16s} {'src/eval 구현':>14s} {'차이':>10s}")
    for k, v in off.items():
        print(f"{k:10s} {float(v):16.5f} {m[k]:14.5f} {m[k]-float(v):+10.5f}")
