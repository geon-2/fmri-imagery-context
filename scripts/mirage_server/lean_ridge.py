"""sklearn Ridge(alpha, fit_intercept=True)와 같은 해를 메모리를 적게 쓰고 구하는 구현.
sklearn은 계수·타깃을 float64로 한꺼번에 만들어 (10세션 학습 피크 48.9GB 실측) retrieval(출력 263k차원)이나 40세션에서 RAM이 모자란다.
수식: X, Y를 열 평균으로 중심화한 뒤 W = (XcᵀXc + αI)⁻¹ Xcᵀ Yc, intercept = mean(Y) - mean(X)·W.
(XcᵀXc+αI)⁻¹Xcᵀ 는 한 번만 float64로 구하고, 출력 차원을 청크로 나눠 곱해 계수는 float32로만 만든다."""
import numpy as np, torch

def lean_ridge(X, Y, alpha, chunk=4096, device="cuda", log=print):
    """X: (N,p) torch/numpy, Y: (N,out) torch(view 가능). 반환 coef (out,p) float32 numpy, intercept (out,) float32 numpy."""
    X = torch.as_tensor(X)
    N, p = X.shape
    Xm = X.double().mean(0)
    Xc = X.double() - Xm
    A = Xc.T @ Xc
    A.diagonal().add_(alpha)
    L = torch.linalg.cholesky(A); del A
    M = torch.cholesky_solve(Xc.T.contiguous(), L)            # (p,N) float64 = (XcᵀXc+αI)⁻¹ Xcᵀ
    del Xc, L
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    Mg = M.to(dev, torch.float32 if dev != "cpu" else torch.float64); del M
    out = Y.shape[1]
    coef = np.empty((out, p), dtype=np.float32); icpt = np.empty((out,), dtype=np.float32)
    Xmg = Xm.to(dev, Mg.dtype)
    for c0 in range(0, out, chunk):
        c1 = min(out, c0 + chunk)
        Yc = Y[:, c0:c1].to(dev, Mg.dtype)
        Ym = Yc.mean(0)
        W = Mg @ (Yc - Ym)                                      # (p,c)
        coef[c0:c1] = W.T.float().cpu().numpy()
        icpt[c0:c1] = (Ym - Xmg @ W).float().cpu().numpy()
        if (c0 // chunk) % 8 == 0: log(f"  ridge chunk {c0}/{out}")
    return coef, icpt
