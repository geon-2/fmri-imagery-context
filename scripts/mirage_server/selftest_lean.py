"""서버(GPU)에서 lean_ridge(float32 GPU 경로)가 sklearn Ridge와 같은 해를 내는지 대조. 사용: $PY selftest_lean.py"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lean_ridge import lean_ridge
from sklearn.linear_model import Ridge
rng = np.random.default_rng(0)
N, p, out, alpha = 3000, 4000, 6000, 100000.0
X = (rng.standard_normal((N, p)) * 3 + 1.5).astype(np.float32)
Y = (X @ (rng.standard_normal((p, out)) * 0.01) + rng.standard_normal((N, out))).astype(np.float32)
t = time.time(); m = Ridge(alpha=alpha, max_iter=50000, random_state=42).fit(X, Y); print("sklearn", round(time.time() - t), "초")
for dev in ("cpu", "cuda"):
    if dev == "cuda" and not torch.cuda.is_available(): continue
    t = time.time(); coef, ic = lean_ridge(torch.from_numpy(X), torch.from_numpy(Y), alpha, chunk=2048, device=dev, log=lambda *a: None)
    rel = np.abs(coef - m.coef_).max() / np.abs(m.coef_).max()
    pr = np.abs((X[:200] @ coef.T + ic) - m.predict(X[:200])).max() / np.abs(m.predict(X[:200])).max()
    print(f"{dev}: {time.time()-t:.0f}초  coef max rel err={rel:.2e}  pred max rel err={pr:.2e}")
