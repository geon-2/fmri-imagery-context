"""NSD 트라이얼(세션 순서) → 이미지(nsdId, cocoId) 대응표를 만든다. 피험자별(기본 subj01).

nsd_expdesign.mat: subjectim[s, k] = 피험자 s의 k번째 이미지의 nsdId(1부터), masterordering[t] = t번째 트라이얼이 보여 준 이미지의 번호 k(1부터).
nsd_stim_info_merged.csv의 nsdId는 0부터 센다. 세션 하나는 750트라이얼(roi_betas/session{NN}.npy의 행 순서와 같다).
출력: data/raw/nsd_fmri/{subj}/trial_map.csv (trial, session, row_in_session, nsdId, cocoId, shared1000)
사용: python3 src/fmri/build_trial_map.py [subj02]
"""
import csv
import sys
from collections import Counter

import numpy as np
import scipy.io as sio

SUBJNAME = sys.argv[1] if len(sys.argv) > 1 else "subj01"
SUBJ = int(SUBJNAME[4:]) - 1
m = sio.loadmat("data/raw/nsd_fmri/nsd_expdesign.mat")
order = m["masterordering"].ravel().astype(int)  # 30000
subjectim = m["subjectim"][SUBJ].astype(int)       # 10000, nsdId 1-based
sharedix = set(int(x) for x in m["sharedix"].ravel())  # nsdId 1-based
stim = {r["nsdId"]: r for r in csv.DictReader(open("data/raw/nsd_stim_info_merged.csv", encoding="utf-8"))}

rows = []
for t, k in enumerate(order):
    nsd1 = int(subjectim[k - 1])
    nsd0 = nsd1 - 1
    r = stim[str(nsd0)]
    rows.append({"trial": t, "session": t // 750 + 1, "row_in_session": t % 750, "nsdId": nsd0, "cocoId": r["cocoId"],
                 "shared1000": int(nsd1 in sharedix)})
with open(f"data/raw/nsd_fmri/{SUBJNAME}/trial_map.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

cnt = Counter(r["nsdId"] for r in rows)
print(f"트라이얼 {len(rows)}, 고유 이미지 {len(cnt)}, 반복 횟수 분포 {dict(Counter(cnt.values()))}")
print("shared1000 트라이얼", sum(r["shared1000"] for r in rows), "고유 이미지", len({r['nsdId'] for r in rows if r['shared1000']}))
flag = {i for i, r in stim.items() if r.get("flagged") == "True"}
print("제외된 flagged 이미지가 이 피험자에 몇 장:", len({r['nsdId'] for r in rows} & {int(i) for i in flag}))
