# fMRI 심상(mental imagery)의 ROI별 지각–심상 공유/변환 문헌 지도 (2026년 10월 기준, 신뢰도 등급 포함)

**결론부터:** 지금까지 가장 직접적인 증거(NSD-Imagery 7T, n=8)를 보면 심상은 지각 신호를 약하게 복사한 것이 아니라 "압축·회전된 지각"에 가깝습니다. 초기 시각피질(V1–hV4)에서는 심상 부분공간의 차원이 지각의 절반에 조금 못 미치고 방향도 회전되어 있습니다(V1 정렬비 25–30%, V4 약 50%). 고차 ventral/lateral 영역에서는 거의 항등(identity)에 가깝고, parietal ROI에서는 정렬이 약 100%에 수렴합니다.\[1\] 다만 이 결과는 **동료심사를 거치지 않은 preprint 한 편**에서 나왔습니다.\[2\] 또 "장면 맥락(장소 유형·활동·암시된 사건)" 수준의 심상 전이는 어떤 연구도 직접 검증하지 않았습니다. 따라서 Geon의 가설(ROI마다 공유 부분공간이 다르다)은 근거가 있지만, 맥락 subspace의 전이는 아직 열린 질문으로 다뤄야 합니다.

## TL;DR
- **ROI 위계:** 초기 시각피질에서는 심상이 "축소·변환"됩니다(차원 약 절반, 부분공간 회전, 공간주파수↓·수용장 크기↑·tuning 폭↑). 고차 ventral/lateral에서는 "대체로 공유", parietal에서는 "거의 동일"합니다. 범주 수준(coarse) 구분은 모든 ROI에서 보존되지만, 범주 내 exemplar 수준 구분은 parietal을 제외하면 약하게만 나타납니다(Saha Roy et al. 2025 preprint, 전문 읽음).
- **이미 알고 있는 항목의 수정 사항:**
  - CCN **2023** 초록 "Mental Imagery: Weak Vision or Compressed Vision?"은 실제로 **존재합니다**(DOI 10.32470/CCN.2023.1693-0).
  - 반대로 CCN 2024 "Do vision and imagery share common principal signal components?"는 **찾지 못했습니다(not verified)**.
  - Lee, Kravitz & Baker는 NeuroImage **2012**(59:4064–4073, 온라인 2011)입니다.
  - Wang et al.(CLIP)은 Nature Machine Intelligence **2023**입니다.
  - Shen et al. CCN 2024의 실제 제목은 "High-dimensional alignment of neural networks and visual cortex"입니다.
  - MIRAGE는 PLoS Computational Biology에 **2026년 5월 22일** 게재되었습니다.
- **본인 연구에 주는 시사점:** CLIP-text 기반 맥락 subspace는 NSD에서 고차 ventral/lateral/parietal stream에 학습시키고, 초기 시각피질은 "회전된 부분공간"을 전제로 별도로 평가하는 것이 문헌과 맞습니다. NSD-Imagery는 상(像)이 12–18개뿐이고 장면 맥락 라벨이 설계되어 있지 않습니다. 따라서 "맥락 전이" 결론은 범주 수준(단순 vs 복잡 자극) 혼입을 통제한 뒤에만 내려야 합니다.

## Key Findings

### 1. "ALREADY KNOWN TO ME" 항목 검증

| 항목 | 검증 결과 | 근거 수준 |
|---|---|---|
| Saha Roy, Breedlove, St-Yves, Kay, Naselaris, "A transformation from vision to imagery in the human brain" | bioRxiv **preprint**, 게시 2025-09-03, DOI 10.1101/2025.09.02.672180. PubMed(PMID 40950062)와 PMC(PMC12424947)에 색인되어 있으나 PubMed/PMC 표기상 "not yet peer reviewed"입니다.\[2\]\[3\] 2026년 10월 현재 저널 버전은 **확인되지 않음**. | 전문 읽음 |
| 사용자 수치: "EVC 차원 약 절반, 고차·parietal은 거의 항등, V1 정렬 25–30%, V4 약 50%" | **대체로 정확.** 원문은 "slightly less than half"이고, V1 25–30% → V4 50% → parietal ROI "converges on 100%"입니다. 두 번째 데이터셋(Breedlove 2020, n=3)에서는 V1/V2 정렬이 50–60%로 더 높습니다.\[1\] 따라서 수치는 데이터셋에 따라 달라집니다. | 전문 읽음 |
| CCN 2024 "Do vision and imagery share common principal signal components?" | **not verified.** 제목 검색과 2024.ccneuro.org 검색 모두 결과 없음(서브에이전트 확인). 같은 연구실의 CCN 2024 imagery 논문은 "fMRI vision reconstruction methods robustly generalize to mental imagery"(저자 미확인, Kneeland 등으로 추정)입니다.\[4\] | 2차 확인 |
| CCN 2023 "Mental Imagery: Weak Vision or Compressed Vision?" | **존재함.** Saha Roy, Breedlove, St-Yves, Kay, Naselaris, CCN 2023(Oxford), 포스터 P-2B.94, DOI 10.32470/CCN.2023.1693-0. 결론은 "provisional support for the 'compressed vision' hypothesis"입니다.\[5\] 관련 항목: CCN 2022 "Quantitative comparison of imagery and perception", J Vis 23(9):5961(2023).\[6\]\[7\] | 2차 확인(서브에이전트) |
| Lee, Kravitz & Baker | NeuroImage 59(4):4064–4073, **2012**(epub 2011), DOI 10.1016/j.neuroimage.2011.10.055\[8\]\[9\] | 초록/2차 |
| Dijkstra, Bosch & van Gerven TiCS | "Shared Neural Mechanisms of Visual Perception and Imagery," TiCS 23(5):423–434, 2019, DOI 10.1016/j.tics.2019.02.004 — 확인됨\[10\]\[11\] | 초록 |
| Spagna et al. 2021 | Spagna, Hajhajate, Liu, Bartolomeo, "Visual mental imagery engages the left fusiform gyrus, but not the early visual cortex: a meta-analysis of neuroimaging evidence," Neurosci Biobehav Rev 122:201–217, DOI 10.1016/j.neubiorev.2020.12.029 — 확인됨\[12\] | 초록/2차 |
| Scrivener, Teed & Silson 2025 | "Visual imagery of familiar people and places in category selective cortex," Neuroscience of Consciousness 2025(1):niaf006, DOI 10.1093/nc/niaf006. **EEG–fMRI fusion** 연구이며 ROI는 MPC, VTC, V1입니다.\[13\]\[14\] | 초록 |
| Wang et al. "2022" | **정정:** Wang, Kay, Naselaris, Tarr, Wehbe, "Better models of human high-level visual cortex emerge from natural language supervision with a large and diverse dataset," Nature Machine Intelligence 5(12):1415–1426, **2023**, DOI 10.1038/s42256-023-00753-y.\[15\]\[16\] 2022는 preprint 연도로 추정됩니다(not verified). | 2차 확인 |
| Doerig et al. 2025 NMI | "High-level visual representations in the human brain are aligned with large language models," NMI 7:1220–1234(2025년 8월), DOI 10.1038/s42256-025-01072-0. preprint는 arXiv 2209.11737(2022) — 확인됨\[17\]\[18\] | 초록+일부 본문 스니펫 |
| Elmoznino & Bonner 2024 | PLoS Comput Biol 20(1):e1011792, DOI 10.1371/journal.pcbi.1011792. preprint는 bioRxiv 2022.07.13.499969 — 확인됨\[19\]\[20\]\[21\] | 2차 확인 |
| Shen et al. CCN 2024 | **제목 정정:** Bonner CV 기준 Shen, Gauthaman, Conwell, Bonner, "High-dimensional alignment of neural networks and visual cortex," CCN 2024.\[22\] "dimensionality of shared representations across cortex"라는 제목은 찾지 못했습니다. **이 초록은 imagery 연구가 아닙니다.** | CV 기재만 확인 |
| Kneeland et al. NSD-Imagery | CVPR 2025, pp. 28852–28862, DOI 10.1109/CVPR52734.2025.02687, arXiv 2506.06898 — 확인됨\[23\]\[24\] | 초록 |
| MIRAGE | Kneeland, Torrico Villanueva, Chen, Ojeda, Khanna, Xu, Scotti, Naselaris, PLoS Comput Biol 22(5):e1014263, 게재 2026-05-22, DOI 10.1371/journal.pcbi.1014263.\[25\]\[26\] 이전 버전은 CCN 2025 초록과 arXiv 2605.17198\[27\]\[28\] | 초록+일부 본문 스니펫 |

### 2. ROI별 요약 표

| ROI 그룹 | 지각 시 표상 (출처) | 심상 상태 | 핵심 수치 (n, 장비) | 출처 / 읽은 범위 | 증거 유형 |
|---|---|---|---|---|---|
| V1 | retinotopic 저수준 특징, 위치·방위·공간주파수 (Breedlove 2020; Favila 2022) | **변환됨/축소됨**: 부분공간 회전, 차원 약 ½\[1\] | 정렬비 25–30%(NSD-Imagery, n=8, 7T, pRF ROI, SNR 상위 2% voxel). V1 심상 차원과 VVIQ의 상관 r=0.71(p<0.05, 95% CI 0.15–0.98, n=8)\[1\] | Saha Roy 2025 preprint / 전문 | (a) 인간 fMRI |
| V1–V3 (공간 tuning) | pRF 크기가 위계에 따라 증가 | **변환됨**: 기억 시 tuning이 넓고 위계적 증가가 사라짐\[29\] | 지각에서는 초기→후기 영역으로 공간 정밀도가 "3배" 감소하지만 기억 인출에서는 이 패턴이 없음.\[30\] 기억 반응 진폭은 모든 영역에서 낮음 | Favila, Kuhl, Winawer 2022 Nat Commun / 초록·스니펫. 참가자 n=9("Nine human participants participated in the experiment"). 9명 평균에서 V1 FWHM이 기억 시 지각보다 2.5배 컸고, 개인별로는 9명 중 7명에서 그랬음. 장비는 not verified | (a) |
| V1–V4 (심상 pRF) | — | **변환됨**: 심상에서 공간주파수 선호↓, 수용장 더 크고 더 중심와 쪽\[31\] | 7T, n=3, 64개 이미지×8개 위치=512 조건(Saha Roy가 기술한 Dataset 2)\[1\] | Breedlove et al. 2020 Curr Biol / 2차 요약 | (a)+(b) 생성모델 |
| V1, V2/3 (aphantasia) | — | **decoding은 가능하지만 지각과 교차되지 않음** | 두 집단 모두 심상 decoding 가능. aphantasia 집단에서만 지각↔심상 cross-decoding 실패, 반응이 동측(ipsilateral)에 나타남.\[32\]\[33\] 스캐너 내 생생함 평정 기준 n은 aphantasia 13명, 대조군 17명. 2차 출처는 총원을 다르게 보고함(aphantasia.com 15 대 16, PsyPost 14 대 18)이라 최종 n은 확정되지 않음. 장비 세기는 확인되지 않음 | Chang et al. 2025 Curr Biol / 초록·스니펫 | (a) |
| EVC (메타분석) | — | **일관된 활성 없음**(univariate) | 27개 fMRI 연구 메타분석에서 EVC가 일관되게 활성화되지 않음\[34\] | Spagna et al. 2021 / 초록 | (c) |
| hV4 | 중간 수준 형태·색 | 부분 공유 | 정렬비 약 50%(NSD-Imagery)\[1\] | Saha Roy 2025 / 전문 | (a) |
| Object (LO, pFs) | 개별 물체 identity | **공유**(상대적으로 보존). 심상 정보 비율이 retinotopic 영역보다 높음\[35\]\[36\] | n=11(2차 출처), 물체 10개. 모든 ROI에서 cross-decoding 유의.\[36\]\[37\]\[38\] 장비는 not verified | Lee, Kravitz, Baker 2012 / 초록·2차 | (a) |
| Ventral stream(NSD streams) | 범주, 장면 의미, 캡션/LLM 임베딩과 정렬 (Doerig 2025; Wang 2023) | **거의 공유**(항등에 가까움). 단, 범주 내 구분은 약함\[1\] | 차원이 지각과 거의 동일(NSD-Imagery)\[1\] | Saha Roy 2025 / 전문 | (a) |
| Lateral stream | 신체, 행위, 장면 layout (Doerig 2025) | 거의 공유 | 위와 동일 | Saha Roy 2025 / 전문 | (a) |
| Parietal stream / IPS | 공간, 위치 | **동일**: 정렬 약 100%, 범주 내 tuning도 지각과 차이 없음\[1\] | Dataset 2에서도 parietal 정렬 "complete"\[1\] | Saha Roy 2025 / 전문 | (a) |
| 얼굴 (FFA/OFA) | 얼굴 | 공유(범주 선택적 재활성화). aphantasia에서는 범주 선택성이 변형됨\[39\] | Liu 2025a(aphantasia 얼굴 패치) — 세부 수치 not verified | 2차 요약(Neuropsychologia 2026 리뷰 표) | (a)/(c) |
| 장면 (PPA/OPA/MPA) + place memory areas (LPMA/VPMA/MPMA) | 장면 지각; PMA는 그 전방에 위치 | **공간적으로 분리됨**: 친숙한 장소 회상은 지각 영역 바로 전방의 PMA를 활성화\[40\]\[41\] | 개인별 fMRI(정밀 수치 not verified)\[42\] | Steel et al. 2021 Nat Commun / 초록·보도자료 | (a) |
| 내측 두정(MPC: Places-1/2, People) | 지각 시 음(−)의 반응이지만 범주 선택적 | 회상 시 양(+)의 반응. +/− pRF의 상반(opponent) 공간 코딩\[43\]\[44\] | Scrivener & Silson 2025 Imaging Neurosci / 초록 | (a) |
| 좌측 fusiform "FIN" | 영역 일반(domain-general) 심상 노드 | **심상 특이적에 가까움**(영역 일반) | 메타분석 MNI 약 (−42, −54, −18). 7T 연구에서도 같은 위치.\[45\] aphantasia에서는 FIN–전두두정 연결 이상 | Spagna 2021; Liu et al. 2025 Cortex / 초록 | (c)+(a) |
| VWFA, EBA/FBA, food | 단어 / 신체 / 음식 | **심상 연구 미발견** | — | 검색 결과 없음 | — |
| 전두(PFC, 전운동) | 과제/주의 | 지각–심상 중첩이 시각피질 밖까지 확장됨(전운동피질 포함)\[9\]\[46\] | — | Dijkstra et al. 2017 J Neurosci; 2021 eNeuro / 초록 | (a) |
| 단일뉴런 ventral temporal(VTC) | 물체 "axis" 코드 | 부분 재활성화 | 시각 반응 뉴런 456개 중 약 80%가 axis-tuned. 심상 과제에서 axis-tuned 뉴런 107개 중 43개(약 40%)가 재활성화. 환자 16명 중 6명이 심상 과제 수행\[47\] | Wadia, Reed, Chung, Bateman, Mamelak, Rutishauser, Tsao, "A shared code for perceiving and imagining objects in human ventral temporal cortex," Science 392:207–215(2026년 4월 9일 게재, DOI 10.1126/science.adt8343) / 초록. 초록 기준 환자 16명에게서 VTC 뉴런 714개를 기록함 | (a) 인간 두개내 기록(fMRI 아님) |

## Details

### A. 초기 시각피질 (V1, V2, V3, hV4) — retinotopy/pRF 계열

1. **Saha Roy et al. 2025, bioRxiv preprint**(2025-09-03, 저널 버전 미확인)
   - **방법 — 1) 데이터와 ROI:** NSD 피험자 8명, 7T, 1.8 mm. 자극 12개(막대·십자 6개, 자연 장면 6개). 지각 runs와 심상 runs를 따로 수집했습니다. ROI는 V1–hV4(NSD pRF)와 ventral/lateral/parietal(해부학적, NSD streams)입니다.\[1\]
   - **방법 — 2) 분석 처리:** 각 ROI에서 NSD-core SNR 상위 2%의 voxel만 썼습니다. 첫 번째 심상 runs(imgA-1, imgB-1)는 SNR이 낮아 제외했습니다.\[1\]
   - **보인 것:**
     - vis2img 모델(reduced-rank ridge)이 모든 영역에서 심상 활동을 예측합니다. 고차 영역의 일부 지점에서는 vis2img 정확도가 vis2vis보다 높았습니다.\[1\]
     - 심상 차원은 영역에 걸쳐 거의 일정하고, 고차 영역의 지각 차원과 비슷합니다.\[1\]
     - 단순 대 복잡 자극의 범주 구분은 모든 ROI에서 보존됩니다. 범주 내 구분은 parietal에서만 지각과 같았습니다.\[1\]
     - V1 심상 차원은 VVIQ와 상관이 있었습니다(r=0.71). 정렬비와 VVIQ의 상관은 어느 ROI에서도 없었습니다.\[1\]
   - **저자 해석:** "transformed vision"이 더 적절한 모델이다. 정렬 불일치는 정보 손실과 "지각/심상 구별" 사이의 trade-off일 수 있다(명시적 speculation). 결과를 시각피질의 총 차원 수 추정으로 해석해서는 안 된다고 저자가 직접 경고합니다.\[1\]
   - **내 추론:**
     - 자극이 12개라 차원의 상한이 12입니다.\[1\] 따라서 "맥락" 같은 미세 의미 축은 이 데이터로 추정할 수 없습니다.
     - 장면 6개만 놓고 맥락 축을 따로 분리하는 것은 통계적으로 매우 취약합니다.
2. **Breedlove, St-Yves, Olman, Naselaris 2020**, "Generative Feedback Explains Distinct Brain Activity Codes for Seen and Mental Images," Current Biology 30(12):2211–2224.e6, DOI 10.1016/j.cub.2020.04.014.\[48\]
   - **보인 것:** 7T, n=3, 64개 물체 이미지를 8개 위치에서 보고 상상하게 했습니다. 심상의 voxel-wise 코드는 공간주파수 선호가 낮고 수용장이 더 크며 중심와 쪽이었습니다.\[31\]\[49\]
   - **해석:** 위계적 생성 네트워크의 feedback으로 이를 설명합니다(모델 기반).\[49\]
   - 읽은 범위: 2차 요약과 초록.
3. **Favila, Kuhl, Winawer 2022**, "Perception and memory have distinct spatial tuning properties in human visual cortex," Nature Communications 13:5864, DOI 10.1038/s41467-022-33161-8.\[50\]
   - **방법과 보인 것:** pRF 모델로 기억 인출 시 활동을 모수화했습니다. 지각에서는 공간 정밀도가 위계를 따라 3배 감소하지만 기억에서는 이 패턴이 없었습니다. 그 결과 초기 영역에서 지각과 기억의 tuning 폭(FWHM)이 크게 달랐습니다.\[29\]\[30\]
   - **해석:** 위계적 feedback 때문이며, 네트워크 시뮬레이션이 이를 뒷받침합니다.\[50\]\[51\]
   - 참가자는 n=9입니다. 9명 평균에서 V1 FWHM은 기억 시 지각보다 2.5배 컸고, 개인별로는 9명 중 7명에서 그랬습니다(Nat Commun 2022). 장비는 not verified.
4. **"Feedback Scales the Spatial Tuning of Cortical Responses during Both Visual Working Memory and Long-Term Memory,"** Woodry, Curtis, Winawer(NYU), J Neurosci 45(17):e0681242025(2025), DOI 10.1523/JNEUROSCI.0681-24.2025(PMC12019112). bioRxiv 2024.04.11.589111이 게재된 버전입니다.
   - **보인 것:** 작업기억과 장기기억 모두에서 초기 지도의 tuning이 지각보다 넓었습니다. 다만 Favila 2022보다 그 차이가 작았습니다. V4–IPS0에서는 두 연구가 일치했습니다.\[52\]
5. **Chang, Zhang, Cao, Pearson, Meng 2025**, "Imageless imagery in aphantasia revealed by early visual cortex decoding," Current Biology 35(3):591–599.e4, DOI 10.1016/j.cub.2024.12.012.\[53\]
   - **방법:** ROI는 V1과 V2/3이고, 심상 대상은 Gabor 등입니다(2차 출처).
   - **보인 것:** "Imagery" 내용은 두 집단 모두에서 decoding되었습니다. aphantasia 집단에서는 신호가 동측이었고 지각과 cross-decoding되지 않았습니다. 지각 BOLD도 aphantasia에서 더 낮았습니다.\[32\]\[33\]\[54\]
   - **해석:** "less or transformed sensory information"을 담은 심상 관련 표상이 존재한다.\[32\]\[54\]
   - **논쟁:** 아래 Citation quality notes 참조.
6. **Lee, Kravitz, Baker 2012** (NeuroImage, 위 표 참조).
   - **보인 것(서브에이전트 2차 확인):** 정보 기울기가 두 조건에서 반대 방향입니다. 지각에서는 retinotopic 영역에 정보가 더 많고, 심상에서는 object-selective 영역에서 상대 비율이 더 높습니다. Cross-decoding은 모든 ROI에서 유의했고 V1도 포함됩니다.\[35\]\[36\]\[38\]
   - 정확도 수치와 장비는 not verified.

### B. 범주 선택 영역 (LO/pFs, OFA/FFA, EBA/FBA, VWFA, PPA/OPA/RSC-MPA, food)

- **물체:** Lee et al. 2012의 LO/pFs 결과는 위와 같습니다. 같은 저자들의 후속 Cerebral Cortex 논문에서 pFs가 가장 안정적이었다는 내용은 2차 출처뿐입니다(not verified).\[55\]
- **얼굴·장소:** O'Craven & Kanwisher, "Mental Imagery of Faces and Places Activates Corresponding Stimulus-Specific Brain Regions"는 **제목만 확인**했고, 세부 수치는 not verified입니다.\[56\]
- **Scrivener, Teed, Silson 2025** (EEG–fMRI fusion)
  - **보인 것:** VTC에서는 초기 시간창에 장소 선택 영역의 fusion이 높았고, 후기 시간창에 얼굴 선택 영역이 높았습니다. V1 fusion은 심상 기간 후반에 나타났습니다.\[14\]
  - **저자 해석:** 범주 영역에서 V1로 내려가는 top-down 진행일 "possibly".\[14\]
  - **2차 요약(2026 bioRxiv 인용):** 사람 대 장소 범주 정보는 첫 약 700 ms에 가장 강했습니다. 개별 항목 정보는 장소가 약 700 ms, 사람이 약 1100 ms에 다시 나타났습니다.\[43\]
  - 자극은 참가자 본인에게 친숙한 사람 6명과 장소 6곳입니다.\[14\]
- **장면 / place memory 계열 (Steel–Robertson–Silson)**
  - **Steel, Billings, Silson, Robertson 2021**, Nat Commun 12:2632, DOI 10.1038/s41467-021-22848-z(Publisher Correction 10.1038/s41467-021-23781-x).\[40\]\[57\]
    - **보인 것:** 친숙한 장소의 시각적 회상은 장면 지각 영역 OPA/PPA/MPA 각각의 **바로 전방**에 있는 세 영역(LPMA/VPMA/MPMA)을 활성화했습니다. 이 영역들은 별도의 네트워크를 이룹니다.\[41\]\[42\]
    - 친숙한 장소 이미지를 볼 때 PMA 활동은 증가했지만 지각 영역에서는 증가가 없었습니다.\[58\]
  - **Steel et al. 2025**, "Relating Scene Memory and Perception Activity to … A Probabilistic Atlas," J Neurosci 45(23):e0028252025. PMA를 확률적 ROI로 공개했습니다.\[59\] **Geon의 NSD 분석에 바로 쓸 수 있는 ROI 자원입니다.**
  - **"Scene Perception–Memory Pairing Extends to the Superior Parietal Cortex,"** eNeuro 13(9), ENEURO.0155-26.2026. bioRxiv 2026.05.08.723871이 게재된 것이며, 저자에 Scrivener, Silson, Steel이 포함됩니다.\[60\]\[61\] 장면 지각–기억 쌍이 상두정(superior parietal)까지 확장되고, 연결성으로 둘을 분리할 수 있음을 보였습니다.\[41\]
  - **"Memory Retrieval of Visuospatial Context is Supported by the Anterior Portions of High-Level Visual Cortex,"** PubMed 38171646. 저자, 연도, 저널은 not verified입니다. 시야 밖의 시공간 맥락이 전방 영역에 표상된다는 주장입니다.\[41\]
  - **"A retinotopic code structures the interaction between perception and memory systems,"** Steel, Silson, Garcia, Robertson, Nature Neuroscience 27:339–347. Robertson 연구실 페이지는 2021로 적었지만 권호상 2024로 보입니다.\[62\] 연도 not verified.
- **Bainbridge, Hall, Baker 2021**, "Distinct representational structure and localization for visual encoding and recall during visual imagery," Cerebral Cortex 31:1898–1913, DOI 10.1093/cercor/bhaa329.
  - **보인 것:** 회상은 부호화(encoding) 패턴의 단순 재활성화가 아닙니다. 표상 구조와 위치가 다르고, 회상 정점은 더 전방에 있습니다. 부호화와 회상의 관계는 모든 시각·기억 영역에서 제한적이었습니다.\[63\]
  - 읽은 범위: 초록·스니펫.
- **VWFA, EBA/FBA, food 영역:** 이번 검색에서 이 ROI들에 대한 지각→심상 decoding, RSA, vividness 연구는 **찾지 못했습니다.** NSD에서 food 선택 영역이 발견되었다는 연구("A highly selective response to food in human visual cortex revealed by hypothesis-free voxel decomposition")는 참고문헌 목록에서 제목만 확인했습니다.\[7\]

### C. Stream·두정·전두·내측두 수준

- **NSD streams ROI 정의:** "A single computational objective can produce specialization of streams in visual cortex," Nature Communications 2026(s41467-026-76098-y). 저자 not verified. EVC는 Wang 아틀라스 V1v–V3d의 합집합이고, 중간·고차 ventral/lateral/dorsal ROI는 해부학적 랜드마크와 noise ceiling을 함께 써서 그렸습니다. 이것이 NSD 공개 데이터의 "streams" ROI입니다.\[64\] **즉 streams ROI는 functional localizer가 아닌 해부·신뢰도 기반 정의입니다.**
- **Doerig et al. 2025:**
  - **보인 것:** NSD streams ROI에서 RSA, encoding, decoding을 수행했고, LLM(MPNet) 캡션 임베딩이 고차 영역 표상과 정렬되었습니다. 이 정렬은 물체 범주 정보 이상의 맥락 정보에 의존합니다.\[65\]\[66\]
  - **저자 해석:** 시각계가 캡션 임베딩과 정렬된 형식으로 수렴한다.\[17\]
  - **내 추론:** 지각에서는 Geon의 "맥락 subspace" 타깃(텍스트 임베딩)의 타당성을 가장 강하게 뒷받침합니다. 하지만 심상 데이터는 쓰지 않았습니다.
- **Dijkstra, Bosch, van Gerven 2017 J Neurosci:** 시행 간(moment-to-moment) 생생함이 전두·두정·시각 네트워크와 관련됩니다.\[67\] 시각계 전체의 지각–심상 중첩이 생생함과 상관이 있었고, 전운동피질에서도 공유 표상이 나타났습니다.\[46\]\[68\]
- **Dijkstra et al. 2017 Sci Rep(DCM):** 심상에서는 top-down 연결이 우세하고, 지각 특유의 bottom-up 증가는 없었습니다.\[11\]\[69\]
- **Dijkstra et al. 2021 eNeuro:** 의식적 지각과 심상 사이의 중첩이 시각·두정·전두에서 확인되었습니다.\[9\]
- **MPC 계열:**
  - Silson et al. 2019 eLife(DOI 10.7554/eLife.47391): 사람과 장소 회상의 하위 영역이 분리됩니다.\[14\]\[43\]
  - Scrivener & Silson 2025 Imaging Neuroscience 3:imag_a_00507: opponent +/− pRF 코딩.\[44\]\[70\]
  - Kidder, Silson, Nau, Baker 2025 eNeuro 12(1): 사람·장소·물체 회상 영역이 분산되어 있습니다.\[70\]
  - Bainbridge & Baker 2022 Nat Commun 13:6508: 일상 기억 영상 수천 개로 MPC의 다차원 기억 지형을 보였습니다.\[71\]
- **해마/MTL:** 지각 대 심상의 ROI별 cross-decoding을 해마에서 직접 보고한 연구는 이번 검색에서 확보하지 못했습니다(not verified).

### D. 좌측 fusiform "imagery node"와 심상 특이 영역

- **Spagna et al. 2021 메타분석:** fMRI 연구 27편을 분석해 좌측 lateral mid-fusiform의 FIN을 보고했습니다. 모든 영역(domain)에서 활성화되며, EVC는 일관되게 활성화되지 않았습니다.\[45\]\[72\]
- **Liu, Zhan, Hajhajate, Spagna, Dehaene, Cohen, Bartolomeo 2025**, "Visual mental imagery in typical imagers and in aphantasia: A millimeter-scale 7-T fMRI study," Cortex 185:113–132, DOI 10.1016/j.cortex.2025.01.013.\[53\]\[73\]
  - **전형적 심상자:** 좌측 전두두정 영역, 지각 영역과 부분적으로 겹치는 domain-preferring VTC 영역, 그리고 FIN이 활성화되었습니다.\[73\]
  - **Aphantasia:** 고차 시각영역에서 심상과 지각의 활성과 표상 내용이 비슷했습니다.\[74\] 그러나 FIN과 전두두정 사이의 연결성에 이상이 있었고, 우반구 활성이 더 높았습니다.\[75\]\[76\]
  - **저자 해석:** 보존된 시각 정보에 의식적으로 접근하지 못하는 것.\[73\]
  - 표본은 전형적 심상자 10명과 선천성 aphantasia 10명이고, 7T 1.2 mm 등방성 해상도로 촬영했습니다(bioRxiv 2023.06.14.544909: "10 typical imagers and 10 individuals with congenital aphantasia").
- **좌측 후두측두 손상 환자 RDS:** "The connectional anatomy of visual mental imagery: evidence from a patient with left occipito-temporal damage," Brain Struct Funct 2022(s00429-022-02505-x). bioRxiv 2022.02.15.480510 저자는 Hajhajate, Kaufmann, Liu, Siuda-Krzywicka, Bartolomeo입니다. 이 환자의 손상이 FIN 부위를 완전히 파괴했습니다.\[77\] 인과적 근거이지만 n=1입니다.\[72\]
- **Bartolomeo, Liu, Spagna, "The Fusiform Imagery Node: Where vision meets concepts in the left temporal lobe,"** Neuropsychologia(PII S0028393226000448, 2026으로 추정). 리뷰입니다.\[45\]

### E. NSD / NSD-Imagery 기반 연구

- **Kneeland et al. CVPR 2025:**
  - **데이터:** NSD 피험자 8명, 7T. 자극 18개(단순 6, 복잡 6, 개념/단어 6)이고 문자 단서로 심상을 유도했습니다.\[78\]\[79\]
  - **보인 것:** MindEye1, MindEye2, Brain Diffuser, iCNN, Takagi 등을 벤치마크했습니다. 심상 재구성 성능은 지각 성능과 "largely decoupled"였습니다. 단순 선형 구조와 multimodal 특징 decoding이 더 잘 일반화되었습니다.\[23\]\[80\]\[81\]
  - Takagi 방법만 NSD streams의 early 및 ventral ROI를 썼습니다(ROI 분해는 이 정도가 전부).\[80\]
- **MIRAGE (PLoS CB 2026):**
  - **방법:** ridge regression, 텍스트+이미지 multimodal 타깃, Stable Cascade를 썼고, 저수준 "layout" 이미지를 먼저 재구성합니다.\[82\]\[83\]
  - **보인 것:** 인간 2AFC 식별(평가자 약 500명, 확률 수준 50%, 모두 p<0.001)에서 모든 자극 유형에 대해 최고 성능을 보였습니다.\[26\] **imagery trial의 개별 % 수치는 not verified**(표 이미지를 읽지 못함). 3시간 분량의 학습 데이터로도 작동하고, 40 세션까지 확장됩니다.\[28\]
  - **ROI별 분해:** 확인되지 않았습니다.
- **ROI/parcel 분해를 시도한 2025–2026 preprint:**
  - **"Seeing the imagined: a latent functional alignment…"** arXiv 2604.15374(2026년 4월, preprint, 심사 미확인). HCP-MMP1 parcel별 심상 decoding을 수행했습니다. 후두 피질이 주된 기여를 했고, 비시각 ROI 단독 성능은 낮았습니다. 피험자 간 변이가 크다고 직접 언급합니다.\[84\]\[85\]
  - **NeuroAdapter ("Towards Interpretable Visual Decoding with Attention to Brain Representations")**, arXiv 2509.23566(preprint). V1–V3, V4, LOC, FFA, PPA, HVC parcel을 썼고 NSD-Imagery 결과는 부록에 있습니다.\[78\]\[86\]
- **비-NSD 재구성:**
  - **Shen, Horikawa, Majima, Kamitani 2019**, PLoS CB 15(1):e1006633. 심상에서 "rudimentary" 재구성을 보였습니다.\[87\]\[88\]
  - **Koide-Majima, Nishimoto, Majima 2024**, Neural Networks 170:349–363, DOI 10.1016/j.neunet.2023.11.024.\[89\] **논쟁 중**(아래 참조).
  - **Caselles-Dupré et al., "Mind-to-Image,"** arXiv 2404.05468(preprint, 심사 여부 not verified). 단일 피험자 약 6시간 분량이며, 기억 기반 심상과 순수 상상 심상을 구분했습니다.\[90\]

### F. Coarse vs fine, 과제 의존성, 위치 의존성

- **Coarse vs fine:**
  - Saha Roy 2025: 범주 구분(단순 vs 장면)은 모든 ROI에서 보존되지만, 범주 내 구분은 parietal 외에서 약하거나 없습니다.\[1\]
  - Scrivener 2025: 범주 정보가 먼저 나타나고 항목 정보는 나중에 나타납니다(EEG).\[43\]
  - Breedlove 2020: 낮은 공간주파수, 즉 gist에 가까운 코드입니다.\[49\]
  - **세 연구는 "심상은 coarse/gist가 우선"이라는 데 서로 독립적으로 수렴합니다.**
- **과제 의존성:** Dijkstra 2019 리뷰와 Sci Rep 2017 서론에 따르면, 초기 영역의 중첩은 과제가 요구하는 세부 수준과 생생함에 따라 달라집니다(리뷰 수준의 주장).\[9\]\[91\] 같은 피험자에서 본 것, 기억한 것, 새로 상상한 것을 체계적으로 비교한 fMRI는 이번 검색에서 확인하지 못했습니다. Mind-to-Image는 기억 대 상상을 구분했지만 preprint이고 n이 매우 작습니다.
- **위치 의존성:** Breedlove 2020과 Saha Roy Dataset 2는 상상한 위치(8곳)가 심상 부분공간에 명시적으로 코딩됨을 보였습니다.\[1\] Favila 2022는 기억 위치의 retinotopic 재활성화를 보였지만 tuning이 넓었습니다.\[29\]\[50\]

## 관련 논문 목록 (ROI 그룹별; "추가하는 것" / "주요 한계")

**초기 시각피질 / pRF**
1. Saha Roy et al. 2025 (bioRxiv preprint) — 지각→심상 변환을 부분공간 차원과 정렬로 정량화함 / preprint, 자극 12개, 고-SNR voxel만 사용.
2. Saha Roy et al. CCN 2023 — 같은 가설("compressed vision")의 초기 버전 / 초록 수준.
3. Breedlove et al. 2020 Curr Biol — 심상 pRF와 공간주파수 변환, 생성 feedback 모델 / n=3.
4. Favila, Kuhl, Winawer 2022 Nat Commun — 기억 인출 시 pRF tuning이 위계와 무관하게 넓음 / 단순 공간 자극, n=9.
5. Woodry, Curtis, Winawer 2025 J Neurosci "Feedback Scales the Spatial Tuning…" — 작업기억과 장기기억으로 일반화 / Favila와 효과 크기 차이.
6. Lee, Kravitz, Baker 2012 NeuroImage — 위계를 따라 반대 방향의 정보 기울기, V1을 포함한 cross-decoding / 물체 10개, 수치·장비 미확인.
7. Chang et al. 2025 Curr Biol — aphantasia에서 V1 decoding은 되지만 cross-decoding은 실패 / 해석 논쟁, 단순 자극.

**범주 선택 영역 / 장면·기억**
8. Scrivener, Teed, Silson 2025 Neurosci Conscious — 범주→항목, VTC→V1 시간 순서 / EEG–fMRI fusion은 간접 지표, 사람 대 장소 2범주.
9. Steel et al. 2021 Nat Commun — place memory areas 발견 / 친숙한 장소 회상에 한정.
10. Steel et al. 2025 J Neurosci — PMA 확률 아틀라스 / 개인 localizer를 대체하지 못함.
11. eNeuro 2026 "Scene Perception–Memory Pairing Extends to the Superior Parietal Cortex" — 상두정 쌍 / 최신, 2026년 7월 이후 재현 없음.
12. Steel, Silson, Garcia, Robertson, Nature Neuroscience 27 — 지각–기억 상호작용의 retinotopic 코드 / 연도 불일치.
13. Bainbridge, Hall, Baker 2021 Cereb Cortex — 회상은 재활성화가 아님 / 장면·물체 수가 제한적(수치 미확인).
14. Silson et al. 2019 eLife — MPC 사람·장소 하위 영역 / 회상 과제 중심.
15. Scrivener & Silson 2025 Imaging Neurosci — MPC opponent pRF 코딩 / 지각–기억 decoding 아님.
16. Kidder, Silson, Nau, Baker 2025 eNeuro — 사람·장소·물체 회상 영역 분포 / univariate 중심(추정).
17. Bainbridge & Baker 2022 Nat Commun — MPC 기억 지형 / 지각 대비 분석이 아님.
18. PubMed 38171646 "Memory Retrieval of Visuospatial Context…" — 시야 밖 맥락이 전방 영역에 / 서지 미확인.
19. Wadia et al. 2026 Science "A shared code for perceiving and imagining objects in human ventral temporal cortex" — 단일뉴런 수준의 재활성화 / fMRI 아님, 환자 표본.

**Stream / 두정 / 전두**
20. Dijkstra, Bosch, van Gerven 2017 J Neurosci — 생생함과 전 시각계 중첩, 전운동피질 / 범주 수준 자극.
21. Dijkstra et al. 2017 Sci Rep — DCM top-down 대 bottom-up / ROI 4개 모델.
22. Dijkstra et al. 2021 eNeuro — 의식적 지각과 심상의 중첩이 시각·두정·전두에 걸침 / 무의식 처리와는 중첩 없음.
23. Nat Commun 2026 streams 논문 — NSD streams ROI 정의 근거 / 심상 무관.

**FIN / 심상 특이 영역 / aphantasia**
24. Spagna et al. 2021 — FIN 메타분석 / univariate, 이질적 과제.
25. Liu et al. 2025 Cortex — 7T, FIN과 aphantasia의 연결성 / n=10+10으로 작음.
26. Brain Struct Funct 2022 RDS 사례 — FIN 손상의 인과 근거 / n=1.
27. Liu, Spagna, Bartolomeo 2021 Brain Struct Funct — 좌반구 편재 리뷰 / 리뷰.
28. Bartolomeo, Liu, Spagna (Neuropsychologia, 2026 추정) FIN 리뷰 / 저자 집단의 자기 리뷰.
29. Neuropsychologia 2026 "A neural model of conscious mental imagery and aphantasia" — aphantasia ROI 소견 표 / 저자 미확인, 리뷰.\[39\]
30. Cortex 2025 Viewpoint "Revisiting the mental imagery debate" — aphantasia 연구 3편에서 V1 decoding 가능 / 의견 논문.\[33\]
31. "Against Unconscious Mental Imagery in Aphantasia"(2025, venue 미확인) — Chang 2025 해석 반박 / 비실증.
32. bioRxiv 2025.05.25.655911 "Reduced Imagery Vividness in Aphantasia is Associated…" — 측두엽 복잡성과 dFC / preprint, 기존 데이터 재분석.\[92\]
33. "The Neural Underpinnings of Aphantasia: A Case Study of Identical Twins"(PMC11463508) — 쌍둥이 간 cross-decoding(지각 66.8%, 회상 67.89%) / n=2, venue 미확인.\[53\]

**모델·의미 / NSD 재구성**
34. Wang et al. 2023 NMI — CLIP 언어 감독이 고차 시각피질을 더 잘 설명 / 지각 전용.\[15\]
35. Doerig et al. 2025 NMI — 캡션 LLM 임베딩과 고차 stream의 정렬 / 지각 전용.
36. Elmoznino & Bonner 2024 PLoS CB — 고차원 잠재 표상과 성능 / 지각 전용, 모델 기반.\[21\]
37. Gauthaman, Ménard, Bonner, "Universal scale-free representations in human visual cortex," PLoS CB(pcbi.1013714; arXiv 2409.06843) — 시각피질 표상의 차원이 사실상 무한 / 지각 전용.\[93\] **Saha Roy의 "저차원" 추정과 함께 해석할 때 주의**(아래 참조).
38. Kneeland et al. CVPR 2025 — NSD-Imagery 벤치마크 / 자극 18개, 세션 1회.\[23\]
39. MIRAGE PLoS CB 2026 — 심상 재구성 SOTA, 단순 구조가 유리 / ROI 분해 없음.\[26\]
40. Shen et al. 2019 PLoS CB — 심상의 deep image reconstruction / 소수 피험자.\[88\]
41. Koide-Majima et al. 2024 Neural Networks — Bayesian 심상 재구성 / 재분석에서 이의 제기.

## Contradictions and open disagreements

1. **초기 시각피질에 심상 내용이 있는가**
   - **"없다" 쪽:** Spagna et al. 2021 메타분석. 제목 그대로 "but not the early visual cortex"입니다.
   - **"있다" 쪽:** Lee 2012의 V1 포함 cross-decoding, Breedlove 2020과 Saha Roy 2025의 V1 예측과 pRF, Favila 2022의 retinotopic 재활성화, Chang 2025의 V1 decoding.
   - **차이의 방법론적 원인:**
     - **univariate 대 multivariate:** 메타분석은 평균 활성(대비)을 모읍니다. MVPA와 encoding은 평균 활성이 낮아도 패턴 정보를 잡아냅니다. Liu 2025 등은 aphantasia에서 foveal V1의 *비활성화(deactivation)*가 강해졌다고도 보고했습니다(2차 출처).
     - **ROI 정의:** 개인 retinotopy(pRF) ROI와 표준공간 좌표 기반 메타분석은 해상도가 다릅니다.
     - **feature selection:** Saha Roy는 SNR 상위 2% voxel만 썼습니다. 이렇게 고르면 V1 결과가 양(+)의 방향으로 치우칠 수 있습니다(내 추론).
     - **자극과 과제:** 단순 방위·위치 자극과 고정 위치에 상상하도록 한 과제는 V1 효과를 키웁니다. 범주 심상이나 무위치 회상은 효과를 줄입니다.
     - **생생함:** V1 효과가 VVIQ나 시행별 생생함에 의존합니다(Saha Roy r=0.71, Dijkstra 2017).
2. **Lee 2012의 생생함 효과 위치:** PubMed 그림 설명은 EVC와 object-selective 둘 다라고 하고, Vision Research 리뷰 그림 설명은 V1–V3만이라고 합니다.\[36\]\[37\] 원문 확인이 필요합니다(not verified).
3. **회상 = 재활성화인가:**
   - **아니라는 쪽:** Bainbridge 2021(구조·위치가 다름), Steel 2021(지각 영역이 아니라 전방 PMA), Favila 2022(tuning이 다름).
   - **그렇다는 쪽:** Dijkstra 2017, 2019의 "공유" 프레임과 Saha Roy의 고차 영역 항등성.
   - **조정:** 고차 영역은 대체로 공유하고, 초기 영역과 장면 영역의 경계는 위치나 형식이 이동합니다(내 종합).
4. **Aphantasia의 V1 신호를 어떻게 해석할까:** Chang 2025는 "imageless imagery"라고 봅니다. 반면 "Against Unconscious Mental Imagery in Aphantasia"와 Scholz·Monzel 등은 decoding이 가능하다는 것만으로 무의식적 심상을 추론할 수 없다고 봅니다.\[39\]\[94\] Liu 2025는 고차 영역에서는 심상과 지각이 비슷하고, 의식 접근이 실패한다고 해석합니다.\[73\]
5. **차원성:** Saha Roy는 차원이 12 이하라고 추정합니다(자극 수 제약). 반면 Bonner 연구실은 NSD 지각 표상이 "scale-free, effectively unbounded"하다고 봅니다.\[93\] 저자가 직접 "총 차원 추정이 아님"이라고 경고했으므로,\[1\] 둘은 서로 모순이라기보다 측정 체계가 다릅니다.
6. **FIN 좌표:** FIN 리뷰는 메타분석 FIN을 (−42, −54, −18)로 적었고, Cortex 2025 본문 스니펫은 "(−40, −55, −11)"로 적었습니다.\[45\]\[74\] z축이 7 mm 차이 납니다. ROI 구성 시 원문 확인이 필요합니다.

## What the literature does NOT cover (장면 맥락 관련 — 이미 답이 나왔다고 가정하면 안 되는 것)

1. **장소 유형, 환경(실내/실외, 자연/도시), 활동, 암시된 사건, 스케일 축이 심상에서 지각과 같은 부분공간에 있는지:** 이를 직접 검증한 fMRI는 찾지 못했습니다. 기존 심상 연구의 의미 수준은 "범주(사람/장소, 단순/복잡)"나 "개별 항목"에 그칩니다.
2. **텍스트(CLIP/LLM) 임베딩 축 → 심상 전이:** Doerig와 Wang은 지각에서만 검증했습니다. MIRAGE는 multimodal 텍스트 특징이 심상 재구성에 유리하다는 공학적 결과일 뿐, ROI별 의미 축 전이 분석은 아닙니다.
3. **PMA/MPC와 맥락 축:** PMA는 "친숙한 실제 장소"의 회상에서 정의되었습니다. COCO 같은 비친숙 장면의 심상이나 맥락 속성(활동·사건)이 PMA에 코딩되는지는 확인되지 않았습니다.
4. **NSD-Imagery의 맥락 해상도:** 복잡 장면은 6개뿐이고 개념 단서도 6개입니다. 맥락 축 수가 자극 수보다 많아지기 쉬워서, 맥락과 범주(단순 대 장면)를 분리하기 어렵습니다(내 추론).
5. **본 것, 기억한 것, 새로 상상한 것의 ROI별 비교, 그리고 위치 대 맥락의 상호작용:** 체계적 fMRI 연구를 찾지 못했습니다.
6. **VWFA, EBA, food 영역의 심상:** 찾지 못했습니다.
7. **Hyperphantasia의 ROI별 차이:** 이번 검색에서 확인한 fMRI 결과가 없습니다.

## Recommendations (Geon의 캡스톤용)

1. **ROI 설계:**
   - NSD streams(early/ventral/lateral/parietal)를 기본으로 씁니다.
   - floc 범주 ROI(PPA/OPA/RSC, FFA, EBA, VWFA)와 Steel 2025 PMA 확률 ROI를 추가합니다.
   - Spagna/Liu FIN 좌표 ROI를 추가하되, 좌표 불일치를 명시합니다.
2. **분석:**
   - 지각에서 학습한 맥락 축을 심상에 투사할 때 "정렬비"(Saha Roy 방식)와 "축별 tuning 보존"을 분리해서 보고합니다.
   - Saha Roy는 정렬이 낮아도 tuning이 근사될 수 있음을 보였습니다.\[1\]
3. **혼입 통제:** 단순 대 복잡 자극의 범주 차이가 모든 ROI에서 지배적이므로, 맥락 축 분석은 **장면 자극 내부에서만** 수행해야 합니다.
4. **voxel 선택:** SNR 필터 기준(예: 상위 2% 대 전체)에 따라 결과가 얼마나 민감한지 사전 등록하는 것을 권합니다.
5. **결론의 수위:** "심상이 맥락 subspace를 재사용한다"는 주장은 parietal과 고차 ventral에서는 선행 근거와 일치합니다. 그러나 EVC 결과가 부정적이어도 "회전된 부분공간" 때문일 수 있으므로, 이를 반증으로 해석하면 안 됩니다.

## Citation quality notes

- **Preprint → 게재:**
  - Doerig(arXiv 2209.11737 → NMI 2025)
  - Elmoznino & Bonner(bioRxiv 2022 → PLoS CB 2024)
  - Feedback scales spatial tuning(bioRxiv 2024 → J Neurosci 2025)
  - superior parietal pairing(bioRxiv 2026.05 → eNeuro 2026)
  - MIRAGE(CCN 2025 / arXiv 2605.17198 → PLoS CB 2026-05-22)
  - Gauthaman et al.(arXiv 2409.06843 → PLoS CB)
  - 연결 해부 사례(bioRxiv 2022 → Brain Struct Funct 2022)
- **아직 preprint:**
  - Saha Roy 2025(PubMed/PMC 표기 "not yet peer reviewed")\[2\]
  - "Seeing the imagined"(arXiv 2026)
  - NeuroAdapter(arXiv 2025)
  - Mind-to-Image(arXiv 2024; 심사 여부 not verified)
  - aphantasia dFC(bioRxiv 2025)
- **정정(Correction):** Steel et al. 2021 Nat Commun의 Publisher Correction(10.1038/s41467-021-23781-x)은 보충자료 링크 오류 수정이며 결과와는 무관합니다.\[57\]
- **논쟁 중인 주장:**
  - Koide-Majima et al. 2024: arXiv 2511.07960의 독립 재분석("Advancing credibility and transparency in brain-to-image reconstruction research…")이 평가 방식(Inception Score, 쌍별 식별)과 대표 재구성 예시에 이의를 제기했습니다.\[95\] 저자 측은 FAQ 페이지로 반론했습니다.\[96\]
  - NSD 기반 재구성 전반: Shirakawa et al. 2025 "Spurious reconstruction from brain activity"(Neural Networks 190:107515)가 일반화 문제를 제기했습니다.\[87\]
  - Chang 2025 "imageless imagery": 이의가 제기되었습니다(위 참조).
- **철회(retraction):** 이번 검색에서 위 논문 중 철회된 것은 발견되지 않았습니다(체계적 Retraction Watch 검색은 하지 않음).

## Caveats

- **전문을 실제로 읽은 것은 Saha Roy et al. 2025 bioRxiv v1 한 편뿐입니다.** 나머지는 초록, 검색 스니펫, 2차 요약(리뷰 표, 보도자료, 서브에이전트 확인)에 근거합니다. 많은 연구의 n, 장비 세기, ROI 정의 방법이 "not verified"인 이유입니다.
- 일부 2026년 논문(Neuropsychologia 리뷰 2편, Nat Commun streams 논문)은 저자 목록을 확인하지 못했습니다.
- 검색 예산 한계로 다음 고전 연구는 이번 지도에서 원자료로 검증하지 못했습니다: Naselaris 2015, Johnson & Johnson 2014, Horikawa & Kamitani 2017, Albers 2013, Reddy 2010, Stokes 2009, O'Craven & Kanwisher 2000. 인용 전에 직접 확인하기 바랍니다.

## Sources

1. <https://www.biorxiv.org/content/10.1101/2025.09.02.672180v1>
2. [A transformation from vision to imagery in the human brain - PubMed](https://pubmed.ncbi.nlm.nih.gov/40950062/)
3. [Electrophysiological Evidence for a Shared Representational Medium for Visual Images and Visual Percepts](https://www.researchgate.net/publication/374664312_Electrophysiological_Evidence_for_a_Shared_Representational_Medium_for_Visual_Images_and_Visual_Percepts)
4. [fMRI vision reconstruction methods robustly generalize to mental imagery](https://2024.ccneuro.org/pdf/308_Paper_authored_Mental_Imagery_CCN_Paper.pdf)
5. [Technical Program: Paper Detail || CCN 2023 || 2023 Conference on Cognitive Computational Neuroscience || Oxford, UK || August 24 - 27, 2023](https://2023.ccneuro.org/view_paper4eea.html?PaperNum=1693)
6. [NSD-Imagery: A benchmark dataset for extending fMRI vision decoding methods to mental imagery](https://arxiv.org/html/2506.06898v1)
7. [Technical Program: Paper Detail || CCN 2022 || 2022 Conference on Cognitive Computational Neuroscience || San Francisco || August 25 - 28, 2022](https://2022.ccneuro.org/view_paper11db.html?PaperNum=1253)
8. [Reconstructing imagined letters from early visual cortex reveals tight topographic correspondence between visual mental…](https://ouci.dntb.gov.ua/en/works/9jeVPNol/)
9. [No Evidence for Neural Overlap between Unconsciously Processed and Imagined Stimuli](https://www.eneuro.org/content/8/5/ENEURO.0228-21.2021)
10. [From dots to faces: Individual differences in visual imagery capacity predict the content of Ganzflicker-induced hallucinations](https://arxiv.org/pdf/2507.09011)
11. [Shared Neural Mechanisms of Visual Perception and Imagery - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1364661319300592)
12. [The connectional anatomy of visual mental imagery: evidence from a patient with left occipito-temporal damage](https://link.springer.com/article/10.1007/s00429-022-02505-x)
13. [Visual imagery of familiar people and places in category selective cortex - PubMed](https://pubmed.ncbi.nlm.nih.gov/40241880/)
14. [Visual mental imagery: Evidence for a heterarchical neural architecture](https://ouci.dntb.gov.ua/en/works/45djb6M7/)
15. [Better models of human high-level visual cortex emerge from natural language supervision with a large and diverse dataset](https://www.nature.com/articles/s42256-023-00753-y)
16. [Better models of human high-level visual cortex emerge from natural language supervision with a large and diverse dataset - Experts@Minnesota](https://experts.umn.edu/en/publications/better-models-of-human-high-level-visual-cortex-emerge-from-natur/)
17. [High-level visual representations in the human brain are aligned with large language models](https://www.nature.com/articles/s42256-025-01072-0)
18. [What Makes Linguistic Representations Good Models of High-Level Visual Perception in the Human Brain?](https://arxiv.org/pdf/2607.16214)
19. [Universal scale-free representations in human visual cortex](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013714)
20. [High-performing neural network models of visual cortex benefit from high latent dimensionality](https://www.biorxiv.org/content/10.1101/2022.07.13.499969v1)
21. [High-performing neural network models of visual cortex benefit from high latent dimensionality](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1011792)
22. [Curriculum Vitae July 2025 1 Michael F. Bonner mfbonner@jhu.edu](https://www.bonnerlab.org/downloads/mick_resume.pdf)
23. [NSD-Imagery - CVPR 2025 Open Access Repository](https://openaccess.thecvf.com/content/CVPR2025/html/Kneeland_NSD-Imagery_A_Benchmark_Dataset_for_Extending_fMRI_Vision_Decoding_Methods_CVPR_2025_paper.html)
24. [NSD-Imagery: A Benchmark Dataset for Extending fMRI Vision Decoding Methods to Mental Imagery](https://mlanthology.org/cvpr/2025/kneeland2025cvpr-nsdimagery/)
25. [MIRAGE: Robust multi-modal architectures translate fMRI-to-image models from vision to mental imagery - PubMed](https://pubmed.ncbi.nlm.nih.gov/42172302/)
26. [MIRAGE: Robust multi-modal architectures translate fMRI-to-image models from vision to mental imagery](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1014263)
27. [MIRAGE: Robust multi-modal architectures translate fMRI-to-image models from](https://2025.ccneuro.org/abstract_pdf/Kneeland_2025_MIRAGE_Robust_multi-modal_architectures_translate_fMRI-to-image.pdf)
28. [MIRAGE: Robust multi-modal architectures translate fMRI-to-image models from vision to mental imagery](https://arxiv.org/html/2605.17198)
29. [(PDF) Perception and memory have distinct spatial tuning properties in human visual cortex](https://www.researchgate.net/publication/364357695_Perception_and_memory_have_distinct_spatial_tuning_properties_in_human_visual_cortex)
30. [Perception and memory have distinct spatial tuning properties in human visual cortex](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9579130/)
31. [Generative Feedback Explains Distinct Brain Activity Codes for Seen and Mental Images](https://www.researchgate.net/publication/341059900_Generative_Feedback_Explains_Distinct_Brain_Activity_Codes_for_Seen_and_Mental_Images)
32. [Imageless imagery in aphantasia revealed by early visual ...](<https://www.cell.com/current-biology/pdfExtended/S0960-9822(24)01652-X>)
33. [Revisiting the mental imagery debate: New evidence from aphantasia and neuroimaging - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0010945225002539)
34. [(PDF) The Connectional Anatomy of Visual Mental Imagery: Evidence From a Patient with Left Occipito-Temporal Damage](https://www.researchgate.net/publication/358989816_The_Connectional_Anatomy_of_Visual_Mental_Imagery_Evidence_From_a_Patient_with_Left_Occipito-Temporal_Damage)
35. [The dynamic contribution of the high‐level visual cortex to imagery and perception - PMC](https://ncbi.nlm.nih.gov/pmc/articles/PMC6865452)
36. [Disentangling visual imagery and perception of real-world objects - PubMed](https://pubmed.ncbi.nlm.nih.gov/22040738/)
37. [Individual differences shape the content of visual representations - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0042698916301250)
38. [Disentangling visual imagery and perception of real-world objects - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1053811911012195)
39. [A neural model of conscious mental imagery and aphantasia - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0028393226000382)
40. [New study reveals where memories of familiar places are stored in the brain](https://www.sciencedaily.com/releases/2021/05/210513100006.htm)
41. [Scene Perception–Memory Pairing Extends to the Superior Parietal Cortex - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13540205/)
42. [A network linking scene perception and spatial memory systems in posterior cerebral cortex](https://www.nature.com/articles/s41467-021-22848-z)
43. [The neural dynamics of familiarity representation during cued recall](https://www.biorxiv.org/content/10.64898/2026.09.21.753245v1.full)
44. [Edward SILSON](https://www.researchgate.net/profile/Edward-Silson)
45. [The Fusiform Imagery Node: Where vision meets concepts in the left temporal lobe - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0028393226000448)
46. [Vividness of Visual Imagery Depends on the Neural Overlap with Perception in Visual Areas](https://www.jneurosci.org/content/37/5/1367)
47. [A shared code for perceiving and imagining objects in human ventral temporal cortex](https://www.science.org/doi/10.1126/science.adt8343)
48. [Generative Feedback Explains Distinct Brain Activity Codes for Seen and Mental Images](https://www.mendeley.com/catalogue/1cd99e9a-89d1-3b5b-9df6-a79d6e1c0585/)
49. [Article Generative Feedback Explains Distinct Brain Activity](http://www.naselarislab.net/s/breedlove_generative_2020.pdf)
50. [Perception and memory have distinct spatial tuning properties in human visual cortex](https://www.nature.com/articles/s41467-022-33161-8)
51. [publications](https://sfavila.github.io/publications/)
52. [Feedback Scales the Spatial Tuning of Cortical Responses during Both Visual Working Memory and Long-Term Memory](https://www.jneurosci.org/content/45/17/e0681242025)
53. [The Neural Underpinnings of Aphantasia: A Case Study of Identical Twins - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11463508/)
54. [Imageless imagery in aphantasia revealed by early visual cortex decoding - ADS](https://ui.adsabs.harvard.edu/abs/2025CBio...35..591C/abstract)
55. [Differential Representations of Perceived and Retrieved Visual Information in Hippocampus and Cortex - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7302513/)
56. [Mental Imagery of Faces and Places Activates Corresponding Stimulus-Specific Brain Regions](https://www.researchgate.net/publication/12154049_Mental_Imagery_of_Faces_and_Places_Activates_Corresponding_Stimulus-Specific_Brain_Regions)
57. [Publisher Correction: A network linking scene perception and spatial memory systems in posterior cerebral cortex](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8172901/)
58. [Newly-Identified Brain Circuits Helps Find Our Place in the World](https://www.technologynetworks.com/neuroscience/news/newly-identified-brain-circuits-helps-find-our-place-in-the-world-348779)
59. [Relating Scene Memory and Perception Activity to Functional Properties, Networks, and Landmarks of Posterior Cerebral Cortex—A Probabilistic Atlas - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12139592/)
60. [Scene Perception–Memory Pairing Extends to the Superior Parietal Cortex](https://www.eneuro.org/content/13/9/ENEURO.0155-26.2026)
61. [Scene perception-memory pairing extends to superior parietal cortex](https://www.biorxiv.org/content/10.64898/2026.05.08.723871v2.full)
62. [Caroline Robertson](https://pbs.dartmouth.edu/people/caroline-robertson)
63. [recall during visual imagery - Wilma Bainbridge](https://wilmabainbridge.com/sharepapers/cerebralcortex-2020.pdf)
64. [A single computational objective can produce specialization of streams in visual cortex](https://www.nature.com/articles/s41467-026-76098-y)
65. [Nature Machine Intelligence](https://refubium.fu-berlin.de/bitstream/handle/fub188/49579/s42256-025-01072-0.pdf?sequence=1&isAllowed=y)
66. [Visual representations in the human brain](https://arxiv.org/pdf/2209.11737)
67. [Mistaking imagination for reality: Congruent mental imagery leads to more liberal perceptual detection - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0010027721001384)
68. [Google Scholar](https://scholar.google.com/scholar_lookup?title=Vividness+of+visual+imagery+depends+on+the+neural+overlap+with+perception+in+visual+areas&publication_year=2017&author=N.+Dijkstra)
69. [Shared neural mechanisms of visual perception and imagery](https://atelierdesfuturs.org/wp-content/uploads/2025/01/Shared-neural-mechanisms-of-visual-perception-and-imagery-4smjhgw856.pdf)
70. [Edward Silson - University of Edinburgh Research Explorer](https://www.research.ed.ac.uk/en/persons/edward-silson/)
71. [Publications - Wilma Bainbridge](https://www.wilmabainbridge.com/publications.html)
72. [The connectional anatomy of visual mental imagery: evidence from a patient with left occipito-temporal damage](https://www.biorxiv.org/content/10.1101/2022.02.15.480510v3.full)
73. [Visual mental imagery in typical imagers and in aphantasia: A millimeter-scale 7-T fMRI study](https://aphantasia.com/research/10.1016/j.cortex.2025.01.013)
74. [Visual mental imagery in typical imagers and in aphantasia: A millimeter-scale 7-T fMRI study - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0010945225000474)
75. [Aphantasia Might Be Linked to Alterations in Brain Connectivity](https://parisbraininstitute.org/news/aphantasia-might-be-linked-alterations-brain-connectivity)
76. [Aphantasia might be linked to alterations in brain connectivity](https://medicalxpress.com/news/2025-06-aphantasia-linked-brain.html)
77. [The connectional anatomy of visual mental imagery:](https://www.biorxiv.org/content/10.1101/2022.02.15.480510v1.full.pdf)
78. [Towards Interpretable Visual Decoding with Attention to Brain Representations](https://arxiv.org/pdf/2509.23566)
79. [Plos](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1014263&type=printable)
80. [(PDF) NSD-Imagery: A benchmark dataset for extending fMRI vision decoding methods to mental imagery](https://www.researchgate.net/publication/392530538_NSD-Imagery_A_benchmark_dataset_for_extending_fMRI_vision_decoding_methods_to_mental_imagery)
81. [CVPR Poster NSD-Imagery: A Benchmark Dataset for Extending fMRI Vision Decoding Methods to Mental Imagery](https://cvpr.thecvf.com/virtual/2025/poster/35026)
82. [MIRAGE: Robust multi-modal architectures translate fMRI-to-image models from vision to mental imagery - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13218618/)
83. [Reading minds with fMRI and machine learning.](https://conservancy.umn.edu/items/ef901cee-eb83-475f-ab81-299518993ee3)
84. [1 Introduction](https://arxiv.org/html/2604.15374v1)
85. [Seeing the imagined: a latent functional alignment in visual ...](https://arxiv.org/pdf/2604.15374)
86. [Towards Interpretable Visual Decoding with Attention to Brain Representations](https://arxiv.org/html/2509.23566v1)
87. [Kei Majima - Google Scholar](https://scholar.google.com/citations?user=3fsDFcYAAAAJ&hl=en)
88. [Deep image reconstruction from human brain activity](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6347330/)
89. [GitHub - nkmjm/mental\_img\_recon: Mental image reconstruction from human brain activity · GitHub](https://github.com/nkmjm/mental_img_recon)
90. [Mind-to-Image: Projecting Visual Mental Imagination of the Brain from fMRI](https://arxiv.org/pdf/2404.05468)
91. [Distinct Top-down and Bottom-up Brain Connectivity During Visual Perception and Imagery](https://www.nature.com/articles/s41598-017-05888-8)
92. [Reduced Imagery Vividness in Aphantasia is Associated ...](https://www.biorxiv.org/content/10.1101/2025.05.25.655911v1.full.pdf)
93. [Colloquium Speaker- Mick Bonner](https://pbs.jhu.edu/event/colloquium-speaker-mick-bonner/)
94. [(PDF) Against Unconscious Mental Imagery in Aphantasia](https://www.researchgate.net/publication/389448541_Against_Unconscious_Mental_Imagery_in_Aphantasia)
95. [Advancing credibility and transparency in brain-to-image reconstruction research: Reanalysis of Koide-Majima, Nishimoto, and Majima (Neural Networks, 2024)](https://arxiv.org/html/2511.07960)
96. [Mental Image Reconstruction - FAQ (English)](https://sites.google.com/view/mentalimagereconstruction/faq-english)
