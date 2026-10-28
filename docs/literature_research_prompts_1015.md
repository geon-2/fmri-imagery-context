# 문헌 리서치 프롬프트 (Perplexity / ChatGPT Deep Research / Claude Research 교차 검증용, 2026-10-15)

사용법
- 프롬프트 A(ROI별 정보·지각-imagery 겹침)와 프롬프트 B(ROI별 부분공간 정렬·RSA·복셀 압축)를 **각각 세 도구에 같은 문구로** 넣는다. 도구마다 문구를 바꾸지 않는다(교차 검증이 목적).
- 결과를 받으면 세 도구가 **공통으로 낸 논문**과 **한 도구만 낸 논문**을 나눠 표로 정리한다. 한 도구만 낸 논문은 DOI·저널 페이지로 존재를 직접 확인한다(환각 논문 방지).
- 숫자(정렬 비율, 정확도, 표본 수)는 반드시 원문에서 다시 확인한다. 도구가 쓴 수치를 그대로 슬라이드·보고서에 옮기지 않는다.
- 이 저장소의 `docs/roi_literature_review_draft.md`가 이미 다룬 논문은 프롬프트 안에 「이미 아는 것」으로 넣었다(중복 방지 + 기존 해석 검증용).

---

## 공통 배경 (두 프롬프트에 같은 블록으로 들어간다)

```
RESEARCH CONTEXT
I am running a capstone study on fMRI decoding of mental imagery. Data: Natural Scenes Dataset (NSD, 7T, perception of COCO images) and NSD-Imagery (Kneeland et al.). Idea: learn a low-dimensional "context" subspace (scene/place type, environment, activity, implied event, scale) from perception fMRI, using text embeddings (e.g., CLIP text) of context labels as targets, then test whether it transfers to mental imagery. Hypothesis under test: imagery reuses a subspace of perception activity, but this probably differs across ROIs (early visual cortex vs. higher ventral/lateral/parietal cortex).

QUALITY RULES (apply strictly)
1. Prefer peer-reviewed papers (journal or top conference). Preprints (bioRxiv/arXiv) are allowed but must be labeled "preprint", with the posting date and whether a peer-reviewed version exists.
2. For every paper give: authors, year, venue, DOI or stable URL, and whether you actually read the full text, only the abstract, or only a secondary summary.
3. Never invent a paper, number, or ROI result. If you cannot verify something, write "not verified". If a search turns up nothing for a sub-question, say so explicitly instead of padding with loosely related work.
4. Distinguish what the authors showed (their result) from what they speculated (their interpretation), and from your own inference.
5. Report sample size (number of subjects, number of stimuli/targets), field strength, and the ROI definition method (functional localizer, atlas, or retinotopy) for each empirical claim.
6. Separate evidence strength: (a) direct fMRI evidence in humans, (b) model-based or computational evidence, (c) reviews or meta-analyses.
7. Answer in Korean, but keep paper titles, ROI names, and method names in English.

ALREADY KNOWN TO ME (please verify, correct, or extend; do not just repeat)
- Saha Roy et al. 2025 bioRxiv, "A transformation from vision to imagery in the human brain" (also a CCN 2024 poster, "Do vision and imagery share common principal signal components?"). My reading: imagery subspace dimensionality is about half of vision in early visual cortex with a rotated subspace, and close to vision (near identity) in higher visual and parietal areas; alignment ratio roughly 25-30% in V1 and about 50% in V4. I could NOT confirm a "CCN 2023" version with the title "Mental Imagery: Weak Vision or Compressed Vision?" - please check whether it exists.
- Lee, Kravitz & Baker 2011 NeuroImage (imagery vs. perception decoding across the visual hierarchy; cross-decoding works even in V1).
- Dijkstra, Bosch & van Gerven 2019 TiCS (imagery-perception overlap review). Spagna et al. 2021 meta-analysis (left fusiform imagery node).
- Scrivener, Teed & Silson 2025 (imagining familiar people/places; scene-selective ROIs).
- Wang et al. 2022 (CLIP and higher visual cortex), Doerig et al. 2025 Nature Machine Intelligence (captions and visual cortex), Elmoznino & Bonner 2024, Shen et al. CCN 2024 (dimensionality of shared representations across cortex).
- Kneeland et al. (NSD-Imagery, CVPR 2025) and the 2025-2026 "MIRAGE" imagery reconstruction paper.
```

---

## 프롬프트 A — ROI별 정보 내용과 지각-imagery 겹침 (최대한 포괄적인 문헌 수집)

```
[Paste the RESEARCH CONTEXT block above]

TASK A: Build an exhaustive, reliability-graded literature map of ROI-specific findings on (i) what each visual and associated region represents during perception, and (ii) how much of that representation is shared with, or transformed in, mental imagery / visual working memory / memory recall.

Scope - cover ALL of the following ROI groups, each separately:
 - early visual cortex (V1, V2, V3, V4, hV4), including retinotopic and population receptive field (pRF) imagery work (e.g., Breedlove et al. 2020 and related)
 - object-selective (LO, pFs), face (OFA, FFA), body (EBA, FBA), word (VWFA), scene (PPA, OPA, RSC/MPA), and "food" regions
 - ventral, lateral, and parietal streams (as defined in NSD streams), IPS/SPL, frontal (e.g., prefrontal, FEF), and medial temporal/hippocampal regions, retrosplenial and posterior parietal "memory" regions (place memory areas, Steel/Baldassano/Silson lines of work)
 - the left fusiform "imagery node" and any evidence of imagery-specific regions

For each ROI group, find studies that report: (a) decoding or encoding during imagery, (b) perception-to-imagery cross-decoding or generalization, (c) representational similarity (RSA) between perception and imagery, (d) effects of vividness (VVIQ, aphantasia, hyperphantasia) on that ROI, (e) category or semantic content (scene type, place, context) vs. low-level content.

Also cover:
 - NSD and NSD-Imagery-based studies that report ROI-wise results (Kneeland et al.; any follow-ups; reconstruction papers that break results down by ROI or stream; "MindEye", "MindEye2", "MIRAGE", brain-diffuser-style methods applied to imagery)
 - aphantasia studies at 3T and 7T with ROI-specific findings (early visual cortex vs. higher areas, frontal-visual connectivity)
 - studies of "coarse" vs. "fine" information during imagery (category-level vs. exemplar-level), and of low-resolution or gist-like imagery content
 - studies showing that imagery quality depends on task (imagery of seen vs. remembered vs. imagined stimuli), and on retinotopic location

Deliverables
1. A table: ROI group | what it represents in perception (with source) | imagery status (shared / reduced / transformed / absent) | key numbers (accuracy, alignment, correlation, n) | source with full-text-read flag | evidence type (human fMRI / model / review).
2. A list of at least 30 relevant papers if that many exist (do not pad), grouped by ROI group, each with a one-line "what this paper adds" and a one-line "main limitation".
3. A section "Contradictions and open disagreements": places where studies disagree about whether early visual cortex carries imagery content, and the methodological reasons (ROI definition, decoding vs. univariate, feature selection, stimulus set, vividness).
4. A section "What the literature does NOT cover": questions on scene context (place type, environment, activity, implied event) in imagery that I should not assume are already answered.
5. Citation quality notes: preprints that were later published, retractions or corrections, and papers whose headline claim is disputed.
```

---

## 프롬프트 B — ROI별 부분공간 정렬 분석, RSA, 복셀 압축 방법

```
[Paste the RESEARCH CONTEXT block above]

TASK B: Find all reliable work on how to test whether imagery activity lies in (or is a transformation of) a perception-derived subspace, ROI by ROI, and on the analysis methods I could use. Focus on methods and their evidence, not only on conclusions.

Part 1 - Subspace alignment between perception and imagery
 - Saha Roy et al. 2025 and related work: the exact definition of the "alignment ratio" (variance of imagery explained by the perception subspace), how dimensionality was estimated, how voxels were selected, how null/random-subspace baselines were built, and ROI-wise numbers. Also report any critiques, replications, or follow-up work.
 - Any other studies comparing perception and imagery or memory-recall activity in a shared low-dimensional space: principal-angle analyses, CCA/PLS/hyperalignment between conditions, "manifold alignment", "neural subspace" or "communication subspace" methods, cross-condition generalization (CCGP), geometry/abstraction analyses (Bernardi-style), and "rotation" vs. "scaling" of subspaces between conditions (e.g., motor planning vs. execution subspace literature, working-memory "rotation" between encoding and maintenance, Murray/Libby/Buschman-style work), if applied to fMRI or relevant to a subspace argument.
 - Studies that train an encoder/decoder on perception and test it on imagery, memory, dreams or attention conditions, ROI by ROI (e.g., Horikawa & Kamitani 2017 hierarchical features during imagery; Horikawa et al. dream decoding; Naselaris et al. 2015 encoding-model decoding of remembered scenes), including how performance varies across ROIs.

Part 2 - RSA and representational geometry across ROIs
 - RSA between perception and imagery per ROI (RDM correlation, noise ceilings, how RDMs were built from few trials, cross-validated distances such as crossnobis), including studies with very few conditions (6 to 12 stimuli) and how they handle reliability.
 - RSA between language/text-embedding models (CLIP text, caption embeddings, BERT/MPNet) and ROI RDMs in NSD or similar data, and whether these models compress to a low-dimensional geometry that matches ROI geometry. Include best-practice papers for RSA with noisy or small data, and tests for comparing RDMs (bootstrap over conditions, pattern-component modeling, rank-based tests).
 - Methods that "compress the text target by RSA/brain alignment": e.g., choosing embedding dimensions by brain-predictive variance (reduced-rank regression, CCA, PLS, DDR), and any work showing that brain-aligned low-dimensional axes generalize better than PCA axes of the same dimensionality.

Part 3 - Compressing voxel patterns per ROI vs. globally
 - Evidence on whether voxel-space compression should be done separately per ROI or on all voxels jointly (ROI-specific PCA, shared response model, hyperalignment, searchlight vs. ROI, dimension estimates per ROI such as participation ratio, power-law/"alpha" spectra in visual cortex: Stringer et al. 2019 and fMRI analogues, NSD-based dimensionality studies, Elmoznino & Bonner 2024, Shen et al. CCN 2024).
 - How voxel selection (reliability thresholds, noise-ceiling SNR, split-half) affects dimension estimates and alignment ratios.
 - Known pitfalls: voxel count differences across ROIs, SNR differences, regression-to-mean in imagery, run-order SNR drop (first imagery run), using the same data for selection and testing, and leakage between selection and evaluation.

Part 4 - Multi-target and multi-decoder reconstruction
 - How MIRAGE, MindEye/MindEye2, Brain-Diffuser, Takagi & Nishimoto and related papers handle the target (low-level latent, CLIP image embedding including the CLS token, CLIP text/caption embedding), whether they decode them with separate decoders and combine them in a diffusion model, whether they split voxels by ROI or use a whole visual-cortex mask (e.g., NSD "nsdgeneral"), and any ablation showing the contribution of each decoder, especially on imagery data. Please quote the methods section wording for the voxel/ROI choice when you can find it and state if it is only in an appendix.

Deliverables
1. A methods table: method | what it tests | required data | known null baseline | main pitfalls | key reference (flag if read in full text).
2. For Saha Roy et al. 2025: a precise methods summary I can reproduce (subspace estimation, alignment ratio formula, null models, selection of voxels, ROI list, n, number of conditions), and a list of what is unclear.
3. A ranked list of the 10 papers I should read in full first, with the reason for the ranking.
4. A section "ROI-wise vs. global subspace analyses: what is established, what is plausible but untested, what is unknown".
5. A section "What would count as evidence against the hypothesis that imagery reuses a perception subspace" - and which studies, if any, report such evidence.
6. Citation quality notes (preprints, disputed claims, corrections), as in Task A.
```

---

## 도구별 메모
- **Perplexity**: 한 번에 너무 길면 출처가 얕아진다. 프롬프트 B는 Part 1~4를 나눠 4번 돌려도 된다(공통 배경은 매번 포함). 출처 링크를 눌러 DOI를 확인한다.
- **ChatGPT Deep Research**: 위 문구를 그대로 사용. 중간에 되묻는 질문에는 「범위 전체, 사전 인쇄본 포함, 한국어 답변」이라고 답한다.
- **Claude Research**: 같은 문구. 결과의 「full text read」 표시가 실제와 맞는지 의심되는 항목은 직접 연다.

## 결과 정리 틀 (받은 뒤 쓸 표)
| 논문 | 세 도구 중 언급한 도구 | DOI 직접 확인 | 읽은 범위 | 우리 가설에 대한 방향(지지/반대/무관) | 비고 |
|---|---|---|---|---|---|
