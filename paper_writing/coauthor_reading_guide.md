# Co-Author Reading Guide — MalariAI Paper

**Your role:** Paper writing, literature review, diagrams, result analysis.  
**KB's role:** All coding (Phase 1 → 4).  
**Immediate task:** Read the three PDFs in this folder and fill in the extraction tables below.  
This feeds directly into §2 Related Work of `MalariAI_Paper_Draft.tex`.

---

## Why This Matters

The Related Work section is already drafted with our three open problems (P1, P2, P3).
Your job is to verify and sharpen each paper's entry in **Table 1** by going into the
actual PDF and finding the exact numbers and exact quotes. Reviewers check this.

Our three problems (defined in the paper):
- **P1 — Incomplete annotation:** end-to-end detectors treat unannotated cells as background
- **P2 — Dense overlap:** NMS discards valid overlapping cells
- **P3 — No integrated XAI:** no whole-slide detection system produces spatial heatmaps

---

## Paper 1 — Singh, Prabha, Abdulla (2025)
**File:** `1. Optimized CNN Framework-s41598-025-23961-5.pdf`  
**Journal:** Scientific Reports (Nature Publishing Group)

### What to extract:

| Field | What to find | Your notes |
|-------|-------------|------------|
| Full title | Exact title from page 1 | |
| All authors | Last name, First initial for all | |
| DOI | From header or footer | |
| Dataset used | Name, size, source | |
| Input format | Whole slide or pre-cropped patches? | |
| Architecture | Exact model names used | |
| Best metric | Accuracy / Dice / mAP — exact number + metric name | |
| Classes detected | Binary? Multi-class? Which classes? | |
| XAI used? | Yes/No — if yes, which method | |
| Handles imbalance? | Yes/No — if yes, how | |
| P1 addressed? | Do they handle unannotated cells? | |
| P2 addressed? | Do they handle overlapping cells? | |
| P3 addressed? | Do they produce spatial heatmaps? | |
| Future work quote | Copy the exact sentence(s) from their conclusion/future work | |

### Key things to look for:
- They use Otsu thresholding + CNN — confirm this is on pre-cropped patches, not whole slides
- Check if they report per-class AP or just overall accuracy
- Find the exact words they use for "future work" — we cite this as evidence of P3 gap

---

## Paper 2 — Loh, Yong, Yapeter, Subburaj, Chandramohanadas (2021)
**File:** `1-s2.0-S0895611120301403-main.pdf`  
**Journal:** Computerized Medical Imaging and Graphics (Elsevier)

### What to extract:

| Field | What to find | Your notes |
|-------|-------------|------------|
| Full title | Exact title from page 1 | |
| All authors | Last name, First initial for all | |
| DOI | From header or footer | |
| Dataset used | Public or private? How many images? Species of parasite? | |
| Architecture | Mask R-CNN variant? Backbone? | |
| Best metric | Exact numbers reported — note which metric (AP, F1, accuracy) | |
| Classes detected | List all classes they trained for | |
| Overlap handling | Do they filter out overlapping cells or separate them? | |
| "Quality check" | They mention "quality check measures" — find the exact passage | |
| XAI used? | Yes/No | |
| P1 addressed? | Find the word "unannotated" or similar | |
| P2 addressed? | Do they acknowledge NMS limitation? Quote it | |
| P3 addressed? | Any mention of heatmaps / explainability | |
| Future work quote | Exact sentence(s) — especially about stage classification | |

### Key things to look for:
- They use a custom dataset (not public BBBC041) — confirm this
- They describe filtering overlapping cells — find the exact method and quote
- Check if they report mAP at COCO thresholds [0.5:0.95] or just IoU=0.5
- Their speed claim ("15× faster") — what baseline are they comparing to?

---

## Paper 3 — Delgado-Ortet, Molina, Alférez, Rodellar, Merino (2020)
**File:** `2-entropy-22-00657.pdf`  
**Journal:** Entropy (MDPI)

### What to extract:

| Field | What to find | Your notes |
|-------|-------------|------------|
| Full title | Exact title | |
| All authors | All of them | |
| DOI | From header | |
| Dataset used | Name, number of images, number of patients | |
| Architecture | Three-stage pipeline — what are the three stages exactly? | |
| Segmentation metric | Exact Dice / IoU for segmentation stage | |
| Classification metric | Accuracy / specificity — exact numbers | |
| Train/test split | How many patients in train vs test? | |
| Overfitting evidence | Find the train accuracy vs test accuracy numbers | |
| Classes | Binary (infected/healthy) or multi-class? | |
| XAI used? | Yes/No | |
| P1 addressed? | Any mention of unannotated cells | |
| P2 addressed? | Any mention of dense overlap | |
| P3 addressed? | Any mention of heatmaps | |
| Future work quote | Exact sentences — they explicitly list future directions | |

### Key things to look for:
- They report 95% val accuracy but ~75% test — find BOTH numbers in the paper
- Their future work section explicitly asks for more classes + larger dataset
  → This is our strongest evidence for the gap. Find the exact quote.
- They use SegNet → crop → SNN → classifier — confirm the three stages
- How many patients total? (We believe it's very small — 5 train, 5 test)

---

## Paper 4 — Pandiaraj, Kshirsagar, Thiagarajan, Tak, Sivaneasan (2025)
**File:** `4-s10278-024-01311-7_260508_222013.pdf`  
**Journal:** Journal of Imaging Informatics in Medicine, 38:2381–2411, 2025  
**DOI:** https://doi.org/10.1007/s10278-024-01311-7

### Background (already known — verify against the PDF):
KB has read this paper in full. The key findings are:
- **Datasets:** Kaggle Malaria Cell Images Dataset (27,558 pre-cropped patches) + a secondary Kaggle detection dataset. **NOT BBBC041. NOT whole-slide images.**
- **Task:** Binary classification only — parasitised vs. uninfected. No multi-class stage detection (ring/trophozoite/schizont/gametocyte).
- **Best result:** 94.88% accuracy (Dataset 1), 92.86% (Dataset 2). Accuracy on binary patches — not mAP on whole slides.
- **Architecture:** Trans-MobileUNet++ (T-MUNet++) segmentation + AA-CRMV2 detection backbone + URP-FFO metaheuristic optimizer.
- **Gaps confirmed:** No XAI, no class imbalance handling, no mAP metric, no whole-slide pipeline.

### What to extract (verify and supplement KB's notes):

| Field | What to find | Your notes |
|-------|-------------|------------|
| Full title | Exact title from page 1 | |
| All 5 authors | Confirm: Pandiaraj, Kshirsagar, Thiagarajan, Tak, Sivaneasan | |
| Dataset 1 details | Confirm: Kaggle, 27,558 images, binary labels, pre-cropped | |
| Dataset 2 details | Size, source, labels | |
| T-MUNet++ | What does the ++ mean here? How does transformer fit in? | |
| AA-CRMV2 | What does "atrous" and "recurrent" add? Describe in 2 sentences | |
| URP-FFO | What is the Fennec Fox Optimisation? Fixed params Maxiter=25, Npop=10? | |
| Accuracy table | Find the exact accuracy numbers per dataset | |
| mAP reported? | Confirm: No mAP at IoU=0.5 or 0.5:0.95 is reported | |
| XAI? | Confirm: No Grad-CAM or explainability mentioned | |
| Imbalance handling? | Confirm: No Focal Loss, no class weights, no sampling | |
| Limitations (stated) | Find their own stated limitations in the conclusion | |
| Future work quote | Exact sentences from conclusion — especially "real-time" limitation | |
| P1 addressed? | Does their pipeline handle unannotated cells? | |
| P2 addressed? | Do they address NMS / overlap in dense smears? | |
| P3 addressed? | Any spatial heatmaps or explainability? | |

### Key things to look for:
- Confirm the dataset scope — we are claiming they solve an *easier* problem (binary pre-cropped patches from Kaggle) vs. our harder problem (multi-class whole-slide detection on BBBC041). The comparison table in §2.6 depends on this.
- Find their exact stated limitation about "real-time" deployment — currently in §2.4 we paraphrase this; find the exact words for a potential quote.
- URP-FFO: check if the parameter table (Maxiter=25, Npop=10) appears — this supports our reproducibility concern.
- Do they compare against any mAP-based baselines at all? (We believe no — confirm.)

### What NOT to do:
- Do **not** compare their accuracy to our mAP — they are different metrics on different tasks.  
  In the paper, we note the metrics are incomparable, which actually makes our point: they are solving a simpler, incomparable problem.
- Do **not** update the §2.4 prose in the .tex (KB has already updated it). Just fill in the table above and flag any discrepancies.

---

## What to do when you're done

1. Fill in all three tables above
2. Open `MalariAI_Paper_Draft.tex`
3. Go to `\subsection{Summary of Limitations}` and verify Table 1 matches your findings
4. Fix any bibliography entries marked `\placeholder{Author(s)}` — the BibTeX keys are:
   - `singh2025` → Paper 1 authors (Singh, Prabha, Abdulla)
   - `loh2021` → Paper 2 authors (Loh, Yong, Yapeter, Subburaj, Chandramohanadas)
   - `delgado2020` → Paper 3 authors (Delgado-Ortet, Molina, Alférez, Rodellar, Merino)
   - `pandiaraj2024` → **Already fixed by KB** (Pandiaraj, Kshirsagar, Thiagarajan, Tak, Sivaneasan)
5. For each paper, note 1–2 sentences we should add/fix in the Related Work prose

---

## Formatting notes for .tex edits

- Author names in `\author{}` field: `Last1, First1 and Last2, First2`
- If a metric appears in a table: use `\textbf{97.96\%}` for their best result
- Do NOT compile the .tex to PDF — KB will do that at the end
- Add a `%% COAUTHOR NOTE:` comment above any line you're unsure about

---

## Timeline

| Week | Co-author deliverable |
|------|-----------------------|
| 1    | Papers 1–3 extraction tables filled (this doc) |
| 2    | Related Work §2 prose revised in .tex |
| 3    | Methodology §3 outline drafted (after KB shows Stage 1 working) |
| 4–5  | Diagrams: architecture figure for Pipeline B |
| 6    | Results §5 analysis after KB sends experiment numbers |
| 7    | Full paper review pass, abstract polish |
| 8    | Final compile + arXiv submission |
