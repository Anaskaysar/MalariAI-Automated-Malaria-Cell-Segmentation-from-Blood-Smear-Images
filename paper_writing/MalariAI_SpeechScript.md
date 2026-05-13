# MalariAI — Project Demonstration Speech Script

**Audience:** Prospective supervisors / co-authors / academic collaborators  
**Duration:** ~12–15 minutes (plus Q&A)  
**Format:** Slide-by-slide cues. *Italics* = speaking notes / timing reminders.

---

## Slide 1 — Title

*Open with calm confidence. No apology, no "so, today I'm going to…"*

"Thank you for taking the time to meet with me.

What I'm presenting today is MalariAI — a research project I've been developing over the last year that targets automated malaria diagnosis from blood smear microscopy.

The full title is: *A Label-Resilient Decoupled Framework for Universal Cell Segmentation and Explainable Stage Classification in Dense Malaria Blood Smears.* It's a dense title on purpose — I'll unpack every word of it over the next twelve minutes.

The target venue is arXiv with a concurrent MICCAI submission, and the live demo will run on Hugging Face Spaces — so this will be publicly accessible once Phase 4 is complete."

---

## Slide 2 — The Global Burden of Malaria

*Pause after each big number. Let them land.*

"To motivate why this problem matters: in 2022 alone, there were 249 million malaria cases globally, resulting in 608,000 deaths — 95% of them in Sub-Saharan Africa.

The clinical gold standard is still manual microscopy. A trained microscopist examines a blood smear under a 100× oil-immersion objective, counts infected red blood cells by hand, and identifies the parasite stage. That takes 20–30 minutes per slide and requires expert training that simply does not exist at scale in endemic regions.

Now, deep learning approaches do exist. But this number in the bottom right — 537:1 — is the ratio of red blood cells to the rarest parasite class in our dataset. That imbalance, combined with two other compounding failure modes, is why no existing AI system has successfully transitioned to clinical deployment.

This project addresses all three failure modes simultaneously — in a single, deployable, explainable pipeline."

---

## Slide 3 — Three Unsolved Problems

*Take each problem row slowly. The audience needs to understand each before the solution lands.*

"Let me be specific about what those three failure modes are, because they're the entire intellectual foundation of MalariAI.

**P1 — Incomplete Annotation.** In datasets like NIH BBBC041 — which we use — only parasite-infected cells are individually annotated. The vast majority of healthy red blood cells are present in the image but unlabelled. When you train an end-to-end detector like Faster R-CNN or YOLO on this data, it treats every unannotated cell as background. The model learns the wrong signal from day one. MalariAI's Stage 1 — a watershed segmentation algorithm — finds every cell in the smear regardless of whether it was ever labelled.

**P2 — Dense Overlap and NMS Failure.** Non-Maximum Suppression is the standard post-processing step in all detection pipelines. It assumes that two highly overlapping proposals are duplicates of the same object and deletes one. But in dense blood smears, 58% of images contain genuinely overlapping cell pairs — real cells that truly overlap. NMS deletes real detections. Our distance-transform guided watershed splits touching cells at the instance level, so NMS never enters the picture.

**P3 — Black-Box Output.** Every published whole-slide malaria detection system produces a class label: 'ring', 'trophozoite', 'schizont'. That's it. Clinicians cannot see *why* the model made that prediction, which is a fundamental barrier to adoption. MalariAI generates a Grad-CAM++ spatial heatmap for every single detected cell — showing exactly which pixels drove the classification."

---

## Slide 4 — Dataset: NIH BBBC041

*The bar chart does the work here. Point to it.*

"Our dataset is the NIH Broad Bioimage Benchmark Collection 041 — Giemsa-stained *Plasmodium falciparum* thin blood smears, imaged at 100× oil immersion, 1600×1200 pixels per image.

1,208 training images. 79,672 valid annotated bounding boxes. Six cell categories.

Look at that class distribution chart. The Red Blood Cell bar — 77,420 instances — dominates the entire scale. Every other class is almost invisible at this scale. Trophozoites: 1,473. Ring forms: 353. Schizont: 179. Gametocyte: 144. Leukocyte: 103.

That 537:1 ratio isn't just a statistical curiosity. It's the direct motivation for using Focal Loss with per-class inverse-frequency weighting in Stage 2.

One more key finding from our EDA: 58% of training images contain at least one overlapping cell pair with IoU greater than 0.3. The median image has 59 cells; the densest has 223. These are the empirical foundations for why P1 and P2 are real, measurable problems — not hypothetical ones."

---

## Slide 5 — MalariAI: Two-Stage Decoupled Framework

*This is the architecture slide. Walk left to right.*

"The core insight of MalariAI is decoupling segmentation from classification.

**Stage 1** operates on the full 1600×1200 blood smear. It runs Otsu thresholding to globally binarise the image, morphological opening to remove noise, a Euclidean distance transform to build a topology map of cell centres, and marker-based watershed segmentation to separate individual cells. The output is N individual 64×64 pixel crops — one per detected cell. Critically, no ground-truth labels are ever consulted. Stage 1 is purely vision-based.

**Stage 2** receives those crops and runs them through EfficientNet-B0 — 5.3 million parameters, chosen specifically because it runs comfortably on CPU at inference time. It's trained with Focal Loss at γ=2.0, with per-class α weights derived from our EDA count ratios. The output is a class label, a confidence score, and a Grad-CAM++ heatmap — for every cell.

The key architectural guarantee: because Stage 1 produces pre-isolated crops, Stage 2 never needs NMS. The two failure modes P1 and P2 are structurally eliminated before Stage 2 even runs."

---

## Slide 6 — Baseline A: Faster R-CNN — 80 Epochs

*The results slide. Be precise with numbers.*

"Before building MalariAI, I trained a competitive Faster R-CNN baseline — ResNet-50 FPN, COCO pretrained, 80 epochs on a Kaggle T4 GPU. This is Phase 2 of the project, and it's now complete.

The headline result: **mAP@0.5 of 58.99%**, best checkpoint at epoch 23.

Look at the per-class breakdown in the chart. Red Blood Cell: 91%. Leukocyte: 88%. These are common, well-annotated classes. The model handles them well.

Now look at the rare classes — highlighted in red. Schizont: 25%. Gametocyte: 26%. These are the parasite stages that matter most for clinical staging, and they're exactly where the model fails.

This isn't a training failure. More epochs, a lower learning rate, data augmentation — none of these fix a 24.57% AP on a class with 179 training examples against a 77,420-example majority class. This is a structural ceiling imposed by the annotation incompleteness problem. And that ceiling is exactly what MalariAI's decoupled design is built to break through."

---

## Slide 7 — The Evidence: P1 Proven Empirically

*This is the most scientifically rigorous slide. Slow down.*

"This loss curve is the most important diagnostic result in Phase 2, and I want you to look at it carefully.

The blue line is training loss — it falls smoothly from epoch 1 to epoch 80, reaching 0.08 by the end.

The red line is validation loss — it falls from epoch 1, reaches a floor around **epoch 8**, plateaus at 0.23, and then *rises* after the learning rate drop at epoch 50, settling around 0.28.

A reviewer might call this overfitting. It is not overfitting.

The train-val divergence begins at epoch 8 — before the model has had any opportunity to memorise training examples. This is the signature of annotation incompleteness. The model is learning to detect unannotated healthy RBCs, which the validation annotations score as false positives. No learning rate schedule, no regularisation, and no amount of additional training can fix this — because the labels themselves are the problem.

The best checkpoint is epoch 23 — before the LR drop makes the model over-commit to unannotated cells. 57 more epochs produce no improvement and actively degrade performance.

This is not an optimisation problem. This is empirical confirmation of P1. And it is the entire motivation for why Stage 1 of MalariAI must be annotation-agnostic."

---

## Slide 8 — Research Contributions

*Deliver these with confidence — this is the 'what we claim' moment.*

"MalariAI makes three formal research contributions:

**C1 — Label-Resilient Segmentation.** The annotation-agnostic watershed Stage 1 detects every cell in a blood smear without consulting ground-truth labels. P1 is not mitigated — it is structurally bypassed.

**C2 — Density-Invariant Overlap Handling.** Distance-transform watershed operates on image topology, not on predicted bounding boxes, so it never applies NMS. Overlapping cells are separated at the instance level. 58% of BBBC041 images with genuinely overlapping cells are handled correctly — not excluded.

**C3 — Integrated End-to-End Explainability.** MalariAI is, to our knowledge, the first whole-slide malaria detection system to generate per-cell spatial attention heatmaps at inference time. The Grad-CAM++ output is available in two views: a 64×64 crop detail, and a full-image overlay showing all detected cells simultaneously.

Each contribution directly addresses one of the three failure modes identified in the literature. The evidence for C1 is the loss curve you just saw. The evidence for C2 is the 58% overlap prevalence from our EDA. The evidence for C3 is the absence of this feature in every prior published system — which we document in our related work section."

---

## Slide 9 — Project Status & Roadmap

*Be direct about what's done and what's next. Don't hedge.*

"Let me show you exactly where the project stands.

**Phase 1** — complete. Full EDA of NIH BBBC041: class distribution analysis, overlap quantification, density profiling. The findings directly shaped every design decision.

**Phase 2** — complete. Faster R-CNN baseline, 80 epochs, mAP@0.5 = 58.99%. Structural ceiling confirmed empirically. Results, loss curves, and per-class AP tables are all in the repository.

**Phase 3** — this is where we are going next. Stage 1 watershed code, Stage 2 EfficientNet-B0 training with Focal Loss, Grad-CAM++ implementation, and full inference pipeline. Target: RunPod RTX 3090 for training.

**Phase 4** — Flask web application deployed to Hugging Face Spaces. Three-card UI: smear view with watershed outlines, cell crop gallery with filtering, and a clinical report with Grad-CAM++ overlays. EfficientNet-B0 runs in approximately 2 seconds on CPU — no GPU required at inference.

**Phase 5** — evaluation, ablation study, and arXiv / MICCAI submission. This is where co-author collaboration on the Results and Discussion section is most valuable."

---

## Slide 10 — Paper: Current Status

*This slide reassures collaborators that the academic groundwork is solid.*

"The paper draft is already underway in LaTeX. Let me summarise what's written and what remains.

Written sections: the Abstract, Related Work — five subsections covering detection baselines, segmentation approaches, class imbalance methods, XAI in medical imaging, and prior malaria AI work — Dataset and Setup, the full Baseline A methodology section, Evaluation Metrics, and the Experiments section through the Baseline A results.

Waiting on Phase 3: the Pipeline B methodology sections — Stage 1 watershed, Stage 2 EfficientNet, and the Grad-CAM++ subsection. The Pipeline B row in the results table will be filled once training completes.

The Introduction and Conclusion are structural placeholders — these are most effectively written after the experimental results are in hand, to ensure the framing matches the actual outcomes.

The co-author reading brief has been distributed, with specific extraction tasks assigned for four reference papers covering the key prior work we cite.

The target venue is arXiv for immediate visibility, with a concurrent MICCAI 2026 submission."

---

## Slide 11 — Thank You / Collaboration

*End with what you're actually asking for. Be specific.*

"MalariAI is addressing three problems that have persisted across the malaria AI literature simultaneously — and Phase 2 has given us empirical evidence that those problems are real and measurable, not just theoretical.

What I'm looking for from this conversation is one or more of these things, depending on your interest and expertise:

Supervisory support for the arXiv and MICCAI submission process — particularly around framing the contributions for a top-tier venue.

Co-authorship on the Results and Discussion section — specifically the ablation study comparing Pipeline B to Baseline A on dense-region recall.

Clinical validation of the Grad-CAM++ outputs — a domain expert's assessment of whether the attention heatmaps correspond to known parasite morphology features.

And feedback on the experimental design and methodology — particularly the Stage 1 watershed evaluation protocol and how to rigorously measure the improvement over NMS-based detection in dense regions.

My contact information is on screen. The full codebase is on GitHub — the link is at the bottom of this slide. I'm happy to walk through any section of the code or paper draft in detail.

Thank you."

---

## Q&A Preparation

*Anticipate these questions — have answers ready.*

**"Why not use Mask R-CNN instead of watershed for Stage 1?"**  
Mask R-CNN requires instance segmentation masks, which BBBC041 does not provide. Watershed requires zero labels — that's the entire point of C1. Adding a labelling requirement to Stage 1 would reintroduce P1.

**"How do you evaluate Stage 1 if it's annotation-agnostic?"**  
We define a cell recovery metric: of the N bounding boxes in the ground truth, what percentage have their centre point inside a Stage 1 watershed region? Additionally, we report dense-region recall — specifically for images with >100 cells per image (the densest 11% of BBBC041).

**"Why EfficientNet-B0 and not a larger model?"**  
This is a deliberate deployment constraint. EfficientNet-B0 runs at ~2 seconds on CPU, making it viable on Hugging Face Spaces free tier with no GPU. The demo needs to be reproducible without cloud credits. EfficientNet-B0 also outperforms ResNet-18 on small 64×64 crops in the published EfficientNet benchmarks.

**"Why mAP@0.5 and not mAP@[0.5:0.95]?"**  
Baseline A mAP@[0.5:0.95] is substantially lower (~28%) because cell boundaries in blood smears are not crisp enough to reward tight localisation at IoU=0.95. The clinical question is detection and classification, not pixel-precise boundary prediction. mAP@0.5 is the standard metric used in all four reference papers we compare against.

**"What is the expected mAP improvement from Pipeline B?"**  
We do not claim a specific number prior to training — that would be data dredging. The claim is structural: Pipeline B eliminates two of the three failure modes by design. The ablation study will show whether the empirical improvement on mAP@0.5 and dense-region recall is statistically significant.

**"Why Grad-CAM++ over standard Grad-CAM?"**  
Grad-CAM++ provides more accurate attribution when multiple activations in the same spatial location contribute to the class score — which is common in small 64×64 crop classifiers where the object occupies most of the receptive field. The mathematical difference is the use of second-order gradients for weighting, which reduces spatial bias toward a single dominant activation.

---

*End of script. Total reading time: ~12 minutes at a measured pace.*
