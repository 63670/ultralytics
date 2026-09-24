# DACP-Net JIOT manuscript: materials and validation checklist

This draft is intentionally built around evidence already available in the repository and the production workflow you described. It does not invent hardware specifications, deployment throughput, control timing, or multi-seed statistics. Supplying the following material will turn the current first draft into a submission-ready paper.

## Highest-priority material

| Needed item | What to provide | Where it will appear | Why it matters |
|---|---|---|---|
| Production-line overview | One wide, clean photo showing the circular knitting machine and inspection position. Blur people, labels, and confidential work orders if necessary. | Fig. 1, beside or replacing part of the system schematic | Establishes that this is a real on-machine project, not an offline dataset only. |
| Inspection installation | 2-3 close photos: camera/light and fabric field of view; proximity switch and its target; edge box/controller and the interface toward the machine. | Fig. 2, a multi-panel deployment figure | Shows the physical signal-to-inspection link and makes the closed loop credible. |
| Operator interaction | Screenshot or photo of the defect display / confirmation screen, redacted if necessary. | Fig. 3 | Makes the human-in-the-loop contribution concrete. |
| Exact hardware | Edge computer model, CPU/GPU/NPU, RAM, OS, camera model/interface, lens, illumination, image resolution, acquisition rate, and inference runtime (PyTorch/ONNX/TensorRT/etc.). | System implementation table | Required to make a reproducible edge-deployment statement. |
| Control chain | What receives the stop request (PLC, relay, I/O board, machine controller), whether it is a request or safety interlock, and the signal interface. | System text and safety statement | Prevents reviewers from challenging an unsafe or vague stop-control claim. |

## Measurements needed before claiming production performance

Measure at least 30-100 events per condition and report mean, standard deviation, and a high percentile (P95 or P99) where applicable.

| Metric | Start and end timestamps | Suggested table/figure |
|---|---|---|
| Edge inference latency | image available -> final detection | latency table, batch=1 |
| End-to-end reaction time | proximity trigger or frame exposure -> stop command sent | timing diagram/table |
| Physical stopping delay | stop command sent -> machine motion stops | timing diagram/table |
| Effective prevention distance/length | defect position at detection -> final stopped position | deployment study |
| False-stop rate | normal fabric time or normal frames -> stop requests | production reliability table |
| Missed-defect rate | independently reviewed defects -> absent alerts | production reliability table |
| Operator confirmation | candidate alert -> confirmed/rejected/uncertain | feedback-loop table |
| Uptime | inspection available time / scheduled production time | deployment table |

Do not state that the machine is ``safety certified,'' that the stop action is fail-safe, or that feedback automatically retrains the model unless the system has been formally validated for that claim. The draft currently says the cloud feedback is used for later \emph{offline} curation and model updating.

## Dataset provenance to document

The current data audit found 282 training images (329 instances), 40 validation images (48 instances), and 81 test images (91 instances). Classes are now named **horizontal defect**, **vertical defect**, and **hole**. Please provide:

1. Collection period; number of machines, material/product types, lots/rolls, speeds, and illumination conditions.
2. Annotation tool, annotator count, label definitions, review process, and treatment of ambiguous defects.
3. The split unit: image-random, roll, batch, machine, or production-time interval. A grouped split by roll/time is stronger because adjacent production frames can be very similar.
4. Whether any test images were seen during model/hyperparameter selection.
5. Permission/ownership statement for production images and whether any personal or confidential information was removed.

## Experiments that should be rerun for submission

The current comparison table correctly labels all entries as retained test checkpoints. It must not be presented as a matched multi-seed mean.

1. Train **YOLOv8n, +WWTE, +LTFR, and DACP-Net** with exactly the same seeds (recommended: 0, 1, 2, 3, 4 or 0, 3, 11, 17, 42), epochs, image size, data split, and augmentation recipe. Report mean +/- standard deviation for AP and AP50.
2. Keep the direct YOLOv8n comparison central because DACP-Net is derived from YOLOv8n. Keep YOLOv5n/11n/12n/26n, DETR, Deformable DETR, Faster R-CNN, and RT-DETR-R18 as context, but state their framework and training protocol clearly in supplementary material.
3. Record exact code commit, package versions, GPU, CUDA, and pretrained checkpoint provenance. In the current audit, the retained DACP-Net test checkpoint is seed 0 and initializes from `yolov8n.pt`; the retained YOLOv8n record visible in the run directory is seed 11. This is why matched reruns are important.
4. Measure end-to-end latency on the actual edge device at batch 1. GPU-server milliseconds and FLOPs cannot prove production response time.
5. Add one grouped or temporally separated test set, if production data permits. The current test set has only 12 vertical-defect instances.

## Figure package to prepare

1. `F1_System.jpg`: wide production-line photo, at least 300 dpi at final printed width.
2. `F2_Installation.jpg`: 3-panel photo: illumination/camera, proximity switch, and edge/controller wiring. Use callouts but avoid proprietary electrical details.
3. `F3_UI.png`: operator confirmation interface with defect image, class, confidence, timestamp, and confirmed/rejected action. Redact identifiers.
4. `F4_DefectGallery.png`: 3 rows (horizontal, vertical, hole) with 3-5 original images per class, labels displayed consistently.
5. `F5_Qualitative.png`: a final common-renderer comparison. Use a prespecified confidence threshold and include enough true failures, not only successes.
6. Optional: a compact timing sequence diagram built from real timestamps. This is more persuasive than a heatmap when claiming a closed operational loop.

## Terminology fixed in the current draft

- Machine: **circular knitting machine** (not a woven-fabric loom).
- Classes: **horizontal defect**, **vertical defect**, and **hole**.
- WWTE: the implementation name is retained, but in the manuscript it is explicitly described as an **orthogonal texture encoder** for knitted fabric, not as a claim about woven warp/weft threads.
- DACP-Net: **Directional-Aware Content-Adaptive Pyramid Network**.

## Current source map

- `manuscript.tex`: editable English JIOT draft.
- `references.bib`: bibliography, including the four provided papers and the relevant circular-knitting references.
- `figures/system_overview.tex`, `figures/network.tex`, and `figures/modules.tex`: editable TikZ diagrams generated from the confirmed workflow and current source code.
- `figures/qualitative_existing.png`: existing standardized qualitative grid, retained as a provisional figure.
- `make_figures.py`: reproducible AP-vs-efficiency plot based on `../tools/efficiency_results.csv`.
- `build.ps1`: rebuilds figures, BibTeX, and `manuscript.pdf` on Windows PowerShell.
