# SP:35 Bridge Inspection AI — Implementation Plan

Document SP35-AI-PLAN-001 · Rev A · 7 Oct 2026 · status: for approval at gate G0
Stakeholder version (same content, with diagrams): https://claude.ai/artifact/Jnuzxbqd6zDShjnGjfhUR8

> **For the agent.** Work one work package (WP) at a time, in order. Before coding a WP, write an implementation-plan artifact that names the files you will create, the tests you will write and how you will meet its "Done when" line. Tick `[x]` here when a task is finished and link the evidence. Stop at every gate.

---

## 1. Product

A Windows desktop tool. An engineer double-clicks **Bridge Inspector**, picks a bridge's photo folder and gets a draft inspection report scored per IRC:SP:35-2024 Chapter 7:
- Appendix IV proformas (component-wise distress, location, extent, severity, condition state, probable reasons, maintenance actions, urgency);
- Appendix V cover sheet with the inspector's certificate and signature block;
- BHI per span and for the bridge, star rating, critical list (state V) and urgent list (state IV);
- MPI inputs and score for network prioritisation.

**Principle.** The AI proposes, the engineer certifies. SP:35 bases condition states on actual site measurement (§6.5, Appendix V), so every state is confirmed by a person and traced to its evidence (photo, measurement or written judgement).

**Models.**
- SAM 3.1 (Meta; released 27 Mar 2026; checkpoints `facebook/sam3.1`, gated; code `github.com/facebookresearch/sam3`) for promptable concept segmentation of components and distresses.
- NVIDIA Cosmos 3 Reasoner (released 31 May 2026; Nano = 8B reasoner tower of a 16B model; NIM name `nvidia/cosmos3-nano-reasoner`) from Phase 3, behind the `Reasoner` contract defined in Phase 2. Fallback: Cosmos Reason 2 (2B/8B).

---

## 2. Architecture

```
Photo folder ──> Ingest & organise ──> SAM 3.1 inferencer ──> Measure ──> SP:35 engine ──proposed CS──> Engineer review ──> Signed report
 (phone, drone)   dedupe, EXIF, blur     components, distresses   extent,     Tables 7.2/7.3   <──confirmed──   confirm, edit,     App. IV/V, BHI,
                  span tagging           GPU (2050 / 3060)        width,      Eq. 1-5                            sign               star, MPI
                                              │      ▲            count            ▲
                                       crops  ▼      │ verified     ▲              │ weights, limits
                                       Cosmos 3 Reasoner (Phase 3)  │ site measurements (gauge, calliper, level)
                                                                    Knowledge base kb/sp35_2024 (YAML, signed)
Engineer corrections ──> training data (Phase 2) ──> fine-tuned SAM 3.1 (Phase 4)
```

### 2.1 Repository layout (target)
```
kb/sp35_2024/        components, distress_bridge, distress_approach, condition_states, mpi, golden_examples (YAML)
kb/prompts/          sam3_prompts.yaml
sp35/                kb.py (load + validate), condition.py (value -> state), scores.py (Eq. 1-3, spans, stars, lists), mpi.py (Eq. 4-5)
vision/              worker.py (FastAPI), sam.py (model wrapper), postprocess.py, tiling.py, probe.py, jobs.py
ingest/              scan.py, dedupe.py (sha256 + perceptual hash), exif.py (EXIF + DJI XMP), blur.py
app/                 server.py, views/, static/ (canvas viewer), reports/ (templates for App. IV/V, PDF, XLSX, JSON)
data/                (gitignored) projects/<bridge_id>/...
tests/               unit/, property/, golden/, e2e/
tools/               sp35_reference_check.py, launcher scripts
docs/                plan/, reports/, gates/
```

### 2.2 Data model (SQLite)
- `bridge(id, name, type, material, n_spans, location, notes)` — inventory data, kept apart from inspection data (§6.5).
- `span(id, bridge_id, idx, kind[end|intermediate], left_support, right_support)`; `support(id, bridge_id, kind[abutment|pier], label)`.
- `inspection(id, bridge_id, type[routine|principal|special|underwater], date, start, end, inspector, kb_version, status)`.
- `image(id, inspection_id, path, sha256, phash, width, height, exif_json, span_id?, support_id?, duplicate_of?, excluded_reason?)`.
- `detection(id, image_id, kind[component|distress|suppression], class_id, prompt, score, mask_rle, bbox, area_px, tile, parent_detection_id?, review[pending|accepted|rejected|edited])`.
- `observation(id, inspection_id, span_id?, support_id?, sp35_sub, distress_row, value?, unit?, source[site_measured|visual_estimate|engineer_judgement|scale_card], location_text, evidence_detection_ids, proposed_state, final_state, reason, reviewer, ts)`.
- `assessment(id, inspection_id, kb_version, span_results_json, bridge_bhi, stars, critical_json, urgent_json, created_at)`; `mpi_input(bridge_id, subfactor, value, pr)`; `audit_log(ts, user, entity, entity_id, change_json)`.

### 2.3 Service contracts
Vision worker (localhost only):
- `POST /jobs` `{inspection_id, image_ids, passes: ["components","distresses"], tile_classes?: [...], device?: "cuda"|"cpu"}` → `{job_id}`
- `GET /jobs/{id}` → `{status, done, total, errors, vram_peak_gb, sec_per_image}`
- `GET /images/{id}/detections` → list of detections (COCO-RLE)
- `POST /refine` `{image_id, detection_id?, points: [[x,y,1|0]], boxes: [[x0,y0,x1,y1]]}` → refined mask (interactive refinement per the sam3 repo examples)

Reasoner contract (defined in Phase 2, used from Phase 3), output JSON schema:
```json
{
  "detection_id": "str",
  "is_valid": true,
  "sp35_component": "C2.8",
  "sp35_distress_row": "T7.2-12",
  "suggested_state": "I|II|III|IV|V|null",
  "severity_note": "<= 200 chars",
  "confidence": 0.0,
  "rationale": "<= 400 chars",
  "model": "rule-stub|cosmos3-nano-reasoner|cosmos-reason2-8b",
  "latency_ms": 0
}
```
Cosmos 3 Reasoner (NIM or vLLM) is called through the OpenAI-compatible `/v1/chat/completions` endpoint with an image plus text. For reasoning output, ask for `<think>…</think><answer>…</answer>`; use temperature 0.3 and max_tokens up to 4096. Keep image tokens ≤ 16k (each 32×32 patch is one token).

---

## 3. Hardware roles and environments

| Machine | GPU | Role | Runs | Never runs |
|---|---|---|---|---|
| Main laptop | RTX 2050, 4 GB, Ampere sm_86; i5-11400H; 16 GB RAM; Windows 11 | Development and the one-click desk app | SAM 3.1 image model in bf16 for one bridge (if the G0 probe passes), SP:35 engine, review app, reports | Training, Cosmos 3, big tiled batches |
| Company | RTX 3060, 12 GB (verify not the 8 GB variant), sm_86 | Batch and labelling station | Full-res + tiled SAM 3.1 batches, pre-labels, prompt sweeps, evaluation, Cosmos Reason 2 (2B) | — |
| Company | RTX 5070 Ti, 16 GB GDDR7, Blackwell sm_120 | Reasoning and training station | Cosmos 3 Reasoner trials/serving (offload or quantised), SAM 3.1 fine-tuning, FP8 inference | — |

Environments: Python 3.12, PyTorch ≥ 2.7 built for CUDA 12.8 (covers sm_86 and sm_120); WSL2 Ubuntu 24.04 for the laptop's GPU worker; Docker image `sp35-worker` (CUDA 12.8) on company GPUs. vLLM ≥ 0.19.1 (CUDA 12.8) for Cosmos 3 reasoner-only serving:
`vllm serve nvidia/Cosmos3-Nano --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' --tensor-parallel-size 1 --mm-encoder-tp-mode data` (add `--enable-layerwise-offload` when VRAM is short).

GPU days needed: Phase 2 → RTX 3060 × 3 days, RTX 5070 Ti × 2 days (December–January). Use the `gpu-batch-day` skill.

---

## 4. Roadmap and gates (start Mon 12 Oct 2026; holidays not subtracted, expect up to 2 weeks slip)

| Phase | Weeks | Dates | Main machine | Gate |
|---|---|---|---|---|
| 0 Setup & feasibility | W1 | 12–16 Oct 2026 | RTX 2050 | G0 · 16 Oct |
| 1 One-click inferencer + Chapter 7 engine | W2–W7 | 19 Oct – 27 Nov 2026 | RTX 2050 | G1 · 27 Nov |
| 2 Ground truth, measurement, batch | W8–W13 | 30 Nov 2026 – 8 Jan 2027 | 3060 + 5070 Ti days | G2 · 8 Jan |
| 3 Cosmos 3 Reasoner | W14–W18 | 11 Jan – 12 Feb 2027 | 5070 Ti or hosted NIM | G3 · 12 Feb |
| 4 Fine-tuning | W19–W24 | 15 Feb – 26 Mar 2027 | 5070 Ti or rented GPU | G4 · 26 Mar |
| 5 Network priority (MPI) | W25–W28 | 29 Mar – 23 Apr 2027 | CPU | G5 · 23 Apr |
| 6 Product & pilots | W29–W34 | 26 Apr – 4 Jun 2027 | all | G6 · 4 Jun |

---

## 5. Phase 0 — Setup and feasibility (W1)

Goal: prove the model fits on the laptop and lock the reading of SP:35 before building on either.

- [ ] **0.1 Repository.** Create the repo from this pack; private remote; CI runs `tools/sp35_reference_check.py` and `pytest`. _(8 Oct: repo and CI workflow done, CI green in a simulated fresh checkout; **open: private remote** — CI has never run.)_
- [ ] **0.2 Model access.** The account holder accepts the SAM License for `facebook/sam3.1` on Hugging Face and creates a read token; store it as `HF_TOKEN` outside the repo. _(8 Oct: gated access works; **open: rotate the token** that was exposed in chat, audit A1.)_
- [x] **0.3 Runtime.** Try native Windows for at most half a day; otherwise use WSL2 Ubuntu 24.04 (recommended by SAM 3; native Windows has Triton issues). Python 3.12, PyTorch ≥ 2.7 for CUDA 12.8. Write `docs/reports/runtime_decision.md`. _(evidence: `docs/reports/runtime_decision.md` — native Windows standard, WSL2 fallback; env pinned in `requirements/vision-win.txt`.)_
- [x] **0.4 Memory probe.** Run the `vram-probe` skill: 10 photos × all prompts in bf16 plus one tiled pass. Pass: peak allocated ≤ 3.4 GB, no out-of-memory error. Fallbacks in order: smaller input → sequential prompts with cache clearing → text encoder on CPU → CPU-only timing → remote worker on the 3060. _(evidence: `docs/reports/vram_probe_rtx2050_2026-10-08.md` — PASS, peak 2.57 GB, no spill, tiled pass included.)_
- [ ] **0.5 Interpretations.** One-hour session; the bridge engineer signs `docs/plan/SP35_CH7_INTERPRETATIONS.md` (E1–E14) and the bands in `kb/sp35_2024/distress_*.yaml`. Set `status: signed` in the KB files. _(open: needs the bridge engineer; `tools/verify_phase0.py --strict` fails until signed.)_
- [x] **0.6 Pilot set.** Dedupe the demo set (383 files → 167 distinct flyover photos), exclude non-bridge frames (2 DJI M3T thermal frames of a non-bridge site, 1 Windows screenshot), tag span and component in `data/pilot/tags.csv`. _(evidence: `ingest/build_pilot.py`; `data/pilot/manifest.csv` 383 → 167, every exclusion reasoned; span/support tagging moved to WP1.3 with the engineer, probe photos carry SP:35 hints.)_

**G0 acceptance:** probe report with numbers; runtime decision; signed register; pilot set ready.

---

## 6. Phase 1 — One-click SAM 3.1 inferencer and the Chapter 7 engine (W2–W7)

Goal: double-click → pick folder → reviewed draft report per SP:35 Chapter 7, entirely on the RTX 2050 laptop; nothing leaves it.

In scope: still photos (JPG, HEIC, PNG, drone RGB); one bridge at a time; Table 7.2 and 7.3 distresses; BHI and MPI engines; local-only processing.
Out of scope: reasoner (Phase 3), automatic mm from scale cards (Phase 2), fine-tuning (Phase 4), multi-bridge ranking UI (Phase 5), video/thermal/installer/multi-user (Phase 6).

### WP1.1 SP:35 knowledge base (W2, CPU)
- [x] Load `kb/sp35_2024/*.yaml` into pydantic models; validate at start-up (weights present, every row has printed I–V, intervals parse, no gaps or overlaps, aliases resolve). _(evidence: `sp35/kb.py`, `tests/unit/test_engine.py`.)_
- [x] Keep `printed` text beside machine bands; expose `explain(row, value)` returning e.g. "Table 7.2 row 11 (Cracking, mm): 0.6 in (0.3,1.0] → III". _(as `condition_state(...).explanation`.)_
- [x] Include extra rows T7.4-CORR, T7.4-TILT and the alias map (E11, E13).
- [ ] KB version hash (sha256 of the YAML files) stamped on every assessment and report. _(hash done — `KnowledgeBase.version_hash`, line-ending independent, on `BridgeAssessment`; reports come with WP1.7.)_
- **Done when:** schema tests pass and the signed register is committed.

### WP1.2 Scoring engine (W2, CPU, deterministic)
- [x] `condition_state(row, value | values)`; dual-criteria rows take the worse state; `null` bands never match. _(evidence: `sp35/condition.py`.)_
- [x] Eq. 3 (`cs_j`), Eq. 2 (`ci_i`), Eq. 1 (`bhi_span`), span model per `components.yaml` `span_model` (E14), bridge BHI = min over spans, star band (E8), critical list (state V) and urgent list (state IV). _(evidence: `sp35/scores.py` — `assess`, `assess_bridge`.)_
- [x] Eq. 4–5 (`mpi`) with Table 7.8 classes from `mpi.yaml`; F7 from BHI via star bands by default (E9), overridable. _(evidence: `sp35/mpi.py`.)_
- [x] Tests: golden cases from `golden_examples.yaml`; property tests (hypothesis): BHI within [0,100]; a worse state never raises CS/CI/BHI; adding an NA element never lowers BHI; bridge BHI equals min span BHI. _(evidence: `tests/golden`, `tests/property` — 69 tests pass.)_
- **Done when:** all tests pass in CI on each commit, and `tools/sp35_reference_check.py` agrees with the engine.

### WP1.3 Ingest and organise (W2–W3, CPU)
- [x] Folder scan; SHA-256 + perceptual hash (dHash 16×16, Hamming ≤ 20/256) to drop HEIC-conversion twins and resaved `_1` copies; keep the earliest original. _(evidence: `ingest/build_pilot.py`.)_
- [ ] EXIF (camera, time, GPS if any) and DJI XMP (`drone-dji:` GPS, RelativeAltitude, Gimbal*Degree, LRFTargetDistance); orientation fix; thumbnails. _(partial: camera, time, GPS flag, DJI ImageSource and orientation done; full XMP fields and thumbnails open.)_
- [ ] Span/support/component tagging: folders `S01/`, `S02/`, `P01/`, `A1/`… or bulk tagging in the app; stored in SQLite.
- [ ] Face and number-plate blur for every exported image (detector on CPU is fine).
- **Done when:** the Patna folder ingests to 167 distinct photos with no manual cleanup, and excluded files list a reason.

### WP1.4 SAM 3.1 inferencer (W3–W5, RTX 2050)
- [ ] `vision/worker.py` FastAPI service; model loaded once; bf16 autocast; `set_image` once per photo; `set_text_prompt` per prompt from `kb/prompts/sam3_prompts.yaml`. _(partial: `vision/model.py` loads once with bf16 encoders and a VRAM cap, used by the `vision/infer.py` CLI; the FastAPI service is open.)_
- [x] Pass 1 components → host masks; pass 2 distresses → keep masks inside a host; subtract suppression masks (cables, ropes, posters, people, vehicles); merge synonym and tile results (IoU ≥ 0.6). _(evidence: `vision/postprocess.py`, `tests/unit/test_postprocess.py`.)_
- [x] Optional tiled pass at native resolution (20% overlap) for `tile: true` classes on selected photos. _(`--tile`.)_
- [ ] COCO-RLE + overlay PNG per photo; resumable JSONL job manifest; progress events to the app; OOM guard (empty cache → smaller input → CPU → "needs batch GPU"). _(partial: RLE, overlay, resumable JSONL manifest and OOM retry done; app progress events come with WP1.6.)_
- [x] Log VRAM peak and seconds per photo per job.
- **Done when:** the pilot set runs end to end on the laptop without a crash; VRAM peak and time per photo are logged against the G0 numbers.

### WP1.5 Measurement v0 (W4, CPU)
- [ ] Extent % = distress area ÷ host component area in the same photo; per component instance keep the worst view; label "visual estimate".
- [ ] Counts for count-based rows (missing elements, drainage spots, potholes), reviewer-confirmed.
- [ ] Entry fields for crack width, scour, deflection, tilt from gauge / calliper / level (Appendix VI tools).
- [ ] Every value stores `source` and `method`; the engine refuses a mm/m value without a source.
- **Done when:** each proposed state shows value, unit, source and its table row.

### WP1.6 Review app (W4–W6)
- [ ] Screens: bridge setup → photo board (by span/component) → photo review (overlays, accept/reject, click-to-refine via `/refine`) → component sheet (distresses, values, proposed state with rule trace, engineer state + reason) → scores (span BHIs, bridge BHI, stars, critical/urgent lists) → export.
- [ ] Audit log for every change; state-V items cannot be dismissed without a written reason.
- [ ] Foundations: explicit engineer judgement entry (state + reason) when not visible.
- **Done when:** the engineer takes the pilot bridge from raw detections to signed states in one sitting.

### WP1.7 Reports (W5–W6)
- [ ] Appendix IV proformas per component group with the book's columns (Id, sub-component, distress type, location, extent, severity, condition state, probable reasons, maintenance actions, urgency).
- [ ] Appendix V cover: bridge name/ID, inspection type, date, duration, summary, photos (geo-tagged where available), comments, the inspector's certificate and signature block, cross-verification line for the implementing agency.
- [ ] BHI per span and bridge, stars, critical and urgent lists, photo plates labelled with bridge ID and defect location (§6.5), annex with KB version and decisions used.
- [ ] Formats: PDF (HTML → PDF), Excel (openpyxl), JSON.
- **Done when:** the PDF matches the Appendix IV/V structure field by field.

### WP1.8 One-click packaging (W6)
- [ ] Desktop shortcut **Bridge Inspector** → PowerShell launcher: start the WSL2 worker and the app, wait for health checks, open the browser on `http://127.0.0.1:<port>`.
- [ ] First-run setup: checks driver, WSL2, disk; downloads weights with the user's `HF_TOKEN`.
- [ ] "Collect diagnostics" zips logs, `nvidia-smi` output and versions.
- **Done when:** a fresh Windows account reaches the app in under 2 minutes after first-run setup.

### WP1.9 Pilot and demo (W7)
- [ ] Full run on the Patna flyover; engineer review; record machine time, review time, accepted/rejected detections per class, VRAM peak.
- [ ] Demo on Fri 27 Nov 2026; retrospective → Phase 2 backlog.

### Phase 1 week plan
| Week | Dates | Build | Visible at week's end |
|---|---|---|---|
| W2 | 19–23 Oct | KB loader, state mapping, Eq. 1–5, golden tests; ingest and dedupe | Yamuna BHI reproduced from YAML; Patna set reduced to 167 photos |
| W3 | 26–30 Oct | Organiser + SQLite; SAM 3.1 worker, whole-photo pass | Masks on pilot photos, timed |
| W4 | 2–6 Nov | Prompt library v1, post-processing, measurement v0, app skeleton | Overlays per photo |
| W5 | 9–13 Nov | Review screens, refine, rule traces; report templates | Engineer confirms one span |
| W6 | 16–20 Nov | Reports, launcher, first-run setup, hardening | Signed PDF/Excel for one span |
| W7 | 23–27 Nov | Pilot, metrics, demo, retrospective | G1 review |

### G1 acceptance
1. All golden and property tests pass; KB v1 and the register are signed.
2. The 167-photo pilot set processes on the RTX 2050 with no crash or out-of-memory error; time per photo and VRAM peak recorded.
3. The engineer signs a draft report in which every state links to a photo, a measurement or a written judgement.
4. Every state-V item appears in the critical list, whatever the BHI.
5. One-click start works on a clean Windows account.

### Demo script (27 Nov)
1. Double-click Bridge Inspector; select the Patna folder.
2. Progress: 383 files in, 167 photos kept, components and distresses found.
3. Pier-cap photo: exposed-reinforcement masks; the rope across the frame ignored.
4. Accept one detection, refine another with one click, type in a crack-gauge reading.
5. Component sheet with the rule trace to Table 7.2.
6. BHI per span, bridge BHI, star rating, critical list.
7. Export the PDF; walk through Appendix IV/V and the certificate block.

---

## 7. Phase 2 — Ground truth, real measurements, batch runs (W8–W13)

Goal: trustworthy, measured output — labelled data, per-class accuracy, crack widths in mm, batch runs on company GPUs, and an evidence-based Cosmos 3 hosting decision.

People: field team (2 capture days), bridge engineer (labelling QA, ~4 h/week), 1–2 labellers for 3 weeks.

### WP2.1 Field capture protocol v1 (W8)
- [ ] One-page shot list per span: overview → mid → close-up of each Table 7.1 sub-component; GPS on; names `<bridge>_<span>_<component>_<seq>.jpg`.
- [ ] ArUco scale card + crack-width comparator in every close-up; laser distance where available.
- [ ] Drone (DJI M3T): RGB wide + zoom, gimbal angles and stand-off for soffits and bearings; thermal optional, logged separately; laser rangefinder on.
- [ ] Field-team training (1 h); import check flags photos with no GPS or no scale card.
- **Done when:** bridges 2 and 3 are photographed under the protocol.

### WP2.2 Labelling setup (W8)
- [ ] Self-hosted CVAT or Label Studio (Docker); import SAM 3.1 pre-labels.
- [ ] Taxonomy = SP:35 sub-component codes × distress rows (+ aliases); guideline illustrated with Table 7.4/7.5 reference photos and our own examples.
- [ ] 10% of images labelled twice; report agreement (mask IoU, class agreement).

### WP2.3 Dataset v1 (W9–W12)
- [ ] Three bridges; targets ≥ 1,500 photos, ≥ 3,000 instance masks, ≥ 12 component and ≥ 10 distress classes.
- [ ] Split by bridge, one bridge fully held out.
- [ ] DVC versioning on an encrypted drive; data card (sources, consent, known gaps).

### WP2.4 Evaluation harness (W9–W10)
- [ ] Per class: mask AP50, recall at fixed precision, photo-level recall for critical classes (exposed rebar, spalling, wide cracks, bearing damage).
- [ ] Measurement error: extent % vs labels; crack width vs calliper.
- [ ] Outcome: weighted kappa vs engineer states; BHI difference per span.
- [ ] HTML report per run; CI blocks changes that drop any critical class by > 5 points.

### WP2.5 Prompt and threshold tuning (W10–W11, RTX 3060)
- [ ] Sweep synonyms, exemplar box prompts (from labelled data and the SP:35 reference photos), negative prompts, per-class thresholds, tile sizes.
- [ ] Output prompt library v2 with per-class settings and a before/after table.

### WP2.6 Measurement v1 (W10–W12)
- [ ] ArUco detection → mm per pixel + plane homography.
- [ ] Crack width from mask skeleton + distance transform (max and mean); areas in m².
- [ ] Drone ground sample distance from laser range or altitude + camera intrinsics.
- [ ] Uncertainty per value; validate on 50 calliper readings.

### WP2.7 Batch runner (W9–W10)
- [ ] Docker image (CUDA 12.8, PyTorch cu128) for 3060 and 5070 Ti; JSONL job manifests; resume after interruption; per-job VRAM budget.
- [ ] Encrypted SSD/NAS transfer with checksums; `gpu-batch-day` skill.

### WP2.8 Reasoner contract and Cosmos 3 spike (W12–W13, RTX 5070 Ti)
- [ ] Implement the contract (section 2.3) with a rule-based stub used by the app.
- [ ] Spike on 200 labelled crops: (a) Cosmos 3 Nano reasoner-only on the 5070 Ti with layer-wise offload; (b) community 4-bit build (unofficial, flagged); (c) hosted NIM `nvidia/cosmos3-nano-reasoner` (needs client consent, photos leave the premises); (d) Cosmos Reason 2 8B/2B.
- [ ] Measure false-positive reduction, recall loss, latency, VRAM → `docs/reports/reasoner_hosting_memo.md`.

### WP2.9 Laptop-speed option (optional, W12–W13)
- [ ] Train a small permissively licensed segmentation model on SAM 3.1 pseudo-labels; compare speed and quality on the RTX 2050.

### Phase 2 week plan
| Week | Dates | Build | GPU booking |
|---|---|---|---|
| W8 | 30 Nov–4 Dec | Protocol, labelling tool, taxonomy, Docker image | — |
| W9 | 7–11 Dec | Capture bridge 2; batch runner; pre-label; harness starts | 3060 day 1 |
| W10 | 14–18 Dec | Labelling sprint 1; harness done; ArUco + crack width | — |
| W11 | 21–25 Dec | Capture bridge 3; labelling sprint 2; prompt sweeps | 3060 day 2 |
| W12 | 28 Dec–1 Jan | Measurement validation; reasoner contract + stub; Cosmos 3 spike 1 | 5070 Ti day 1 |
| W13 | 4–8 Jan | Baseline eval on held-out bridge; Cosmos 3 spike 2; hosting memo; G2 | 3060 day 3 · 5070 Ti day 2 |

### G2 acceptance
1. Dataset v1 frozen with data card; labelling agreement reported.
2. Baseline per-class accuracy on the held-out bridge (this sets Phase 3–4 targets; no numbers are promised before it).
3. Target, confirmed at G2: crack width from scale-card photos within ±0.1 mm of calliper for ≥ 80% of 50 test cracks (0.2–2 mm).
4. A 1,000-photo batch completes unattended on the RTX 3060 in one session, including a forced restart.
5. Cosmos 3 hosting decision signed off with measured VRAM, latency and quality.

---

## 8. Phases 3–6 (outline; detailed at the gate before each starts)

- **Phase 3 · Cosmos 3 Reasoner (W14–W18).** Verify each detection (drop cables, ropes, posters, shadows), assign SP:35 codes, draft Appendix IV "probable reasons / maintenance actions / urgency", ask-the-photo Q&A with saved reasoning; JSON-schema outputs; reasoner never sets the final state. G3: fewer false positives without meaningful loss on critical classes (targets from G2).
- **Phase 4 · Fine-tuning (W19–W24).** SAM 3.1 adapter or partial fine-tune with Meta's training code (5070 Ti or rented GPU); LoRA for the reasoner on engineer-reviewed SP:35 Q&A; optional distilled laptop model; model registry. G4: beats prompt-only on critical classes at equal precision.
- **Phase 5 · Network priority (W25–W28).** Bridge portfolio; MPI input form (Tables 7.7–7.8); network/regional ranking; budget optimisation per §7.11 with an open-source solver; deterioration rate from repeat inspections. G5: pilot client accepts ranking for ≥ 10 bridges.
- **Phase 6 · Product and pilots (W29–W34).** Installer and updates, roles, audit trail, backups, security review; drone module (M3T checklist, thermal R-JPEG for delamination candidates, video via SAM 3.1 multi-object tracking); mobile capture checklist; training; 2–3 paid pilots. G6: v1.0.

---

## 9. Risks (owner checks at the listed gate)

| Risk | L | I | Response | Gate |
|---|---|---|---|---|
| SAM 3.1 exceeds 4 GB on the laptop | M | H | bf16 image-only, smaller input, sequential prompts, CPU fallback, batch on 3060 | G0 |
| Domain terms missed; cables read as cracks/rebar | H | H | exemplar + negative prompts, component clipping, reasoner, fine-tuning | G2 |
| No scale in photos → no mm/m² | H | H | capture protocol, scale cards, laser; gauge entry; never invent mm | G2 |
| Cosmos 3 too big for 16 GB (BF16-only release) | H | M | offload, flagged quantisation, hosted NIM with consent, Cosmos Reason 2 | G2 |
| Company GPUs rarely free | H | M | resumable batches, booked days, cloud overflow, dedicated machine | G1 |
| Book ambiguities → wrong scores | M | H | signed register E1–E14, versioned KB, golden tests | G0 |
| Over-trust in AI | M | H | proposals only, evidence trace, state V can't be silently dismissed, App. V certificate | all |
| Licences (SAM License, OpenMDW-1.1) | L | H | legal review; "Built on NVIDIA Cosmos" attribution | G1 |
| Privacy (people, infra data) | M | M | local by default, blur exports, cloud only with consent | G1 |
| Windows/WSL2/driver friction | M | M | WSL2 standard, Docker image, diagnostics | G0 |
| Agent invents thresholds / skips tests | M | M | AGENTS.md, KB-only values, golden tests, plan review | continuous |
| SP:35 amendments | L | M | versioned KB; reports state edition | continuous |

## 10. Decisions needed from the co-founders

1. Approve phases and gates G0–G2 — by 12 Oct 2026.
2. Name the bridge engineer for sign-off and labelling QA (~4 h/week in Phases 0–2) — by 12 Oct.
3. Accept the SAM 3.1 licence and create a read token (account holder) — by 13 Oct.
4. Book GPU days (3060 × 3, 5070 Ti × 2, Dec–Jan) or approve a cloud-GPU budget — by 27 Nov.
5. Approve photographing two more bridges; data rules (local by default, cloud only with consent) — by 30 Nov.
6. Legal review of SAM License and OpenMDW-1.1 — by G1.
7. Buy field kit (ArUco scale cards, crack-width comparators, laser distance meter, encrypted SSD) — by 30 Nov.
8. Consider a dedicated GPU workstation — by G2.

## 11. Metrics to log from day one
Seconds per photo (encode / per prompt / tiled), VRAM peak, detections per class (accepted / rejected / edited), engineer review minutes per bridge, number of states set from measurement vs visual estimate vs judgement, BHI difference between AI-proposed and final states.

## 12. Demo data notes (yugmi_images.zip)
383 files: 183 iPhone 16 Pro originals (12 MP and 24 MP, March 2026, no GPS) of one urban flyover in Patna, plus 181 HEIC-conversion twins and 15 resaved `_1` copies → 167 distinct photos after near-duplicate merge. Also 15 Street View screenshots of the approach retaining walls, 2 DJI M3T thermal frames of a non-bridge site (Nov 2025), 1 unrelated Windows screenshot. Dense cables, ropes, posters and people in frame.
