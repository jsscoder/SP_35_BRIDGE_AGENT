# NVIDIA Cosmos 3 Reasoner Integration — Context, Guardrails & Agent Specification

**Project:** SP:35 Bridge Inspection AI (`sp35-bridge-ai`)  
**Target Module:** Reasoner Service / Cosmos 3 Reasoner Agent (`reasoner/`)  
**Standard:** Indian Roads Congress IRC:SP:35-2024 Chapter 7  
**Hardware Environment:** Desktop with NVIDIA GeForce RTX 3060 (12 GB VRAM) / Hosted NIM Fallback  

---

## 1. Mission & Objective

Build the **Cosmos Reasoner Integration** that sits downstream of the SAM 3.1 segmentation engine and upstream of the deterministic SP:35 scoring engine.

### What the Reasoner DOES:
1. **Verifies Detections:** Analyzes SAM 3.1 image crops (in the context of the full frame) to confirm whether a detection is a real bridge distress or a false positive (e.g., rejecting cables, construction joints, shadows, posters, or dirt).
2. **Assigns Standard Codes:** Maps the defect to official SP:35 component (`C1.1`–`C7.3`) and distress rows (`T7.2-01`–`T7.2-28`, `T7.3-01`–`T7.3-34`).
3. **Proposes Condition State:** Suggests a state (`I`, `II`, `III`, `IV`, `V`) based strictly on IRC:SP:35-2024 Table 7.2 & 7.3 descriptions.
4. **Drafts Appendix IV Proforma Narrative:** Generates engineering text for:
   - **Position & Location description:** Exact placement on the structural element.
   - **Probable Reasons:** Engineering root cause (e.g., carbonation, moisture ponding, overload).
   - **Maintenance Actions & Urgency:** Recommended repair method and urgency level.

---

## 2. Strict Architectural Guardrails (Non-Negotiable)

The coding agent **MUST NOT** violate these rules under any circumstance:

### ⛔ Guardrail 1: Never Calculate $S_k$ or BHI in the LLM
* **Rule:** An LLM must **never** perform mathematical calculations for $S_k$, $CS_j$, $CI_i$, $BHI$, star ratings, or MPI.
* **Why:** In IRC:SP:35-2024, $S_k$ is legally fixed ($I=0, II=0.25, III=0.5, IV=0.75, V=1.0$). LLMs hallucinate numbers (e.g., guessing $S_k = 0.35$).
* **Requirement:** The Reasoner only outputs the discrete condition state string (`"I"`, `"II"`, `"III"`, `"IV"`, `"V"`). The deterministic scoring engine in [`sp35/scores.py`](file:///c:/Users/omkar/Desktop/sp-35%20reasoner/sp35-bridge-ai/sp35/scores.py) takes the state and computes the math.

### ⛔ Guardrail 2: The AI Proposes; The Engineer Certifies
* **Rule:** The Reasoner output is **always marked as proposed/unverified** (`is_final = False`).
* **Requirement:** Detections, suggested states, and written rationales are presented to the review dashboard. Only a human bridge engineer can click "Confirm" to finalize a state.

### ⛔ Guardrail 3: Zero Hardcoded Weights or Limits
* **Rule:** Do not type component weights, distress thresholds, or scoring bands into the reasoner Python code.
* **Requirement:** All references must validate against the versioned YAML files in [`kb/sp35_2024/`](file:///c:/Users/omkar/Desktop/sp-35%20reasoner/sp35-bridge-ai/kb/sp35_2024) and [`kb/prompts/sam3_prompts.yaml`](file:///c:/Users/omkar/Desktop/sp-35%20reasoner/sp35-bridge-ai/kb/prompts/sam3_prompts.yaml).

### ⛔ Guardrail 4: VRAM Budget on the RTX 3060 (12 GB VRAM)
* **Rule:** Peak VRAM must stay $\le 10.5\text{ GB}$ to prevent CUDA Out-of-Memory (OOM) or system paging.
* **Requirement:**
  - If serving locally via vLLM or Ollama, use **4-bit AWQ / FP8** quantisation or the **2B Cosmos Reason 2** model.
  - Full BF16 8B/16B models must not be loaded without layer-wise offloading or via remote NIM API.

### ⛔ Guardrail 5: Don't Break Existing Tests
* **Requirement:** Existing verification scripts and tests must remain 100% green:
  - `uv run --with pyyaml python tools/sp35_reference_check.py` $\to$ `ALL PASS`
  - `uv run pytest` $\to$ `69 passed`
  - `uv run --with pyyaml python tools/verify_phase0.py` $\to$ `ALL PASS`

---

## 3. Data Contract & JSON Schemas

The Reasoner interface must conform to the contract defined in Section 2.3 of [`docs/plan/IMPLEMENTATION_PLAN.md`](file:///c:/Users/omkar/Desktop/sp-35%20reasoner/sp35-bridge-ai/docs/plan/IMPLEMENTATION_PLAN.md), augmented with Appendix IV reporting fields.

### 3.1 Input Payload to Reasoner
```json
{
  "image_id": "IMG_4231",
  "image_path": "data/pilot/probe10/IMG_4231.jpg",
  "detection_id": "det_4231_004",
  "bbox": [147, 584, 649, 731],
  "sam_class": "crack",
  "sam_score": 0.613,
  "host_component": {
    "code": "C2.8",
    "name": "Pier cap"
  },
  "crop_path": "data/pilot/crops/det_4231_004.jpg",
  "measured_values": {
    "crack_width_mm": 0.45,
    "extent_pct": 2.8,
    "source": "scale_card"
  }
}
```

### 3.2 Output Schema from Reasoner
```json
{
  "detection_id": "det_4231_004",
  "is_valid": true,
  "reject_reason": null,
  "sp35_component": "C2.8",
  "sp35_distress_row": "T7.2-11",
  "distress_name": "Cracking (structural)",
  "suggested_state": "III",
  "severity_note": "Crack width ~0.45 mm exceeds 0.3 mm threshold",
  "confidence": 0.88,
  "rationale": "Diagonal shear crack extending from bearing pedestal across pier cap face. Not an electrical cable or superficial stain.",
  "appendix_iv": {
    "location": "Span 1 Pier Cap P1 west bearing shelf",
    "probable_reasons": "Shear stress under dynamic wheel loading combined with thermal movement constraint",
    "maintenance_actions": "Epoxy injection grouting and surface seal with elastomeric coating",
    "urgency": "Urgent"
  },
  "model": "cosmos3-nano-reasoner-4bit",
  "latency_ms": 1420
}
```

---

## 4. Proposed Module Architecture (`reasoner/`)

Create a streamlined, modular directory under the root:

```
reasoner/
├── __init__.py
├── contract.py          # Pydantic models for ReasonerInput & ReasonerOutput
├── prompts.py           # IRC:SP:35 system prompt, few-shot examples & JSON format specs
├── client.py            # Client interface supporting 3 backends:
│                         # 1. RuleStubClient (deterministic mock for CI & CPU testing)
│                         # 2. LocalVLLMClient (calls local RTX 3060 desktop OpenAI-compatible endpoint)
│                         # 3. NvidiaNIMClient (calls hosted NVIDIA NIM API if consent flag is true)
├── cropper.py           # Generates padded visual crops + marked full-frame images for VLM input
└── service.py           # High-level pipeline taking SAM detections -> Reasoner -> SP:35 candidate list
```

---

## 5. Cosmos 3 System Prompt Template

```markdown
You are an expert structural bridge inspection engineer adhering to Indian Roads Congress code IRC:SP:35-2024.
You will inspect a bridge defect crop alongside the full bridge component context.

Your tasks:
1. Verify if the detection is a genuine structural bridge defect (true) or non-structural noise (false: shadow, cable, rope, stain, poster).
2. Identify the exact SP:35 Component Code (C1.1 to C7.3) and Distress Table Row (T7.2-01 to T7.2-28).
3. Suggest a Condition State strictly from: ["I", "II", "III", "IV", "V"]:
   - State I: Sound / Minor (no immediate action)
   - State II: Minor defect (routine maintenance)
   - State III: Moderate defect (maintenance needed)
   - State IV: Severe defect (urgent rehabilitation)
   - State V: Critical defect (immediate safety hazard / critical list)
4. Draft engineering notes for Appendix IV Proforma: Location, Probable Reasons, Maintenance Actions, Urgency.

Rules:
- NEVER output numerical S_k or BHI scores.
- Base condition state on the visual evidence and measured crack width/extent if provided.
- If uncertain, explain why in the rationale.
- Always output valid JSON strictly matching the requested schema within <answer>...</answer> tags.
```

---

## 6. How to Prompt the Coding Agent

Copy and paste the prompt below to initiate the coding agent:

```text
You are building the Cosmos Reasoner Integration for "SP:35 Bridge Inspection AI".
Read the specification in COSMOS_REASONER_AGENT_SPEC.md, AGENTS.md, and docs/plan/IMPLEMENTATION_PLAN.md §2.3.

Task Scope:
1. Create reasoner/contract.py with Pydantic models for ReasonerInput, ReasonerOutput, and AppendixIVDraft.
2. Implement reasoner/prompts.py encoding the IRC:SP:35-2024 reasoning guidelines.
3. Implement reasoner/client.py with an abstract BaseReasonerClient and two implementations:
   - RuleStubClient: A pure Python, deterministic mock used in CI and unit tests (maps SAM detection to default SP:35 rows without GPU).
   - OpenAICompatibleClient: Connects to local vLLM / Ollama on the RTX 3060 desktop (or hosted NIM) via /v1/chat/completions.
4. Implement reasoner/cropper.py to extract defect crops with context padding.
5. Write unit tests in tests/unit/test_reasoner.py verifying:
   - Valid JSON parsing from LLM responses (handling <think>...</think><answer>...</answer> blocks).
   - Error handling on malformed JSON or network timeouts (graceful fallback).
   - Strict validation: Condition state must be one of ["I", "II", "III", "IV", "V", null], never an arbitrary number.
6. Guardrails:
   - Do NOT edit kb/sp35_2024/*.yaml or sp35/scores.py.
   - Do NOT calculate Sk or BHI in the reasoner.
   - Ensure `uv run pytest` continues to pass 100%.
```
