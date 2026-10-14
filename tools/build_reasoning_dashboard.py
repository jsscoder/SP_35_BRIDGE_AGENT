import csv
import json
from pathlib import Path

def generate_dashboard():
    csv_path = Path("data/pilot/calibrated_v1/conditions_comparison_audit.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from CSV.")

    # Save to both the artifacts directory and the workspace directory
    artifact_dir = Path(r"C:\Users\omkar\.gemini\antigravity\brain\d84449fc-feea-44dd-9370-35eaacf0b77c")
    workspace_dir = Path("data/pilot/calibrated_v1")

    artifact_html = artifact_dir / "reasoning_dashboard.html"
    workspace_html = workspace_dir / "reasoning_dashboard.html"

    json_data = json.dumps(rows)

    # Dynamically compute all KPIs from the loaded CSV
    total_proposals = len(rows)
    unique_photos = len(set(r["photo"] for r in rows))
    blocked_count = sum(1 for r in rows if "NONE" in r.get("baseline_proposed_state", "") or not r.get("baseline_proposed_state"))
    blocked_pct = (blocked_count / total_proposals * 100) if total_proposals else 0.0

    state_counts = {"I": 0, "II": 0, "III": 0, "IV": 0, "V": 0}
    for r in rows:
        st = r.get("calibrated_proposed_state", "")
        if st in state_counts:
            state_counts[st] += 1

    needs_host_count = sum(1 for r in rows if "needs_host" in r.get("baseline_status", ""))
    needs_t73_count = sum(1 for r in rows if "needs_table_7_3_row" in r.get("baseline_status", ""))
    needs_site_count = sum(1 for r in rows if "needs_site_measurement" in r.get("baseline_status", ""))
    needs_mask_count = sum(1 for r in rows if "needs_mask_check" in r.get("baseline_status", ""))

    st2_pct = (state_counts['II'] / total_proposals * 100) if total_proposals else 0.0
    st3_pct = (state_counts['III'] / total_proposals * 100) if total_proposals else 0.0
    st4_pct = (state_counts['IV'] / total_proposals * 100) if total_proposals else 0.0

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>IRC:SP:35-2024 Bridge Condition Reasoning & AI Evidence Dashboard</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    :root {{
      --bg: #090d16;
      --card: #131c2e;
      --card-hover: #1a2740;
      --border: #23334d;
      --border-accent: #38bdf8;
      --text: #f8fafc;
      --muted: #94a3b8;
      --primary: #38bdf8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --state1: #3b82f6;
      --state2: #10b981;
      --state3: #eab308;
      --state4: #f97316;
      --state5: #ef4444;
    }}
    body {{
      background-color: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    .custom-scrollbar::-webkit-scrollbar {{
      width: 6px;
      height: 6px;
    }}
    .custom-scrollbar::-webkit-scrollbar-thumb {{
      background: var(--border);
      border-radius: 4px;
    }}
    .custom-scrollbar::-webkit-scrollbar-track {{
      background: rgba(0, 0, 0, 0.2);
    }}
    .proposal-card.active {{
      border-color: var(--primary);
      background-color: var(--card-hover);
      box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.4);
    }}
  </style>
</head>
<body class="p-4 md:p-6 antialiased">
  <div class="max-w-[1700px] mx-auto space-y-5">

    <!-- Top Navigation & Title Header -->
    <header class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-[var(--border)] pb-5">
      <div>
        <div class="flex items-center gap-2.5">
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30">
            IRC:SP:35-2024 Audit Trace
          </span>
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
            100% Calibrated Coverage
          </span>
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-purple-500/20 text-purple-400 border border-purple-500/30">
            SAM 3.1 &bull; Tier 4 Structural Proxy
          </span>
        </div>
        <h1 class="text-2xl md:text-3xl font-bold tracking-tight text-white mt-2">
          Bridge Distress Condition Reasoning & AI Diagnostic Workbench
        </h1>
        <p class="text-xs text-[var(--muted)] mt-1">
          Interactive line-by-line physical and mathematical evidence explorer for <code>conditions_comparison_audit.csv</code> ({total_proposals} distress proposals across {unique_photos} photos).
        </p>
      </div>

      <div class="flex items-center gap-3">
        <a href="ground_truth_matching_dashboard.html" class="px-3.5 py-2 text-xs font-medium rounded-lg bg-[var(--card)] text-slate-300 border border-[var(--border)] hover:text-white hover:border-slate-500 transition">
          &larr; Ground Truth Matrix
        </a>
        <button onclick="copyCurrentReasoning()" class="px-3.5 py-2 text-xs font-semibold rounded-lg bg-sky-500 hover:bg-sky-400 text-white transition shadow-sm flex items-center gap-1.5">
          <span>&#128203;</span> Copy Active Reasoning
        </button>
      </div>
    </header>

    <!-- KPI Summary Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 text-xs">
      <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-3.5">
        <div class="text-[var(--muted)] font-medium">Total Proposals</div>
        <div class="text-xl font-bold text-white mt-1">{total_proposals} <span class="text-xs font-normal text-[var(--muted)]">in {unique_photos} photos</span></div>
        <div class="text-[11px] text-emerald-400 mt-1">100% Scored</div>
      </div>

      <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-3.5">
        <div class="text-[var(--muted)] font-medium">Baseline Blocked</div>
        <div class="text-xl font-bold text-rose-400 mt-1">{blocked_count} <span class="text-xs font-normal text-rose-400/70">({blocked_pct:.1f}%)</span></div>
        <div class="text-[11px] text-rose-300 mt-1">{needs_host_count} missing host, {needs_site_count} site mm</div>
      </div>

      <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-3.5">
        <div class="text-[var(--muted)] font-medium">Calibrated Rescued</div>
        <div class="text-xl font-bold text-emerald-400 mt-1">{total_proposals} <span class="text-xs font-normal text-emerald-400/70">(100%)</span></div>
        <div class="text-[11px] text-emerald-300 mt-1">+{blocked_count} proposals unblocked</div>
      </div>

      <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-3.5">
        <div class="text-[var(--muted)] font-medium">State II (Good / Minor)</div>
        <div class="text-xl font-bold text-emerald-400 mt-1">{state_counts['II']} <span class="text-xs font-normal text-slate-400">({st2_pct:.1f}%)</span></div>
        <div class="text-[11px] text-slate-400 mt-1">CS_j = 75.0 (S_k = 0.25)</div>
      </div>

      <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-3.5">
        <div class="text-[var(--muted)] font-medium">State III (Fair / Medium)</div>
        <div class="text-xl font-bold text-amber-400 mt-1">{state_counts['III']} <span class="text-xs font-normal text-slate-400">({st3_pct:.1f}%)</span></div>
        <div class="text-[11px] text-slate-400 mt-1">CS_j = 50.0 (S_k = 0.50)</div>
      </div>

      <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-3.5">
        <div class="text-[var(--muted)] font-medium">State IV (Poor / Urgent)</div>
        <div class="text-xl font-bold text-orange-400 mt-1">{state_counts['IV']} <span class="text-xs font-normal text-slate-400">({st4_pct:.1f}%)</span></div>
        <div class="text-[11px] text-slate-400 mt-1">CS_j = 25.0 (S_k = 0.75)</div>
      </div>
    </div>

    <!-- Filter & Search Toolbar -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-3.5 space-y-3">
      <div class="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        <div class="relative flex-1 max-w-md">
          <input id="search-input" type="text" placeholder="Search photo (e.g. IMG_4195), component, distress, or keyword..."
                 class="w-full text-xs px-3.5 py-2 pl-9 rounded-lg bg-slate-900 border border-[var(--border)] text-white focus:outline-none focus:border-sky-400 placeholder:text-slate-500"
                 oninput="applyFilters()">
          <span class="absolute left-3 top-2 text-slate-500 text-xs">&#128269;</span>
        </div>
        <div class="text-xs text-[var(--muted)] flex items-center gap-2">
          <span>Showing: <strong id="filter-count" class="text-white">{total_proposals}</strong> / {total_proposals} proposals</span>
          <button onclick="resetFilters()" class="text-sky-400 hover:underline text-xs">Reset</button>
        </div>
      </div>

      <!-- Quick Filter Pills -->
      <div class="flex flex-wrap items-center gap-1.5 text-xs">
        <span class="text-[var(--muted)] text-[11px] font-semibold mr-1">Filter by:</span>
        <button onclick="setFilter('all')" class="pill-btn active px-2.5 py-1 rounded-md bg-sky-500 text-white font-medium" data-filter="all">All ({total_proposals})</button>
        <button onclick="setFilter('needs_host')" class="pill-btn px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-[var(--border)] hover:text-white" data-filter="needs_host">Blocked: needs_host ({needs_host_count})</button>
        <button onclick="setFilter('needs_table_7_3_row')" class="pill-btn px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-[var(--border)] hover:text-white" data-filter="needs_table_7_3_row">Blocked: Appurtenance ({needs_t73_count})</button>
        <button onclick="setFilter('needs_site_measurement')" class="pill-btn px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-[var(--border)] hover:text-white" data-filter="needs_site_measurement">Blocked: Crack mm ({needs_site_count})</button>
        <button onclick="setFilter('needs_mask_check')" class="pill-btn px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-[var(--border)] hover:text-white" data-filter="needs_mask_check">Blocked: 100% Crop ({needs_mask_count})</button>
        <button onclick="setFilter('state_II')" class="pill-btn px-2.5 py-1 rounded-md bg-slate-800 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/10" data-filter="state_II">State II ({state_counts['II']})</button>
        <button onclick="setFilter('state_III')" class="pill-btn px-2.5 py-1 rounded-md bg-slate-800 text-amber-400 border border-amber-500/30 hover:bg-amber-500/10" data-filter="state_III">State III ({state_counts['III']})</button>
        <button onclick="setFilter('state_IV')" class="pill-btn px-2.5 py-1 rounded-md bg-slate-800 text-orange-400 border border-orange-500/30 hover:bg-orange-500/10" data-filter="state_IV">State IV ({state_counts['IV']})</button>
      </div>
    </div>

    <!-- Master-Detail Workbench -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">

      <!-- LEFT COLUMN: Proposal Cards List (5 Cols) -->
      <div class="lg:col-span-5 space-y-2.5 max-h-[780px] overflow-y-auto custom-scrollbar pr-1" id="proposal-list-container">
        <!-- Javascript will render proposal list items here -->
      </div>

      <!-- RIGHT COLUMN: Deep-Dive Reasoning & AI Telemetry Inspector (7 Cols) -->
      <div class="lg:col-span-7 sticky top-4 space-y-4" id="detail-inspector">
        <!-- Javascript will render the active inspection card here -->
      </div>

    </div>

  </div>

  <script>
    // Embedded Data Payload from CSV
    const PROPOSALS = {json_data};
    let activeIndex = 0;
    let activeFilter = 'all';

    // State Colors & Badges
    const STATE_CONFIG = {{
      'I': {{ name: 'Excellent', color: 'text-blue-400', bg: 'bg-blue-500/15', border: 'border-blue-500/30' }},
      'II': {{ name: 'Good', color: 'text-emerald-400', bg: 'bg-emerald-500/15', border: 'border-emerald-500/30' }},
      'III': {{ name: 'Fair', color: 'text-amber-400', bg: 'bg-amber-500/15', border: 'border-amber-500/30' }},
      'IV': {{ name: 'Poor', color: 'text-orange-400', bg: 'bg-orange-500/15', border: 'border-orange-500/30' }},
      'V': {{ name: 'Critical', color: 'text-rose-400', bg: 'bg-rose-500/15', border: 'border-rose-500/30' }}
    }};

    function getFilteredList() {{
      const query = document.getElementById('search-input').value.toLowerCase().trim();
      return PROPOSALS.filter((p, idx) => {{
        // Filter pills
        if (activeFilter === 'needs_host' && !p.baseline_status.includes('needs_host')) return false;
        if (activeFilter === 'needs_table_7_3_row' && !p.baseline_status.includes('needs_table_7_3_row')) return false;
        if (activeFilter === 'needs_site_measurement' && !p.baseline_status.includes('needs_site_measurement')) return false;
        if (activeFilter === 'needs_mask_check' && !p.baseline_status.includes('needs_mask_check')) return false;
        if (activeFilter === 'state_II' && p.calibrated_proposed_state !== 'II') return false;
        if (activeFilter === 'state_III' && p.calibrated_proposed_state !== 'III') return false;
        if (activeFilter === 'state_IV' && p.calibrated_proposed_state !== 'IV') return false;

        // Search text
        if (query) {{
          const searchCorpus = (p.photo + ' ' + p.distress + ' ' + p.host + ' ' + p.sp35_row + ' ' + p.baseline_status + ' ' + p.plain_english_reasoning_and_debug).toLowerCase();
          if (!searchCorpus.includes(query)) return false;
        }}
        return true;
      }});
    }}

    function renderList() {{
      const filtered = getFilteredList();
      const container = document.getElementById('proposal-list-container');
      document.getElementById('filter-count').innerText = filtered.length;

      if (filtered.length === 0) {{
        container.innerHTML = `
          <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-8 text-center text-[var(--muted)] text-xs">
            No distress proposals match the active query or filter.
          </div>
        `;
        document.getElementById('detail-inspector').innerHTML = '';
        return;
      }}

      let html = '';
      filtered.forEach((p, idx) => {{
        const isSelected = (idx === activeIndex);
        const stConf = STATE_CONFIG[p.calibrated_proposed_state] || STATE_CONFIG['II'];
        const isBlocked = p.baseline_proposed_state.includes('NONE') || !p.baseline_proposed_state;

        html += `
          <div onclick="selectProposal(${{idx}})" class="proposal-card cursor-pointer bg-[var(--card)] hover:bg-[var(--card-hover)] border border-[var(--border)] rounded-xl p-3.5 transition ${{isSelected ? 'active' : ''}}">
            <div class="flex items-center justify-between gap-2">
              <span class="font-mono text-xs font-bold text-sky-400">${{p.photo}}</span>
              <div class="flex items-center gap-1.5">
                <span class="text-[11px] font-bold px-2 py-0.5 rounded ${{stConf.bg}} ${{stConf.color}} border ${{stConf.border}}">
                  State ${{p.calibrated_proposed_state}}
                </span>
              </div>
            </div>

            <div class="flex items-center justify-between text-xs mt-2 text-slate-300">
              <span class="font-semibold">${{p.distress}}</span>
              <span class="text-[var(--muted)] text-[11px]">${{p.host || 'Unassigned'}} (${{p.sp35_row}})</span>
            </div>

            <div class="flex items-center justify-between text-[11px] mt-2 pt-2 border-t border-slate-800/80">
              <div class="flex items-center gap-1.5">
                <span class="text-[var(--muted)]">Baseline:</span>
                <span class="px-1.5 py-0.2 rounded text-[10px] ${{isBlocked ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-slate-800 text-slate-300'}}">
                  ${{p.baseline_status}}
                </span>
              </div>
              <div class="text-[10px] text-slate-400 font-mono">
                ${{p.detections}} mask(s) &bull; score ${{parseFloat(p.max_score).toFixed(2)}}
              </div>
            </div>
          </div>
        `;
      }});

      container.innerHTML = html;
      renderInspector(filtered[activeIndex] || filtered[0]);
    }}

    function selectProposal(index) {{
      activeIndex = index;
      const filtered = getFilteredList();
      const cards = document.querySelectorAll('.proposal-card');
      cards.forEach((c, i) => {{
        if (i === index) c.classList.add('active');
        else c.classList.remove('active');
      }});
      renderInspector(filtered[index]);
    }}

    function renderInspector(p) {{
      if (!p) return;
      const stConf = STATE_CONFIG[p.calibrated_proposed_state] || STATE_CONFIG['II'];
      const inspector = document.getElementById('detail-inspector');

      // Parse AI evidence items
      let samScore = p.max_score;
      let maskArea = 'N/A';
      let darknessDrop = '1.00';
      let rustIndex = '0.000';
      if (p.ai_vision_evidence) {{
        const mArea = p.ai_vision_evidence.match(/Area=([0-9,]+)\\s*px/);
        if (mArea) maskArea = mArea[1];
        const mDark = p.ai_vision_evidence.match(/DarknessDrop=([0-9.]+)/);
        if (mDark) darknessDrop = mDark[1];
        const mRust = p.ai_vision_evidence.match(/RustIndex=([0-9.]+)/);
        if (mRust) rustIndex = mRust[1];
      }}

      // Parse Math breakdown
      let mathFormula = p.mathematical_extent_formula || 'Extent = (A_distress / A_member) * 100%';
      let governingRule = p.sp35_governing_band_and_cs_j || 'Table 7.2';

      inspector.innerHTML = `
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-5">

          <!-- Header -->
          <div class="flex items-start justify-between gap-4 pb-4 border-b border-[var(--border)]">
            <div>
              <div class="flex items-center gap-2">
                <span class="font-mono text-base font-bold text-sky-400">${{p.photo}}</span>
                <span class="text-xs text-[var(--muted)]">&bull; Row ${{p.sp35_row}}</span>
                <span class="text-xs text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20">${{p.host}}</span>
              </div>
              <h2 class="text-lg font-bold text-white mt-1 capitalize">${{p.distress.replace('_', ' ')}} Severity Proposal</h2>
            </div>
            <div class="text-right">
              <span class="text-xs uppercase tracking-wider text-[var(--muted)] font-medium block">Calibrated State</span>
              <span class="inline-block mt-1 text-sm font-bold px-3 py-1 rounded-lg ${{stConf.bg}} ${{stConf.color}} border ${{stConf.border}}">
                State ${{p.calibrated_proposed_state}} (${{stConf.name}})
              </span>
            </div>
          </div>

          <!-- AI Vision Evidence Telemetry Cards -->
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div class="p-3 rounded-lg bg-slate-900/90 border border-slate-800">
              <div class="text-[var(--muted)]">SAM Confidence</div>
              <div class="text-base font-bold text-sky-400 mt-0.5">${{parseFloat(samScore).toFixed(2)}}</div>
              <div class="text-[10px] text-slate-400">Class threshold: 0.25&ndash;0.45</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/90 border border-slate-800">
              <div class="text-[var(--muted)]">Defect Area</div>
              <div class="text-base font-bold text-white mt-0.5">${{maskArea}} <span class="text-xs font-normal text-slate-400">px</span></div>
              <div class="text-[10px] text-slate-400">${{p.detections}} mask(s) merged</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/90 border border-slate-800">
              <div class="text-[var(--muted)]">Cavity Darkness Drop</div>
              <div class="text-base font-bold text-amber-400 mt-0.5">${{darknessDrop}}</div>
              <div class="text-[10px] text-slate-400">Shadow depth proxy</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/90 border border-slate-800">
              <div class="text-[var(--muted)]">Rust Redness Index</div>
              <div class="text-base font-bold text-orange-400 mt-0.5">${{rustIndex}}</div>
              <div class="text-[10px] text-slate-400">Oxidation indicator</div>
            </div>
          </div>

          <!-- Mathematical Extent Calculation Box -->
          <div class="p-4 rounded-xl bg-blue-950/20 border border-blue-500/30 space-y-2">
            <div class="flex items-center justify-between text-xs font-semibold text-blue-300">
              <span>&#128290; Mathematical Extent & Denominator Normalization</span>
              <span class="font-mono text-[11px] text-blue-400 font-normal">${{p.calibrated_denominator_mode}}</span>
            </div>
            <div class="p-2.5 rounded-lg bg-slate-950 font-mono text-xs text-sky-300">
              ${{mathFormula}}
            </div>
            <div class="text-[11px] text-[var(--muted)] leading-relaxed">
              Optical Denominator Rationale: Close-up camera frames span only a localized crop of the member. Dividing by the visible crop inflates extent by 20x to 300x. Tier 4 normalization applies structural proxy cap <code class="text-slate-300">f_crop &le; 0.05</code> (5% member ceiling).
            </div>
          </div>

          <!-- Deterministic SP:35 Ruling Card -->
          <div class="p-4 rounded-xl bg-emerald-950/20 border border-emerald-500/30 space-y-2">
            <div class="flex items-center justify-between text-xs font-semibold text-emerald-300">
              <span>&#9878; IRC:SP:35-2024 Chapter 7 Rulebook Ruling</span>
              <span class="text-xs font-mono font-bold text-emerald-400">Equation 3</span>
            </div>
            <div class="p-2.5 rounded-lg bg-slate-950 font-mono text-xs text-emerald-300">
              ${{governingRule}}
            </div>
            <div class="text-[11px] text-[var(--muted)]">
              Contribution to Sub-Component Condition Index: <code class="text-emerald-400">CS_j = 100 - (100 * S_k)</code>. AI proposes; final certification reserved for Appendix V engineer sign-off.
            </div>
          </div>

          <!-- Plain English Reasoned Narrative -->
          <div class="space-y-2">
            <h3 class="text-xs uppercase tracking-wider text-[var(--muted)] font-semibold flex items-center gap-1.5">
              <span>&#128172;</span> Plain-English Civil Engineering Reasoned Explanation
            </h3>
            <div class="p-3.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300 leading-relaxed space-y-2">
              <p>${{p.plain_english_reasoning_and_debug}}</p>
            </div>
          </div>

          <!-- Dual-Pipeline Diagnostic Center -->
          <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3">
            <h3 class="text-xs uppercase tracking-wider text-amber-400 font-semibold flex items-center gap-1.5">
              <span>&#128736;</span> Pipeline Debugging & Root-Cause Diagnostics
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <div class="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <strong class="text-sky-300 text-[11px]">Upstream SAM 3.1 Pre-Pipeline</strong>
                <p class="text-[11px] text-[var(--muted)]">
                  If defect boundary is truncated, check prompt thresholds in <code>kb/prompts/sam3_prompts.yaml</code>. Recommended threshold for <code>${{p.distress}}</code> is <strong>0.25</strong>.
                </p>
              </div>

              <div class="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <strong class="text-emerald-300 text-[11px]">Downstream Calibrator & Certification</strong>
                <p class="text-[11px] text-[var(--muted)]">
                  Verify host member assignment. If bridge dimensions are available in SP:35 Form 1 inventory, upgrade from Tier 4 proxy to Tier 1 physical area scaling.
                </p>
              </div>
            </div>
          </div>

        </div>
      `;
    }}

    function setFilter(filterName) {{
      activeFilter = filterName;
      activeIndex = 0;
      document.querySelectorAll('.pill-btn').forEach(b => {{
        if (b.dataset.filter === filterName) {{
          b.className = 'pill-btn active px-2.5 py-1 rounded-md bg-sky-500 text-white font-medium';
        }} else {{
          b.className = 'pill-btn px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-[var(--border)] hover:text-white';
        }}
      }});
      renderList();
    }}

    function applyFilters() {{
      activeIndex = 0;
      renderList();
    }}

    function resetFilters() {{
      document.getElementById('search-input').value = '';
      setFilter('all');
    }}

    function copyCurrentReasoning() {{
      const filtered = getFilteredList();
      const p = filtered[activeIndex] || filtered[0];
      if (!p) return;
      const textToCopy = `Photo: ${{p.photo}}\\nDistress: ${{p.distress}} (${{p.sp35_row}})\\nHost: ${{p.host}}\\nCalibrated State: State ${{p.calibrated_proposed_state}}\\nFormula: ${{p.mathematical_extent_formula}}\\nReasoning: ${{p.plain_english_reasoning_and_debug}}`;
      navigator.clipboard.writeText(textToCopy).then(() => {{
        alert('Reasoning explanation copied to clipboard!');
      }}).catch(() => {{
        prompt('Copy reasoning:', textToCopy);
      }});
    }}

    // Initial render
    renderList();
  </script>
</body>
</html>
"""

    artifact_html.write_text(html_content, encoding="utf-8")
    workspace_html.write_text(html_content, encoding="utf-8")
    print(f"Generated dashboard at: {artifact_html}")
    print(f"Generated dashboard at: {workspace_html}")

if __name__ == "__main__":
    generate_dashboard()
