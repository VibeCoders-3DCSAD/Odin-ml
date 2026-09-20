# Synthetic Generation v2 Implementation Plan

> **Status: Tasks 0-6 implemented and committed (2026-09-17). Self-reviewed and closed out.**
>
> Self-review (2026-09-17) found and resolved two gaps:
>
> 1. `training/config/hfce_quarterly_indices.json` had never been `git add`ed — fixed
>    (committed separately).
> 2. v2's transaction generator depended on an *unrelated, then-uncommitted* working-tree
>    change to v1 (`employment_type` → `income_pattern` rename + `year_month` field on
>    `Transaction`/`MonthlySummary`). Against the last **committed** v1 state at the time,
>    `generate_persona_transactions_v2` silently produced zero transactions per persona
>    (the broad `except Exception: warnings.warn(...); continue` in the per-month loop
>    swallowed the resulting `KeyError`/`TypeError`). Confirmed via a fresh `git clone` +
>    test run. Per user decision, that pending v1 rename (already complete and covered by
>    its own tests: `test_multi_year_pipeline.py`, `test_forecaster_parallel.py`) was
>    committed separately (`bbefdf1`, `2f5e29c`, `0404481`, `ba939c1` — training/api/docs/
>    tests scopes). Re-verified via a second fresh clone: `synthesizer_v2.py` now generates
>    real transactions end-to-end (7,113 in a 60-persona smoke run) and all 33 v2 tests pass.

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship **Synthetic Generation v2** as a **parallel** pipeline — FIES-anchored, HFCE-calibrated temporal disaggregation with proportional benchmarking — without changing behavior of existing v1 synth scripts.

**Architecture:** Leave `generate_personas.py`, `generate_transactions.py`, `synthesizer.py`, and `preprocessor.py` **behaviorally untouched**. Add new v2 modules that may *import* shared types/helpers from v1 but must not edit v1 expense logic. V2 writes to `synth_v2/` by default (v1 keeps `synth/`). Personas may be produced by calling v1 `generate_all_personas` from a new orchestrator (import-only). Expense months come from HFCE schedules in `generate_transactions_v2.py`.

**Tech Stack:** Python 3.14, NumPy, pandas, pytest, JSON HFCE config.

**Canonical methodology:** [`training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md`](../../training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md)

**Canonical HFCE config (already populated):** [`training/config/hfce_quarterly_indices.json`](../../training/config/hfce_quarterly_indices.json)

---

## Hard Isolation Rule (non-negotiable)

| May touch? | Path |
| :---: | :--- |
| **No** | `training/scripts/generate_personas.py` |
| **No** | `training/scripts/generate_transactions.py` |
| **No** | `training/scripts/synthesizer.py` |
| **No** | `training/scripts/preprocessor.py` |
| **Yes (new only)** | `training/scripts/temporal_disaggregation.py` |
| **Yes (new only)** | `training/scripts/generate_transactions_v2.py` |
| **Yes (new only)** | `training/scripts/synthesizer_v2.py` |
| **Yes (new only)** | `tests/test_temporal_disaggregation.py`, `tests/test_synth_v2_*.py` |
| **Yes** | `training/config/hfce_quarterly_indices.json` (already present) |
| **Yes** | docs under `training/docs/data-collection/`, `README.md`, `INDEX.md` |

**Allowed:** `from generate_transactions import Transaction, MonthlySummary, inject_anomalies, ...` (read-only reuse).

**Forbidden:** editing v1 files to add `--hfce`, remove `Normal(1,0.15)`, or change defaults.

**Regression gate:** a test (or CI step in this plan) asserts v1 `generate_transactions.py` still contains the legacy expense variation path (`rng.normal(1.0, 0.15)`), proving v1 was not “upgraded in place.”

---

## Global Constraints

- Methodology label: **FIES-anchored, HFCE-calibrated temporal disaggregation with proportional benchmarking**
- Synth label: **Synthetic Generation v2** / `synth_version = "2.0.0"`
- Default output directory: **`synth_v2/`** (never overwrite `synth/` unless user passes an explicit path)
- Never claim Visit 1 / Visit 2 / semester household anchors from the public PUF
- V2 expense baseline: no `Normal(1, 0.15)`, no `0.5×–2.0×` caps
- Income inside v2 txn generator: may reuse v1 income helpers via import, or copy income functions into v2 — either way **do not edit v1**
- Essentials = `food`, `housing`, `health`, `transport`, `education`
- `other` = non-essential residual HFCE = Total − essentials
- Pre-anomaly: \(\sum_m E_{c,m} = A_c\); \(\sum_i T_{c,m,i} = E_{c,m}\)
- Anomalies after reconcile only; no re-reconcile after anomalies
- Commits: scopes `data` / `training` / `tests` / `docs` / `config`
- Always: `source .venv/bin/activate` and `PYTHONPATH=training/scripts`

## Locked Design Decisions

| Decision | Choice | Why |
| :--- | :--- | :--- |
| Isolation | Parallel scripts + `synth_v2/` | User requirement: v1 scripts unaffected |
| Personas | Import v1 `generate_all_personas` from `synthesizer_v2` | Same FIES-calibrated personas; no personas_v2 required for v2.0 |
| \(A_c\) | `persona[c]_expense * 12` | Honest calibrated annual target |
| HFCE `other` | Residual Total − essentials | Matches essential vs non-essential |
| Q→month | Equal thirds | Conservative |
| Multi-year | Calendar month → HFCE month index `(month-1) % 12` | Documented limitation |
| Downstream | User points feature eng / preprocessor **inputs** at `synth_v2/` manually | Avoids changing `preprocessor.py` defaults |
| Docs | Methodology + v2 runbook; v1 docs left as historical | No dual expense truth inside v2 |

## Already Done

| Artifact | Status |
| :--- | :--- |
| Methodology markdown | Present |
| HFCE JSON + residual `other` | Present |
| INDEX methodology pointer | Present |

## File Structure

### Untouched (v1)

```text
training/scripts/generate_personas.py
training/scripts/generate_transactions.py
training/scripts/synthesizer.py
training/scripts/preprocessor.py
synth/                          # v1 default outputs
```

### New (v2)

| File | Responsibility |
| :--- | :--- |
| `training/scripts/temporal_disaggregation.py` | HFCE load, weights, annual→month, reconcile |
| `training/scripts/generate_transactions_v2.py` | V2 txn/summary generation from HFCE schedules |
| `training/scripts/synthesizer_v2.py` | Orchestrate personas (via v1 import) + v2 txns → `synth_v2/` |
| `tests/test_temporal_disaggregation.py` | Unit tests |
| `tests/test_synth_v2_transactions.py` | V2 integration + §13 checks |
| `tests/test_synth_v1_untouched.py` | Guard: v1 expense noise path still present |
| `training/docs/data-collection/synthetic-generation-v2.md` | Operator runbook |
| `synth_v2/` | Default v2 outputs (gitignored like `synth/`) |

## End-to-End (v2 only)

```text
FIES NCR
  → generate_all_personas()          # imported from v1, unchanged
  → generate_transactions_v2()       # NEW HFCE path
       A_c = monthly_c × 12
       → W_{c,q} from HFCE config (other = residual)
       → Q = A×W → E = Q/3 → reconcile
       → transactions Σ = E
       → anomalies (optional)
  → synth_v2/{personas, transactions, monthly_summaries, synthesis_report.json}
```

V1 path remains:

```text
preprocessor / synthesizer → generate_transactions (noise) → synth/
```

---

### Task 0: Docs — methodology bridge + v2 runbook + isolation statement

**Files:**
- Modify: `training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md`
- Create: `training/docs/data-collection/synthetic-generation-v2.md`
- Modify: `README.md`, `INDEX.md` (additive links only)

- [x] **Step 1: Implementation Bridge** in methodology

Must state:

1. Synthetic Generation **v2** is a **parallel** pipeline; v1 scripts remain available and unchanged.
2. \(A_c = \texttt{persona monthly} \times 12\) is FIES-calibrated, not 1:1 FIES `SEQ_NO` rows in v2.0.
3. `other` = residual Total − essentials.
4. Default artifacts land in `synth_v2/`.
5. Bump methodology metadata to `1.1.0`.

- [x] **Step 2: Runbook** `synthetic-generation-v2.md`

Must include:

- Isolation rule (v1 untouched)
- Commands for `synthesizer_v2.py`
- Output layout under `synth_v2/`
- How to point downstream tools at `synth_v2/` **without** editing preprocessor defaults
- Validation commands
- Non-claims (no Visit 1/2; income not HFCE-calibrated)

- [x] **Step 3: README / INDEX** — additive “Synthetic Generation v2” links; do not rewrite v1 setup as deleted

- [x] **Step 4: Commit**

```bash
git add training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md \
  training/docs/data-collection/synthetic-generation-v2.md README.md INDEX.md
git commit -m "$(cat <<'EOF'
docs(data): document parallel synthetic generation v2

EOF
)"
```

---

### Task 1: V1 untouched guard + `temporal_disaggregation` module

**Files:**
- Create: `training/scripts/temporal_disaggregation.py`
- Create: `tests/test_temporal_disaggregation.py`
- Create: `tests/test_synth_v1_untouched.py`

**Interfaces:**
- `SYNTH_VERSION = "2.0.0"`
- `HFCE_CATEGORIES`, `ESSENTIAL_CATEGORIES`, `DEFAULT_HFCE_PATH`
- `load_hfce_levels`, `quarter_weights`, `disaggregate_annual`, `reconcile_to_annual`, `build_year_schedule`, `month_amount`
- `HFCEConfigError`

- [x] **Step 1: Write `test_synth_v1_untouched.py`**

```python
from pathlib import Path

def test_v1_generate_transactions_still_has_expense_gaussian():
    src = Path("training/scripts/generate_transactions.py").read_text()
    assert "rng.normal(1.0, 0.15)" in src
    assert "max(0.5, min(2.0, variation))" in src


def test_v1_scripts_exist():
    for name in (
        "generate_personas.py",
        "generate_transactions.py",
        "synthesizer.py",
        "preprocessor.py",
    ):
        assert (Path("training/scripts") / name).is_file()
```

- [x] **Step 2: Write temporal unit tests** (load real HFCE; `other` Q1 == 1255736; weights; reconcile; month 13 wraps; zeros)

- [x] **Step 3: Implement `temporal_disaggregation.py`**

- [x] **Step 4: pytest both — PASS**

- [x] **Step 5: Commit**

```bash
git add training/scripts/temporal_disaggregation.py \
  tests/test_temporal_disaggregation.py tests/test_synth_v1_untouched.py
git commit -m "$(cat <<'EOF'
feat(data): add HFCE disaggregation module for synth v2

EOF
)"
```

---

### Task 2: `generate_transactions_v2.py` (new file only)

**Files:**
- Create: `training/scripts/generate_transactions_v2.py`
- Create: `tests/test_synth_v2_transactions.py`

**Interfaces:**
- May import from v1: `Transaction`, `MonthlySummary`, `load_personas`, `inject_anomalies`, `compute_monthly_summary`, income helpers, `EXPENSE_CATEGORIES` **if needed**
- Must **not** monkeypatch or rewrite v1 module globals
- Public API:

```python
def generate_persona_transactions_v2(
    persona: dict,
    *,
    start_year: int = 2023,
    start_month: int = 1,
    num_months: int = 12,
    seed: int = 42,
    hfce_path: str | Path | None = None,
    inject_anomalies_flag: bool = True,
) -> tuple[list[Transaction], list[MonthlySummary]]:
    ...


def export_transactions_v2(...) -> None:
    ...


def build_synth_v2_report(...) -> dict:
    ...
```

**Behavior:**

1. `annual_by_category = {c: persona.get(f"{c}_expense", 0) * 12 for c in HFCE_CATEGORIES}` (map `other_expense` → `other`)
2. `schedule = build_year_schedule(...)`
3. Each month expenses from `month_amount(schedule, c, calendar_month)`
4. Weekly splits: residual on last txn
5. Income: call imported v1 income generator **or** local copy — do not edit v1
6. Anomalies last
7. Docstring states Synthetic Generation v2 + methodology name
8. CLI: `--input`, `--output` default `synth_v2/`, `--hfce`, `--months`, `--seed`, `--no-anomalies`

- [x] **Step 1: Failing integration tests** (annual sums, weekly exactness, quarterly weights, report `synth_version`)

- [x] **Step 2: Implement v2 generator**

- [x] **Step 3: pytest — PASS** (including v1 untouched guard)

- [x] **Step 4: Commit**

```bash
git add training/scripts/generate_transactions_v2.py tests/test_synth_v2_transactions.py
git commit -m "$(cat <<'EOF'
feat(training): add parallel synth v2 transaction generator

EOF
)"
```

---

### Task 3: `synthesizer_v2.py` orchestrator (new file only)

**Files:**
- Create: `training/scripts/synthesizer_v2.py`

**Interfaces:**
- Imports `generate_all_personas`, FIES loaders from v1 personas module (**import only**)
- Imports `generate_persona_transactions_v2`, export helpers from v2
- Writes under `synth_v2/` by default:

```text
synth_v2/
  personas.json
  personas.parquet
  transactions.parquet
  monthly_summaries.parquet
  synthesis_report.json
  expense_ratios.json          # if generated
  fies_stats.json              # if generated
```

`synthesis_report.json` must include:

```json
{
  "synth_version": "2.0.0",
  "pipeline": "synthesizer_v2",
  "methodology": "FIES-anchored, HFCE-calibrated temporal disaggregation with proportional benchmarking",
  "hfce_path": "training/config/hfce_quarterly_indices.json",
  "other_construction": "total - essentials",
  "annual_anchor": "persona_monthly_category_expense * 12",
  "within_quarter": "equal_thirds",
  "income_path": "v1_helpers_imported",
  "anomalies": "post_reconcile",
  "parallel_to": "training/scripts/synthesizer.py",
  "v1_untouched": true
}
```

CLI example:

```bash
PYTHONPATH=training/scripts python training/scripts/synthesizer_v2.py \
  --input <fies.csv|parquet> \
  --output synth_v2/ \
  --hfce training/config/hfce_quarterly_indices.json \
  --personas-per-archetype 1000 \
  --months 12 \
  --seed 42
```

- [x] **Step 1: Implement orchestrator + argparse**

- [x] **Step 2: Smoke test** with tiny `--personas-per-archetype 2` and `--skip-fies` if supported (mirror v1 skip behavior via import)

- [x] **Step 3: Commit**

```bash
git add training/scripts/synthesizer_v2.py
git commit -m "$(cat <<'EOF'
feat(training): add synthesizer_v2 parallel entry point

EOF
)"
```

---

### Task 4: Validation + gitignore

**Files:**
- Extend: `tests/test_synth_v2_transactions.py` (V1–V5 checks from methodology §13)
- Modify: `.gitignore` **only if** `synth_v2/` not already covered — add `synth_v2/` beside `synth/` if needed

**Checks (anomalies off):**

| ID | Rule |
| :--- | :--- |
| V1 | Per persona×category annual sum |
| V2 | Population quarterly shares match HFCE weights |
| V3 | Monthly txn sums == schedule |
| V4 | Deterministic expenses given seed + HFCE |
| V5 | Report `synth_version == "2.0.0"` |
| V0 | V1 untouched guard still green |

- [x] **Step 1: Implement / finish tests**

- [x] **Step 2: Ensure `synth_v2/` gitignored**

- [x] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
test(data): validate synth v2 identities and keep v1 frozen

EOF
)"
```

---

### Task 5: Documentation sweep (no lapses, no v1 rewrites)

**Checklist:**

- [x] Methodology: parallel v2 + bridge + residual `other`
- [x] Runbook: exact `synthesizer_v2.py` commands; `synth_v2/` layout
- [x] README: v1 still default; v2 optional parallel path
- [x] INDEX: both methodology + runbook + plan
- [x] TEAMMATE-GUIDE: additive note only if it documents synth (do not remove v1 instructions)
- [x] Grep docs: do not claim v1 generator now uses HFCE
- [x] This plan remains the execution source of truth

- [x] **Commit**

```bash
git commit -m "$(cat <<'EOF'
docs(data): finish synth v2 parallel-pipeline documentation

EOF
)"
```

---

### Task 6: Local regen smoke (v2 only)

- [x] Run `synthesizer_v2.py` → `synth_v2/`
- [x] Confirm `synthesis_report.json` stamps v2
- [x] Confirm `synth/` unchanged if it already existed
- [x] Confirm v1 scripts’ mtimes/contents unchanged (`git diff training/scripts/generate_*.py synthesizer.py preprocessor.py` empty for those files)

---

## Explicit Non-Goals (v2.0)

| Non-goal | Reason |
| :--- | :--- |
| Editing v1 synth scripts | Isolation requirement |
| Changing `preprocessor.py` defaults | Would affect v1 path; users pass `synth_v2/` explicitly downstream |
| Deleting or deprecating v1 in code | v1 remains usable |
| Visit 1/2 anchors | Not in PUF |
| HFCE income seasonality | Expenditure-only |
| Auto-retrain models | Separate step after choosing v2 artifacts |

## Definition of Done

1. Tasks 0–5 complete  
2. `git diff` shows **no** modifications to the four v1 scripts listed in the isolation table  
3. `tests/test_synth_v1_untouched.py` passes  
4. V2 unit/integration tests pass  
5. `synth_v2/` regen smoke OK  
6. Docs state parallel coexistence clearly  

## Self-Review

1. Isolation requirement covered by Hard Isolation Rule + V0 guard + DoD item 2  
2. Methodology / residual other / HFCE / validation / runbook assigned  
3. No “modify generate_transactions.py” tasks remain  
4. Downstream consumption documented without forcing preprocessor edits  

---

## Execution

**1. Subagent-driven** — one agent per task  
**2. Inline** — this chat  

Start at Task 0, then Task 1 (guard + core module).
