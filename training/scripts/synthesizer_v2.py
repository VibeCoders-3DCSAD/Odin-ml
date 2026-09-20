"""
Odin ML Synthetic Data Pipeline v2 — Synthetic Generation v2

Orchestrates persona generation (imported, unchanged, from v1's
`generate_personas.py`) with the new HFCE-calibrated transaction generator
(`generate_transactions_v2.py`) to produce `synth_v2/` artifacts.

This script runs **in parallel** to `training/scripts/synthesizer.py` (v1).
v1 remains available, unedited, and fully usable on its own. This script
does not import from or modify `synthesizer.py`.

Methodology: FIES-anchored, HFCE-calibrated temporal disaggregation with
proportional benchmarking.
See: training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md
Runbook: training/docs/data-collection/synthetic-generation-v2.md

Usage:
    PYTHONPATH=training/scripts python training/scripts/synthesizer_v2.py \
        --input training/datasets/unprocessed/puf.parquet \
        --output synth_v2/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add scripts directory to path for sibling imports (mirrors v1 scripts' pattern).
sys.path.insert(0, str(Path(__file__).parent))

from fies_columns import (  # noqa: E402  (v1, read-only reuse)
    FIESColumnError,
    validate_columns,
)
from generate_personas import (  # noqa: E402  (v1, read-only reuse)
    PersonaGenerationError,
    SyntheticPersona,
    compute_expense_ratios,
    compute_fies_statistics,
    export_personas,
    filter_ncr,
    generate_all_personas,
    load_fies_data,
    validate_personas,
)
from generate_transactions_v2 import (  # noqa: E402
    METHODOLOGY,
    build_synth_v2_report,
    export_transactions_v2,
    generate_persona_transactions_v2,
)
from temporal_disaggregation import (  # noqa: E402
    DEFAULT_HFCE_PATH,
    SYNTH_VERSION,
    validate_hfce_coverage,
)


class PipelineErrorV2(Exception):
    """Raised when the Synthetic Generation v2 pipeline fails."""

    pass


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


def _print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f" {text}")
    print(f"{'=' * 60}")


def _print_step(step: int, total: int, text: str) -> None:
    print(f"\n[{step}/{total}] {text}")


def run_pipeline_v2(
    input_path: str,
    output_path: str,
    personas_per_archetype: int = 1000,
    num_months: int = 42,
    seed: int = 42,
    skip_fies: bool = False,
    strict: bool = False,
    hfce_path: str | Path | None = None,
    inject_anomalies_flag: bool = True,
) -> dict:
    """
    Run the full Synthetic Generation v2 pipeline: personas (v1, imported,
    unchanged) + HFCE-calibrated v2 transactions -> `output_path`.

    Returns:
        The v2 synthesis report dict (also written to
        ``<output_path>/synthesis_report.json``).

    Raises:
        PipelineErrorV2: If persona generation fails outright.
    """
    start_time = time.time()
    results: dict = {"errors": [], "warnings": []}

    _print_header("Odin ML Synthetic Data Pipeline v2 (Synthetic Generation v2)")
    print(f"Methodology: {METHODOLOGY}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Personas per archetype: {personas_per_archetype}")
    print(f"Months: {num_months}")
    print(f"Seed: {seed}")
    print(f"HFCE config: {hfce_path or DEFAULT_HFCE_PATH}")

    try:
        validate_hfce_coverage(2023, 1, num_months, hfce_path)
    except Exception as error:
        raise PipelineErrorV2(f"Requested HFCE timeline is unavailable: {error}") from error

    total_steps = 5

    # Step 1: Load FIES data (identical semantics to v1, imported read-only)
    _print_step(1, total_steps, "Loading FIES NCR data")
    fies_stats = None
    expense_ratios = None

    if not skip_fies:
        try:
            df = load_fies_data(input_path)
            report = validate_columns(df, strict=strict)
            print(f"  Found {report['found']}/{report['total_columns']} expected columns")

            if report["missing_ids"]:
                results["warnings"].append(f"Missing FIES columns: {report['missing_ids']}")

            if not report["is_valid"]:
                missing_critical = report["critical_missing"]
                results["errors"].append(f"Missing critical columns: {missing_critical}")
                if strict:
                    raise PipelineErrorV2(
                        f"Strict mode: Missing critical columns: {missing_critical}\n"
                        "Use --skip-fies to bypass FIES data validation"
                    )

            ncr_df = filter_ncr(df)
            fies_stats = compute_fies_statistics(ncr_df)
            expense_ratios = compute_expense_ratios(ncr_df)

            results["fies_loaded"] = True
            results["fies_rows"] = len(ncr_df)
            results["fies_columns"] = report
        except FileNotFoundError as e:
            print(f"  ERROR: File not found: {e}")
            results["errors"].append(f"File not found: {e}")
            print("  Falling back to default statistics")
        except FIESColumnError as e:
            print(f"  ERROR: FIES column error: {e}")
            results["errors"].append(f"FIES column error: {e}")
            print("  Falling back to default statistics")
        except PipelineErrorV2:
            raise
        except Exception as e:
            print(f"  ERROR: Unexpected error loading FIES data: {e}")
            results["errors"].append(f"Unexpected error: {e}")
            print("  Falling back to default statistics")
    else:
        print("  Skipping FIES data loading (--skip-fies)")
        results["fies_loaded"] = False

    # Step 2: Generate personas (v1, imported unchanged)
    _print_step(2, total_steps, "Generating synthetic personas (v1 generator, imported)")
    try:
        personas = generate_all_personas(
            personas_per_archetype=personas_per_archetype,
            fies_stats=fies_stats,
            expense_ratios=expense_ratios,
            seed=seed,
        )

        persona_validation = validate_personas(personas)
        results["persona_count"] = persona_validation["total_personas"]
        results["persona_validation"] = persona_validation

        export_personas(personas, str(output_path))

        if not skip_fies and expense_ratios is not None:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "expense_ratios.json", "w") as f:
                json.dump(expense_ratios, f, indent=2, cls=NumpyEncoder)
            with open(output_dir / "fies_stats.json", "w") as f:
                json.dump(fies_stats, f, indent=2, cls=NumpyEncoder)
    except PersonaGenerationError as e:
        print(f"  ERROR: Persona generation failed: {e}")
        raise PipelineErrorV2(f"Persona generation failed: {e}") from e
    except Exception as e:
        print(f"  ERROR: Unexpected error generating personas: {e}")
        raise PipelineErrorV2(f"Unexpected error generating personas: {e}") from e

    # Step 3: Generate v2 (HFCE-calibrated) transactions
    _print_step(3, total_steps, "Generating HFCE-calibrated transaction histories (v2)")
    all_transactions = []
    all_summaries = []
    failed_personas = 0

    for i, persona in enumerate(personas):
        if (i + 1) % 1000 == 0:
            print(f"  Processing persona {i + 1:,}/{len(personas):,}...")

        try:
            persona_dict = asdict(persona) if isinstance(persona, SyntheticPersona) else persona

            transactions, summaries = generate_persona_transactions_v2(
                persona_dict,
                start_year=2023,
                start_month=1,
                num_months=num_months,
                seed=seed,
                hfce_path=hfce_path,
                inject_anomalies_flag=inject_anomalies_flag,
            )

            all_transactions.extend(transactions)
            all_summaries.extend(summaries)
        except Exception as e:
            warnings.warn(f"Failed v2 transaction generation for persona {i}: {e}", stacklevel=2)
            failed_personas += 1

    results["transaction_count"] = len(all_transactions)
    results["summary_count"] = len(all_summaries)
    results["failed_personas"] = failed_personas

    if failed_personas > 0:
        print(f"  WARNING: {failed_personas} personas failed v2 transaction generation")
        results["warnings"].append(f"{failed_personas} personas failed")

    # Step 4: Export v2 transactions
    _print_step(4, total_steps, "Exporting v2 transaction data")
    try:
        export_transactions_v2(all_transactions, all_summaries, str(output_path))
    except Exception as e:
        print(f"  ERROR: Failed to export v2 transactions: {e}")
        results["errors"].append(f"Export failed: {e}")

    # Step 5: Build and save synthesis report
    _print_step(5, total_steps, "Generating v2 synthesis report")
    report = build_synth_v2_report(all_summaries, hfce_path=hfce_path)
    report["persona_count"] = results.get("persona_count", 0)
    report["errors"] = results.get("errors", [])
    report["warnings"] = results.get("warnings", [])
    report["elapsed_seconds"] = round(time.time() - start_time, 2)

    report_path = Path(output_path) / "synthesis_report.json"
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        print(f"Saved v2 synthesis report to {report_path}")
    except Exception as e:
        print(f"  ERROR: Failed to save report: {e}")

    elapsed_time = time.time() - start_time
    _print_header("Synthetic Generation v2 Pipeline Complete")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Personas generated: {results.get('persona_count', 0):,}")
    print(f"Transactions generated: {results.get('transaction_count', 0):,}")
    print(f"Monthly summaries: {results.get('summary_count', 0):,}")
    print(f"\nOutput directory: {output_path}")
    print("  personas.json / personas.parquet")
    print("  transactions.parquet / monthly_summaries.parquet")
    print(f"  synthesis_report.json (synth_version = {SYNTH_VERSION})")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Odin ML Synthetic Data Pipeline v2 — FIES-anchored, HFCE-calibrated "
            "temporal disaggregation (parallel to synthesizer.py; v1 untouched)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with FIES parquet data
    python scripts/synthesizer_v2.py --input datasets/unprocessed/puf.parquet

    # Skip FIES loading (use default archetype statistics)
    python scripts/synthesizer_v2.py --skip-fies --personas-per-archetype 50

    # Custom HFCE config + output directory
    python scripts/synthesizer_v2.py --hfce training/config/hfce_quarterly_indices.json \\
        --output synth_v2/

    # Disable anomaly injection
    python scripts/synthesizer_v2.py --skip-fies --no-anomalies
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/unprocessed/puf.parquet",
        help="Path to FIES data file (CSV or Parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synth_v2/",
        help="Output directory for all generated v2 data",
    )
    parser.add_argument(
        "--hfce",
        type=str,
        default=None,
        help="Path to HFCE quarterly indices config (defaults to "
        "training/config/hfce_quarterly_indices.json)",
    )
    parser.add_argument(
        "--personas-per-archetype",
        type=int,
        default=1000,
        help="Number of personas to generate per archetype (default: 1000)",
    )
    parser.add_argument(
        "--months", type=int, default=42, help="Number of months to generate (default: 42)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip-fies",
        action="store_true",
        help="Skip loading FIES data and use default statistics",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Strict mode: fail on missing critical columns"
    )
    parser.add_argument("--no-anomalies", action="store_true", help="Disable anomaly injection")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit total number of personas (for testing)"
    )

    args = parser.parse_args()

    personas_per_archetype = args.personas_per_archetype
    if args.limit:
        personas_per_archetype = args.limit // 12 + 1
        print(f"Limiting to ~{args.limit} personas ({personas_per_archetype} per archetype)")

    try:
        report = run_pipeline_v2(
            input_path=args.input,
            output_path=args.output,
            personas_per_archetype=personas_per_archetype,
            num_months=args.months,
            seed=args.seed,
            skip_fies=args.skip_fies,
            strict=args.strict,
            hfce_path=args.hfce,
            inject_anomalies_flag=not args.no_anomalies,
        )

        if report.get("errors"):
            print("\nPipeline completed with errors.")
            sys.exit(1)
        else:
            print("\nPipeline completed successfully!")
            print(f"synth_version = {SYNTH_VERSION}")
            sys.exit(0)
    except PipelineErrorV2 as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected pipeline failure: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
