"""
Synthetic Transaction Generator v2 — Synthetic Generation v2

Generates transaction histories using the FIES-anchored, HFCE-calibrated
temporal disaggregation with proportional benchmarking methodology, instead
of v1's flat Gaussian monthly noise (``Normal(1, 0.15)``, ``0.5x-2.0x`` bounds).

This module runs **in parallel** to `training/scripts/generate_transactions.py`
(v1). It imports read-only helpers from v1 (``Transaction``, ``MonthlySummary``,
``load_personas``, ``inject_anomalies``, ``compute_monthly_summary``,
``generate_income_transactions``, ``EXPENSE_CATEGORIES``) but does not edit,
monkeypatch, or rewrite anything in v1.

Methodology: FIES-anchored, HFCE-calibrated temporal disaggregation with
proportional benchmarking.
See: training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md

Usage:
    python training/scripts/generate_transactions_v2.py \
        --input synth_v2/personas.json --output synth_v2/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add scripts directory to path for sibling imports (mirrors v1 scripts' pattern).
sys.path.insert(0, str(Path(__file__).parent))

from generate_transactions import (  # noqa: E402  (v1, read-only reuse)
    EXPENSE_CATEGORIES,
    MonthlySummary,
    PersonaLoadError,
    Transaction,
    TransactionGenerationError,
    compute_monthly_summary,
    generate_income_transactions,
    inject_anomalies,
    load_personas,
)
from temporal_disaggregation import (  # noqa: E402
    DEFAULT_HFCE_PATH,
    HFCE_CATEGORIES,
    SYNTH_VERSION,
    available_months,
    build_year_schedule,
    month_amount,
    validate_hfce_coverage,
)

METHODOLOGY = (
    "FIES-anchored, HFCE-calibrated temporal disaggregation with proportional benchmarking"
)


class TransactionGenerationErrorV2(TransactionGenerationError):
    """Raised when Synthetic Generation v2 transaction generation fails."""

    pass


def generate_expense_transactions_v2(
    persona: dict,
    month: int,
    year: int,
    schedule: dict[str, dict[int, float]],
    rng: np.random.Generator,
) -> list[Transaction]:
    """
    Generate a month's expense transactions from the HFCE-derived schedule.

    Unlike v1, no random monthly variation is applied to the category totals
    (methodology §9: sigma_random = 0) — the scheduled monthly amount is a hard
    constraint. Only the intra-month split (date, subcategory, and — for
    weekly categories — the per-week breakdown) is randomized. Any rounding
    residual from the weekly split is absorbed by the last transaction so the
    weekly amounts still sum exactly to the scheduled monthly amount.
    """
    transactions: list[Transaction] = []
    persona_id = persona["persona_id"]

    for category in HFCE_CATEGORIES:
        amount = month_amount(schedule, category, month)
        if amount <= 0:
            continue

        cat_info = EXPENSE_CATEGORIES[category]
        subcategory = rng.choice(cat_info["subcategories"])

        if cat_info["frequency"] == "weekly":
            # 4 transactions; residual on the last absorbs rounding so the
            # weekly amounts still sum exactly to `amount`.
            rounded_shares = [round(amount / 4, 2) for _ in range(3)]
            last_share = round(amount - sum(rounded_shares), 2)
            weekly_amounts = rounded_shares + [last_share]

            for week, weekly_amount in enumerate(weekly_amounts):
                day = int(rng.integers(1, 29))
                date = f"{year}-{month:02d}-{day:02d}"
                transactions.append(
                    Transaction(
                        transaction_id=(
                            f"txn_v2_{persona_id}_{year}_{month:02d}_{category}_{week}"
                        ),
                        persona_id=persona_id,
                        month=month,
                        year=year,
                        year_month=f"{year}-{month:02d}",
                        date=date,
                        category=category,
                        subcategory=subcategory,
                        amount=weekly_amount,
                        transaction_type="expense",
                        description=f"{category} expense",
                        is_anomalous=False,
                        anomaly_type=None,
                    )
                )
        else:
            day = int(rng.integers(1, 29))
            date = f"{year}-{month:02d}-{day:02d}"
            transactions.append(
                Transaction(
                    transaction_id=f"txn_v2_{persona_id}_{year}_{month:02d}_{category}",
                    persona_id=persona_id,
                    month=month,
                    year=year,
                    year_month=f"{year}-{month:02d}",
                    date=date,
                    category=category,
                    subcategory=subcategory,
                    amount=round(amount, 2),
                    transaction_type="expense",
                    description=f"{category} expense",
                    is_anomalous=False,
                    anomaly_type=None,
                )
            )

    return transactions


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
    """
    Generate a full v2 transaction history for a persona.

    Each calendar year uses its own current-price HFCE profile. Complete years
    benchmark twelve persona-months; partial 2026 benchmarks its published
    Q1-Q2 period to six persona-months. Anomalies are injected last, after the
    reconciled monthly amounts are fixed.

    Args:
        persona: Persona dict (same schema as v1 personas).
        start_year: Starting calendar year.
        start_month: Starting calendar month (1-12).
        num_months: Number of months to generate.
        seed: Random seed for reproducibility.
        hfce_path: Optional override for the HFCE config path.
        inject_anomalies_flag: Whether to inject anomalies (default True).

    Returns:
        Tuple of (transactions, monthly_summaries).

    Raises:
        TransactionGenerationErrorV2: If the persona data is invalid.
    """
    try:
        persona_id = persona["persona_id"]
        rng = np.random.default_rng(seed + hash(persona_id) % 10000)
    except (KeyError, TypeError) as e:
        raise TransactionGenerationErrorV2(f"Invalid persona data: {e}") from e

    try:
        validate_hfce_coverage(start_year, start_month, num_months, hfce_path)
    except Exception as error:
        raise TransactionGenerationErrorV2(str(error)) from error

    months_by_year: dict[int, set[int]] = {}
    for month_offset in range(num_months):
        month = ((start_month - 1 + month_offset) % 12) + 1
        year = start_year + (start_month - 1 + month_offset) // 12
        months_by_year.setdefault(year, set()).add(month)

    schedules: dict[int, dict[str, dict[int, float]]] = {}
    for year, requested_months in months_by_year.items():
        published_months = set(available_months(year, hfce_path))
        unavailable_months = requested_months - published_months
        if unavailable_months:
            unavailable = ", ".join(str(month) for month in sorted(unavailable_months))
            raise TransactionGenerationErrorV2(
                f"HFCE data is unavailable for {year} month(s): {unavailable}"
            )
        benchmark_months = len(published_months)
        benchmark_by_category = {
            category: float(persona.get(f"{category}_expense", 0)) * benchmark_months
            for category in HFCE_CATEGORIES
        }
        schedules[year] = build_year_schedule(benchmark_by_category, year, hfce_path)

    all_transactions: list[Transaction] = []
    all_summaries: list[MonthlySummary] = []
    previous_balance = 0.0

    for month_offset in range(num_months):
        month = ((start_month - 1 + month_offset) % 12) + 1
        year = start_year + (start_month - 1 + month_offset) // 12

        try:
            income_txns = generate_income_transactions(persona, month, year, rng)
            expense_txns = generate_expense_transactions_v2(persona, month, year, schedules[year], rng)

            month_txns = income_txns + expense_txns
            if inject_anomalies_flag:
                month_txns = inject_anomalies(month_txns, persona, rng)

            summary = compute_monthly_summary(persona, month_txns, month, year, previous_balance)
            previous_balance = summary.balance

            all_transactions.extend(month_txns)
            all_summaries.append(summary)
        except Exception as e:
            warnings.warn(
                f"Error generating v2 transactions for {persona_id} month {month}/{year}: {e}",
                stacklevel=2,
            )
            continue

    return all_transactions, all_summaries


def export_transactions_v2(
    all_transactions: list[Transaction],
    all_summaries: list[MonthlySummary],
    output_dir: str | Path,
) -> None:
    """Export v2 transactions and monthly summaries (Parquet + JSON preview)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_path = output_path / "transactions.parquet"
    txn_df = pd.DataFrame([asdict(t) for t in all_transactions])
    txn_df.to_parquet(parquet_path, index=False)
    print(f"Exported {len(all_transactions):,} v2 transactions to {parquet_path}")

    summary_path = output_path / "monthly_summaries.parquet"
    summary_df = pd.DataFrame([asdict(s) for s in all_summaries])
    summary_df.to_parquet(summary_path, index=False)
    print(f"Exported {len(all_summaries):,} v2 monthly summaries to {summary_path}")

    json_path = output_path / "transactions.json"
    preview_count = min(1000, len(all_transactions))
    with open(json_path, "w") as f:
        json.dump([asdict(t) for t in all_transactions[:preview_count]], f, indent=2)
    print(f"Exported v2 transaction preview ({preview_count} rows) to {json_path}")


def build_synth_v2_report(
    all_summaries: list[MonthlySummary],
    hfce_path: str | Path | None = None,
) -> dict:
    """Build the Synthetic Generation v2 synthesis report dict."""
    resolved_hfce_path = str(hfce_path) if hfce_path is not None else str(DEFAULT_HFCE_PATH)

    report: dict = {
        "synth_version": SYNTH_VERSION,
        "pipeline": "generate_transactions_v2",
        "methodology": METHODOLOGY,
        "hfce_path": resolved_hfce_path,
        "other_construction": "total - essentials",
        "benchmark_anchor": "persona_monthly_category_expense * published_months_in_year",
        "hfce_price_basis": "current_prices",
        "hfce_coverage": "2023-Q1 through 2026-Q2",
        "within_quarter": "equal_thirds",
        "income_path": "v1_helpers_imported",
        "anomalies": "post_reconcile",
        "parallel_to": "training/scripts/generate_transactions.py",
        "v1_untouched": True,
    }

    if all_summaries:
        df = pd.DataFrame([asdict(s) for s in all_summaries])
        report["total_months"] = len(all_summaries)
        report["total_transaction_count"] = int(df["transaction_count"].sum())
        report["avg_monthly_income"] = float(df["total_income"].mean())
        report["avg_monthly_expenses"] = float(df["total_expenses"].mean())
        report["anomaly_rate"] = float(df["is_anomalous"].mean())
    else:
        report["total_months"] = 0
        report["total_transaction_count"] = 0

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Synthetic Generation v2 transactions "
            "(FIES-anchored, HFCE-calibrated temporal disaggregation)"
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="synth_v2/personas.json",
        help="Path to personas JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synth_v2/",
        help="Output directory for v2 transactions",
    )
    parser.add_argument(
        "--hfce",
        type=str,
        default=None,
        help="Path to HFCE quarterly indices config (defaults to "
        "training/config/hfce_quarterly_indices.json)",
    )
    parser.add_argument(
        "--months", type=int, default=42, help="Number of months to generate (default: 42)"
    )
    parser.add_argument("--start-year", type=int, default=2023, help="Start year")
    parser.add_argument("--start-month", type=int, default=1, help="Start month (1-12)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of personas")
    parser.add_argument(
        "--no-anomalies",
        action="store_true",
        help="Disable anomaly injection",
    )

    args = parser.parse_args()

    try:
        validate_hfce_coverage(args.start_year, args.start_month, args.months, args.hfce)
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)

    try:
        personas = load_personas(args.input)
    except PersonaLoadError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.limit:
        personas = personas[: args.limit]
        print(f"Limited to {args.limit} personas")

    all_transactions: list[Transaction] = []
    all_summaries: list[MonthlySummary] = []
    failed_count = 0

    for i, persona in enumerate(personas):
        if (i + 1) % 100 == 0:
            print(f"Processing persona {i + 1:,}/{len(personas):,}...")

        try:
            transactions, summaries = generate_persona_transactions_v2(
                persona,
                start_year=args.start_year,
                start_month=args.start_month,
                num_months=args.months,
                seed=args.seed,
                hfce_path=args.hfce,
                inject_anomalies_flag=not args.no_anomalies,
            )
            all_transactions.extend(transactions)
            all_summaries.extend(summaries)
        except TransactionGenerationErrorV2 as e:
            warnings.warn(f"Failed to generate v2 transactions for persona {i}: {e}", stacklevel=2)
            failed_count += 1
        except Exception as e:
            warnings.warn(f"Unexpected error for persona {i}: {e}", stacklevel=2)
            failed_count += 1

    if failed_count > 0:
        print(f"Warning: {failed_count} personas failed to generate v2 transactions")

    print(f"\nGenerated {len(all_transactions):,} total v2 transactions")
    print(f"Generated {len(all_summaries):,} v2 monthly summaries")

    export_transactions_v2(all_transactions, all_summaries, args.output)

    report = build_synth_v2_report(all_summaries, hfce_path=args.hfce)
    report_path = Path(args.output) / "synthesis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved v2 synthesis report to {report_path}")

    print("\nSynthetic Generation v2 transaction generation complete!")


if __name__ == "__main__":
    main()
