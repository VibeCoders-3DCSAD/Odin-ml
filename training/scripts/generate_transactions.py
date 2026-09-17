"""
Synthetic Transaction Generator for Odin ML Models

Generates 12-month transaction histories for each synthetic persona.
Implements the 20 injection rules from synthetic-injection-rules.md.

Uses 3-dimension PFP labels (Stability x Obligation x Tolerance):
  - Stability: CV < 0.5 = Stable, >= 0.5 = Variable
  - Obligation: ratio > 0.6 = Obligated, <= 0.6 = Flexible
  - Tolerance: runway >= 3 months = Tolerant, < 3 months = At-Risk

Usage:
    python scripts/generate_transactions.py --input <personas_json> --output <output_dir>
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from generate_personas import (
    STABILITY_CV_THRESHOLD,
    OBLIGATION_RATIO_THRESHOLD,
    TOLERANCE_RUNWAY_MONTHS,
    _compute_pfp_label,
)


class TransactionGenerationError(Exception):
    """Raised when transaction generation fails."""
    pass


class PersonaLoadError(Exception):
    """Raised when persona loading fails."""
    pass


@dataclass
class Transaction:
    """A single financial transaction."""
    transaction_id: str
    persona_id: str
    month: int
    year: int
    year_month: str
    date: str
    category: str
    subcategory: str
    amount: float
    transaction_type: str  # income or expense
    description: str
    is_anomalous: bool
    anomaly_type: Optional[str]


@dataclass
class MonthlySummary:
    """Monthly financial summary for a persona."""
    persona_id: str
    month: int
    year: int
    year_month: str
    total_income: float
    total_expenses: float
    food_expense: float
    housing_expense: float
    transport_expense: float
    health_expense: float
    education_expense: float
    other_expense: float
    savings: float
    debt_payment: float
    balance: float
    transaction_count: int
    income_stability_cv: float
    obligation_ratio: float
    runway_months: float
    financial_tolerance: str
    pfp_label: str
    is_anomalous: bool


# Expense categories and their typical distributions
EXPENSE_CATEGORIES = {
    "food": {
        "subcategories": [
            "groceries", "rice", "meat", "fish", "vegetables",
            "fruits", "bread", "cooking_oil", "condiments", "snacks"
        ],
        "frequency": "weekly",
        "typical_range": (2000, 8000),
    },
    "housing": {
        "subcategories": [
            "rent", "electricity", "water", "gas", "maintenance"
        ],
        "frequency": "monthly",
        "typical_range": (3000, 15000),
    },
    "transport": {
        "subcategories": [
            "jeepney", "bus", "train", "tricycle", "taxi", "gasoline", "parking"
        ],
        "frequency": "daily",
        "typical_range": (1000, 5000),
    },
    "health": {
        "subcategories": [
            "medicine", "doctor_visit", "hospital", "vitamins", "dental"
        ],
        "frequency": "monthly",
        "typical_range": (500, 3000),
    },
    "education": {
        "subcategories": [
            "tuition", "books", "supplies", "uniform", "transport"
        ],
        "frequency": "monthly",
        "typical_range": (1000, 8000),
    },
    "other": {
        "subcategories": [
            "clothing", "phone", "internet", "entertainment", "personal_care"
        ],
        "frequency": "monthly",
        "typical_range": (500, 3000),
    },
}

# Income patterns (maps to PersonaArchetype.income_pattern)
INCOME_PATTERNS = {
    "regular": {"frequency": "monthly", "variation": 0.05},
    "irregular": {"frequency": "irregular", "variation": 0.40},
    "volatile": {"frequency": "irregular", "variation": 0.60},
    "project_based": {"frequency": "project_based", "variation": 0.50},
    "commission_variable": {"frequency": "monthly", "variation": 0.35},
}


def load_personas(json_path: str) -> list[dict]:
    """
    Load personas from JSON file with error handling.
    
    Args:
        json_path: Path to personas JSON file
    
    Returns:
        List of persona dicts
    
    Raises:
        PersonaLoadError: If loading fails
    """
    path = Path(json_path)
    
    if not path.exists():
        raise PersonaLoadError(
            f"Personas file not found: {json_path}\n"
            "Run generate_personas.py first to create personas."
        )
    
    if path.stat().st_size == 0:
        raise PersonaLoadError(f"Personas file is empty: {json_path}")
    
    try:
        with open(path, "r") as f:
            personas = json.load(f)
    except json.JSONDecodeError as e:
        raise PersonaLoadError(f"Invalid JSON in personas file: {e}")
    except Exception as e:
        raise PersonaLoadError(f"Failed to load personas: {e}")
    
    if not isinstance(personas, list):
        raise PersonaLoadError(f"Expected list of personas, got {type(personas).__name__}")
    
    if len(personas) == 0:
        raise PersonaLoadError("Personas file contains no personas")
    
    # Validate required fields
    required_fields = ["persona_id", "archetype_id", "monthly_income", "income_pattern"]
    missing_fields = []
    
    for i, persona in enumerate(personas[:5]):  # Check first 5
        for field in required_fields:
            if field not in persona:
                missing_fields.append(f"persona[{i}].{field}")
    
    if missing_fields:
        raise PersonaLoadError(
            f"Personas missing required fields: {missing_fields}\n"
            "Ensure personas were generated with generate_personas.py"
        )
    
    print(f"Loaded {len(personas):,} personas from {path.name}")
    return personas


def validate_persona(persona: dict, index: int) -> list[str]:
    """
    Validate a single persona has all required fields.
    
    Args:
        persona: Persona dict
        index: Index in list (for error messages)
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    required_fields = {
        "persona_id": str,
        "archetype_id": str,
        "archetype_name": str,
        "monthly_income": (int, float),
        "income_cv": (int, float),
        "obligation_ratio": (int, float),
        "savings_rate": (int, float),
        "runway_months": (int, float),
        "household_size": (int, float),
        "income_pattern": str,
        "food_expense": (int, float),
        "housing_expense": (int, float),
        "transport_expense": (int, float),
        "health_expense": (int, float),
        "education_expense": (int, float),
        "other_expense": (int, float),
    }
    
    for field, expected_type in required_fields.items():
        if field not in persona:
            errors.append(f"persona[{index}] missing field: {field}")
        elif not isinstance(persona[field], expected_type):
            errors.append(f"persona[{index}].{field} has wrong type: expected {expected_type}, got {type(persona[field]).__name__}")
    
    # Validate ranges
    if "monthly_income" in persona and persona["monthly_income"] < 0:
        errors.append(f"persona[{index}].monthly_income is negative: {persona['monthly_income']}")
    
    if "obligation_ratio" in persona and not (0 <= persona["obligation_ratio"] <= 1):
        errors.append(f"persona[{index}].obligation_ratio out of range: {persona['obligation_ratio']}")
    
    return errors


def generate_income_transactions(
    persona: dict,
    month: int,
    year: int,
    rng: np.random.Generator,
) -> list[Transaction]:
    """Generate income transactions for a month."""
    transactions = []
    persona_id = persona["persona_id"]
    base_income = persona["monthly_income"]
    income_pattern = persona["income_pattern"]
    income_cv = persona.get("income_cv", 0.1)

    pattern = INCOME_PATTERNS.get(income_pattern, INCOME_PATTERNS["regular"])

    if pattern["frequency"] == "none":
        return transactions

    # Apply monthly variation based on CV
    if pattern["frequency"] == "monthly":
        variation = rng.normal(1.0, income_cv)
        monthly_income = base_income * max(0.5, min(2.0, variation))
    elif pattern["frequency"] == "irregular":
        # Some months may have zero income
        if rng.random() < 0.15:  # 15% chance of zero income month
            monthly_income = 0
        else:
            variation = rng.normal(1.0, pattern["variation"])
            monthly_income = base_income * max(0.2, min(2.5, variation))
    elif pattern["frequency"] == "project_based":
        # Income comes in chunks
        if rng.random() < 0.3:  # 30% chance of project payment
            monthly_income = base_income * rng.uniform(1.5, 3.0)
        elif rng.random() < 0.2:  # 20% chance of partial payment
            monthly_income = base_income * rng.uniform(0.3, 0.7)
        else:
            monthly_income = 0
    elif pattern["frequency"] == "multiple":
        # Multiple income sources
        main_income = base_income * 0.5 * rng.normal(1.0, 0.05)
        side_income = base_income * 0.3 * rng.normal(1.0, 0.4)
        rental_income = base_income * 0.2 * rng.normal(1.0, 0.1)
        monthly_income = main_income + side_income + rental_income
    else:
        monthly_income = base_income

    if monthly_income > 0:
        # Generate income transaction
        day = rng.integers(1, 29)
        date = f"{year}-{month:02d}-{day:02d}"

        transactions.append(Transaction(
            transaction_id=f"txn_{persona_id}_{year}_{month:02d}_income",
            persona_id=persona_id,
            month=month,
            year=year,
            year_month=f"{year}-{month:02d}",
            date=date,
            category="income",
            subcategory="salary",
            amount=round(monthly_income, 2),
            transaction_type="income",
            description="Monthly income",
            is_anomalous=False,
            anomaly_type=None,
        ))

    return transactions


def generate_expense_transactions(
    persona: dict,
    month: int,
    year: int,
    total_income: float,
    rng: np.random.Generator,
) -> list[Transaction]:
    """Generate expense transactions for a month."""
    transactions = []
    persona_id = persona["persona_id"]

    # Get persona's expense ratios
    food_expense = persona.get("food_expense", 0)
    housing_expense = persona.get("housing_expense", 0)
    transport_expense = persona.get("transport_expense", 0)
    health_expense = persona.get("health_expense", 0)
    education_expense = persona.get("education_expense", 0)
    other_expense = persona.get("other_expense", 0)

    # Generate transactions for each category
    for category, amount in [
        ("food", food_expense),
        ("housing", housing_expense),
        ("transport", transport_expense),
        ("health", health_expense),
        ("education", education_expense),
        ("other", other_expense),
    ]:
        if amount <= 0:
            continue

        cat_info = EXPENSE_CATEGORIES[category]
        subcategory = rng.choice(cat_info["subcategories"])

        # Apply monthly variation
        variation = rng.normal(1.0, 0.15)
        adjusted_amount = amount * max(0.5, min(2.0, variation))

        # Generate transaction date
        if cat_info["frequency"] == "weekly":
            # Generate 4 transactions (weekly)
            for week in range(4):
                day = rng.integers(1, 29)
                date = f"{year}-{month:02d}-{day:02d}"
                weekly_amount = adjusted_amount / 4 * rng.uniform(0.7, 1.3)

                transactions.append(Transaction(
                    transaction_id=f"txn_{persona_id}_{year}_{month:02d}_{category}_{week}",
                    persona_id=persona_id,
                    month=month,
                    year=year,
                    year_month=f"{year}-{month:02d}",
                    date=date,
                    category=category,
                    subcategory=subcategory,
                    amount=round(weekly_amount, 2),
                    transaction_type="expense",
                    description=f"{category} expense",
                    is_anomalous=False,
                    anomaly_type=None,
                ))
        else:
            # Single monthly transaction
            day = rng.integers(1, 29)
            date = f"{year}-{month:02d}-{day:02d}"

            transactions.append(Transaction(
                transaction_id=f"txn_{persona_id}_{year}_{month:02d}_{category}",
                persona_id=persona_id,
                month=month,
                year=year,
                year_month=f"{year}-{month:02d}",
                date=date,
                category=category,
                subcategory=subcategory,
                amount=round(adjusted_amount, 2),
                transaction_type="expense",
                description=f"{category} expense",
                is_anomalous=False,
                anomaly_type=None,
            ))

    return transactions


def inject_anomalies(
    transactions: list[Transaction],
    persona: dict,
    rng: np.random.Generator,
) -> list[Transaction]:
    """Inject anomalous transactions (3% target rate)."""
    anomaly_rate = 0.03  # 3% of transactions should be anomalous
    modified_transactions = []

    for txn in transactions:
        # Decide if this transaction should be anomalous
        if rng.random() < anomaly_rate:
            # Choose anomaly type
            anomaly_type = rng.choice([
                "amount_spike",
                "category_mismatch",
                "frequency_change",
                "new_merchant",
            ])

            if anomaly_type == "amount_spike":
                # 3-5x normal amount
                txn.amount *= rng.uniform(3.0, 5.0)
                txn.is_anomalous = True
                txn.anomaly_type = anomaly_type
                txn.description = f"ANOMALY: {txn.description} (amount spike)"

            elif anomaly_type == "category_mismatch":
                # Unusual category for this persona
                unusual_categories = ["luxury", "gambling", "investment"]
                txn.category = rng.choice(unusual_categories)
                txn.subcategory = "anomalous"
                txn.is_anomalous = True
                txn.anomaly_type = anomaly_type
                txn.description = f"ANOMALY: Unusual category {txn.category}"

            elif anomaly_type == "frequency_change":
                # Unusually frequent transactions
                txn.amount *= 0.3  # Smaller amounts but more frequent
                txn.is_anomalous = True
                txn.anomaly_type = anomaly_type
                txn.description = f"ANOMALY: {txn.description} (frequency spike)"

            elif anomaly_type == "new_merchant":
                # Transaction from unknown merchant
                txn.is_anomalous = True
                txn.anomaly_type = anomaly_type
                txn.description = f"ANOMALY: New merchant {txn.subcategory}"

        modified_transactions.append(txn)

    return modified_transactions


def compute_monthly_summary(
    persona: dict,
    transactions: list[Transaction],
    month: int,
    year: int,
    previous_balance: float,
) -> MonthlySummary:
    """Compute monthly summary from transactions, including 3-dimension PFP label."""
    income_txns = [t for t in transactions if t.transaction_type == "income"]
    expense_txns = [t for t in transactions if t.transaction_type == "expense"]

    total_income = sum(t.amount for t in income_txns)
    total_expenses = sum(t.amount for t in expense_txns)

    # Category breakdown
    food = sum(t.amount for t in expense_txns if t.category == "food")
    housing = sum(t.amount for t in expense_txns if t.category == "housing")
    transport = sum(t.amount for t in expense_txns if t.category == "transport")
    health = sum(t.amount for t in expense_txns if t.category == "health")
    education = sum(t.amount for t in expense_txns if t.category == "education")
    other = sum(t.amount for t in expense_txns if t.category == "other")

    # Savings and debt
    savings = max(0, total_income - total_expenses)
    debt_payment = total_expenses * 0.1  # Assume 10% goes to debt

    # Balance
    balance = previous_balance + total_income - total_expenses

    # Income CV: use persona's base CV with small noise (consistent with label generation)
    income_cv = float(np.clip(
        persona.get("income_cv", 0.1) + np.random.normal(0, 0.02), 0.0, 1.5
    ))

    # Obligation ratio: (Essential + Obligatory) / Total
    essential_expenses = food + housing + transport + health + education
    obligatory_expenses = debt_payment  # Debt/loan repayments per MDD §4
    obligation_ratio = (essential_expenses + obligatory_expenses) / total_expenses if total_expenses > 0 else 0

    # Financial Tolerance: emergency runway in months
    # runway = current balance / avg monthly expenses
    avg_monthly_expenses = total_expenses if total_expenses > 0 else 1
    runway_months = balance / avg_monthly_expenses if avg_monthly_expenses > 0 else 0
    runway_months = max(0.0, runway_months)

    # Financial tolerance label
    financial_tolerance = "Tolerant" if runway_months >= TOLERANCE_RUNWAY_MONTHS else "At-Risk"

    # 3-dimension PFP label
    pfp_label = _compute_pfp_label(income_cv, obligation_ratio, runway_months)

    # Check for anomalies
    is_anomalous = any(t.is_anomalous for t in transactions)

    return MonthlySummary(
        persona_id=persona["persona_id"],
        month=month,
        year=year,
        year_month=f"{year}-{month:02d}",
        total_income=round(total_income, 2),
        total_expenses=round(total_expenses, 2),
        food_expense=round(food, 2),
        housing_expense=round(housing, 2),
        transport_expense=round(transport, 2),
        health_expense=round(health, 2),
        education_expense=round(education, 2),
        other_expense=round(other, 2),
        savings=round(savings, 2),
        debt_payment=round(debt_payment, 2),
        balance=round(balance, 2),
        transaction_count=len(transactions),
        income_stability_cv=round(income_cv, 3),
        obligation_ratio=round(obligation_ratio, 3),
        runway_months=round(runway_months, 2),
        financial_tolerance=financial_tolerance,
        pfp_label=pfp_label,
        is_anomalous=is_anomalous,
    )


def generate_persona_transactions(
    persona: dict,
    start_year: int = 2023,
    start_month: int = 1,
    num_months: int = 12,
    seed: int = 42,
) -> tuple[list[Transaction], list[MonthlySummary]]:
    """
    Generate full transaction history for a persona.
    
    Args:
        persona: Persona dict
        start_year: Starting year
        start_month: Starting month (1-12)
        num_months: Number of months to generate
        seed: Random seed
    
    Returns:
        Tuple of (transactions, monthly_summaries)
    
    Raises:
        TransactionGenerationError: If generation fails
    """
    try:
        persona_id = persona["persona_id"]
        rng = np.random.default_rng(seed + hash(persona_id) % 10000)
    except (KeyError, TypeError) as e:
        raise TransactionGenerationError(f"Invalid persona data: {e}")

    all_transactions = []
    all_summaries = []
    previous_balance = 0.0

    for month_offset in range(num_months):
        month = ((start_month - 1 + month_offset) % 12) + 1
        year = start_year + (start_month - 1 + month_offset) // 12

        try:
            # Generate income
            income_txns = generate_income_transactions(persona, month, year, rng)

            # Compute total income for expense generation
            total_income = sum(t.amount for t in income_txns)

            # Generate expenses
            expense_txns = generate_expense_transactions(persona, month, year, total_income, rng)

            # Inject anomalies
            all_month_txns = income_txns + expense_txns
            all_month_txns = inject_anomalies(all_month_txns, persona, rng)

            # Compute monthly summary
            summary = compute_monthly_summary(
                persona, all_month_txns, month, year, previous_balance
            )
            previous_balance = summary.balance

            all_transactions.extend(all_month_txns)
            all_summaries.append(summary)
        except Exception as e:
            warnings.warn(f"Error generating transactions for {persona_id} month {month}/{year}: {e}")
            continue

    return all_transactions, all_summaries


def validate_transactions(
    all_summaries: list[MonthlySummary],
) -> dict:
    """Validate generated transactions."""
    if not all_summaries:
        return {
            "total_months": 0,
            "total_transactions": 0,
            "error": "No summaries to validate",
        }
    
    df = pd.DataFrame([asdict(s) for s in all_summaries])

    validation = {
        "total_months": len(all_summaries),
        "total_transactions": int(df["transaction_count"].sum()),
        "avg_monthly_income": float(df["total_income"].mean()),
        "avg_monthly_expenses": float(df["total_expenses"].mean()),
        "avg_savings_rate": float((df["total_income"] - df["total_expenses"]).mean() / df["total_income"].mean()) if df["total_income"].mean() > 0 else 0,
        "anomaly_rate": float(df["is_anomalous"].mean()),
        "income_cv_stats": {
            "mean": float(df["income_stability_cv"].mean()),
            "std": float(df["income_stability_cv"].std()),
        },
        "obligation_ratio_stats": {
            "mean": float(df["obligation_ratio"].mean()),
            "std": float(df["obligation_ratio"].std()),
        },
        "runway_months_stats": {
            "mean": float(df["runway_months"].mean()),
            "std": float(df["runway_months"].std()),
        },
        "pfp_label_counts": df["pfp_label"].value_counts().to_dict(),
    }

    return validation


def export_transactions(
    all_transactions: list[Transaction],
    all_summaries: list[MonthlySummary],
    output_dir: str,
) -> None:
    """Export transactions and summaries."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export transactions as Parquet
    parquet_path = output_path / "transactions.parquet"
    txn_df = pd.DataFrame([asdict(t) for t in all_transactions])
    txn_df.to_parquet(parquet_path, index=False)
    print(f"Exported {len(all_transactions):,} transactions to {parquet_path}")

    # Export summaries as Parquet
    summary_path = output_path / "monthly_summaries.parquet"
    summary_df = pd.DataFrame([asdict(s) for s in all_summaries])
    summary_df.to_parquet(summary_path, index=False)
    print(f"Exported {len(all_summaries):,} monthly summaries to {summary_path}")

    # Export as JSON for easy loading (first 1000 for preview)
    json_path = output_path / "transactions.json"
    preview_count = min(1000, len(all_transactions))
    with open(json_path, "w") as f:
        json.dump([asdict(t) for t in all_transactions[:preview_count]], f, indent=2)
    print(f"Exported transaction preview ({preview_count} rows) to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transactions")
    parser.add_argument(
        "--input",
        type=str,
        default="synth/personas.json",
        help="Path to personas JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synth/",
        help="Output directory for transactions",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Number of months to generate",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2023,
        help="Start year for transactions",
    )
    parser.add_argument(
        "--start-month",
        type=int,
        default=1,
        help="Start month for transactions (1-12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of personas to process",
    )

    args = parser.parse_args()

    # Load personas
    try:
        personas = load_personas(args.input)
    except PersonaLoadError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.limit:
        personas = personas[:args.limit]
        print(f"Limited to {args.limit} personas")

    # Validate personas
    print("Validating personas...")
    validation_errors = []
    for i, persona in enumerate(personas):
        errors = validate_persona(persona, i)
        validation_errors.extend(errors)
    
    if validation_errors:
        print(f"Warning: Found {len(validation_errors)} validation errors:")
        for error in validation_errors[:10]:  # Show first 10
            print(f"  - {error}")
        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more")

    # Generate transactions for each persona
    all_transactions = []
    all_summaries = []
    failed_count = 0

    for i, persona in enumerate(personas):
        if (i + 1) % 100 == 0:
            print(f"Processing persona {i + 1:,}/{len(personas):,}...")

        try:
            transactions, summaries = generate_persona_transactions(
                persona,
                start_year=args.start_year,
                start_month=args.start_month,
                num_months=args.months,
                seed=args.seed,
            )

            all_transactions.extend(transactions)
            all_summaries.extend(summaries)
        except TransactionGenerationError as e:
            warnings.warn(f"Failed to generate transactions for persona {i}: {e}")
            failed_count += 1
        except Exception as e:
            warnings.warn(f"Unexpected error for persona {i}: {e}")
            failed_count += 1

    if failed_count > 0:
        print(f"Warning: {failed_count} personas failed to generate transactions")

    print(f"\nGenerated {len(all_transactions):,} total transactions")
    print(f"Generated {len(all_summaries):,} monthly summaries")

    # Validate
    validation = validate_transactions(all_summaries)
    print("\nValidation Results:")
    print(json.dumps(validation, indent=2))

    # Export
    export_transactions(all_transactions, all_summaries, args.output)

    # Save validation results
    output_path = Path(args.output)
    validation_path = output_path / "validation.json"
    with open(validation_path, "w") as f:
        json.dump(validation, f, indent=2)
    print(f"Saved validation results to {validation_path}")

    print("\nTransaction generation complete!")


if __name__ == "__main__":
    main()
