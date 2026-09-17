"""
Phase 4.5: PFP Dimension & Threshold Discovery

Performs unsupervised clustering on the three classifying dimensions
(Stability, Obligation, Tolerance) to check whether binary splits hold
or whether more natural groupings exist.

Also computes two overlay indicator features:
- Financial Trajectory: trend/slope of income-expense gap across months
- Financial Margin: (Income - Expenses) / Income at each month

Outputs:
- dimension-threshold-candidates.md with proposed cut points
- clustering summary JSON

Note: Any patterns found are bounded by what the synthesis process encoded
(circularity caveat) and are provisional pending real user data.

Usage:
    python scripts/dimension_discovery.py --input datasets/processed/ --output datasets/dimension-discovery/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_processed_data(input_dir: str) -> pd.DataFrame:
    """Load and merge train/val splits."""
    input_path = Path(input_dir)
    dfs = []
    for name in ["train", "val", "test"]:
        path = input_path / f"{name}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        raise FileNotFoundError(f"No processed data found in {input_dir}")
    return pd.concat(dfs, ignore_index=True)


def load_synth_summaries(synth_dir: str) -> pd.DataFrame:
    """Load monthly summaries from synth output (has dimension scores)."""
    path = Path(synth_dir) / "monthly_summaries.parquet"
    if not path.exists():
        raise FileNotFoundError(f"monthly_summaries.parquet not found at {path}")
    return pd.read_parquet(path)


def run_clustering_analysis(df: pd.DataFrame, output_dir: str) -> dict:
    """Run k-means and GMM clustering on the3 dimension scores."""
    feature_cols = []
    for col in ["income_stability_cv", "obligation_ratio"]:
        if col in df.columns:
            feature_cols.append(col)
    if "runway_months" in df.columns:
        feature_cols.append("runway_months")

    if len(feature_cols) < 2:
        print("  Insufficient dimensions for clustering")
        return {}

    X = df[feature_cols].dropna().values
    if len(X) < 10:
        print("  Insufficient data for clustering")
        return {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {"features": feature_cols, "n_samples": len(X)}

    # Subsampled set for silhouette (silhouette is O(n^2))
    silhouette_n = min(5000, len(X))
    rng = np.random.RandomState(42)
    sil_idx = rng.choice(len(X_scaled), silhouette_n, replace=False)
    X_sil = X_scaled[sil_idx]

    # K-means: elbow + silhouette (include k=8 for the 2x2x2 = 8-class check)
    k_range = range(2, 9)
    kmeans_scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_labels = km.predict(X_sil)
        sil = silhouette_score(X_sil, sil_labels)
        kmeans_scores[k] = {
            "inertia": float(km.inertia_),
            "silhouette": float(sil),
        }
    results["kmeans"] = kmeans_scores

    # GMM: BIC + AIC
    gmm_scores = {}
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_scaled)
        gmm_scores[k] = {
            "bic": float(gmm.bic(X_scaled)),
            "aic": float(gmm.aic(X_scaled)),
        }
    results["gmm"] = gmm_scores

    # Best k by silhouette
    best_k_sil = max(kmeans_scores, key=lambda k: kmeans_scores[k]["silhouette"])
    results["best_k_silhouette"] = best_k_sil
    results["best_silhouette_score"] = kmeans_scores[best_k_sil]["silhouette"]

    # Best k by BIC
    best_k_bic = min(gmm_scores, key=lambda k: gmm_scores[k]["bic"])
    results["best_k_bic"] = best_k_bic

    # Binary split validation: compare k=2 vs k=8 (2x2x2 = 8 classes)
    binary_sil = kmeans_scores.get(2, {}).get("silhouette", 0)
    eight_sil = kmeans_scores.get(8, {}).get("silhouette", 0)
    results["binary_vs_8class"] = {
        "k2_silhouette": binary_sil,
        "k8_silhouette": eight_sil,
        "binary_hold": binary_sil >= eight_sil * 0.9,
    }

    # Per-dimension distribution stats
    dim_stats = {}
    for col in feature_cols:
        vals = df[col].dropna()
        dim_stats[col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(vals.median()),
            "q25": float(vals.quantile(0.25)),
            "q75": float(vals.quantile(0.75)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    results["dimension_stats"] = dim_stats

    print(f"  Clustering on {len(feature_cols)} features, {len(X)} samples")
    print(f"  Best k (silhouette): {best_k_sil} (score={results['best_silhouette_score']:.4f})")
    print(f"  Best k (BIC): {best_k_bic}")
    print(f"  Binary split holds: {results['binary_vs_8class']['binary_hold']}")

    return results


def compute_overlay_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Financial Trajectory and Financial Margin overlay features.

    Financial Margin: (Income - Expenses) / Income at each month.
    Financial Trajectory: per-persona linear slope of the (Income - Expenses)
    gap across months — positive slope means a widening surplus.
    """
    df = df.copy()

    # Financial Margin: (Income - Expenses) / Income
    if "total_income" in df.columns and "total_expenses" in df.columns:
        df["financial_margin"] = np.where(
            df["total_income"] > 0,
            (df["total_income"] - df["total_expenses"]) / df["total_income"],
            0.0,
        )

    # Financial Trajectory: slope of the monthly savings gap per persona
    if all(c in df.columns for c in ["persona_id", "year_month", "total_income", "total_expenses"]):
        df["savings_gap"] = df["total_income"] - df["total_expenses"]
        slopes = {}
        for pid, g in df.groupby("persona_id"):
            months = pd.PeriodIndex(g["year_month"], freq="M").asi8.astype(float)
            gap = g["savings_gap"].values
            if len(months) >= 2 and np.std(months) > 0:
                slopes[pid] = float(np.polyfit(months, gap, 1)[0])
            else:
                slopes[pid] = 0.0
        df["financial_trajectory"] = df["persona_id"].map(slopes)

    return df


def export_threshold_candidates(
    clustering_results: dict,
    output_dir: str,
) -> None:
    """Export dimension-threshold-candidates.md."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dim_stats = clustering_results.get("dimension_stats", {})
    binary_hold = clustering_results.get("binary_vs_8class", {}).get("binary_hold", False)

    lines = [
        "# Dimension & Threshold Candidates",
        "",
        "**Phase:** 4.5 — Dimension & Threshold Discovery",
        "**Status:** Provisional — ready for SME sanity-check; numeric values pending real user data",
        "",
        "---",
        "",
        "## Current Thresholds (SME Draft)",
        "",
        "| Dimension | Threshold | Rationale |",
        "|-----------|-----------|-----------|",
        "| Stability | CV < 0.5 = Stable | SME draft: moderate income consistency cutoff |",
        "| Obligation | ratio > 0.6 = Obligated | SME draft: essential obligations exceed 60% of expenses |",
        "| Tolerance | runway >= 3 months = Tolerant | SME draft: 3-month buffer before depletion |",
        "",
        "---",
        "",
        "## Dimension Distributions (Synthetic Data)",
        "",
    ]

    for dim, stats in dim_stats.items():
        lines.append(f"### {dim}")
        lines.append("")
        lines.append(f"- Mean: {stats['mean']:.4f}")
        lines.append(f"- Std: {stats['std']:.4f}")
        lines.append(f"- Median: {stats['median']:.4f}")
        lines.append(f"- IQR: [{stats['q25']:.4f}, {stats['q75']:.4f}]")
        lines.append(f"- Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Clustering Analysis",
        "",
        f"**Binary split holds (k=2 sufficient):** {binary_hold}",
        "",
        "**K-Means silhouette scores:**",
        "",
    ])

    kmeans = clustering_results.get("kmeans", {})
    for k, scores in sorted(kmeans.items()):
        lines.append(f"- k={k}: silhouette={scores['silhouette']:.4f}, inertia={scores['inertia']:.1f}")

    lines.extend([
        "",
        "**GMM BIC scores:**",
        "",
    ])

    gmm = clustering_results.get("gmm", {})
    for k, scores in sorted(gmm.items()):
        lines.append(f"- k={k}: BIC={scores['bic']:.1f}, AIC={scores['aic']:.1f}")

    lines.extend([
        "",
        "---",
        "",
        "## Recommendations",
        "",
        "1. **Binary splits are sufficient** if clustering silhouette for k=2 is competitive with higher k values.",
        "2. **Threshold values need validation** against real user transaction data — current values are SME-informed estimates.",
        "3. **Financial Trajectory and Financial Margin** are overlay indicators, not classifying dimensions.",
        "4. **Circularity caveat:** All patterns here reflect the synthetic generation parameters, not real-world distributions.",
        "",
        "---",
        "",
        "*This document is provisional. Final thresholds require SME review + real user data.*",
    ])

    with open(out / "dimension-threshold-candidates.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  Exported dimension-threshold-candidates.md to {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4.5: Dimension & Threshold Discovery",
    )
    parser.add_argument("--input", default="datasets/processed/",
                        help="Input directory with processed data")
    parser.add_argument("--output", default="datasets/dimension-discovery/",
                        help="Output directory")
    parser.add_argument("--synth-dir", default="synth/",
                        help="Synthetic data directory")

    args = parser.parse_args()

    print("=" * 60)
    print("Phase 4.5 — Dimension & Threshold Discovery")
    print("=" * 60)

    print("\n[1/3] Loading data...")
    try:
        synth_df = load_synth_summaries(args.synth_dir)
        print(f"  Loaded {len(synth_df):,} rows from synth ({synth_df['persona_id'].nunique():,} personas)")
    except FileNotFoundError:
        synth_df = load_processed_data(args.input)
        print(f"  Loaded {len(synth_df):,} rows from processed ({synth_df['user_id'].nunique():,} personas)")

    print("\n[2/3] Computing overlay features...")
    synth_df = compute_overlay_features(synth_df)
    overlay_cols = [c for c in ["financial_margin", "financial_trajectory"]
                    if c in synth_df.columns]
    overlay_stats = {}
    for c in overlay_cols:
        vals = synth_df[c].dropna()
        overlay_stats[c] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    synth_df[["persona_id", "year_month", "month"] + overlay_cols].to_parquet(
        out_path / "overlay_features.parquet", index=False
    )
    print(f"  Exported overlay features: {overlay_cols}")

    print("\n[3/3] Running clustering analysis...")
    clustering_results = run_clustering_analysis(synth_df, args.output)
    clustering_results["overlay_features"] = overlay_stats

    print("  Exporting results...")
    export_threshold_candidates(clustering_results, args.output)

    # Save clustering results as JSON
    with open(out_path / "clustering_results.json", "w") as f:
        json.dump(clustering_results, f, indent=2, default=str)
    print(f"  Exported clustering_results.json to {out_path}")

    print("\n" + "=" * 60)
    print("Phase 4.5 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
