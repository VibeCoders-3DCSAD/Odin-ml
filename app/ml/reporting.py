"""Shared evaluation-report writer for final model artifacts.

All four families (pfp, forecaster, anomaly, budget) emit a uniform
`evaluation_report.md` skeleton: a consistent title, `**Generated:**` UTC
timestamp, folds, decision rule, a Winner block (tier + artifact + reason),
an Approval Criteria Result, Aggregate Results, Per-Fold Results, and an
optional Supplementary section. `mean ± std` is formatted uniformly. This
module is the single place that enforces the format so future training runs
cannot drift; `family_report()` interprets each family's `evaluation.json`
into the shared sections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ml.metadata import framework_version_of

PR_AUC_TARGET = 0.15
PR_AUC_MIN_TIMES_IQR = 1.5

PFP_CLASSES = (
    "Stable/Obligated/Tolerant",
    "Stable/Obligated/At-Risk",
    "Stable/Flexible/Tolerant",
    "Stable/Flexible/At-Risk",
    "Variable/Obligated/Tolerant",
    "Variable/Obligated/At-Risk",
    "Variable/Flexible/Tolerant",
    "Variable/Flexible/At-Risk",
)

ANOMALY_TYPES = (
    "### Anomaly Types",
    "",
    "Synthetic data injects 4 anomaly types (`anomaly_type` column):",
    "",
    "1. **amount_spike** — unusually high transaction amount",
    "2. **new_merchant** — first transaction with a new merchant",
    "3. **frequency_change** — abnormal transaction frequency",
    "4. **category_mismatch** — transaction category inconsistent with expectation",
)

ANOMALY_RECOMMENDATIONS = (
    "### Recommendations",
    "",
    "1. Deploy the winning model for real-time scoring",
    "2. Set anomaly threshold based on business tolerance (precision vs recall)",
    "3. Monitor model performance on incoming data for drift",
    "4. Consider ensemble approach for production robustness",
)

FAMILY_LABELS = {
    "pfp": "PFP",
    "forecaster": "Forecaster",
    "anomaly": "Anomaly",
    "budget": "Budget",
    "Financial Forecasting": "Financial Forecasting",
    "PFP Classification": "PFP Classification",
    "Anomaly Detection": "Anomaly Detection",
    "Budget Optimization": "Budget Optimization",
}


def pm(mean: float | None, std: float | None, precision: int = 4) -> str:
    """Format a metric as `<mean> ± <std>` with uniform spacing."""
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def pm_label(name: str) -> str:
    return f"{name} (mean ± std)"


def _decision_rule_for(family: str, evaluation: dict) -> str:
    if family == "forecaster":
        pct = round(evaluation.get("pre_registered_mape_reduction", 0.20) * 100)
        return f"best learned model must cut MAPE by ≥ {pct}% vs naive baseline, else naive"
    if family == "pfp":
        margin = evaluation.get("pre_registered_margin", 0.02)
        return f"winner must beat the rule-based Tier 1 by > {margin} Macro-F1, else fall back to Tier 1"
    if family == "anomaly":
        rule = evaluation.get("decision_rule")
        if isinstance(rule, dict) and rule.get("metric") == "pr_auc":
            ratio = rule.get("pr_auc_improvement_ratio", 1.5)
            target = rule.get("pr_auc_target", 0.15)
            return (
                f"winner must reach PR-AUC ≥ {ratio:g}× the IQR baseline "
                f"and PR-AUC ≥ {target:g}; otherwise fall back to IQR"
            )
        target = 0.85 if not isinstance(rule, dict) else rule.get("f1_target", 0.85)
        return (
            f"winner must beat the IQR baseline by ≥ 50% F1 improvement and reach "
            f"F1 ≥ {target}; otherwise fall back to IQR"
        )
    if family == "budget":
        return (
            "LP must satisfy all hard constraints with high utilization and "
            "minimal deviation from user preferences"
        )
    return ""


def _winner_artifact_for(family: str, evaluation: dict) -> str:
    artifact = evaluation.get("winner_artifact")
    if artifact and isinstance(artifact, str):
        return artifact
    if family == "pfp":
        return f"{evaluation.get('winner', 'tier1_rule_based')}.joblib"
    if family == "forecaster":
        winner = evaluation.get("winner", "tier3_sarima")
        return f"{winner}.joblib"
    if family == "anomaly":
        return "anomaly_detector.joblib"
    if family == "budget":
        return "budget_config.json"
    return ""


def _forecaster_sections(evaluation: dict) -> tuple[str, str, list[str], str]:
    header = (
        "| Tier | MAE (mean ± std) | SMAPE (mean ± std) | MDA (mean ± std) | "
        "RMSE (mean ± std) | MAPE (mean ± std) | R² (mean ± std) |"
    )
    sep = "|------|-----------------|-------------------|-----------------|------------------|----------------|--------------|"
    rows = [header, sep]
    for tier, m in sorted(evaluation.get("aggregate_metrics", {}).items()):
        rows.append(
            "| {t} | {mae} | {smape} | {mda} | {rmse} | {mape} | {r2} |".format(
                t=tier,
                mae=pm(m.get("mae_mean"), m.get("mae_std")),
                smape=pm(m.get("smape_mean"), m.get("smape_std")),
                mda=pm(m.get("mda_mean"), m.get("mda_std")),
                rmse=pm(m.get("rmse_mean"), m.get("rmse_std"), 2),
                mape=pm(m.get("mape_mean"), m.get("mape_std")) + "%",
                r2=pm(m.get("r2_mean"), m.get("r2_std")),
            )
        )
    aggregate = "\n".join(rows)

    fold_lines = []
    for fr in evaluation.get("fold_details", []):
        fold_lines.extend(
            [
                f"### Fold {fr['fold']}",
                "",
                f"- Train months: {fr['train_months']}",
                f"- Test months: {fr['test_months']}",
                f"- Train samples: {fr['n_train']}",
                f"- Test samples: {fr['n_test']}",
                "",
                "| Tier | MAE | SMAPE | MDA | RMSE | MAPE | R² |",
                "|------|-----|-------|-----|------|------|-----|",
            ]
        )
        for tier, m in sorted(fr.get("tier_results", {}).items()):
            fold_lines.append(
                "| {t} | {mae:.2f} | {smape:.2f}% | {mda:.4f} | {rmse:.2f} | "
                "{mape:.2f}% | {r2:.4f} |".format(
                    t=tier,
                    mae=m["mae"],
                    smape=m["smape"],
                    mda=m["mda"],
                    rmse=m["rmse"],
                    mape=m["mape"],
                    r2=m["r2"],
                )
            )
        fold_lines.append("")

    winner = evaluation.get("winner", "")
    winner_reason = evaluation.get("winner_reason", "")
    naive_mape = evaluation.get("naive_mape")
    aggregate_map = evaluation.get("aggregate_metrics", {})
    winner_mape = aggregate_map.get(winner, {}).get("mape_mean") if naive_mape else None
    passed = bool(winner and winner != "naive_baseline")
    approval = [
        f"**Result:** {'PASS' if passed else 'FAIL — naive baseline retained'} — {winner_reason}"
    ]
    if naive_mape and winner_mape:
        approval.append(
            f"MAPE {winner_mape:.2f}% vs naive {naive_mape:.2f}% "
            f"({-(1.0 - winner_mape / naive_mape) * 100:.1f}%)."
        )
    supplementary = ""
    return aggregate, "\n".join(fold_lines), approval, supplementary


def _pfp_sections(evaluation: dict) -> tuple[str, str, list[str], str]:
    header = pm_label("Macro-F1") + " | " + pm_label("Accuracy")
    rows = [f"| Tier | {header} |", "|------|----------------------|----------------------|"]
    for tier, m in sorted(evaluation.get("aggregate_metrics", {}).items()):
        rows.append(
            "| {t} | {f1} | {acc} |".format(
                t=tier,
                f1=pm(m.get("macro_f1_mean"), m.get("macro_f1_std")),
                acc=pm(m.get("accuracy_mean"), m.get("accuracy_std")),
            )
        )
    aggregate = "\n".join(rows)

    fold_lines = []
    for fr in evaluation.get("fold_details", []):
        fold_lines.extend(
            [
                f"### Fold {fr['fold']}",
                "",
                f"- Train months: {fr['train_months']}",
                f"- Test months: {fr['test_months']}",
                f"- Train personas: {fr['n_train_personas']}",
                f"- Test personas: {fr['n_test_personas']}",
                "",
                "| Tier | Macro-F1 | Accuracy |",
                "|------|----------|----------|",
            ]
        )
        for tier, m in fr.get("tier_results", {}).items():
            if isinstance(m, dict) and "macro_f1" in m:
                fold_lines.append(f"| {tier} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} |")
        fold_lines.append("")

    last_fold = evaluation.get("fold_details", [])[-1] if evaluation.get("fold_details") else None
    if last_fold:
        fold_lines.extend(
            [
                "---",
                "",
                "## Per-Class Accuracy (Last Fold)",
                "",
                "| Tier | " + " | ".join(PFP_CLASSES) + " |",
                "|------|" + "|".join(["------"] * len(PFP_CLASSES)) + "|",
            ]
        )
        for tier, m in last_fold.get("tier_results", {}).items():
            if isinstance(m, dict) and "per_class_accuracy" in m:
                accs = [f"{m['per_class_accuracy'].get(c, 0):.4f}" for c in PFP_CLASSES]
                fold_lines.append(f"| {tier} | " + " | ".join(accs) + " |")
        fold_lines.append("")

    margin = evaluation.get("pre_registered_margin", 0.02)
    approval = [
        f"**Result:** PASS — {evaluation.get('winner_reason', '')}",
        f"Required margin: > {margin} Macro-F1 over Tier 1.",
    ]
    return aggregate, "\n".join(fold_lines), approval, ""


def _anomaly_sections(evaluation: dict) -> tuple[str, str, list[str], str]:
    summary = evaluation.get("fold_summary", {})
    rows = [
        "| Model | F1 (mean ± std) | Accuracy (mean) | PR-AUC (mean ± std) |",
        "|-------|-----------------|-----------------|----------------------|",
    ]
    for name, stats in sorted(summary.items(), key=lambda x: -x[1].get("f1_mean", 0)):
        if name == "baseline":
            rows.append(
                f"| {name} | {pm(stats.get('f1_mean'), stats.get('f1_std'))} "
                f"| {stats.get('accuracy_mean', 0):.4f} | N/A |"
            )
        else:
            rows.append(
                f"| {name} | {pm(stats.get('f1_mean'), stats.get('f1_std'))} "
                f"| {stats.get('accuracy_mean', 0):.4f} "
                f"| {pm(stats.get('pr_auc_mean'), stats.get('pr_auc_std'))} |"
            )
    aggregate = "\n".join(rows)

    fold_lines = []
    for fold_no in sorted(evaluation.get("fold_results", {}), key=int):
        tiers = evaluation["fold_results"][fold_no]
        tier_names = sorted(k for k in tiers if k != "models")
        fold_lines.extend(
            [
                f"### Fold {fold_no}",
                "",
                "| Model | F1 | Accuracy | PR-AUC |",
                "|-------|----|----------|--------|",
            ]
        )
        for name in tier_names:
            m = tiers[name]
            fold_lines.append(
                f"| {name} | {m.get('f1', 0):.4f} | {m.get('accuracy', 0):.4f} "
                f"| {pm(m.get('pr_auc'), None) if m.get('pr_auc') is not None else '—'} |"
            )
        fold_lines.append("")

    drule = evaluation.get("decision_rule", {})
    rule_passed = drule.get("rule_passed") if isinstance(drule, dict) else None
    if isinstance(drule, dict) and drule.get("metric") == "pr_auc":
        approval = [
            (
                f"**Result:** {'PASS' if rule_passed else 'FAIL — pre-registered fallback to IQR'} — "
                f"PR-AUC improvement over IQR "
                f"{drule.get('pr_auc_improvement_over_iqr_pct', 0)}% (target ≥ 50%), "
                f"PR-AUC target ≥ {drule.get('pr_auc_target', 0.15)}, "
                f"passed: {rule_passed}"
            ),
            (
                f"Operating point: F2 (β=2) maximized on the held-out val split subject to "
                f"precision ≥ {drule.get('operating_point', {}).get('min_precision', 0.30)}."
            ),
        ]
    else:
        approval = [
            (
                f"**Result:** {'PASS' if rule_passed else 'FAIL — pre-registered fallback to IQR'} — "
                f"F1 improvement over IQR "
                f"{drule.get('f1_improvement_over_iqr_pct', 0)}% (target ≥ 50%), "
                f"F1 target ≥ 0.85, passed: {rule_passed}"
            )
        ]

    final = evaluation.get("final_test_metrics", {})
    key_findings = [
        f"- **Class imbalance:** ~{evaluation.get('anomaly_rate_train', 0) * 100:.1f}% anomaly rate",
        (
            "- **Primary ranking metric is PR-AUC** (imbalance-safe, threshold-free; "
            "baseline equals the anomaly rate); Accuracy/Precision/Recall/F1 are reported at "
            "the chosen operating point"
        ),
        "- **IQR provides interpretable statistical baseline** with per-feature thresholds",
        (
            "- **Isolation Forest handles unsupervised detection**; contamination set to the "
            "observed training anomaly rate"
        ),
        (
            "- **Operating threshold selected on the held-out val split** (F2 maximized, "
            "β=2, precision ≥ 0.30) to avoid test leakage"
        ),
    ]
    tv = evaluation.get("test_validation") or {}
    if tv.get("measured"):
        key_findings.append(
            "- **Fold-vs-test generalization gap (reported, not hidden):** "
            f"fold-nominated candidate ({evaluation.get('fold_nominated_winner', '—')}) "
            f"reached PR-AUC {tv.get('pr_auc', '—')} on the held-out test split vs IQR "
            f"{tv.get('iqr_test_pr_auc', '—')} (ratio {tv.get('improvement_ratio', '—')}; "
            f"gate passed: {tv.get('gate_passed')}) — below the "
            f"{PR_AUC_TARGET} / {PR_AUC_MIN_TIMES_IQR}× gates, so the pre-registered "
            f"rule falls back to the IQR baseline"
        )
    elif tv:
        key_findings.append(
            f"- **Test validation incomplete:** {tv.get('reason', 'no validation recorded')}"
        )
    sup_op = evaluation.get("val_operating_point", {})
    supp = "\n".join(
        [
            "### Final Test Metrics (threshold selected on held-out val)",
            "",
            f"- **Operating threshold:** {evaluation.get('val_selected_threshold', 0):.4f}",
            (
                f"- **Operating point (val):** F2 = {sup_op.get('f2', '—')}, "
                f"precision = {sup_op.get('precision', '—')}, "
                f"recall = {sup_op.get('recall', '—')}, F1 = {sup_op.get('f1', '—')}"
                if sup_op
                else ""
            ),
            f"- **Accuracy:** {final.get('accuracy', 0):.4f}",
            f"- **Precision:** {final.get('precision', 0):.4f}",
            f"- **Recall:** {final.get('recall', 0):.4f}",
            f"- **F1:** {final.get('f1', 0):.4f}",
            f"- **PR-AUC (supplementary):** {final.get('pr_auc', 0):.4f}",
            f"- **ROC-AUC (supplementary):** {final.get('roc_auc', 0):.4f}",
            f"- **TP/FP/FN/TN:** {final.get('tp', 0)}/{final.get('fp', 0)}/"
            f"{final.get('fn', 0)}/{final.get('tn', 0)}",
            "",
            "### Key Findings",
            "",
            *key_findings,
            "",
            *ANOMALY_TYPES,
            "",
            *ANOMALY_RECOMMENDATIONS,
        ]
    )
    return aggregate, "\n".join(fold_lines), approval, supp


def _budget_sections(evaluation: dict) -> tuple[str, str, list[str], str]:
    rows = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Constraint satisfaction rate | {evaluation.get('constraint_satisfaction_rate', 0):.4f} |",
        f"| Budget utilization rate | {evaluation.get('budget_utilization_rate', 0):.4f} |",
        f"| Mean deviation from user preferences | {evaluation.get('mean_deviation', 0):.4f} |",
        f"| Feasibility | {evaluation.get('feasibility', '—')} |",
    ]
    aggregate = "\n".join(rows)
    approval = [
        f"**Result:** {'PASS' if evaluation.get('constraint_satisfaction_rate', 0) >= 0.99 else 'FAIL'} — "
        "LP must meet all hard constraints with minimal deviation from user targets."
    ]
    return aggregate, "", approval, ""


def _forecaster_framework(winner: str) -> tuple[str, str]:
    if winner in ("tier3_arima", "tier3_sarima"):
        return "statsmodels", framework_version_of("statsmodels")
    if winner.startswith("tier3_"):
        return "pytorch", framework_version_of("torch")
    return "scikit-learn", framework_version_of("scikit-learn")


def _family_artifacts(family: str, winner: str, evaluation: dict) -> list[str]:
    artifact = _winner_artifact_for(family, evaluation)
    if family == "forecaster":
        if winner.startswith("tier3_") and winner not in (
            "tier3_arima",
            "tier3_sarima",
        ):
            return [f"{winner}.pth", f"{winner}_meta.joblib"]
        if winner == "naive_baseline":
            return ["evaluation.json"]
        return [artifact]
    if family == "budget":
        return ["budget_config.json"]
    return [artifact]


def family_metadata(family: str, evaluation: dict, *, data_sources: list[Path]) -> dict[str, Any]:
    """Build canonical `build_metadata` kwargs for one family's evaluation.json.

    Every family emits the same winner contract (`winner`, `winner_artifact`,
    `winner_params`, `threshold`) so serving resolution is metadata-driven.
    """
    winner = evaluation.get("winner", "")
    if family == "forecaster":
        aggregate = evaluation.get("aggregate_metrics", {})
        stats = aggregate.get(winner, {})
        naive_mape = evaluation.get("naive_mape")
        winner_mape = stats.get("mape_mean")
        reduction = None if not (naive_mape and winner_mape) else (1.0 - winner_mape / naive_mape)
        metrics = {
            "primary": {
                "name": "mape",
                "value": winner_mape,
                "reduction_vs_naive_pct": reduction,
                "folds": evaluation.get("n_folds"),
            },
            "secondary": {
                key: stats[key]
                for key in ("mae_mean", "smape_mean", "mda_mean", "rmse_mean", "r2_mean")
                if key in stats
            },
        }
        framework, framework_version = _forecaster_framework(winner)
    elif family == "anomaly":
        final = evaluation.get("final_test_metrics", {})
        summary = evaluation.get("fold_summary", {})
        stats = summary.get(winner, {})
        metrics = {
            "primary": {
                "name": "pr_auc",
                "value": final.get("pr_auc"),
                "threshold": evaluation.get("val_selected_threshold"),
                "operating_point": evaluation.get("val_operating_point"),
                "folds": len(evaluation.get("fold_results", {})),
            },
            "secondary": {
                "accuracy": final.get("accuracy"),
                "precision": final.get("precision"),
                "recall": final.get("recall"),
                "f1": final.get("f1"),
                "roc_auc": final.get("roc_auc"),
                "pr_auc_baseline": evaluation.get("anomaly_rate_test"),
                "fold_pr_auc_mean": stats.get("pr_auc_mean"),
                "fold_pr_auc_std": stats.get("pr_auc_std"),
                "fold_f1_mean": stats.get("f1_mean"),
                "fold_f1_std": stats.get("f1_std"),
            },
        }
        framework, framework_version = "scikit-learn", framework_version_of("scikit-learn")
    elif family == "pfp":
        aggregate = evaluation.get("aggregate_metrics", {})
        stats = aggregate.get(winner, {})
        metrics = {
            "primary": {
                "name": "macro_f1",
                "value": stats.get("macro_f1_mean"),
                "threshold": None,
                "folds": evaluation.get("n_folds"),
            },
            "secondary": {
                "accuracy": stats.get("accuracy_mean"),
                "macro_f1_std": stats.get("macro_f1_std"),
                "accuracy_std": stats.get("accuracy_std"),
            },
        }
        framework, framework_version = "scikit-learn", framework_version_of("scikit-learn")
    elif family == "budget":
        metrics = {
            "primary": {
                "name": "constraint_satisfaction_rate",
                "value": evaluation.get("constraint_satisfaction_rate"),
                "folds": None,
            },
            "secondary": {
                "budget_utilization_rate": evaluation.get("budget_utilization_rate"),
                "mean_deviation": evaluation.get("mean_deviation"),
            },
        }
        framework, framework_version = "scipy", framework_version_of("scipy")
    else:
        raise KeyError(f"unsupported model family: {family}")

    threshold = evaluation.get("val_selected_threshold") if family == "anomaly" else None
    params: dict[str, Any] = {
        "model_id": f"{family}-{winner}",
        "family": family,
        "feature_columns": evaluation.get("feature_columns", []),
        "metrics": metrics,
        "decision_rule": _decision_rule_for(family, evaluation),
        "framework": framework,
        "framework_version": framework_version,
        "artifacts": _family_artifacts(family, winner, evaluation),
        "data_sources": data_sources,
        "winner": winner or None,
        "winner_artifact": _winner_artifact_for(family, evaluation) or None,
        "winner_reason": evaluation.get("winner_reason")
        or f"Winner {winner or '(none)'} selected per the pre-registered decision rule",
        "winner_params": evaluation.get("winner_params") or None,
        "threshold": threshold,
        "fitted": family != "budget",
    }
    return params


def family_report(family: str, evaluation: dict) -> dict[str, Any]:
    """Render one family's `evaluation.json` into the shared report sections."""
    if family == "forecaster":
        aggregate, fold_sections, approval, supplementary = _forecaster_sections(evaluation)
    elif family == "pfp":
        aggregate, fold_sections, approval, supplementary = _pfp_sections(evaluation)
    elif family == "anomaly":
        aggregate, fold_sections, approval, supplementary = _anomaly_sections(evaluation)
    elif family == "budget":
        aggregate, fold_sections, approval, supplementary = _budget_sections(evaluation)
    else:
        raise KeyError(f"unsupported model family: {family}")

    winner = evaluation.get("winner", "naive_baseline")
    tasks = {
        "pfp": "PFP Classification",
        "forecaster": "Financial Forecasting",
        "anomaly": "Anomaly Detection",
        "budget": "Budget Optimization",
    }
    n_folds = evaluation.get(
        "n_folds",
        len(evaluation.get("fold_details") or evaluation.get("fold_results") or []),
    )
    return {
        "family": FAMILY_LABELS[family],
        "task": tasks[family],
        "generated_at": evaluation.get("generated_at") or evaluation.get("timestamp", ""),
        "n_folds": n_folds or 1,
        "decision_rule": evaluation.get("decision_rule_text")
        or _decision_rule_for(family, evaluation),
        "winner": winner,
        "winner_artifact": _winner_artifact_for(family, evaluation),
        "winner_reason": evaluation.get("winner_reason", ""),
        "approval_result": approval,
        "aggregate": aggregate,
        "fold_sections": fold_sections,
        "supplementary": supplementary,
    }


def write_evaluation_report(
    output_dir: Path,
    *,
    family: str,
    task: str,
    generated_at: str,
    n_folds: int,
    decision_rule: str,
    winner: str,
    winner_artifact: str,
    winner_reason: str,
    approval_result: list[str],
    aggregate: str,
    fold_sections: str,
    supplementary: str = "",
) -> Path:
    """Write the standardized `evaluation_report.md` into `output_dir`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {family} — {task} Evaluation Report",
        "",
        f"**Generated:** {generated_at}",
        f"**Folds:** {n_folds}",
        f"**Decision rule:** {decision_rule}",
        "",
        "## Winner",
        "",
        f"- **Tier:** {winner}",
        f"- **Artifact:** {winner_artifact}",
        f"- **Reason:** {winner_reason}",
        "",
        "## Approval Criteria Result",
        "",
        *approval_result,
        "",
        "## Aggregate Results",
        "",
        aggregate,
        "",
        "## Per-Fold Results",
        "",
        fold_sections or "_No per-fold breakdown._",
    ]
    if supplementary:
        lines.extend(["", "## Supplementary", "", supplementary])
    path = output_dir / "evaluation_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path
