"""
Anomaly Detector Training Pipeline

Trains and compares the lightest-tier anomaly detectors for transaction-level
anomaly detection on synthetic financial data.

Scope (top-3-lightest, per docs/models/model-candidate-report.md §6):
  Tier 0: Majority-class baseline (sanity floor)
  Tier 1: IQR (statistical, per-feature) — decision-rule baseline
  Tier 2: Adaptive threshold calibration (IQR-magnitude scoring, ~0 params),
          Isolation Forest, One-Class SVM (kernel), Autoencoder (PyTorch)
  Tier 3: Hybrid ensemble (score-average of available Tier 1 + Tier 2)

Heavier candidates (deep reconstruction ensembles, LLM-style) are documented
as "research only" in the roster and are not trained in the default scope.

Evaluation:
  - 5-fold expanding window (temporal_folds.json)
  - Primary ranking metric: PR-AUC (imbalance-safe, threshold-free; baseline
    equals the anomaly rate). Supplementary: ROC-AUC, and Accuracy/Precision/
    Recall/F1 at the chosen operating point.
  - Operating point selected on held-out val split (no test leakage):
    maximize F2 (beta=2, recall-prioritized) subject to precision >= 0.30.
  - Decision rule (Option A, see docs/thesis/anomaly-decision-rule-rationale.md):
    adopt the best-learned tier iff PR-AUC(winner) >= 1.5 x PR-AUC(IQR) and
    PR-AUC(winner) >= 0.15; otherwise fall back to the IQR baseline.

Usage:
    python training/scripts/train_anomaly.py --input training/datasets/anomaly/ --output models/anomaly/
    python training/scripts/train_anomaly.py --skip-ae   # CPU host without AE in hybrid
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.svm import OneClassSVM

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ml.metadata import build_metadata, write_metadata
from app.ml.models import AdaptiveThresholdDetector, HybridEnsemble, IQRDetector, _Autoencoder
from app.ml.reporting import family_metadata, family_report, write_evaluation_report

HAS_PYTORCH = False
torch = None
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_PYTORCH = True
except ImportError:
    warnings.warn("torch unavailable — Tier 2 Autoencoder will be skipped")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "mean_income_rolling",
    "std_income_rolling",
    "mean_expenses_rolling",
    "std_expenses_rolling",
    "category_dist",
    "txn_frequency_rolling",
    "avg_txn_size_rolling",
    "category_entropy",
    "volatility_index",
    "spending_concentration",
    "amount_deviation",
    "category_deviation",
    "frequency_deviation",
    "income_deviation",
    "expense_deviation",
    "is_novel_category",
    "amount_vs_category_mean",
    "amount_vs_category_std",
    "category_frequency_change",
    "amount_percentile_in_category",
    "days_since_last_txn",
    "is_weekend",
    "amount_zscore_overall",
    "amount_zscore_category",
]

LABEL_COL = "is_anomalous"
META_COLS = [
    "user_id",
    "transaction_id",
    "month",
    "date",
    "category",
    "amount",
    "transaction_type",
    "is_anomalous",
    "anomaly_type",
]

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data(data_dir: str):
    """Load parquet splits and feature metadata."""
    data_path = Path(data_dir)

    train_df = pd.read_parquet(data_path / "train.parquet")
    val_df = pd.read_parquet(data_path / "val.parquet")
    test_df = pd.read_parquet(data_path / "test.parquet")

    with open(data_path / "feature_columns.json") as f:
        meta = json.load(f)

    return train_df, val_df, test_df, meta


def load_folds(folds_path: str):
    """Load temporal fold definitions."""
    with open(folds_path) as f:
        return json.load(f)


def prepare_fold_data(train_df: pd.DataFrame, fold: dict):
    """Split train_df into fold-train and fold-test based on months."""
    train_months = set(fold["train_months"])
    test_months = set(fold["test_months"])

    fold_train = train_df[train_df["month"].isin(train_months)]
    fold_test = train_df[train_df["month"].isin(test_months)]

    return fold_train, fold_test


def extract_features(df: pd.DataFrame):
    """Extract X (features) and y (labels) from a dataframe."""
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5):
    """Compute anomaly detection metrics.

    Primary (MDD v2.3): Accuracy, Precision, Recall, F1 at the given
    threshold. Supplementary: PR-AUC, ROC-AUC, best-F1 operating point.
    """
    y_pred = (y_scores >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_scores)
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc_auc = float("nan")

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Best-F1 operating point (used for ranking only, never for final eval)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else threshold
    best_f1 = float(f1_scores[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall = float(recalls[best_idx])
    y_pred_best = (y_scores >= best_threshold).astype(int)
    tp_b = int(np.sum((y_pred_best == 1) & (y_true == 1)))
    tn_b = int(np.sum((y_pred_best == 0) & (y_true == 0)))
    best_accuracy = (tp_b + tn_b) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "threshold": round(threshold, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "best_f1": round(best_f1, 4),
        "best_threshold": round(best_threshold, 4),
        "best_precision": round(best_precision, 4),
        "best_recall": round(best_recall, 4),
        "best_accuracy": round(best_accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_positives": int(np.sum(y_true)),
        "n_predictions_positive": int(np.sum(y_pred)),
    }


def compute_baseline_metrics(y_true: np.ndarray):
    """Tier 0: always predict majority class (all normal)."""
    y_scores = np.zeros_like(y_true, dtype=float)
    return compute_metrics(y_true, y_scores)


def _select_operating_point(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    min_precision: float = 0.30,
    beta: float = 2.0,
) -> tuple[float, dict]:
    """Choose the val operating threshold that maximizes F-beta subject to a
    precision floor (Option A operating-point policy).

    Returns ``(threshold, point_metrics)`` where ``point_metrics`` carries the
    precision/recall/F1/F-beta/accuracy at the chosen threshold.
    """
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    beta2 = float(beta) ** 2
    fbeta = (1 + beta2) * (precisions * recalls) / (
        beta2 * precisions + recalls + 1e-12
    )

    # The PR-curve terminal point (precision=1.0, recall=0.0) is not an
    # operating point — it flags nothing. Exclude recall==0 so a detector whose
    # precision floor is unreachable falls back to argmax F-beta over real
    # operating points instead of "flag everything" (threshold = min score).
    real = np.asarray(recalls) > 0
    eligible = np.where(real & (np.asarray(precisions) >= min_precision))[0]
    if len(eligible) > 0:
        precision_floor_met = True
        best_idx = int(eligible[np.argmax(fbeta[eligible])])
    else:  # no point meets the floor: fall back to argmax F-beta over real points
        precision_floor_met = False
        eligible = np.where(real)[0]
        best_idx = int(eligible[np.argmax(fbeta[eligible])])
    threshold = (
        float(thresholds[best_idx])
        if best_idx < len(thresholds)
        else float(y_scores.min())
    )  # PR curve terminal point carries no threshold
    precision = float(precisions[best_idx])
    recall = float(recalls[best_idx])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    f2 = float(fbeta[best_idx])
    y_pred = (y_scores >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return threshold, {
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "beta": beta,
        "min_precision": min_precision,
        "precision_floor_met": precision_floor_met,
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": int(np.sum((y_pred == 1) & (y_true == 0))),
        "fn": int(np.sum((y_pred == 0) & (y_true == 1))),
        "tn": tn,
        "n_positives": int(np.sum(y_true)),
        "n_predictions_positive": int(np.sum(y_pred)),
    }


# ---------------------------------------------------------------------------
# Tier 1: IQR Detector (canonical class lives in app.ml.models)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tier 2: Adaptive threshold calibration (Zhong 2025, B6) — ~0 learned params.
# Scores each row by magnitude-normalized per-feature IQR deviation, so the
# operating threshold decouples from raw feature units and can be calibrated
# on the held-out val split (per-feature adaptive bounds). Canonical class
# lives in app.ml.models so the serving app can unpickle an adaptive winner.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tier 2: Isolation Forest
# ---------------------------------------------------------------------------


def train_isolation_forest(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
):
    """Train Isolation Forest with contamination set from train label rate."""
    contamination = float(np.clip(np.mean(y_train), 0.001, 0.5))
    model = IsolationForest(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        max_features=0.8,
    )
    model.fit(X_train)
    val_scores = -model.decision_function(X_val)
    val_scores = (val_scores - val_scores.min()) / (val_scores.max() - val_scores.min() + 1e-8)
    metrics = compute_metrics(y_val, val_scores)
    return model, {"contamination": contamination, "n_estimators": 200}, metrics["best_f1"]


# ---------------------------------------------------------------------------
# Tier 2: One-Class SVM
# ---------------------------------------------------------------------------


def train_ocsvm(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Train One-Class SVM with subsampling for speed."""
    max_samples = min(3000, X_train.shape[0])
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(X_train.shape[0], max_samples, replace=False)
    X_sub = X_train[idx]

    best_score = -1
    best_model = None
    best_params = None

    for nu in [0.01, 0.05]:
        try:
            model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
            model.fit(X_sub)
            val_scores = -model.decision_function(X_val)
            val_scores = (val_scores - val_scores.min()) / (
                val_scores.max() - val_scores.min() + 1e-8
            )
            metrics = compute_metrics(y_val, val_scores)
            if metrics["best_f1"] > best_score:
                best_score = metrics["best_f1"]
                best_model = model
                best_params = {"nu": nu}
        except Exception:
            continue

    return best_model, best_params, best_score


# ---------------------------------------------------------------------------
# Tier 2: Autoencoder (PyTorch) — canonical model class lives in app.ml.models
# ---------------------------------------------------------------------------


def build_autoencoder(input_dim: int, encoding_dim: int = 7):
    """Build a simple autoencoder for anomaly detection (PyTorch)."""
    if not HAS_PYTORCH:
        return None
    return _Autoencoder(input_dim, encoding_dim)


def train_autoencoder(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
):
    """Train autoencoder; anomalies have higher reconstruction error."""
    if not HAS_PYTORCH:
        return None, None, -1

    input_dim = X_train.shape[1]
    model = build_autoencoder(input_dim)
    if model is None:
        return None, None, -1

    device = torch.device("cpu")
    model = model.to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    n_val = max(1, int(len(X_t) * 0.1))
    X_tr = X_t[: len(X_t) - n_val]
    X_v = X_t[len(X_t) - n_val :]

    train_ds = TensorDataset(X_tr, X_tr)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(30):
        model.train()
        for xb, _ in train_dl:
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_recon = model(X_v)
            val_loss = criterion(val_recon, X_v).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        val_recon = model(X_val_t).numpy()
    val_scores = np.mean((X_val - val_recon) ** 2, axis=1)
    val_scores = (val_scores - val_scores.min()) / (val_scores.max() - val_scores.min() + 1e-8)

    metrics = compute_metrics(y_val, val_scores)
    return model, "mse", metrics["best_f1"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_confusion_matrices(results: dict, output_dir: Path):
    """Plot confusion matrix heatmaps for all models."""
    if not HAS_MATPLOTLIB:
        return

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
        )
        ax.set_title(f"{name}\nPR-AUC={res['pr_auc']:.3f}", fontsize=10)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curves(y_true: np.ndarray, score_dict: dict, output_dir: Path):
    """Plot precision-recall curves for all models."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, scores in score_dict.items():
        precisions, recalls, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        ax.plot(recalls, precisions, label=f"{name} (AUC={pr_auc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Anomaly Detection")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training/datasets/anomaly/")
    parser.add_argument("--output", default="models/anomaly/")
    parser.add_argument("--folds", default="training/datasets/processed/temporal_folds.json")
    parser.add_argument(
        "--skip-ae",
        action="store_true",
        help="Skip the PyTorch Autoencoder tier (also excluded from the hybrid). "
        "Use on CPU-constrained hosts; run the full tier set on a GPU-capable "
        "teammate machine instead.",
    )
    args = parser.parse_args()

    t0 = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading data...")
    train_df, val_df, test_df, meta = load_data(args.input)
    folds = load_folds(args.folds)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"  Features: {len(FEATURE_COLS)}")

    all_fold_results = {}
    all_fold_scores = {}

    print(f"\n[2/6] Training across {len(folds)} folds...")

    for fold_info in folds:
        fold_num = fold_info["fold"]
        print(
            f"\n  Fold {fold_num}: train months {fold_info['train_months']}, "
            f"test months {fold_info['test_months']}"
        )

        fold_train, fold_test = prepare_fold_data(train_df, fold_info)
        X_train, y_train = extract_features(fold_train)
        X_test, y_test = extract_features(fold_test)

        print(
            f"    Train: {len(X_train)} ({int(y_train.sum())} anomalies), "
            f"Test: {len(X_test)} ({int(y_test.sum())} anomalies)"
        )

        fold_scores = {}

        # Tier 0: Baseline
        baseline = compute_baseline_metrics(y_test)
        print(f"    Tier 0 (Baseline): F1={baseline['f1']:.4f}, Acc={baseline['accuracy']:.4f}")
        fold_scores["tier0_baseline"] = np.zeros_like(y_test, dtype=float)

        # Tier 1: IQR
        iqr = IQRDetector(iqr_multiplier=1.5)
        iqr.fit(X_train)
        iqr_scores = iqr.score(X_test)
        iqr_metrics = compute_metrics(y_test, iqr_scores)
        print(
            f"    Tier 1 (IQR): F1={iqr_metrics['best_f1']:.4f}, "
            f"Acc={iqr_metrics['best_accuracy']:.4f}, "
            f"PR-AUC={iqr_metrics['pr_auc']:.4f}"
        )
        fold_scores["tier1_iqr"] = iqr_scores

        # Tier 2: Adaptive threshold calibration (IQR-magnitude, ~0 params)
        adaptive = AdaptiveThresholdDetector(iqr_multiplier=1.5)
        adaptive.fit(X_train)
        adaptive_scores = adaptive.score(X_test)
        adaptive_metrics = compute_metrics(y_test, adaptive_scores)
        print(
            f"    Tier 2 (AdaptiveThr): F1={adaptive_metrics['best_f1']:.4f}, "
            f"Acc={adaptive_metrics['best_accuracy']:.4f}, "
            f"PR-AUC={adaptive_metrics['pr_auc']:.4f}"
        )
        fold_scores["tier2_adaptive_threshold"] = adaptive_scores

        # Tier 2: Isolation Forest
        if_model, if_params, if_val_score = train_isolation_forest(X_train, y_train, X_test, y_test)
        if_scores = -if_model.decision_function(X_test)
        if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
        if_metrics = compute_metrics(y_test, if_scores)
        print(
            f"    Tier 2 (IF): F1={if_metrics['best_f1']:.4f}, "
            f"Acc={if_metrics['best_accuracy']:.4f}, "
            f"PR-AUC={if_metrics['pr_auc']:.4f}, params={if_params}"
        )
        fold_scores["tier2_isolation_forest"] = if_scores

        # Tier 2: One-Class SVM
        ocsvm_model, ocsvm_params, ocsvm_val_score = train_ocsvm(X_train, y_train, X_test, y_test)
        if ocsvm_model is not None:
            ocsvm_scores = -ocsvm_model.decision_function(X_test)
            ocsvm_scores = (ocsvm_scores - ocsvm_scores.min()) / (
                ocsvm_scores.max() - ocsvm_scores.min() + 1e-8
            )
            ocsvm_metrics = compute_metrics(y_test, ocsvm_scores)
            print(
                f"    Tier 2 (OCSVM): F1={ocsvm_metrics['best_f1']:.4f}, "
                f"Acc={ocsvm_metrics['best_accuracy']:.4f}, "
                f"PR-AUC={ocsvm_metrics['pr_auc']:.4f}, params={ocsvm_params}"
            )
            fold_scores["tier2_ocsvm"] = ocsvm_scores
        else:
            print("    Tier 2 (OCSVM): FAILED")
            ocsvm_metrics = {"pr_auc": 0, "best_f1": 0, "best_accuracy": 0}

        # Tier 2: Autoencoder
        if HAS_PYTORCH and not args.skip_ae:
            ae_model, ae_loss, ae_val_score = train_autoencoder(X_train, y_train, X_test, y_test)
            if ae_model is not None:
                ae_model.eval()
                with torch.no_grad():
                    ae_recon = ae_model(torch.tensor(X_test, dtype=torch.float32)).numpy()
                ae_scores = np.mean((X_test - ae_recon) ** 2, axis=1)
                ae_scores = (ae_scores - ae_scores.min()) / (
                    ae_scores.max() - ae_scores.min() + 1e-8
                )
                ae_metrics = compute_metrics(y_test, ae_scores)
                print(
                    f"    Tier 2 (AE): F1={ae_metrics['best_f1']:.4f}, "
                    f"Acc={ae_metrics['best_accuracy']:.4f}, "
                    f"PR-AUC={ae_metrics['pr_auc']:.4f}"
                )
                fold_scores["tier2_autoencoder"] = ae_scores
            else:
                print("    Tier 2 (AE): FAILED")
                ae_metrics = {"pr_auc": 0, "best_f1": 0, "best_accuracy": 0}
        else:
            ae_metrics = {"pr_auc": 0, "best_f1": 0, "best_accuracy": 0}

        # Tier 3: Hybrid ensemble — score-average of available Tier 1 + Tier 2
        _norm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-8)
        _members = [_norm(iqr_scores), adaptive_scores, if_scores]
        if "tier2_ocsvm" in fold_scores:
            _members.append(fold_scores["tier2_ocsvm"])
        if "tier2_autoencoder" in fold_scores:
            _members.append(fold_scores["tier2_autoencoder"])
        hybrid_scores = np.mean(_members, axis=0)
        hybrid_metrics = compute_metrics(y_test, hybrid_scores)
        print(
            f"    Tier 3 (Hybrid): F1={hybrid_metrics['best_f1']:.4f}, "
            f"Acc={hybrid_metrics['best_accuracy']:.4f}, "
            f"PR-AUC={hybrid_metrics['pr_auc']:.4f} (members={len(_members)})"
        )
        fold_scores["tier3_hybrid"] = hybrid_scores

        # Store fold results
        all_fold_results[fold_num] = {
            "baseline": baseline,
            "tier1_iqr": iqr_metrics,
            "tier2_adaptive_threshold": adaptive_metrics,
            "tier2_isolation_forest": if_metrics,
            "tier2_ocsvm": ocsvm_metrics,
            "tier2_autoencoder": ae_metrics,
            "tier3_hybrid": hybrid_metrics,
            "models": {
                "iqr_bounds": iqr.bounds,
                "if_params": if_params,
                "ocsvm_params": ocsvm_params,
            },
        }
        all_fold_scores[fold_num] = fold_scores

    # Aggregate across folds
    print(f"\n[3/6] Aggregating results across {len(folds)} folds...")
    model_names = [
        k for k in all_fold_results[folds[0]["fold"]].keys() if k not in ("baseline", "models")
    ]

    def _fold_stats(key, stat):
        return [all_fold_results[f["fold"]][key][stat] for f in folds]

    summary = {}
    for model_name in model_names:
        f1s = _fold_stats(model_name, "best_f1")
        accs = _fold_stats(model_name, "best_accuracy")
        precs = _fold_stats(model_name, "best_precision")
        recs = _fold_stats(model_name, "best_recall")
        pr_aucs = _fold_stats(model_name, "pr_auc")
        summary[model_name] = {
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "precision_mean": round(float(np.mean(precs)), 4),
            "recall_mean": round(float(np.mean(recs)), 4),
            "pr_auc_mean": round(float(np.mean(pr_aucs)), 4),
            "pr_auc_std": round(float(np.std(pr_aucs)), 4),
        }

    baseline_f1s = _fold_stats("baseline", "f1")
    baseline_accs = _fold_stats("baseline", "accuracy")
    summary["baseline"] = {
        "f1_mean": round(float(np.mean(baseline_f1s)), 4),
        "f1_std": round(float(np.std(baseline_f1s)), 4),
        "accuracy_mean": round(float(np.mean(baseline_accs)), 4),
        "pr_auc_mean": 0.0,
        "pr_auc_std": 0.0,
    }

    # Print summary table (PR-AUC is primary per Option A)
    print("\n  Model Summary (mean ± std across folds):")
    print(f"  {'Model':<28} {'F1':>12} {'Acc':>12} {'PR-AUC':>12}")
    print(f"  {'-' * 66}")
    for name, stats in sorted(summary.items(), key=lambda x: -x[1].get("pr_auc_mean", 0)):
        f1_str = f"{stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}"
        acc_str = f"{stats['accuracy_mean']:.4f}"
        pr_str = f"{stats['pr_auc_mean']:.4f} ± {stats['pr_auc_std']:.4f}"
        print(f"  {name:<28} {f1_str:>12} {acc_str:>12} {pr_str:>12}")

    # Select winner by primary ranking metric (PR-AUC), then enforce the
    # pre-registered Option A rule (see docs/thesis/anomaly-decision-rule-rationale.md):
    # winner must reach PR-AUC >= 1.5 x IQR PR-AUC AND PR-AUC >= 0.15
    best_model_name = max(
        [k for k in summary if k != "baseline"], key=lambda k: summary[k]["pr_auc_mean"]
    )
    best_stats = summary[best_model_name]
    iqr_stats = summary["tier1_iqr"]
    pr_auc_improvement = (best_stats["pr_auc_mean"] - iqr_stats["pr_auc_mean"]) / max(
        iqr_stats["pr_auc_mean"], 1e-8
    )
    target_met = best_stats["pr_auc_mean"] >= 0.15
    rule_passed = (pr_auc_improvement >= 0.50) and target_met

    if not rule_passed:
        winner_reason = (
            f"Best learned tier ({best_model_name}) improved PR-AUC by "
            f"{pr_auc_improvement * 100:.1f}% over IQR but reached only "
            f"{best_stats['pr_auc_mean']:.4f} (target >= 0.15); pre-registered rule failed, "
            f"retaining the interpretable IQR baseline."
        )
        print(
            f"\n  Decision rule NOT satisfied for {best_model_name}: "
            f"PR-AUC improvement {pr_auc_improvement * 100:.1f}% (need >=50%), "
            f"PR-AUC {best_stats['pr_auc_mean']:.4f} (need >=0.15). "
            f"Falling back to interpretable IQR baseline."
        )
        best_model_name = "tier1_iqr"
        best_stats = summary["tier1_iqr"]
    else:
        winner_reason = (
            f"{best_model_name} improved PR-AUC by {pr_auc_improvement * 100:.1f}% over IQR "
            f"(target >= 50%) and reached PR-AUC {best_stats['pr_auc_mean']:.4f} "
            f"(target >= 0.15); the recall-prioritized operating point is selected on the "
            f"held-out val split maximizing F2 subject to precision >= 0.30."
        )

    print(f"\n  Winner: {best_model_name}")
    print(f"  PR-AUC: {best_stats['pr_auc_mean']:.4f} ± {best_stats['pr_auc_std']:.4f}")

    # Retrain winner on full training data, select threshold on val, eval on test
    print(f"\n[4/6] Retraining {best_model_name} on full training set...")
    X_full_train, y_full_train = extract_features(train_df)
    X_val_final, y_val_final = extract_features(val_df)
    X_test_final, y_test_final = extract_features(test_df)

    contamination = float(np.clip(np.mean(y_full_train), 0.001, 0.5))

    if best_model_name == "tier1_iqr":
        winner = IQRDetector(iqr_multiplier=1.5)
        winner.fit(X_full_train)
        score_fn = lambda X: winner.score(X)
        winner_params = {"iqr_multiplier": 1.5}
    elif best_model_name == "tier2_isolation_forest":
        winner = IsolationForest(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            n_estimators=200,
            max_samples="auto",
            contamination=contamination,
            max_features=0.8,
        )
        winner.fit(X_full_train)
        score_fn = lambda X: -winner.decision_function(X)
        winner_params = {"contamination": contamination, "n_estimators": 200}
    elif best_model_name == "tier2_ocsvm":
        winner = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        winner.fit(X_full_train)
        score_fn = lambda X: -winner.decision_function(X)
        winner_params = {"nu": 0.05}
    elif best_model_name == "tier2_autoencoder" and HAS_PYTORCH:
        winner = build_autoencoder(X_full_train.shape[1])
        # Train on full training data
        device = torch.device("cpu")
        winner = winner.to(device)
        X_t = torch.tensor(X_full_train, dtype=torch.float32)
        n_val = max(1, int(len(X_t) * 0.1))
        X_tr, X_v = X_t[: len(X_t) - n_val], X_t[len(X_t) - n_val :]
        train_ds = TensorDataset(X_tr, X_tr)
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(winner.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for epoch in range(50):
            winner.train()
            for xb, _ in train_dl:
                optimizer.zero_grad()
                loss = criterion(xb, winner(xb))
                loss.backward()
                optimizer.step()
        winner.eval()

        def ae_score(X):
            with torch.no_grad():
                recon = winner(torch.tensor(X, dtype=torch.float32)).numpy()
            return np.mean((X - recon) ** 2, axis=1)

        score_fn = ae_score
        winner_params = {"loss": "mse", "epochs": 50}
    elif best_model_name == "tier2_adaptive_threshold":
        winner = AdaptiveThresholdDetector(iqr_multiplier=1.5)
        winner.fit(X_full_train)
        score_fn = lambda X: winner.score(X)
        winner_params = {"iqr_multiplier": 1.5}
    elif best_model_name == "tier3_hybrid":
        # Rebuild each member on the full training set and average via HybridEnsemble
        members = {}
        members["iqr"] = IQRDetector(iqr_multiplier=1.5)
        members["iqr"].fit(X_full_train)
        members["adaptive"] = AdaptiveThresholdDetector(iqr_multiplier=1.5)
        members["adaptive"].fit(X_full_train)
        members["if"] = IsolationForest(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            n_estimators=200,
            max_samples="auto",
            contamination=contamination,
            max_features=0.8,
        )
        members["if"].fit(X_full_train)
        members["ocsvm"] = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        members["ocsvm"].fit(X_full_train)
        if HAS_PYTORCH and not args.skip_ae:
            members["ae"] = build_autoencoder(X_full_train.shape[1])
            members["ae"].train()
            X_t = torch.tensor(X_full_train, dtype=torch.float32)
            n_val = max(1, int(len(X_t) * 0.1))
            X_tr, X_v = X_t[: len(X_t) - n_val], X_t[len(X_t) - n_val :]
            train_ds = TensorDataset(X_tr, X_tr)
            train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
            optimizer = torch.optim.Adam(members["ae"].parameters(), lr=0.001)
            criterion = nn.MSELoss()
            for _ in range(30):
                for xb, _ in train_dl:
                    optimizer.zero_grad()
                    loss = criterion(xb, members["ae"](xb))
                    loss.backward()
                    optimizer.step()
            members["ae"].eval()
        detector_order = ["iqr", "adaptive", "if", "ocsvm"] + (["ae"] if "ae" in members else [])
        winner = HybridEnsemble(
            detectors=[members[k] for k in detector_order],
            weights=[1.0 / len(detector_order)] * len(detector_order),
        )
        score_fn = lambda X: winner.score(X)
        winner_params = {"members": detector_order}
    else:
        winner = None
        score_fn = lambda X: np.zeros(X.shape[0], dtype=float)
        winner_params = {}

    # Select operating threshold on held-out val split (no test leakage):
    # maximize F2 (recall-prioritized) subject to precision >= 0.30 (Option A)
    val_scores = score_fn(X_val_final)
    threshold, op_point = _select_operating_point(
        y_val_final, val_scores, min_precision=0.30, beta=2.0
    )

    test_scores = score_fn(X_test_final)
    val_metrics = compute_metrics(y_val_final, val_scores, threshold=threshold)
    final_metrics = compute_metrics(y_test_final, test_scores, threshold=threshold)
    print(f"  Val-selected threshold: {threshold:.4f}")
    print(
        f"  Val F2 @ threshold: {op_point['f2']:.4f} "
        f"(P={op_point['precision']:.4f}, R={op_point['recall']:.4f}, F1={op_point['f1']:.4f})"
    )
    print(f"  Test F1: {final_metrics['f1']:.4f}")
    print(f"  Test Precision: {final_metrics['precision']:.4f}")
    print(f"  Test Recall: {final_metrics['recall']:.4f}")
    print(f"  Test Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Test PR-AUC (supplementary): {final_metrics['pr_auc']:.4f}")

    # Save model
    print("\n[5/6] Saving model and artifacts...")
    model_path = output_dir / "anomaly_detector.joblib"
    if winner is not None:
        joblib.dump(winner, model_path)
        print(f"  Saved: {model_path}")

    # Save evaluation report
    report = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "task": "anomaly_detection",
        "task_type": "unsupervised_binary_classification",
        "n_features": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "anomaly_rate_train": float(train_df[LABEL_COL].mean()),
        "anomaly_rate_val": float(val_df[LABEL_COL].mean()),
        "anomaly_rate_test": float(test_df[LABEL_COL].mean()),
        "winner": best_model_name,
        "winner_artifact": "anomaly_detector.joblib",
"winner_params": winner_params,
        "winner_reason": winner_reason,
        "decision_rule": {
            "metric": "pr_auc",
            "pr_auc_target": 0.15,
            "pr_auc_improvement_ratio": 1.5,
            "pr_auc_improvement_over_iqr_pct": round(pr_auc_improvement * 100, 1),
            "rule_passed": rule_passed,
            "operating_point": {
                "method": "F2 maximization with precision floor",
                "beta": 2.0,
                "min_precision": 0.30,
                "selected_on": "held-out val split",
            },
        },
        "val_selected_threshold": threshold,
        "val_operating_point": op_point,
        "val_metrics": val_metrics,
        "final_test_metrics": final_metrics,
        "fold_results": {str(k): v for k, v in all_fold_results.items()},
    }
    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(report, f, indent=2)

    # Generate plots
    print("  Generating plots...")
    # Use last fold for plots
    last_fold_num = folds[-1]["fold"]
    plot_confusion_matrices(
        {
            k: all_fold_results[last_fold_num][k]
            for k in ["baseline", "tier1_iqr", "tier2_isolation_forest", best_model_name]
            if k in all_fold_results[last_fold_num]
        },
        output_dir,
    )
    plot_pr_curves(y_test_final, {"winner": test_scores}, output_dir)

    # Save markdown report
    write_evaluation_report(output_dir, **family_report("anomaly", report))

    # Emit metadata.json (Phase 8 provenance: hash, commit, metrics, rule)
    metadata = build_metadata(
        **family_metadata(
            "anomaly",
            report,
            data_sources=[
                Path(args.input) / name for name in ("train.parquet", "val.parquet", "test.parquet")
            ],
        )
    )
    write_metadata(metadata, output_dir)

    elapsed = time.time() - t0
    print(
        f"\n[6/6] Done. Winner: {best_model_name}, "
        f"Test F1: {final_metrics['f1']:.4f}, {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
