"""Internal synthetic-target candidate evaluation for the unserved v3 forecaster."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from logging_v3 import configure_logging, get_logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

LOGGER = get_logger("training")

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def _load_deterministic_household_sample(path: Path, modulus: int = 20) -> pd.DataFrame:
    """Load a stable household-level sample without loading the full PUF output."""
    selected: list[pd.DataFrame] = []
    fallback: pd.DataFrame | None = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000):
        frame = batch.to_pandas()
        if fallback is None:
            fallback = frame
        hashes = frame["household_id_v3"].map(
            lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)
        )
        sample = frame.loc[hashes % modulus == 0]
        if not sample.empty:
            selected.append(sample)
    if selected:
        return pd.concat(selected, ignore_index=True)
    return fallback if fallback is not None else pd.DataFrame()


def _mae(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(mean_absolute_error(actual, predicted))


def _classical_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit pooled classical models and scale their normalized paths to each household."""
    if not HAS_STATSMODELS:
        LOGGER.warning("statsmodels is unavailable; skipping ARIMA and SARIMA")
        return {}
    history = train.groupby("year_month", sort=True)["target_expenses"].mean()
    if len(history) < 12:
        LOGGER.warning(
            "Only %d monthly observations available; skipping classical models", len(history)
        )
        return {}
    test_levels = test.groupby("household_id_v3")["lag_1"].mean().clip(lower=1.0)
    default_level = float(train["target_expenses"].mean())
    level = test["household_id_v3"].map(test_levels).fillna(default_level).to_numpy()
    normalized = history / float(history.mean())
    predictions: dict[str, np.ndarray] = {}
    for name, factory in {
        "arima": lambda: ARIMA(normalized, order=(1, 1, 0)).fit(),
        "sarima": lambda: SARIMAX(normalized, order=(1, 1, 0), seasonal_order=(1, 0, 0, 12)).fit(
            disp=False
        ),
    }.items():
        try:
            fitted = factory()
            normalized_prediction = fitted.predict(start=0, end=len(normalized) - 1)
            path = dict(zip(history.index, normalized_prediction, strict=True))
            predictions[name] = np.asarray(
                [
                    float(path.get(period, 1.0)) * user_level
                    for period, user_level in zip(test["year_month"], level, strict=True)
                ]
            )
        except Exception as error:
            LOGGER.warning("%s fitting failed: %s", name.upper(), error)
    return predictions


class _LSTM(nn.Module if HAS_TORCH else object):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 24, batch_first=True)
        self.output = nn.Linear(24, 1)

    def forward(self, features):
        encoded, _ = self.lstm(features)
        return self.output(encoded[:, -1]).squeeze(1)


def _sequence_samples(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for _, household in frame.sort_values(["household_id_v3", "year_month"]).groupby(
        "household_id_v3"
    ):
        values = household[features].to_numpy(dtype=np.float32)
        labels = household["target_expenses"].to_numpy(dtype=np.float32)
        for index in range(2, len(household)):
            rows.append(values[index - 2 : index + 1])
            targets.append(labels[index])
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _lstm_predictions(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], batch_size: int
) -> np.ndarray | None:
    if not HAS_TORCH:
        LOGGER.warning("PyTorch is unavailable; skipping LSTM")
        return None
    train_x, train_y = _sequence_samples(train, features)
    test_x, _ = _sequence_samples(test, features)
    if not len(train_x) or not len(test_x):
        LOGGER.warning("Insufficient sequence rows for LSTM")
        return None
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x.reshape(-1, len(features))).reshape(train_x.shape)
    test_x = scaler.transform(test_x.reshape(-1, len(features))).reshape(test_x.shape)
    target_scaler = StandardScaler()
    train_y = target_scaler.fit_transform(train_y.reshape(-1, 1)).ravel()
    torch.manual_seed(42)
    model = _LSTM(len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss()
    training_data = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)
    )
    model.train()
    for _ in range(10):
        for batch_features, batch_targets in DataLoader(
            training_data, batch_size=batch_size, shuffle=True
        ):
            optimizer.zero_grad()
            loss(model(batch_features), batch_targets).backward()
            optimizer.step()
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch_features in DataLoader(
            torch.tensor(test_x, dtype=torch.float32), batch_size=batch_size
        ):
            predictions.append(model(batch_features).numpy())
    return target_scaler.inverse_transform(np.concatenate(predictions).reshape(-1, 1)).ravel()


def train_and_evaluate(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    run_rf: bool = True,
    run_torch: bool = True,
    sample_modulus: int = 20,
    rf_workers: int = 1,
    rf_estimators: int = 100,
    lstm_batch_size: int = 1024,
) -> dict:
    source, destination = Path(input_dir), Path(output_dir)
    if sample_modulus < 1 or rf_workers < 1 or rf_estimators < 1 or lstm_batch_size < 1:
        raise ValueError(
            "sample modulus, worker count, estimator count, and batch size must be positive"
        )
    LOGGER.info("Starting internal synthetic-target evaluation from %s", source)
    features = json.loads((source / "feature_columns.json").read_text())["feature_columns"]
    train = _load_deterministic_household_sample(source / "train.parquet", sample_modulus).dropna(
        subset=["target_expenses"]
    )
    test = _load_deterministic_household_sample(source / "test.parquet", sample_modulus).dropna(
        subset=["target_expenses"]
    )
    LOGGER.info("Loaded deterministic sample: train=%d rows, test=%d rows", len(train), len(test))
    metrics = {"naive": _mae(test["target_expenses"], test["lag_1"].to_numpy())}
    for name, prediction in _classical_predictions(train, test).items():
        metrics[name] = _mae(test["target_expenses"], prediction)
    if run_rf:
        LOGGER.info(
            "Training Random Forest with %d estimators and %d worker(s)", rf_estimators, rf_workers
        )
        model = RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=8,
            random_state=42,
            n_jobs=rf_workers,
        )
        model.fit(train[features], train["target_expenses"])
        metrics["random_forest"] = _mae(test["target_expenses"], model.predict(test[features]))
        del model
        gc.collect()
    else:
        LOGGER.info("Skipping Random Forest by request")
    if run_torch:
        LOGGER.info("Training LSTM with batch size %d", lstm_batch_size)
        prediction = _lstm_predictions(train, test, features, lstm_batch_size)
        if prediction is not None:
            _, test_y = _sequence_samples(test, features)
            metrics["lstm"] = float(mean_absolute_error(test_y, prediction))
    else:
        LOGGER.info("Skipping PyTorch LSTM by request")
    result = {
        "evaluation_level": "internal_synthetic_target_only",
        "external_validation_gate": "not_satisfied",
        "served_forecaster_unchanged": True,
        "feature_columns": features,
        "test_households": int(test["household_id_v3"].nunique()),
        "training_sample": f"deterministic 1-in-{sample_modulus} household sample",
        "resource_limits": {
            "sample_modulus": sample_modulus,
            "rf_workers": rf_workers,
            "rf_estimators": rf_estimators,
            "lstm_batch_size": lstm_batch_size,
        },
        "candidate_mae": metrics,
        "candidates_skipped": {"random_forest": not run_rf, "torch": not run_torch},
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluation.json").write_text(json.dumps(result, indent=2))
    LOGGER.info("Internal candidate evaluation complete: %s", metrics)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Internally evaluate survey-only v3 forecast features."
    )
    parser.add_argument("--input", default="training/datasets/forecaster_v3")
    parser.add_argument("--output", default="models/forecaster_v3")
    parser.add_argument("--skip-rf", action="store_true", help="Skip the Random Forest candidate.")
    parser.add_argument(
        "--skip-torch", action="store_true", help="Skip the PyTorch LSTM candidate."
    )
    parser.add_argument("--sample-modulus", type=int, default=20)
    parser.add_argument("--rf-workers", type=int, default=1)
    parser.add_argument("--rf-estimators", type=int, default=100)
    parser.add_argument("--lstm-batch-size", type=int, default=1024)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    train_and_evaluate(
        args.input,
        args.output,
        run_rf=not args.skip_rf,
        run_torch=not args.skip_torch,
        sample_modulus=args.sample_modulus,
        rf_workers=args.rf_workers,
        rf_estimators=args.rf_estimators,
        lstm_batch_size=args.lstm_batch_size,
    )
