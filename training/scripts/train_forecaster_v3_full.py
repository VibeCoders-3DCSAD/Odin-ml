"""Full-corpus, one-candidate-at-a-time v3 forecaster evaluation."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from logging_v3 import configure_logging, get_logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

LOGGER = get_logger("full_training")
BATCH_SIZE = 100_000
RF_PROGRESS_TREES = 5

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class _LSTM(nn.Module if HAS_TORCH else object):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 24, batch_first=True)
        self.output = nn.Linear(24, 1)

    def forward(self, features):
        encoded, _ = self.lstm(features)
        return self.output(encoded[:, -1]).squeeze(1)


def _batches(path: Path, columns: list[str]) -> Iterator[pd.DataFrame]:
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=BATCH_SIZE):
        yield batch.to_pandas()


def _row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _mae_stream(path: Path, predictor, columns: list[str] | None = None) -> tuple[float, int]:
    absolute_error = 0.0
    count = 0
    required = ["household_id_v3", "year_month", "lag_1", "target_expenses"]
    for frame in _batches(path, list(dict.fromkeys([*required, *(columns or [])]))):
        usable = frame.dropna(subset=["target_expenses"])
        if usable.empty:
            continue
        predicted = predictor(usable)
        absolute_error += float(np.abs(usable["target_expenses"].to_numpy() - predicted).sum())
        count += len(usable)
    if not count:
        raise ValueError("test data has no forecasting targets")
    return absolute_error / count, count


def _pooled_history(path: Path) -> pd.Series:
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for frame in _batches(path, ["year_month", "target_expenses"]):
        usable = frame.dropna(subset=["target_expenses"])
        grouped = usable.groupby("year_month")["target_expenses"].agg(["sum", "count"])
        for period, row in grouped.iterrows():
            totals[period][0] += float(row["sum"])
            totals[period][1] += float(row["count"])
    return pd.Series(
        {period: total / count for period, (total, count) in totals.items()}
    ).sort_index()


def _classical_mae(train_path: Path, test_path: Path, candidate: str) -> tuple[float, int]:
    if not HAS_STATSMODELS:
        raise RuntimeError("statsmodels is required for ARIMA and SARIMA")
    history = _pooled_history(train_path)
    if len(history) < 12:
        raise ValueError("at least 12 monthly observations are required for classical candidates")
    normalized = history / float(history.mean())
    model = (
        ARIMA(normalized, order=(1, 1, 0)).fit()
        if candidate == "arima"
        else SARIMAX(normalized, order=(1, 1, 0), seasonal_order=(1, 0, 0, 12)).fit(disp=False)
    )
    path = dict(zip(history.index, model.predict(start=0, end=len(history) - 1), strict=True))
    levels: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for frame in _batches(test_path, ["household_id_v3", "lag_1", "target_expenses"]):
        usable = frame.dropna(subset=["target_expenses"])
        grouped = usable.groupby("household_id_v3")["lag_1"].agg(["sum", "count"])
        for household, row in grouped.iterrows():
            levels[household][0] += float(row["sum"])
            levels[household][1] += float(row["count"])
    default_level = float(history.mean())

    def predictor(frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [
                float(path.get(period, 1.0)) * (levels[household][0] / levels[household][1])
                if household in levels
                else default_level
                for household, period in zip(
                    frame["household_id_v3"], frame["year_month"], strict=True
                )
            ]
        )

    return _mae_stream(test_path, predictor)


def _rf_mae(
    train_path: Path, test_path: Path, features: list[str], estimators: int, workers: int
) -> tuple[float, int]:
    train_rows = sum(
        len(frame.dropna(subset=["target_expenses"]))
        for frame in _batches(train_path, [*features, "target_expenses"])
    )
    LOGGER.info("Preparing disk-backed RF training matrix with %d rows", train_rows)
    with tempfile.TemporaryDirectory(prefix="odin-v3-rf-") as temporary:
        x_path, y_path = Path(temporary) / "x.dat", Path(temporary) / "y.dat"
        x = np.memmap(x_path, dtype="float32", mode="w+", shape=(train_rows, len(features)))
        y = np.memmap(y_path, dtype="float32", mode="w+", shape=(train_rows,))
        offset = 0
        for frame in _batches(train_path, [*features, "target_expenses"]):
            usable = frame.dropna(subset=["target_expenses"])
            next_offset = offset + len(usable)
            x[offset:next_offset] = usable[features].to_numpy(dtype=np.float32)
            y[offset:next_offset] = usable["target_expenses"].to_numpy(dtype=np.float32)
            offset = next_offset
        model = RandomForestRegressor(
            n_estimators=0,
            max_depth=12,
            random_state=42,
            n_jobs=workers,
            warm_start=True,
        )
        LOGGER.info("Training RF on all %d rows with %d worker(s)", train_rows, workers)
        for tree_count in range(
            RF_PROGRESS_TREES, estimators + RF_PROGRESS_TREES, RF_PROGRESS_TREES
        ):
            model.n_estimators = min(tree_count, estimators)
            model.fit(x, y)
            LOGGER.info("Completed RF trees: %d/%d", model.n_estimators, estimators)

        def predictor(frame: pd.DataFrame) -> np.ndarray:
            return model.predict(frame[features].to_numpy(dtype=np.float32))

        return _mae_stream(test_path, predictor, features)


def _sequence_batches(
    path: Path,
    features: list[str],
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    carry = pd.DataFrame()
    pending_samples: list[np.ndarray] = []
    pending_targets: list[float] = []
    columns = ["household_id_v3", "year_month", *features, "target_expenses"]

    def add_households(frame: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for _, household in frame.groupby("household_id_v3"):
            usable = household.dropna(subset=["target_expenses"])
            if len(usable) < 3:
                continue
            values = feature_scaler.transform(usable[features].to_numpy(dtype=np.float32))
            targets = target_scaler.transform(
                usable[["target_expenses"]].to_numpy(dtype=np.float32)
            ).ravel()
            pending_samples.extend(values[index - 2 : index + 1] for index in range(2, len(values)))
            pending_targets.extend(targets[2:])
            while len(pending_samples) >= batch_size:
                yield (
                    np.asarray(pending_samples[:batch_size], dtype=np.float32),
                    np.asarray(pending_targets[:batch_size], dtype=np.float32),
                )
                del pending_samples[:batch_size]
                del pending_targets[:batch_size]

    for source in _batches(path, columns):
        frame = pd.concat([carry, source], ignore_index=True)
        last_household = frame.iloc[-1]["household_id_v3"]
        complete = frame.loc[frame["household_id_v3"] != last_household]
        carry = frame.loc[frame["household_id_v3"] == last_household]
        yield from add_households(complete)
    if not carry.empty:
        yield from add_households(carry)
    if pending_samples:
        yield (
            np.asarray(pending_samples, dtype=np.float32),
            np.asarray(pending_targets, dtype=np.float32),
        )


def _lstm_mae(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    features: list[str],
    batch_size: int,
    epochs: int,
    patience: int,
) -> tuple[float, int]:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for LSTM")
    feature_scaler, target_scaler = StandardScaler(), StandardScaler()
    for frame in _batches(train_path, [*features, "target_expenses"]):
        usable = frame.dropna(subset=["target_expenses"])
        feature_scaler.partial_fit(usable[features].to_numpy(dtype=np.float32))
        target_scaler.partial_fit(usable[["target_expenses"]].to_numpy(dtype=np.float32))
    torch.manual_seed(42)
    model = _LSTM(len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss()
    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()
        for samples, targets in _sequence_batches(
            train_path, features, feature_scaler, target_scaler, batch_size
        ):
            x = torch.tensor(samples, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.float32)
            optimizer.zero_grad()
            loss(model(x), y).backward()
            optimizer.step()
        model.eval()
        validation_loss = 0.0
        validation_count = 0
        with torch.no_grad():
            for samples, targets in _sequence_batches(
                validation_path, features, feature_scaler, target_scaler, batch_size
            ):
                predicted = model(torch.tensor(samples, dtype=torch.float32))
                validation_loss += float(
                    loss(predicted, torch.tensor(targets, dtype=torch.float32))
                ) * len(targets)
                validation_count += len(targets)
        mean_validation_loss = validation_loss / validation_count
        if mean_validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = mean_validation_loss
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        LOGGER.info(
            "Completed LSTM epoch %d/%d; validation loss=%.6f",
            epoch + 1,
            epochs,
            mean_validation_loss,
        )
        if epochs_without_improvement >= patience:
            LOGGER.info("Early stopping LSTM after %d epochs", epoch + 1)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    absolute_error = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for samples, targets in _sequence_batches(
            test_path, features, feature_scaler, target_scaler, batch_size
        ):
            predicted = model(torch.tensor(samples, dtype=torch.float32))
            actual = target_scaler.inverse_transform(targets.reshape(-1, 1)).ravel()
            forecast = target_scaler.inverse_transform(predicted.numpy().reshape(-1, 1)).ravel()
            absolute_error += float(np.abs(actual - forecast).sum())
            count += len(actual)
    return absolute_error / count, count


def run(
    candidate: str,
    input_dir: str | Path,
    output_dir: str | Path,
    rf_estimators: int,
    rf_workers: int,
    lstm_batch_size: int,
    lstm_epochs: int,
    lstm_patience: int,
) -> dict:
    source, destination = Path(input_dir), Path(output_dir)
    train_path, validation_path, test_path = (
        source / "train.parquet",
        source / "val.parquet",
        source / "test.parquet",
    )
    features = json.loads((source / "feature_columns.json").read_text())["feature_columns"]
    LOGGER.info("Running full-corpus %s candidate", candidate)
    if candidate == "naive":
        mae, test_rows = _mae_stream(test_path, lambda frame: frame["lag_1"].to_numpy())
    elif candidate in {"arima", "sarima"}:
        mae, test_rows = _classical_mae(train_path, test_path, candidate)
    elif candidate == "random_forest":
        mae, test_rows = _rf_mae(train_path, test_path, features, rf_estimators, rf_workers)
    else:
        mae, test_rows = _lstm_mae(
            train_path,
            validation_path,
            test_path,
            features,
            lstm_batch_size,
            lstm_epochs,
            lstm_patience,
        )
    result = {
        "candidate": candidate,
        "evaluation_level": "internal_synthetic_target_only",
        "training_scope": "full_household_corpus",
        "train_rows": _row_count(train_path),
        "test_rows": test_rows,
        "mae": mae,
        "served_forecaster_unchanged": True,
    }
    candidates = destination / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    (candidates / f"{candidate}.json").write_text(json.dumps(result, indent=2))
    LOGGER.info("Completed full-corpus %s: MAE=%.2f", candidate, mae)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one full-corpus v3 forecast candidate.")
    parser.add_argument(
        "--candidate", choices=["naive", "arima", "sarima", "random_forest", "lstm"], required=True
    )
    parser.add_argument("--input", default="training/datasets/forecaster_v3")
    parser.add_argument("--output", default="models/forecaster_v3")
    parser.add_argument("--rf-estimators", type=int, default=100)
    parser.add_argument("--rf-workers", type=int, default=1)
    parser.add_argument("--lstm-batch-size", type=int, default=512)
    parser.add_argument("--lstm-epochs", type=int, default=100)
    parser.add_argument("--lstm-patience", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    run(
        args.candidate,
        args.input,
        args.output,
        args.rf_estimators,
        args.rf_workers,
        args.lstm_batch_size,
        args.lstm_epochs,
        args.lstm_patience,
    )
