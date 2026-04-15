from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

from ctecho import CtEchoClassifier, CtEchoConfig


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    timestamps_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    timestamps_test: np.ndarray
    y_test: np.ndarray


def build_config(
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: str | None = None,
) -> CtEchoConfig:
    """CtEcho parameters copied from run_shared_config.py without importing it."""
    config_batch_size = 1024 if batch_size is None else batch_size
    config_num_workers = 24 if num_workers is None else num_workers

    return CtEchoConfig(
        retain_rate=0.5,
        n_reservoir=100,
        connectivity=1.0,
        spectral_radius=0.95,
        alpha=1e-3,
        leaky=1.0,
        ode_method="dopri5",
        rtol=1e-3,
        atol=1e-4,
        n_forget_points=5,
        batch_size=config_batch_size,
        num_workers=config_num_workers,
        seed=42,
        device=device,
    )


def _get_first(data: np.lib.npyio.NpzFile, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in data.files:
            return data[name]
    available = ", ".join(data.files)
    expected = ", ".join(names)
    raise KeyError(f"Missing one of [{expected}] in npz. Available fields: [{available}]")


def _as_float3(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, length, channels], got {arr.shape}")
    return arr


def _as_int2(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, observed_length], got {arr.shape}")
    return arr


def _as_labels(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D labels, got {arr.shape}")
    return arr.astype(np.int64)


def load_split_npz(npz_path: Path) -> SplitData:
    if not npz_path.exists():
        raise FileNotFoundError(f"npz_path does not exist: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        x_train = _as_float3(
            "x_train",
            _get_first(data, ("x_train_irregular", "X_train_irregular", "x_train", "X_train")),
        )
        x_test = _as_float3(
            "x_test",
            _get_first(data, ("x_test_irregular", "X_test_irregular", "x_test", "X_test")),
        )
        timestamps_train = _as_int2(
            "timestamps_train",
            _get_first(data, ("timestamps_train", "t_train", "timestep_train", "timesteps_train")),
        )
        timestamps_test = _as_int2(
            "timestamps_test",
            _get_first(data, ("timestamps_test", "t_test", "timestep_test", "timesteps_test")),
        )
        y_train = _as_labels("y_train", _get_first(data, ("y_train", "labels_train", "train_labels")))
        y_test = _as_labels("y_test", _get_first(data, ("y_test", "labels_test", "test_labels", "true_labels")))

    if x_train.shape[0] != timestamps_train.shape[0] or x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Train x/timestamps/labels batch sizes do not match")
    if x_test.shape[0] != timestamps_test.shape[0] or x_test.shape[0] != y_test.shape[0]:
        raise ValueError("Test x/timestamps/labels batch sizes do not match")

    return SplitData(
        x_train=x_train,
        timestamps_train=timestamps_train,
        y_train=y_train,
        x_test=x_test,
        timestamps_test=timestamps_test,
        y_test=y_test,
    )


def run(
    npz_path: Path,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: str | None = None,
) -> dict[str, object]:
    split = load_split_npz(npz_path)
    config = build_config(batch_size=batch_size, num_workers=num_workers, device=device)
    model = CtEchoClassifier(config=config)

    print("npz_path =", npz_path)
    print("x_train.shape =", split.x_train.shape)
    print("timestamps_train.shape =", split.timestamps_train.shape)
    print("y_train.shape =", split.y_train.shape)
    print("x_test.shape =", split.x_test.shape)
    print("timestamps_test.shape =", split.timestamps_test.shape)
    print("y_test.shape =", split.y_test.shape)
    print("ctecho_config =", config)

    train_features = model.extract_features(split.x_train, split.timestamps_train)
    test_features = model.extract_features(split.x_test, split.timestamps_test)

    clf = SVC(kernel="rbf", decision_function_shape="ovo", random_state=config.seed)
    clf.fit(train_features, split.y_train)
    train_pred = clf.predict(train_features)
    test_pred = clf.predict(test_features)

    train_acc = float(accuracy_score(split.y_train, train_pred))
    test_acc = float(accuracy_score(split.y_test, test_pred))

    print("train_features.shape =", train_features.shape)
    print("test_features.shape =", test_features.shape)
    print("classifier = SVM")
    print("train_acc =", f"{train_acc:.4f}")
    print("test_acc =", f"{test_acc:.4f}")
    print("classification_report:")
    print(classification_report(split.y_test, test_pred, zero_division=0))
    print("confusion_matrix:")
    print(confusion_matrix(split.y_test, test_pred))

    return {
        "train_features": train_features,
        "test_features": test_features,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load split irregular sequences from npz, extract CtEcho features, and run SVM classification."
    )
    parser.add_argument(
        "npz_path",
        type=Path,
        help="Path to a split npz containing train/test irregular series, timestamps, and labels.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override CtEcho batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override CtEcho num_workers.")
    parser.add_argument("--device", default=None, help="Override device, for example 'cpu' or 'cuda'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.npz_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
