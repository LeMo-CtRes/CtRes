from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchcde
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import LinearSVC, SVC
from torchdiffeq import odeint

from utils_data import sample_data_with_nans, seed_everything
from utils_speed import compute_hermite_coeffs_robust


@dataclass
class CtEchoConfig:
    retain_rate: float = 0.9
    n_reservoir: int = 10
    connectivity: float = 1.0
    spectral_radius: float = 0.95
    alpha: float = 1e-3
    leaky: float = 1.0
    ode_method: str = "dopri5"
    rtol: float = 1e-3
    atol: float = 1e-4
    n_forget_points: int = 5
    batch_size: int = 4096
    seed: int = 42
    num_workers: int = 4
    device: Optional[str] = None


def configure_reproducibility(seed: int) -> None:
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_per_sample_observed(x_irregular: np.ndarray) -> np.ndarray:
    x_irregular = np.asarray(x_irregular, dtype=np.float32)
    if x_irregular.ndim != 3:
        raise ValueError(f"x_irregular must have shape [batch, length, num_variables], got {x_irregular.shape}")

    x_tensor = torch.from_numpy(x_irregular)
    mask = ~torch.isnan(x_tensor)
    count = mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)

    safe_x = torch.where(mask, x_tensor, torch.zeros_like(x_tensor))
    mean = safe_x.sum(dim=1, keepdim=True) / count

    sq_diff = torch.where(mask, (x_tensor - mean) ** 2, torch.zeros_like(x_tensor))
    std = torch.sqrt(sq_diff.sum(dim=1, keepdim=True) / count).clamp_min(1e-8)

    normalized = torch.where(mask, (x_tensor - mean) / std, x_tensor)
    return normalized.numpy()


class _CtEchoReservoirFunc(torch.nn.Module):
    def __init__(self, input_dim: int, n_reservoir: int, leaky: float, activation=torch.tanh):
        super().__init__()
        self.leaky = leaky
        self.activation = activation
        self.W_in = torch.nn.Parameter(torch.zeros(input_dim, n_reservoir), requires_grad=False)
        self.W_res = torch.nn.Parameter(torch.zeros(n_reservoir, n_reservoir), requires_grad=False)
        self.register_buffer("spline", None)

    def set_weights(self, W_in: torch.Tensor, W_res: torch.Tensor) -> None:
        self.W_in.data.copy_(W_in)
        self.W_res.data.copy_(W_res)

    def set_spline(self, spline: torchcde.CubicSpline) -> None:
        self.spline = spline

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x_t = self.spline.evaluate(t)
        pre_activation = x_t @ self.W_in + z @ self.W_res
        return -self.leaky * z + self.activation(pre_activation)


def _validate_inputs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    if x.ndim != 3:
        raise ValueError(f"x must have shape [batch, length, num_variables], got {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D labels, got {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have the same batch size, got {x.shape[0]} and {y.shape[0]}")
    return x, y


def _validate_irregular_inputs(x_irregular: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_irregular = np.asarray(x_irregular, dtype=np.float32)
    timestamps = np.asarray(timestamps)
    if x_irregular.ndim != 3:
        raise ValueError(f"x_irregular must have shape [batch, length, num_variables], got {x_irregular.shape}")
    if timestamps.ndim != 2:
        raise ValueError(f"timestamps must have shape [batch, retained_length], got {timestamps.shape}")
    if x_irregular.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"x_irregular and timestamps must have the same batch size, got {x_irregular.shape[0]} and {timestamps.shape[0]}"
        )
    return x_irregular, timestamps


def _validate_coeffs_inputs(coeffs: np.ndarray, x_irregular: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float32)
    if coeffs.ndim != 3:
        raise ValueError(f"coeffs must have shape [batch, length-1, num_variables*4], got {coeffs.shape}")
    if coeffs.shape[0] != x_irregular.shape[0]:
        raise ValueError(
            f"coeffs and x_irregular must have the same batch size, got {coeffs.shape[0]} and {x_irregular.shape[0]}"
        )
    return coeffs


def _build_reservoir_weights(input_dim: int, config: CtEchoConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(config.seed)
    W_in = 2.0 * (rng.rand(input_dim, config.n_reservoir) - 0.5)
    W_res = rng.rand(config.n_reservoir, config.n_reservoir) - 0.5
    W_res *= (rng.rand(config.n_reservoir, config.n_reservoir) < config.connectivity)
    rho = np.max(np.abs(np.linalg.eigvals(W_res)))
    if rho > 0:
        W_res = W_res * (config.spectral_radius / (rho + 1e-12))
    return W_in.astype(np.float32), W_res.astype(np.float32)


def _find_valid_pairs(ts_batch: torch.LongTensor, n_forget_points: int) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    bs, T = ts_batch.shape
    device = ts_batch.device
    pos = torch.arange(T - 1, device=device)
    valid = (ts_batch[:, 1:] - ts_batch[:, :-1] == 1) & (pos[None, :] >= n_forget_points)
    counts = valid.sum(dim=1)
    max_count = int(counts.max().item())
    if max_count == 0:
        raise ValueError("No valid consecutive observation pairs were found. Increase retain_rate or reduce n_forget_points.")

    score = torch.where(valid, -pos.unsqueeze(0), torch.full((bs, T - 1), -T, device=device))
    _, origin_idx = torch.topk(score, k=max_count, dim=1)
    mask = torch.arange(max_count, device=device).unsqueeze(0) < counts.unsqueeze(1)
    return origin_idx, mask


def _batch_extract_readout_weights(
    coeffs_batch: torch.Tensor,
    x_miss_batch: torch.Tensor,
    ts_batch: torch.LongTensor,
    W_in: torch.Tensor,
    W_res: torch.Tensor,
    config: CtEchoConfig,
) -> torch.Tensor:
    device = coeffs_batch.device
    batch_size, seq_len, input_dim = x_miss_batch.shape
    spline = torchcde.CubicSpline(coeffs_batch)
    func = _CtEchoReservoirFunc(input_dim, config.n_reservoir, config.leaky).to(device)
    func.set_weights(W_in, W_res)
    func.set_spline(spline)

    z0 = torch.zeros(batch_size, config.n_reservoir, device=device)
    t = torch.linspace(0, seq_len - 1, seq_len, device=device)
    z_all = odeint(func, z0, t, method=config.ode_method, rtol=config.rtol, atol=config.atol)
    states = torch.tanh(z_all).permute(1, 0, 2)

    origin_idx, mask = _find_valid_pairs(ts_batch, config.n_forget_points)
    gathered_ts = ts_batch.gather(1, origin_idx)

    Xf = states.gather(1, gathered_ts.unsqueeze(-1).expand(batch_size, origin_idx.size(1), config.n_reservoir))
    Y = x_miss_batch.gather(1, gathered_ts.unsqueeze(-1).expand(batch_size, origin_idx.size(1), input_dim))

    mask3 = mask.unsqueeze(-1)
    Xf = Xf * mask3
    Y = Y * mask3

    XtX = Xf.transpose(1, 2) @ Xf
    XtY = Xf.transpose(1, 2) @ Y
    eye = torch.eye(config.n_reservoir, device=device).unsqueeze(0)
    return torch.linalg.solve(XtX + config.alpha * eye, XtY)


class CtEchoClassifier:
    def __init__(self, config: Optional[CtEchoConfig] = None):
        self.config = config or CtEchoConfig()
        self.classifiers_: Dict[str, object] = {}
        self.features_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def extract_features(self, x_irregular: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        x_irregular, timestamps = _validate_irregular_inputs(x_irregular, timestamps)
        configure_reproducibility(self.config.seed)
        x_norm = normalize_per_sample_observed(x_irregular)
        coeffs = compute_hermite_coeffs_robust(
            x_norm,
            strategy="multiprocess",
            num_workers=self.config.num_workers,
            verbose=False,
        )
        return self.extract_features_from_coeffs(coeffs, x_irregular, timestamps)

    def extract_features_from_coeffs(
        self,
        coeffs: np.ndarray,
        x_irregular: np.ndarray,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        x_irregular, timestamps = _validate_irregular_inputs(x_irregular, timestamps)
        coeffs = _validate_coeffs_inputs(coeffs, x_irregular)
        configure_reproducibility(self.config.seed)
        x_norm = normalize_per_sample_observed(x_irregular)

        device_name = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_name)

        W_in_np, W_res_np = _build_reservoir_weights(x_irregular.shape[2], self.config)
        W_in = torch.from_numpy(W_in_np).to(device)
        W_res = torch.from_numpy(W_res_np).to(device)
        coeffs_t = torch.from_numpy(coeffs).to(device)
        x_norm_t = torch.from_numpy(x_norm).to(device)
        ts_t = torch.from_numpy(timestamps).to(device=device, dtype=torch.long)

        readouts = []
        total = x_irregular.shape[0]
        for start in range(0, total, self.config.batch_size):
            end = min(start + self.config.batch_size, total)
            W_out = _batch_extract_readout_weights(
                coeffs_t[start:end],
                x_norm_t[start:end],
                ts_t[start:end],
                W_in,
                W_res,
                self.config,
            )
            readouts.append(W_out.detach().cpu().numpy())

        features = np.concatenate(readouts, axis=0).reshape(total, -1)
        return features

    def fit(
        self,
        x_irregular: np.ndarray,
        timestamps: np.ndarray,
        y: np.ndarray,
        *,
        test_size: float = 0.2,
        stratify: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        configure_reproducibility(self.config.seed)
        x_irregular, timestamps = _validate_irregular_inputs(x_irregular, timestamps)
        _, y = _validate_inputs(x_irregular, y)
        features = self.extract_features(x_irregular, timestamps)
        return self.fit_from_features(features, y, test_size=test_size, stratify=stratify)

    def fit_from_coeffs(
        self,
        coeffs: np.ndarray,
        x_irregular: np.ndarray,
        timestamps: np.ndarray,
        y: np.ndarray,
        *,
        test_size: float = 0.2,
        stratify: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        configure_reproducibility(self.config.seed)
        x_irregular, timestamps = _validate_irregular_inputs(x_irregular, timestamps)
        _, y = _validate_inputs(x_irregular, y)
        features = self.extract_features_from_coeffs(coeffs, x_irregular, timestamps)
        return self.fit_from_features(features, y, test_size=test_size, stratify=stratify)

    def fit_from_features(
        self,
        features: np.ndarray,
        y: np.ndarray,
        *,
        test_size: float = 0.2,
        stratify: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        configure_reproducibility(self.config.seed)
        features = np.asarray(features, dtype=np.float32)
        y = np.asarray(y)
        if features.ndim != 2:
            raise ValueError(f"features must have shape [batch, feature_dim], got {features.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D labels, got {y.shape}")
        if features.shape[0] != y.shape[0]:
            raise ValueError(f"features and y must have the same batch size, got {features.shape[0]} and {y.shape[0]}")

        self.features_ = features
        self.labels_ = y

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            y,
            test_size=test_size,
            random_state=self.config.seed,
            stratify=y if stratify else None,
        )

        classifiers = {
            "LinearSVM": LinearSVC(random_state=self.config.seed, max_iter=3000),
            "RidgeClassifierCV": RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=self.config.seed),
            "SVM": SVC(kernel="rbf", decision_function_shape="ovo", random_state=self.config.seed),
            "KNN": KNeighborsClassifier(n_neighbors=10),
        }

        results: Dict[str, Dict[str, float]] = {}
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            train_pred = clf.predict(X_train)
            test_pred = clf.predict(X_test)
            self.classifiers_[name] = clf
            results[name] = {
                "train_acc": float(accuracy_score(y_train, train_pred)),
                "test_acc": float(accuracy_score(y_test, test_pred)),
                "precision": float(precision_score(y_test, test_pred, average="macro", zero_division=0)),
                "recall": float(recall_score(y_test, test_pred, average="macro", zero_division=0)),
                "f1_score": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
            }

        return results

    def fit_with_split(
        self,
        x_train_irregular: np.ndarray,
        ts_train: np.ndarray,
        y_train: np.ndarray,
        x_test_irregular: np.ndarray,
        ts_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        configure_reproducibility(self.config.seed)
        x_train_irregular, ts_train = _validate_irregular_inputs(x_train_irregular, ts_train)
        x_test_irregular, ts_test = _validate_irregular_inputs(x_test_irregular, ts_test)
        _, y_train = _validate_inputs(x_train_irregular, y_train)
        _, y_test = _validate_inputs(x_test_irregular, y_test)
        train_features = self.extract_features(x_train_irregular, ts_train)
        test_features = self.extract_features(x_test_irregular, ts_test)

        classifiers = {
            "LinearSVM": LinearSVC(random_state=self.config.seed, max_iter=3000),
            "RidgeClassifierCV": RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=self.config.seed),
            "SVM": SVC(kernel="rbf", decision_function_shape="ovo", random_state=self.config.seed),
            "KNN": KNeighborsClassifier(n_neighbors=10),
        }

        self.features_ = np.concatenate([train_features, test_features], axis=0)
        self.labels_ = np.concatenate([y_train, y_test], axis=0)

        results: Dict[str, Dict[str, float]] = {}
        for name, clf in classifiers.items():
            clf.fit(train_features, y_train)
            train_pred = clf.predict(train_features)
            test_pred = clf.predict(test_features)
            self.classifiers_[name] = clf
            results[name] = {
                "train_acc": float(accuracy_score(y_train, train_pred)),
                "test_acc": float(accuracy_score(y_test, test_pred)),
                "precision": float(precision_score(y_test, test_pred, average="macro", zero_division=0)),
                "recall": float(recall_score(y_test, test_pred, average="macro", zero_division=0)),
                "f1_score": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
            }

        return results

    def tsne_embedding(self, features: Optional[np.ndarray] = None, random_state: Optional[int] = None) -> np.ndarray:
        configure_reproducibility(self.config.seed if random_state is None else random_state)
        data = self.features_ if features is None else np.asarray(features)
        if data is None:
            raise ValueError("No features available. Run fit/extract_features first or pass features explicitly.")
        tsne = TSNE(n_components=2, random_state=self.config.seed if random_state is None else random_state, init="pca")
        return tsne.fit_transform(data)
