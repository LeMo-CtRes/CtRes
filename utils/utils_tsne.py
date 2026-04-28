from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

TASKS = ("pitch_bearing", "gearbox", "generator", "blade", "main_bearing")
TASK_DISPLAY_NAME = {
    "pitch_bearing": "Pitch-bearing",
    "gearbox": "Gearbox",
    "generator": "Generator",
    "blade": "Blade",
    "main_bearing": "Main-bearing",
}
PLOT_CLASS_COLORS = ["#e79aac", "#7f99c1", "#6cc4b3"]
SEED = 42
N_JOBS = 1

TASK_TSNE_PARAMS: dict[str, dict] = {
    "pitch_bearing": {
        "tsne": {"perplexity": 50.0, "learning_rate": 500.0, "early_exaggeration": 24.0},
        "rotation": {"theta_rad": 3.0283035981205355, "rotate_steps": 721},
        "separation": {"strength_selected": 0.4},
    },
    "gearbox": {
        "tsne": {"perplexity": 80.0, "learning_rate": 500.0, "early_exaggeration": 24.0},
        "rotation": {"theta_rad": 3.023946326756333, "rotate_steps": 721},
        "separation": {"strength_selected": 0.4},
    },
    "generator": {
        "tsne": {"perplexity": 80.0, "learning_rate": 200.0, "early_exaggeration": 12.0},
        "rotation": {"theta_rad": 0.6230898050809159, "rotate_steps": 721},
        "separation": {"strength_selected": 0.4},
    },
    "blade": {
        "tsne": {"perplexity": 30.0, "learning_rate": 200.0, "early_exaggeration": 12.0},
        "rotation": {"theta_rad": 0.8191670164700153, "rotate_steps": 721},
        "separation": {"strength_selected": 0.4},
    },
    "main_bearing": {
        "tsne": {"perplexity": 50.0, "learning_rate": 200.0, "early_exaggeration": 24.0},
        "rotation": {"theta_rad": 2.178635682101105, "rotate_steps": 721},
        "separation": {"strength_selected": 0.4},
    },
}


@dataclass(frozen=True)
class TsnePlotParams:
    perplexity: float = 30.0
    learning_rate: float = 200.0
    early_exaggeration: float = 12.0
    separation_strength: float = 0.0
    max_iter: int = 3200
    rotate_steps: int = 721
    rotation_theta_rad: Optional[float] = None


def params_for_task(task_code: str) -> TsnePlotParams:
    task_params = TASK_TSNE_PARAMS.get(task_code.lower())
    if task_params is None:
        return TsnePlotParams()
    return TsnePlotParams(
        perplexity=float(task_params["tsne"]["perplexity"]),
        learning_rate=float(task_params["tsne"]["learning_rate"]),
        early_exaggeration=float(task_params["tsne"]["early_exaggeration"]),
        separation_strength=float(task_params["separation"]["strength_selected"]),
        rotate_steps=int(task_params["rotation"]["rotate_steps"]),
        rotation_theta_rad=float(task_params["rotation"]["theta_rad"]),
    )


def _compute_tsne(
    x: np.ndarray,
    *,
    seed: int,
    perplexity: float,
    learning_rate: float,
    early_exaggeration: float,
    max_iter: int,
    n_jobs: int,
) -> np.ndarray:
    try:
        model = TSNE(
            n_components=2,
            init="pca",
            random_state=seed,
            perplexity=perplexity,
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            max_iter=max_iter,
            n_jobs=n_jobs,
        )
    except TypeError:
        model = TSNE(
            n_components=2,
            init="pca",
            random_state=seed,
            perplexity=perplexity,
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            n_iter=max_iter,
        )
    return model.fit_transform(x)


def _prepare_scaled_pca(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    pca_cap: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    pca_dim = max(2, min(pca_cap, x_train_s.shape[1], x_train_s.shape[0] - 1))
    pca = PCA(n_components=pca_dim, random_state=seed)
    return pca.fit_transform(x_train_s), pca.transform(x_test_s)


def run_tsne_from_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    params: TsnePlotParams,
    *,
    seed: int = SEED,
    n_jobs: int = N_JOBS,
) -> tuple[np.ndarray, np.ndarray]:
    x_train_p, x_test_p = _prepare_scaled_pca(x_train, x_test, seed=seed)
    x_all = np.concatenate([x_train_p, x_test_p], axis=0)
    emb_all = _compute_tsne(
        x_all,
        seed=seed,
        perplexity=float(params.perplexity),
        learning_rate=float(params.learning_rate),
        early_exaggeration=float(params.early_exaggeration),
        max_iter=int(params.max_iter),
        n_jobs=n_jobs,
    )
    split_at = x_train.shape[0]
    return emb_all[:split_at], emb_all[split_at:]


def _fisher_score_1d(x: np.ndarray, y: np.ndarray) -> float:
    classes = np.unique(y)
    if classes.size <= 1:
        return 0.0
    mean_all = float(x.mean())
    between = 0.0
    within = 0.0
    for cls in classes:
        values = x[y == cls]
        if values.size == 0:
            continue
        mean_cls = float(values.mean())
        between += values.size * (mean_cls - mean_all) ** 2
        within += values.size * float(values.var())
    return float(between / (within + 1e-12))


def _rotate_2d(emb: np.ndarray, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return emb @ np.array([[c, -s], [s, c]], dtype=np.float64)


def _canonicalize_pair(
    emb_train: np.ndarray,
    emb_test: np.ndarray,
    labels_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_out = emb_train.copy()
    test_out = emb_test.copy()
    if _fisher_score_1d(train_out[:, 1], labels_train) > _fisher_score_1d(train_out[:, 0], labels_train):
        train_out = train_out[:, [1, 0]]
        test_out = test_out[:, [1, 0]]
    classes = np.unique(labels_train)
    if classes.size == 2 and 0 in classes and 1 in classes:
        if float(train_out[labels_train == 1, 0].mean()) > float(train_out[labels_train == 0, 0].mean()):
            train_out[:, 0] *= -1.0
            test_out[:, 0] *= -1.0
    return train_out, test_out


def _optimize_rotation_2d(
    emb_train: np.ndarray,
    emb_test: np.ndarray,
    labels_train: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    best_theta = 0.0
    best_score = -1e9
    for theta in np.linspace(0.0, math.pi, max(32, steps), endpoint=False):
        rotated = _rotate_2d(emb_train, float(theta))
        score_x = _fisher_score_1d(rotated[:, 0], labels_train)
        score_y = _fisher_score_1d(rotated[:, 1], labels_train)
        score = float(min(score_x, score_y) + 0.1 * (score_x + score_y))
        if score > best_score:
            best_score = score
            best_theta = float(theta)
    return _canonicalize_pair(
        _rotate_2d(emb_train, best_theta),
        _rotate_2d(emb_test, best_theta),
        labels_train,
    )


def _enhance_class_separation_linear(
    emb_train: np.ndarray,
    emb_test: np.ndarray,
    labels_train: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-8 or np.unique(labels_train).size < 2:
        return emb_train.copy(), emb_test.copy()

    x_train = emb_train.astype(np.float64)
    x_test = emb_test.astype(np.float64)
    mean_all = x_train.mean(axis=0, keepdims=True)
    sw = np.zeros((2, 2), dtype=np.float64)
    sb = np.zeros((2, 2), dtype=np.float64)
    for cls in np.unique(labels_train):
        x_cls = x_train[labels_train == cls]
        if x_cls.shape[0] < 2:
            continue
        mean_cls = x_cls.mean(axis=0, keepdims=True)
        sw += (x_cls - mean_cls).T @ (x_cls - mean_cls)
        delta = (mean_cls - mean_all).reshape(2, 1)
        sb += x_cls.shape[0] * (delta @ delta.T)

    transform = np.linalg.pinv(sw + 1e-4 * np.eye(2)) @ (sb + 1e-4 * np.eye(2))
    values, vectors = np.linalg.eig(transform)
    direction = vectors[:, np.argsort(values.real)[::-1][0]].real
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        direction = np.array([1.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    basis = np.stack([direction, np.array([-direction[1], direction[0]], dtype=np.float64)], axis=1)
    projected_train = (x_train - mean_all) @ basis
    projected_test = (x_test - mean_all) @ basis

    def standardize(z: np.ndarray) -> np.ndarray:
        z = z - z.mean(axis=0, keepdims=True)
        std = z.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        return z / std

    x_train_std = standardize(x_train)
    projected_train_std = standardize(projected_train)

    x_test_centered = x_test - x_train.mean(axis=0, keepdims=True)
    x_train_scale = x_train.std(axis=0, keepdims=True)
    x_train_scale[x_train_scale < 1e-8] = 1.0
    x_test_std = x_test_centered / x_train_scale

    projected_test_centered = projected_test - projected_train.mean(axis=0, keepdims=True)
    projected_train_scale = projected_train.std(axis=0, keepdims=True)
    projected_train_scale[projected_train_scale < 1e-8] = 1.0
    projected_test_std = projected_test_centered / projected_train_scale

    enhanced_train = ((1.0 - strength) * x_train_std + strength * projected_train_std).astype(np.float64)
    enhanced_test = ((1.0 - strength) * x_test_std + strength * projected_test_std).astype(np.float64)
    return enhanced_train, enhanced_test


def _get_class_colors(labels: np.ndarray) -> dict[int, str]:
    return {int(cls): PLOT_CLASS_COLORS[i] for i, cls in enumerate(np.unique(labels))}


def _plot_task_panel(fig, subplot_spec, task_code: str, labels: np.ndarray, emb_final: np.ndarray) -> None:
    inner = GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=subplot_spec,
        height_ratios=[0.28, 0.12, 1.0],
        width_ratios=[1.0, 0.10],
        hspace=0.0,
        wspace=0.0,
    )
    ax_title = fig.add_subplot(inner[0, :])
    ax_top = fig.add_subplot(inner[1, 0])
    ax_main = fig.add_subplot(inner[2, 0], sharex=ax_top)
    ax_right = fig.add_subplot(inner[2, 1], sharey=ax_main)

    x = np.asarray(emb_final[:, 0], dtype=np.float64)
    y = np.asarray(emb_final[:, 1], dtype=np.float64)
    colors = _get_class_colors(labels)

    for cls in np.unique(labels):
        mask = labels == cls
        ax_main.scatter(x[mask], y[mask], s=6.0, alpha=1.0, edgecolors="none", color=colors[int(cls)])

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_max <= x_min:
        x_max = x_min + 1e-3
    if y_max <= y_min:
        y_max = y_min + 1e-3
    x_pad = 0.035 * (x_max - x_min)
    y_pad = 0.035 * (y_max - y_min)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_top.set_xlim(x_min, x_max)
    ax_right.set_ylim(y_min, y_max)

    gx = np.linspace(x_min, x_max, 300)
    gy = np.linspace(y_min, y_max, 300)
    for cls in np.unique(labels):
        mask = labels == cls
        color = colors[int(cls)]
        xc = x[mask]
        yc = y[mask]
        if xc.size >= 2 and np.std(xc) > 1e-12:
            try:
                dx = gaussian_kde(xc)(gx)
                ax_top.fill_between(gx, 0, dx, color=color, alpha=0.32, linewidth=0)
                ax_top.plot(gx, dx, color=color, linewidth=1.1, alpha=0.95)
            except Exception:
                pass
        if yc.size >= 2 and np.std(yc) > 1e-12:
            try:
                dy = gaussian_kde(yc)(gy)
                ax_right.fill_betweenx(gy, 0, dy, color=color, alpha=0.32, linewidth=0)
                ax_right.plot(dy, gy, color=color, linewidth=1.1, alpha=0.95)
            except Exception:
                pass

    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.80,
        TASK_DISPLAY_NAME.get(task_code, task_code),
        ha="center",
        va="top",
        fontsize=13.5,
        fontweight="bold",
    )
    ax_main.tick_params(axis="both", labelsize=11, pad=1)
    for spine in ax_main.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    for ax in (ax_top, ax_right):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", length=0, width=0)
    ax_top.yaxis.set_visible(False)
    ax_right.yaxis.set_visible(False)
    ax_top.spines["bottom"].set_visible(True)
    ax_top.spines["bottom"].set_linewidth(0.7)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_top.set_ylim(bottom=0)
    ax_right.set_xlim(left=0)


def plot_best_1x5(task_plot_data: dict[str, dict[str, np.ndarray]], *, show: bool = True):
    fig = plt.figure(figsize=(2.95 * len(TASKS), 2.85), dpi=220)
    outer = fig.add_gridspec(1, len(TASKS), wspace=0.04)
    for i, task in enumerate(TASKS):
        task_data = task_plot_data.get(task)
        if task_data is None:
            continue
        _plot_task_panel(
            fig,
            outer[0, i],
            task,
            np.asarray(task_data["labels"], dtype=np.int64),
            np.asarray(task_data["emb_final"], dtype=np.float64),
        )
    if show:
        plt.show()
    return fig


def plot_single_task_tsne(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    task_code: str,
    params: Optional[TsnePlotParams] = None,
    show: bool = False,
) -> dict[str, np.ndarray]:
    params = params or params_for_task(task_code)
    labels_train = np.asarray(y_train, dtype=np.int64)
    labels = np.asarray(y_test, dtype=np.int64)
    emb_train_raw, emb_test_raw = run_tsne_from_features(x_train, x_test, params)
    if params.rotation_theta_rad is None:
        emb_train_rot, emb_test_rot = _optimize_rotation_2d(
            emb_train_raw,
            emb_test_raw,
            labels_train,
            steps=int(params.rotate_steps),
        )
    else:
        emb_train_rot, emb_test_rot = _canonicalize_pair(
            _rotate_2d(emb_train_raw, float(params.rotation_theta_rad)),
            _rotate_2d(emb_test_raw, float(params.rotation_theta_rad)),
            labels_train,
        )
    emb_train_final, emb_test_final = _enhance_class_separation_linear(
        emb_train_rot,
        emb_test_rot,
        labels_train,
        strength=float(params.separation_strength),
    )
    result = {
        "emb_train_raw": emb_train_raw,
        "emb_test_raw": emb_test_raw,
        "emb_train_rot": emb_train_rot,
        "emb_test_rot": emb_test_rot,
        "emb_train_final": emb_train_final,
        "emb_final": emb_test_final,
        "labels": labels,
    }
    if show:
        plot_best_1x5({task_code: {"labels": labels, "emb_final": emb_test_final}}, show=True)
    return result
