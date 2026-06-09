"""Utilitários: reprodutibilidade, medição de tempo, E/S de resultados e gráficos."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np


def set_seeds(seed: int) -> None:
    """Fixa as sementes de NumPy, random e TensorFlow para reprodutibilidade.

    Conforme a Seção 6.3.1 do TCC (random_seed = 42 em NumPy, TensorFlow e na
    divisão do dataset).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        # Operações determinísticas quando possível (pode reduzir desempenho).
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    except ImportError:
        pass


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Stopwatch:
    """Cronômetro simples para medir tempos de treinamento/inferência."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start


def count_params(model) -> Dict[str, int]:
    """Conta parâmetros totais e treináveis de um modelo Keras."""
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def plot_learning_curves(history: Dict, title: str, out_path: str) -> None:
    """Salva as curvas de aprendizado (acurácia e perda por época)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(out_path) or ".")
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, history["accuracy"], "o-", label="treino")
    if "val_accuracy" in history:
        ax1.plot(epochs, history["val_accuracy"], "s-", label="validação")
    ax1.set_title(f"{title} — Acurácia")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Acurácia")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["loss"], "o-", label="treino")
    if "val_loss" in history:
        ax2.plot(epochs, history["val_loss"], "s-", label="validação")
    ax2.set_title(f"{title} — Perda")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Perda")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], title: str, out_path: str) -> None:
    """Salva a matriz de confusão como heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(out_path) or ".")
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Matriz de Confusão — {title}")
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
