"""Avaliação preditiva e custo de inferência (Etapas 4 e 5 do protocolo).

Computa acurácia, precisão, recall e F1-Score (por classe e agregados em macro e
weighted), a matriz de confusão, e o tempo médio de inferência por imagem.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from .config import CLASSES
from .utils import plot_confusion_matrix


def _predictions(model, test_gen) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (y_true, y_pred) como índices de classe, na ordem do gerador."""
    test_gen.reset()
    probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_gen.classes[: len(y_pred)]
    return np.asarray(y_true), np.asarray(y_pred)


def evaluate_model(model, test_gen, name: str, outputs_dir: str) -> Dict:
    """Avalia o modelo no conjunto de teste e devolve um dicionário de métricas.

    Inclui relatório por classe (precision/recall/f1/support), métricas agregadas
    (macro e weighted), acurácia, matriz de confusão e tempo de inferência por
    imagem. Também salva a figura da matriz de confusão.
    """
    import os

    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)

    y_true, y_pred = _predictions(model, test_gen)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASSES))),
        target_names=list(CLASSES),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    acc = float(accuracy_score(y_true, y_pred))

    cm_path = os.path.join(outputs_dir, name, "confusion_matrix.png")
    plot_confusion_matrix(cm, list(CLASSES), name, cm_path)

    inference = measure_inference_time(model, test_gen)

    per_class = {
        cls: {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }
        for cls in CLASSES
    }

    return {
        "model": name,
        "accuracy": acc,
        "per_class": per_class,
        "macro": {
            "precision": float(report["macro avg"]["precision"]),
            "recall": float(report["macro avg"]["recall"]),
            "f1": float(report["macro avg"]["f1-score"]),
        },
        "weighted": {
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1": float(report["weighted avg"]["f1-score"]),
        },
        "confusion_matrix": cm.tolist(),
        "inference_time_ms_per_image": inference,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def measure_inference_time(model, test_gen, warmup: int = 1, runs: int = 3) -> float:
    """Mede o tempo médio de inferência por imagem (em milissegundos).

    Faz algumas passagens completas pelo conjunto de teste (após um warmup) e
    divide o tempo total pelo número de imagens processadas.
    """
    n_images = len(test_gen.classes)
    if n_images == 0:
        return 0.0

    for _ in range(warmup):
        test_gen.reset()
        model.predict(test_gen, verbose=0)

    t0 = time.perf_counter()
    for _ in range(runs):
        test_gen.reset()
        model.predict(test_gen, verbose=0)
    elapsed = time.perf_counter() - t0

    return (elapsed / (runs * n_images)) * 1000.0


def build_comparison_table(results: List[Dict]) -> List[Dict]:
    """Monta a tabela comparativa consolidada entre arquiteturas (Seção 6.4).

    Cada linha reúne desempenho preditivo (acurácia, F1 macro/weighted) e custo
    computacional (parâmetros, tempo de treino, inferência).
    """
    rows = []
    for r in results:
        rows.append({
            "modelo": r["eval"]["model"],
            "clahe": r.get("clahe", False),
            "acuracia": round(r["eval"]["accuracy"], 4),
            "f1_macro": round(r["eval"]["macro"]["f1"], 4),
            "f1_weighted": round(r["eval"]["weighted"]["f1"], 4),
            "f1_good": round(r["eval"]["per_class"]["good"]["f1"], 4),
            "f1_worn": round(r["eval"]["per_class"]["worn"]["f1"], 4),
            "f1_cracked": round(r["eval"]["per_class"]["cracked"]["f1"], 4),
            "params_total": r["train"]["params"]["total"],
            "params_trainable": r["train"]["params"]["trainable"],
            "epocas": r["train"]["epochs_trained"],
            "tempo_por_epoca_s": round(r["train"]["time_per_epoch_s"], 2),
            "tempo_total_treino_s": round(r["train"]["total_train_time_s"], 2),
            "inferencia_ms_img": round(r["eval"]["inference_time_ms_per_image"], 3),
        })
    return rows


def comparison_table_to_markdown(rows: List[Dict]) -> str:
    """Renderiza a tabela comparativa em Markdown."""
    if not rows:
        return "_(sem resultados)_\n"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"
