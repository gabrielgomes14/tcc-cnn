"""Análise de erros entre arquiteturas (Etapa 6 do protocolo).

Verifica se os erros das três arquiteturas são correlacionados (mesmas imagens
difíceis para todos) ou distintos (cada arquitetura erra em conjuntos diferentes),
conforme a Seção 6.4 do TCC ("sobreposição dos exemplos errados entre arquiteturas").
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import numpy as np


def error_overlap(evals: Dict[str, Dict]) -> Dict:
    """Calcula a sobreposição dos erros entre os modelos (sem CLAHE, por padrão).

    Parameters
    ----------
    evals : dict
        Mapa nome_do_modelo -> dicionário de avaliação (com ``y_true`` e ``y_pred``).

    Returns
    -------
    dict com:
        per_model_errors  : nº de erros de cada modelo;
        common_errors     : nº de imagens erradas por TODOS os modelos;
        unique_errors     : nº de imagens erradas por apenas um modelo;
        pairwise_jaccard  : índice de Jaccard das máscaras de erro entre pares;
        common_indices    : índices das imagens erradas por todos.
    """
    names = list(evals.keys())
    if not names:
        return {}

    # Máscara booleana de erro por modelo (alinhadas pela ordem do test_gen).
    masks: Dict[str, np.ndarray] = {}
    n = None
    for name in names:
        y_true = np.asarray(evals[name]["y_true"])
        y_pred = np.asarray(evals[name]["y_pred"])
        m = y_true != y_pred
        masks[name] = m
        n = len(m) if n is None else min(n, len(m))

    # Trunca para o mesmo comprimento (defensivo).
    for name in names:
        masks[name] = masks[name][:n]

    stacked = np.vstack([masks[name] for name in names])  # (n_models, n_images)
    all_wrong = np.all(stacked, axis=0)
    sum_wrong = np.sum(stacked, axis=0)
    unique_wrong = sum_wrong == 1

    pairwise = {}
    for a, b in combinations(names, 2):
        inter = np.sum(masks[a] & masks[b])
        union = np.sum(masks[a] | masks[b])
        pairwise[f"{a}__{b}"] = float(inter / union) if union > 0 else 0.0

    return {
        "models": names,
        "n_images": int(n),
        "per_model_errors": {name: int(masks[name].sum()) for name in names},
        "common_errors": int(all_wrong.sum()),
        "unique_errors": int(unique_wrong.sum()),
        "pairwise_jaccard": pairwise,
        "common_indices": np.where(all_wrong)[0].tolist(),
    }


def summarize_error_overlap(overlap: Dict) -> str:
    """Texto interpretativo da análise de sobreposição de erros."""
    if not overlap:
        return "Sem dados suficientes para análise de erros.\n"

    lines = ["### Análise de sobreposição de erros\n"]
    lines.append(f"- Imagens de teste avaliadas: **{overlap['n_images']}**")
    for name, e in overlap["per_model_errors"].items():
        lines.append(f"- Erros do modelo `{name}`: **{e}**")
    lines.append(f"- Erros comuns a TODOS os modelos: **{overlap['common_errors']}** "
                 "(imagens intrinsecamente difíceis)")
    lines.append(f"- Erros exclusivos de um único modelo: **{overlap['unique_errors']}** "
                 "(específicos da arquitetura)")
    lines.append("\n**Jaccard par a par (interseção/união das máscaras de erro):**")
    for pair, j in overlap["pairwise_jaccard"].items():
        a, b = pair.split("__")
        lines.append(f"- `{a}` vs `{b}`: {j:.3f}")
    lines.append("")
    return "\n".join(lines)
