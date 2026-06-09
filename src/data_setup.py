"""Consolidação e divisão do dataset (Etapa 1 do protocolo experimental).

Combina os dois datasets públicos do Kaggle em um conjunto unificado de três
classes (good, worn, cracked), remove duplicatas entre as classes ``good`` dos
dois conjuntos e realiza a divisão estratificada em treino/validação/teste com
semente fixa (random_seed = 42).

Datasets de origem:
  * Warcoder (2023) — Tyre Quality Classification: classes ``good`` e ``defective``.
  * Bhathena (2021) — Tire Texture Image Recognition: classes ``normal`` e ``cracked``.

Mapeamento (Seção 6.1 do TCC):
  good (Warcoder) + normal (Bhathena) -> good
  defective (Warcoder)                -> worn
  cracked (Bhathena)                  -> cracked
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .config import CLASSES, RANDOM_SEED, SOURCE_LABEL_MAP
from .utils import ensure_dir

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _iter_images(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMAGE_EXTS):
                yield os.path.join(dirpath, fn)


def _file_hash(path: str, chunk: int = 1 << 16) -> str:
    """Hash MD5 do conteúdo do arquivo, para deduplicação exata."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def discover_source_class(path: str) -> str | None:
    """Infere a classe de origem (good/defective/normal/cracked) pelo caminho.

    Procura, sem diferenciar maiúsculas, por um dos rótulos de origem em qualquer
    componente do caminho. Robusto às variações de estrutura de pasta dos dois
    datasets do Kaggle.
    """
    parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
    # Ordena por especificidade para não casar "good" dentro de outra palavra.
    for label in ("defective", "cracked", "normal", "good"):
        for part in parts:
            if part == label or label in part.split("_"):
                return label
    return None


def consolidate(raw_dir: str, consolidated_dir: str, dedup: bool = True) -> Dict[str, int]:
    """Consolida os datasets brutos em ``consolidated_dir`` com 3 classes.

    Retorna a contagem de imagens por classe-alvo. Aplica deduplicação exata por
    hash de conteúdo (relevante sobretudo para a classe ``good``, oriunda de dois
    datasets, evitando redundância — Seção 6.1 do TCC).
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Diretório de dados brutos não encontrado: {raw_dir!r}. "
            "Baixe os datasets do Kaggle (ver scripts/download_data.py) ou gere "
            "dados sintéticos com scripts/make_synthetic_data.py."
        )

    for cls in CLASSES:
        ensure_dir(os.path.join(consolidated_dir, cls))

    seen_hashes: set[str] = set()
    counts: Counter = Counter()
    skipped = 0

    for img_path in _iter_images(raw_dir):
        src_label = discover_source_class(img_path)
        if src_label is None:
            continue
        target = SOURCE_LABEL_MAP.get(src_label)
        if target is None:
            continue

        if dedup:
            digest = _file_hash(img_path)
            if digest in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(digest)

        ext = os.path.splitext(img_path)[1].lower()
        dst_name = f"{target}_{counts[target]:05d}{ext}"
        dst_path = os.path.join(consolidated_dir, target, dst_name)
        shutil.copy2(img_path, dst_path)
        counts[target] += 1

    result = {cls: counts.get(cls, 0) for cls in CLASSES}
    result["_duplicates_skipped"] = skipped
    return result


def split_dataset(
    consolidated_dir: str,
    split_dir: str,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, int]]:
    """Divisão estratificada em train/val/test, preservando a proporção de classes.

    Implementação própria (sem dependência externa obrigatória) que mantém a
    estratificação por classe e usa semente fixa. Cria a estrutura::

        split_dir/{train,val,test}/{good,worn,cracked}/*.jpg
    """
    import random as _random

    assert abs(sum(ratios) - 1.0) < 1e-6, "As proporções devem somar 1.0"
    train_r, val_r, _ = ratios
    rng = _random.Random(seed)

    summary: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for split in ("train", "val", "test"):
        for cls in CLASSES:
            ensure_dir(os.path.join(split_dir, split, cls))

    for cls in CLASSES:
        cls_dir = os.path.join(consolidated_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(IMAGE_EXTS)
        )
        rng.shuffle(files)  # embaralhamento determinístico

        n = len(files)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        # garante que o teste recebe o restante (evita perder amostras por arredondamento)
        buckets = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, items in buckets.items():
            for fn in items:
                shutil.copy2(
                    os.path.join(cls_dir, fn),
                    os.path.join(split_dir, split, cls, fn),
                )
            summary[split][cls] = len(items)

    return summary


def count_split(split_dir: str) -> Dict[str, Dict[str, int]]:
    """Conta imagens por split e classe a partir do diretório dividido."""
    out: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        out[split] = {}
        for cls in CLASSES:
            d = os.path.join(split_dir, split, cls)
            out[split][cls] = (
                len([f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS)])
                if os.path.isdir(d) else 0
            )
    return out


def compute_class_weights(split_dir: str) -> Dict[int, float]:
    """Pesos por classe inversamente proporcionais à frequência no treino.

    Usado para compensar desbalanceamento na função de perda (Buda, Maki e
    Mazurowski, 2018 — citado na Etapa 1 do protocolo).
    """
    counts = count_split(split_dir)["train"]
    total = sum(counts.values())
    n_classes = len([c for c in counts.values() if c > 0]) or 1
    weights: Dict[int, float] = {}
    for idx, cls in enumerate(CLASSES):
        c = counts.get(cls, 0)
        weights[idx] = (total / (n_classes * c)) if c > 0 else 0.0
    return weights
