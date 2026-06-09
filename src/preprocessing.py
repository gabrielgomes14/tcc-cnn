"""Pré-processamento e geração de batches (Etapa 2 do protocolo experimental).

Implementa:
  * Redimensionamento para 224x224 e normalização para [0, 1];
  * Data augmentation (rotação, espelhamento horizontal, zoom) — somente no treino;
  * CLAHE no canal de luminância (experimento auxiliar — Seção 9.5 do TCC).

O mesmo pré-processamento é aplicado às três arquiteturas, de modo que diferenças
de desempenho sejam atribuíveis à arquitetura, e não ao tratamento das imagens.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import CLASSES, RANDOM_SEED, Config


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Aplica CLAHE ao canal de luminância de uma imagem RGB.

    A imagem é convertida para o espaço LAB; o CLAHE é aplicado apenas ao canal L
    (luminância), preservando a informação cromática (canais a, b). Conforme a
    Seção 9.5 do TCC, isso realça as texturas locais (sulcos, rachaduras) sem
    distorcer as cores.

    Parameters
    ----------
    image : np.ndarray
        Imagem RGB ``uint8`` (H, W, 3) com valores em [0, 255].
    clip_limit : float
        Limiar de corte do histograma local (Tabela de parâmetros do CLAHE).
    tile_grid : tuple
        Tamanho da grade de tiles.

    Returns
    -------
    np.ndarray
        Imagem RGB ``uint8`` com contraste local realçado.
    """
    import cv2

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def make_clahe_preprocessing(cfg: Config):
    """Cria a função de pré-processamento CLAHE para o ImageDataGenerator.

    O ImageDataGenerator chama esta função com a imagem (float, [0,255]) antes do
    rescale. Retornamos a imagem em [0,255]; a normalização [0,1] é feita pelo
    parâmetro ``rescale`` do gerador, mantendo o protocolo idêntico ao caso sem
    CLAHE (apenas a etapa de realce é adicionada).
    """
    clip = cfg.clahe_clip_limit
    grid = cfg.clahe_tile_grid

    def _fn(img: np.ndarray) -> np.ndarray:
        out = apply_clahe(img.astype(np.uint8), clip_limit=clip, tile_grid=grid)
        return out.astype(np.float32)

    return _fn


def build_generators(cfg: Config, split_dir: str, use_clahe: bool = False):
    """Cria os geradores de dados de treino, validação e teste.

    Usa ``ImageDataGenerator`` do Keras. Augmentation é aplicado apenas ao treino;
    validação e teste recebem somente resize + normalização (e CLAHE, se ativo).

    Returns
    -------
    (train_gen, val_gen, test_gen) : tuple de DirectoryIterator
    """
    import os
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    preprocessing_fn = make_clahe_preprocessing(cfg) if use_clahe else None

    train_datagen = ImageDataGenerator(
        rescale=cfg.rescale,
        rotation_range=cfg.rotation_range,
        horizontal_flip=cfg.horizontal_flip,
        zoom_range=cfg.zoom_range,
        preprocessing_function=preprocessing_fn,
    )
    eval_datagen = ImageDataGenerator(
        rescale=cfg.rescale,
        preprocessing_function=preprocessing_fn,
    )

    common = dict(
        target_size=cfg.image_size,
        batch_size=cfg.batch_size,
        class_mode="categorical",
        classes=list(CLASSES),
        color_mode="rgb",
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(split_dir, "train"), shuffle=True, seed=RANDOM_SEED, **common
    )
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(split_dir, "val"), shuffle=False, **common
    )
    test_gen = eval_datagen.flow_from_directory(
        os.path.join(split_dir, "test"), shuffle=False, **common
    )
    return train_gen, val_gen, test_gen


def preview_clahe(image_path: str, cfg: Config, out_path: str) -> None:
    """Salva uma figura comparando a imagem original e a versão com CLAHE."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from .utils import ensure_dir

    ensure_dir(__import__("os").path.dirname(out_path) or ".")
    img = np.array(Image.open(image_path).convert("RGB").resize(cfg.image_size))
    clahe_img = apply_clahe(img, cfg.clahe_clip_limit, cfg.clahe_tile_grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    ax1.imshow(img); ax1.set_title("Original"); ax1.axis("off")
    ax2.imshow(clahe_img); ax2.set_title("Com CLAHE"); ax2.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
