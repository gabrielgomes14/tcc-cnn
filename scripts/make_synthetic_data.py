"""Gera um dataset SINTÉTICO de pneus para validar o pipeline (smoke test).

ATENÇÃO: estas imagens NÃO são pneus reais — são texturas procedurais com
características visuais distintas por classe (good = sulcos profundos/contraste
alto; worn = textura homogênea/baixo contraste; cracked = linhas de alto
contraste simulando rachaduras). Servem apenas para verificar que a ferramenta
roda de ponta a ponta. Para o experimento real, use os datasets do Kaggle
(scripts/download_data.py).

Cria a estrutura no formato dos datasets brutos (com rótulos de ORIGEM no
caminho), de modo que src.data_setup.consolidate consiga mapeá-las:
  good, defective (->worn), cracked
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image


def _base_tire(size: int, rng: np.random.Generator) -> np.ndarray:
    """Fundo cinza tipo borracha com leve granulado."""
    base = rng.normal(90, 8, (size, size)).clip(0, 255)
    return base


def gen_good(size: int, rng: np.random.Generator) -> np.ndarray:
    """Sulcos profundos e regulares (alto contraste)."""
    img = _base_tire(size, rng)
    period = max(6, size // 16)
    for x in range(0, size, period):
        img[:, x:x + period // 2] -= 55  # sulcos escuros profundos
    img += rng.normal(0, 4, img.shape)
    return img


def gen_worn(size: int, rng: np.random.Generator) -> np.ndarray:
    """Banda desgastada: textura homogênea, sulcos rasos."""
    img = _base_tire(size, rng) + 25
    period = max(10, size // 8)
    for x in range(0, size, period):
        img[:, x:x + period // 3] -= 12  # sulcos rasos, baixo contraste
    img += rng.normal(0, 6, img.shape)
    return img


def gen_cracked(size: int, rng: np.random.Generator) -> np.ndarray:
    """Rachaduras: linhas finas de alto contraste em direções aleatórias."""
    img = _base_tire(size, rng) + 10
    n_cracks = rng.integers(5, 12)
    for _ in range(n_cracks):
        x, y = rng.integers(0, size, 2)
        length = rng.integers(size // 4, size)
        angle = rng.uniform(0, np.pi)
        for t in range(length):
            xx = int(x + t * np.cos(angle))
            yy = int(y + t * np.sin(angle))
            if 0 <= xx < size and 0 <= yy < size:
                img[yy, max(0, xx - 1):xx + 1] -= 70  # rachadura escura
    img += rng.normal(0, 4, img.shape)
    return img


GENERATORS = {
    "good": gen_good,
    "defective": gen_worn,   # rótulo de origem Warcoder -> worn
    "cracked": gen_cracked,
}


def to_rgb_image(arr: np.ndarray) -> Image.Image:
    arr = arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr] * 3, axis=-1), mode="RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join("data", "raw"))
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--size", type=int, default=96,
                    help="Tamanho da imagem sintética (será reescalada a 224 no pipeline)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for label, gen in GENERATORS.items():
        out_dir = os.path.join(args.dest, "synthetic", label)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(args.per_class):
            arr = gen(args.size, rng)
            to_rgb_image(arr).save(os.path.join(out_dir, f"{label}_{i:04d}.png"))
        print(f"[synthetic] {label}: {args.per_class} imagens em {out_dir}")
    print(f"[ok] Dataset sintético gerado em {args.dest}")


if __name__ == "__main__":
    main()
