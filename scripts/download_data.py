"""Download dos datasets públicos do Kaggle usados no TCC.

Requer a API do Kaggle configurada (arquivo ~/.kaggle/kaggle.json ou variáveis
KAGGLE_USERNAME / KAGGLE_KEY). No Google Colab, faça upload do kaggle.json.

Datasets:
  * warcoder/tyre-quality-classification   (good / defective)
  * jehanbhathena/tire-texture-...         (normal / cracked)

Após o download, organize as imagens em data/raw/ de modo que cada caminho
contenha o rótulo de origem (good, defective, normal, cracked). O módulo
src.data_setup.discover_source_class infere a classe pelo caminho.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Slugs no Kaggle. O segundo pode variar; ajuste conforme o dataset escolhido.
DATASETS = {
    "warcoder/tyre-quality-classification": "tyre_quality",
    "jehanbhathena/tire-texture-image-recognition": "tire_texture",
}


def download(dest: str):
    os.makedirs(dest, exist_ok=True)
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Instale a API do Kaggle: pip install kaggle", file=sys.stderr)
        sys.exit(1)

    for slug, sub in DATASETS.items():
        out = os.path.join(dest, sub)
        os.makedirs(out, exist_ok=True)
        print(f"[kaggle] Baixando {slug} -> {out}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", out, "--unzip"],
            check=True,
        )
    print(f"[ok] Datasets em {dest}. Confira se os caminhos contêm os rótulos de "
          "origem (good/defective/normal/cracked).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join("data", "raw"))
    download(ap.parse_args().dest)
