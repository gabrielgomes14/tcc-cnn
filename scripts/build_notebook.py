"""Gera o notebook do Google Colab (notebooks/TCC_Pneus_Colab.ipynb)."""

import json
import os

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}

cells = [
    md("# Detecção de Desgaste em Pneus com CNNs — TCC",
       "**Gabriel Gomes Galikosky** · IFC Campus Araquari · 2026",
       "",
       "Comparação de três arquiteturas (Baseline, VGG16, ResNet50) para classificar",
       "a condição de pneus em `good`, `worn` e `cracked`.",
       "",
       "> Em **Ambiente de execução → Alterar tipo de hardware**, selecione **GPU (T4)**."),

    md("## 1. Setup — clonar/enviar o projeto e instalar dependências"),
    code("# Se o projeto estiver no GitHub, clone-o; senão, faça upload da pasta src/ e scripts/.",
         "# !git clone https://github.com/<seu-usuario>/tcc-cnn.git && cd tcc-cnn",
         "!pip -q install opencv-python-headless split-folders"),

    md("## 2. Dados",
       "Escolha **A** (datasets reais do Kaggle) ou **B** (sintéticos, só para testar)."),
    code("# --- Opção A: Kaggle (requer kaggle.json) ---",
         "# from google.colab import files; files.upload()  # envie kaggle.json",
         "# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json",
         "# !python scripts/download_data.py --dest data/raw",
         "",
         "# --- Opção B: dados sintéticos (smoke test) ---",
         "!python scripts/make_synthetic_data.py --dest data/raw --per-class 60"),

    md("## 3. Consolidação + divisão estratificada (70/15/15, seed=42)"),
    code("from src import data_setup",
         "from src.config import default_config, RANDOM_SEED",
         "cfg = default_config()",
         "print(data_setup.consolidate(cfg.raw_dir, cfg.consolidated_dir))",
         "print(data_setup.split_dataset(cfg.consolidated_dir, cfg.split_dir,",
         "      ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), seed=RANDOM_SEED))"),

    md("## 4. Visualizar efeito do CLAHE (experimento auxiliar)"),
    code("import glob",
         "from src.preprocessing import preview_clahe",
         "sample = glob.glob('data/consolidated/good/*')[0]",
         "preview_clahe(sample, cfg, 'outputs/clahe_preview.png')",
         "from IPython.display import Image; Image('outputs/clahe_preview.png')"),

    md("## 5. Executar o experimento completo",
       "Os 3 modelos, com e sem CLAHE. Em GPU, use as 50 épocas padrão (sem `--quick`)."),
    code("!python -m scripts.run_experiment --models baseline vgg16 resnet50 --clahe both"),

    md("## 6. Resultados",
       "Tabela comparativa, relatório e gráficos em `outputs/`."),
    code("from IPython.display import Markdown",
         "Markdown(open('outputs/RELATORIO.md', encoding='utf-8').read())"),
    code("from IPython.display import Image",
         "for m in ['baseline','vgg16','resnet50']:",
         "    display(Image(f'outputs/{m}/confusion_matrix.png'))",
         "    display(Image(f'outputs/{m}/learning_curves.png'))"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join("notebooks", "TCC_Pneus_Colab.ipynb")
os.makedirs("notebooks", exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("ok:", out)
