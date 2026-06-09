"""Orquestrador do experimento completo (protocolo da Seção 6.3 do TCC).

Executa, sob protocolo controlado, a comparação entre as três arquiteturas
(baseline, VGG16, ResNet50), opcionalmente com e sem CLAHE, e consolida os
resultados em tabela comparativa, gráficos e relatório.

Exemplos
--------
# Experimento completo (3 modelos, com e sem CLAHE), a partir de dados já divididos:
python -m scripts.run_experiment --split-dir data/split --clahe both

# Apenas um modelo, rápido (poucas épocas) — útil para validar o pipeline:
python -m scripts.run_experiment --models baseline --epochs 2 --quick

# Reconstruir o dataset a partir dos brutos do Kaggle antes de treinar:
python -m scripts.run_experiment --raw-dir data/raw --rebuild
"""

from __future__ import annotations

import argparse
import os
import sys

# Permite executar como `python scripts/run_experiment.py` ou `-m scripts.run_experiment`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CLASSES, RANDOM_SEED, default_config  # noqa: E402
from src import data_setup  # noqa: E402
from src.utils import (ensure_dir, save_json, set_seeds,  # noqa: E402
                       plot_learning_curves)


def parse_args():
    p = argparse.ArgumentParser(description="Comparação de arquiteturas CNN — TCC Pneus")
    p.add_argument("--raw-dir", default=None, help="Datasets brutos do Kaggle")
    p.add_argument("--consolidated-dir", default=None, help="Dataset consolidado (3 classes)")
    p.add_argument("--split-dir", default=None, help="Dataset dividido train/val/test")
    p.add_argument("--outputs-dir", default=None, help="Diretório de saída")
    p.add_argument("--rebuild", action="store_true",
                   help="Reconstrói consolidação + divisão a partir de --raw-dir")
    p.add_argument("--models", nargs="+", default=["baseline", "vgg16", "resnet50"],
                   choices=["baseline", "vgg16", "resnet50"])
    p.add_argument("--clahe", choices=["none", "on", "both"], default="none",
                   help="Experimento auxiliar CLAHE: sem, com, ou ambos")
    p.add_argument("--epochs", type=int, default=None, help="Sobrescreve max_epochs")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--quick", action="store_true",
                   help="Modo rápido: reduz épocas/paciência para validar o pipeline")
    return p.parse_args()


def build_config(args):
    cfg = default_config()
    if args.raw_dir:
        cfg.raw_dir = args.raw_dir
    if args.consolidated_dir:
        cfg.consolidated_dir = args.consolidated_dir
    if args.split_dir:
        cfg.split_dir = args.split_dir
    if args.outputs_dir:
        cfg.outputs_dir = args.outputs_dir
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.quick:
        cfg.max_epochs = min(cfg.max_epochs, 3)
        cfg.early_stopping_patience = 2
    return cfg


def prepare_data(cfg, args):
    """Garante que exista um dataset dividido; (re)constrói se solicitado."""
    if args.rebuild or not os.path.isdir(os.path.join(cfg.split_dir, "train")):
        if os.path.isdir(cfg.raw_dir):
            print(f"[dados] Consolidando datasets de {cfg.raw_dir} ...")
            counts = data_setup.consolidate(cfg.raw_dir, cfg.consolidated_dir)
            print(f"[dados] Consolidado: {counts}")
            print("[dados] Dividindo (70/15/15, estratificado, seed=42) ...")
            summary = data_setup.split_dataset(
                cfg.consolidated_dir, cfg.split_dir,
                ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio),
                seed=RANDOM_SEED,
            )
            print(f"[dados] Divisão: {summary}")
        else:
            raise FileNotFoundError(
                f"Nenhum dataset dividido em {cfg.split_dir} e nenhum bruto em "
                f"{cfg.raw_dir}. Gere dados sintéticos ou baixe os do Kaggle."
            )
    return data_setup.count_split(cfg.split_dir)


def run_single(cfg, name, use_clahe, train_gen, val_gen, test_gen, class_weights):
    """Treina + avalia um modelo e devolve o dicionário consolidado de resultados."""
    from src.models import build_model
    from src.train import train_model
    from src.evaluate import evaluate_model

    tag = f"{name}{'_clahe' if use_clahe else ''}"
    out_sub = os.path.join(cfg.outputs_dir, tag)
    ensure_dir(out_sub)
    print(f"\n=== Treinando {tag} ===")

    model = build_model(name, cfg)
    train_res = train_model(model, cfg, train_gen, val_gen, class_weights)

    plot_learning_curves(train_res["history"], tag,
                         os.path.join(out_sub, "learning_curves.png"))

    eval_res = evaluate_model(model, test_gen, tag, cfg.outputs_dir)
    save_json({"train": train_res, "eval": eval_res},
              os.path.join(out_sub, "results.json"))

    print(f"[{tag}] acurácia={eval_res['accuracy']:.4f} "
          f"f1_macro={eval_res['macro']['f1']:.4f} "
          f"params={train_res['params']['total']:,} "
          f"inf={eval_res['inference_time_ms_per_image']:.2f}ms/img")

    return {"clahe": use_clahe, "train": train_res, "eval": eval_res}


def main():
    args = parse_args()
    cfg = build_config(args)
    set_seeds(RANDOM_SEED)
    ensure_dir(cfg.outputs_dir)

    split_counts = prepare_data(cfg, args)
    print(f"[dados] Contagem por split: {split_counts}")

    class_weights = (data_setup.compute_class_weights(cfg.split_dir)
                     if cfg.use_class_weights else None)
    print(f"[dados] Pesos de classe: {class_weights}")

    clahe_modes = {"none": [False], "on": [True], "both": [False, True]}[args.clahe]

    from src.preprocessing import build_generators
    from src.evaluate import (build_comparison_table, comparison_table_to_markdown)
    from src.error_analysis import error_overlap, summarize_error_overlap

    all_results = []
    evals_no_clahe = {}

    for use_clahe in clahe_modes:
        train_gen, val_gen, test_gen = build_generators(cfg, cfg.split_dir, use_clahe)
        for name in args.models:
            res = run_single(cfg, name, use_clahe, train_gen, val_gen,
                             test_gen, class_weights)
            all_results.append(res)
            if not use_clahe:
                evals_no_clahe[name] = res["eval"]

    # Tabela comparativa consolidada.
    table = build_comparison_table(all_results)
    save_json(table, os.path.join(cfg.outputs_dir, "comparison_table.json"))

    # Análise de erros (sobre os modelos sem CLAHE).
    overlap = error_overlap(evals_no_clahe) if len(evals_no_clahe) >= 2 else {}
    if overlap:
        save_json(overlap, os.path.join(cfg.outputs_dir, "error_overlap.json"))

    # Relatório em Markdown.
    report_path = os.path.join(cfg.outputs_dir, "RELATORIO.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Relatório do Experimento — Comparação de Arquiteturas CNN\n\n")
        f.write("Classificação da condição de pneus (good / worn / cracked).\n\n")
        f.write(f"- Modelos: {', '.join(args.models)}\n")
        f.write(f"- CLAHE: {args.clahe}\n")
        f.write(f"- Semente: {RANDOM_SEED}\n")
        f.write(f"- Imagens por split: `{split_counts}`\n\n")
        f.write("## Tabela comparativa\n\n")
        f.write(comparison_table_to_markdown(table))
        f.write("\n")
        if overlap:
            f.write(summarize_error_overlap(overlap))
    print(f"\n[ok] Relatório salvo em {report_path}")
    print(comparison_table_to_markdown(table))


if __name__ == "__main__":
    main()
