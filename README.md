# Detecção de Desgaste em Pneus Automotivos com CNNs

Ferramenta de **comparação sistemática de três arquiteturas de Redes Neurais
Convolucionais** para classificação automática da condição de pneus automotivos
em três classes — `good` (bom estado), `worn` (desgastado) e `cracked` (rachado).

Implementação do Trabalho de Conclusão de Curso de **Gabriel Gomes Galikosky**,
Bacharelado em Sistemas de Informação — Instituto Federal Catarinense, Campus
Araquari (2026). Orientador: Prof. Dr. Paulo Cesar Fernandes de Oliveira.

> O objetivo central **não** é propor um sistema de inspeção, mas **gerar
> conhecimento sobre o comportamento relativo das arquiteturas** num domínio de
> classificação de imagens com texturas visualmente semelhantes entre classes.

## Arquiteturas comparadas

| Modelo | Estratégia | Paradigma |
|---|---|---|
| **A — Baseline** | Treinada do zero | CNN simples (controle experimental) |
| **B — VGG16** | Transfer learning (ImageNet) + fine-tuning do bloco 5 | Redes profundas com filtros pequenos |
| **C — ResNet50** | Transfer learning (ImageNet) + fine-tuning do conv5 | Conexões residuais |

Todas são treinadas sob **protocolo controlado**: mesmos dados, mesmos
hiperparâmetros gerais e mesmo pré-processamento, isolando a arquitetura como
única variável (Tabela 2 do TCC).

## O que a ferramenta mede

- **Desempenho preditivo:** acurácia, precisão, recall e F1-Score — por classe e
  agregados (macro e weighted) — além da matriz de confusão e curvas de aprendizado.
- **Custo computacional:** nº de parâmetros (total e treináveis), tempo por época,
  tempo total de treinamento e tempo médio de inferência por imagem.
- **Análise de erros:** sobreposição dos erros entre as três arquiteturas (erros
  comuns vs. específicos de cada modelo; Jaccard par a par).
- **Experimento auxiliar — CLAHE:** comparação com e sem realce de contraste local
  (CLAHE no canal de luminância L do espaço LAB).

## Estrutura do projeto

```
tcc-cnn/
├── src/
│   ├── config.py          # hiperparâmetros (Tabela 2) e estratégia de TL (Tabela 6)
│   ├── data_setup.py      # consolidação dos 2 datasets + divisão estratificada 70/15/15
│   ├── preprocessing.py   # resize/normalização/augmentation + CLAHE
│   ├── models.py          # Baseline (Tabela 4), VGG16, ResNet50
│   ├── train.py           # treino + EarlyStopping + medição de custo
│   ├── evaluate.py        # métricas, matriz de confusão, tabela comparativa
│   ├── error_analysis.py  # sobreposição de erros entre modelos
│   └── utils.py           # seeds, gráficos, E/S
├── scripts/
│   ├── run_experiment.py    # orquestrador principal (CLI)
│   ├── download_data.py     # download dos datasets do Kaggle
│   └── make_synthetic_data.py  # dados sintéticos para smoke test
├── notebooks/
│   └── TCC_Pneus_Colab.ipynb   # notebook pronto para o Google Colab
├── requirements.txt
└── outputs/                 # resultados (gerado em runtime)
```

## Instalação

```bash
pip install -r requirements.txt
```

Recomenda-se **Google Colab com GPU T4** (ambiente previsto no TCC) ou máquina
com GPU dedicada. Para execução local em CPU, troque `tensorflow` por
`tensorflow-cpu` no `requirements.txt` (treino mais lento).

## Como usar

### 1. Obter os dados

**Opção A — datasets reais do Kaggle** (resultado científico):

```bash
python scripts/download_data.py --dest data/raw
```

Requer a [API do Kaggle](https://www.kaggle.com/docs/api) configurada. Os dois
datasets são:
- `warcoder/tyre-quality-classification` (good / defective);
- Tire Texture Image Recognition (normal / cracked).

**Opção B — dados sintéticos** (apenas para validar o pipeline):

```bash
python scripts/make_synthetic_data.py --dest data/raw --per-class 60
```

### 2. Rodar o experimento

```bash
# Experimento completo: 3 modelos, com e sem CLAHE, reconstruindo o dataset
python -m scripts.run_experiment --raw-dir data/raw --rebuild --clahe both

# Validação rápida do pipeline (poucas épocas, 1 modelo)
python -m scripts.run_experiment --raw-dir data/raw --rebuild \
    --models baseline --epochs 2 --quick
```

Principais flags:

| Flag | Descrição |
|---|---|
| `--models` | Subconjunto de `baseline vgg16 resnet50` |
| `--clahe`  | `none` / `on` / `both` (experimento auxiliar) |
| `--rebuild`| Reconsolida e redivide o dataset a partir de `--raw-dir` |
| `--epochs` | Sobrescreve o nº máximo de épocas (padrão 50) |
| `--quick`  | Reduz épocas/paciência para um smoke test |

### 3. Resultados

Em `outputs/` são gerados:
- `RELATORIO.md` — relatório consolidado (tabela comparativa + análise de erros);
- `comparison_table.json` — tabela comparativa entre arquiteturas;
- `error_overlap.json` — sobreposição de erros;
- por modelo: `results.json`, `learning_curves.png`, `confusion_matrix.png`.

## Reprodutibilidade

Toda a aleatoriedade é fixada com `random_seed = 42` (NumPy, TensorFlow e divisão
do dataset), conforme a Seção 6.3.1 do TCC. A divisão é estratificada (70/15/15),
preservando a proporção de classes em todos os subconjuntos.

## Hiperparâmetros (Tabela 2 do TCC)

| Hiperparâmetro | Valor |
|---|---|
| Entrada | 224 × 224 × 3 |
| Otimizador | Adam |
| Learning rate | 1e-4 (transfer) / 1e-3 (baseline) |
| Batch size | 32 |
| Épocas (máx.) | 50 |
| Early stopping | paciência 10 (val_loss) |
| Dropout | 0,5 |
| Perda | Categorical cross-entropy |
| Cabeçalho (TL) | GAP → Dense(256, ReLU) → Dropout(0,5) → Dense(3, Softmax) |
