"""Configuração central do experimento.

Reúne todos os hiperparâmetros e constantes do protocolo experimental definidos
na metodologia do TCC (Tabela 2 — Hiperparâmetros, e Tabela 6 — Estratégia de
transfer learning). Centralizar esses valores garante que os três modelos sejam
treinados sob condições idênticas, isolando a arquitetura como única variável.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


# Classes do problema, na ordem canônica usada em todo o pipeline.
CLASSES: Tuple[str, ...] = ("good", "worn", "cracked")
NUM_CLASSES: int = len(CLASSES)

# Semente única usada em NumPy, TensorFlow e na divisão do dataset (reprodutibilidade).
RANDOM_SEED: int = 42


@dataclass
class Config:
    """Hiperparâmetros e caminhos do experimento (Tabela 2 do TCC)."""

    # ----- Dados -----
    raw_dir: str = os.path.join("data", "raw")          # datasets brutos do Kaggle
    consolidated_dir: str = os.path.join("data", "consolidated")  # dataset unificado (3 classes)
    split_dir: str = os.path.join("data", "split")      # train/val/test estratificado
    outputs_dir: str = "outputs"

    # ----- Divisão treino/val/teste (estratificada) -----
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ----- Entrada / pré-processamento -----
    image_size: Tuple[int, int] = (224, 224)            # 224 x 224 x 3
    channels: int = 3
    rescale: float = 1.0 / 255.0                         # normalização para [0, 1]

    # ----- Data augmentation (somente no treino) -----
    rotation_range: int = 20                             # rotações
    horizontal_flip: bool = True                         # espelhamentos horizontais
    zoom_range: float = 0.2                              # variações de zoom

    # ----- Treinamento -----
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 10                    # monitorando val_loss
    early_stopping_monitor: str = "val_loss"
    dropout_rate: float = 0.5                            # camadas densas
    loss: str = "categorical_crossentropy"

    # Learning rates distintos: baixo para transfer learning, maior para o baseline.
    lr_transfer: float = 1e-4
    lr_baseline: float = 1e-3

    # ----- Cabeçalho de classificação (transfer learning) -----
    head_dense_units: int = 256                          # GlobalAvgPool -> Dense(256, ReLU) -> Dropout -> Dense(3, Softmax)

    # ----- Desbalanceamento -----
    use_class_weights: bool = True                       # pesos proporcionais na perda, se houver desbalanceamento

    # ----- CLAHE (experimento auxiliar) -----
    clahe_clip_limit: float = 2.0                        # limiar de corte do histograma local
    clahe_tile_grid: Tuple[int, int] = (8, 8)            # grade de tiles

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.image_size[0], self.image_size[1], self.channels)

    def to_dict(self) -> Dict:
        return asdict(self)


# Estratégia de transfer learning por modelo (Tabela 6 do TCC).
# Para cada arquitetura pré-treinada indicamos o prefixo das camadas que devem
# permanecer TREINÁVEIS (fine-tuning); as demais ficam congeladas.
TRANSFER_STRATEGY: Dict[str, Dict] = {
    "vgg16": {
        # Bloco 5 descongelado (block5_conv1..3); blocos 1-4 congelados.
        "unfreeze_prefixes": ("block5",),
    },
    "resnet50": {
        # Último estágio residual descongelado (conv5_block1..3); demais congelados.
        "unfreeze_prefixes": ("conv5",),
    },
}


# Mapeamento das classes de origem dos datasets do Kaggle para as três classes-alvo.
# Dataset Warcoder 2023 (Tyre Quality): {good, defective}
# Dataset Bhathena 2021 (Tire Texture): {normal, cracked}
SOURCE_LABEL_MAP: Dict[str, str] = {
    "good": "good",        # Warcoder
    "defective": "worn",   # Warcoder -> desgastado
    "normal": "good",      # Bhathena
    "cracked": "cracked",  # Bhathena
}


def default_config() -> Config:
    return Config()
