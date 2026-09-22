"""Configuração central do experimento.

Reúne todos os hiperparâmetros e constantes do protocolo experimental definidos
na metodologia do TCC (tabelas de hiperparâmetros e de estratégia de
transfer learning, Seção 3.3.1). Centralizar esses valores garante que os três modelos sejam
treinados sob condições idênticas, isolando a arquitetura como única variável.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

CLASSES: Tuple[str, ...] = ("good", "worn", "cracked")
NUM_CLASSES: int = len(CLASSES)

# Semente única usada em NumPy, TensorFlow e na divisão do dataset (reprodutibilidade).
RANDOM_SEED: int = 42


@dataclass
class Config:
    """Hiperparâmetros e caminhos do experimento (Tabela 3.3.1 do TCC)."""

    raw_dir: str = os.path.join("data", "raw")
    consolidated_dir: str = os.path.join("data", "consolidated")
    split_dir: str = os.path.join("data", "split") 
    outputs_dir: str = "outputs"

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    image_size: Tuple[int, int] = (224, 224)            
    channels: int = 3
    rescale: float = 1.0 / 255.0                         

    rotation_range: int = 20                             
    horizontal_flip: bool = True                         
    zoom_range: float = 0.2                             

    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 10                    
    early_stopping_monitor: str = "val_loss"
    dropout_rate: float = 0.5                            
    loss: str = "categorical_crossentropy"

    lr_transfer: float = 1e-4
    lr_baseline: float = 1e-4

    head_dense_units: int = 256                          

    use_class_weights: bool = True                       

    clahe_clip_limit: float = 2.0                        
    clahe_tile_grid: Tuple[int, int] = (8, 8)            

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.image_size[0], self.image_size[1], self.channels)

    def to_dict(self) -> Dict:
        return asdict(self)


# Estratégia de transfer learning por modelo (Seção 3.3.1 6 do TCC).
# Para cada arquitetura pré-treinada indicamos o prefixo das camadas que devem
# permanecer TREINÁVEIS (fine-tuning); as demais ficam congeladas.
TRANSFER_STRATEGY: Dict[str, Dict] = {
    "vgg16": {
        # Bloco 5 descongelado (block5_conv1..3); blocos 1-4 congelados.
        "unfreeze_prefixes": ("block5",),
    },
    "resnet50": {
        # Último estágio residual descongelado (conv5_block1..3); demais congelados.
        "unfreeze_prefixes": ("conv5_block",),
    },
}


# Mapeamento das classes de origem dos datasets do Kaggle para as três classes-alvo.
# Dataset Warcoder 2023 (Tyre Quality): {good, defective}
# Dataset Bhathena 2021 (Tire Texture): {normal, cracked}
SOURCE_LABEL_MAP: Dict[str, str] = {
    "good": "good",        # Warcoder
    "defective": "worn",   # Warcoder 
    "normal": "good",      # Bhathena
    "cracked": "cracked",  # Bhathena
}


def default_config() -> Config:
    return Config()
