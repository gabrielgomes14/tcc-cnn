"""Construção das três arquiteturas comparadas (Seção 9.2 e Tabelas 4 e 6 do TCC).

  * Modelo A — CNN Baseline treinada do zero (Tabela 4);
  * Modelo B — VGG16 com transfer learning (ImageNet) + fine-tuning do bloco 5;
  * Modelo C — ResNet50 com transfer learning (ImageNet) + fine-tuning do conv5.

O cabeçalho de classificação dos modelos de transfer learning é idêntico
(GlobalAvgPool -> Dense(256, ReLU) -> Dropout(0,5) -> Dense(3, Softmax)),
conforme a Tabela 6.
"""

from __future__ import annotations

from typing import Iterable

from .config import NUM_CLASSES, TRANSFER_STRATEGY, Config


def build_baseline(cfg: Config):
    """Modelo A — CNN Baseline (Tabela 4 do TCC).

    Quatro blocos convolucionais (Conv 3x3 + BatchNorm + ReLU + MaxPool 2x2) com
    32, 64, 128 e 128 filtros, seguidos de Flatten, Dense(128) + Dropout(0,5) e
    camada de saída Dense(3, Softmax). Da ordem de ~3 milhões de parâmetros.
    """
    from tensorflow.keras import Input, Model, layers

    inputs = Input(shape=cfg.input_shape, name="input")
    x = inputs
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(cfg.dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    return Model(inputs, outputs, name="baseline")


def _classification_head(x, cfg: Config):
    """Cabeçalho comum aos modelos de transfer learning (Tabela 6)."""
    from tensorflow.keras import layers

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(cfg.head_dense_units, activation="relu", name="head_dense")(x)
    x = layers.Dropout(cfg.dropout_rate, name="head_dropout")(x)
    return layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)


def _apply_finetuning(base_model, unfreeze_prefixes: Iterable[str],
                      freeze_bn: bool = True) -> None:
    """Congela toda a base e descongela apenas as camadas com os prefixos dados.

    Args:
        freeze_bn: Se True (padrão, usado no VGG16), mantém BatchNormalization
                   congelada mesmo nas camadas descongeladas — prática recomendada
                   quando o BN foi treinado com grandes datasets e o domínio-alvo
                   é pequeno. Se False (usado no ResNet50), descongela o BN junto
                   com as demais camadas do bloco, necessário porque na ResNet50
                   o BN está intercalado nos blocos residuais e congelá-lo impede
                   o fluxo correto do gradiente durante o fine-tuning.
    """
    from tensorflow.keras import layers

    prefixes = tuple(unfreeze_prefixes)
    base_model.trainable = True
    for layer in base_model.layers:
        unfreeze = layer.name.startswith(prefixes)
        if freeze_bn and isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = bool(unfreeze)


def build_vgg16(cfg: Config):
    """Modelo B — VGG16 (ImageNet), fine-tuning do bloco 5 (block5_conv1..3).

    BN mantido congelado (freeze_bn=True): VGG16 não possui BN nas camadas
    convolucionais, portanto o parâmetro não tem efeito prático aqui, mas
    mantém a interface consistente.
    """
    from tensorflow.keras import Model
    from tensorflow.keras.applications import VGG16

    base = VGG16(weights="imagenet", include_top=False, input_shape=cfg.input_shape)
    _apply_finetuning(base, TRANSFER_STRATEGY["vgg16"]["unfreeze_prefixes"],
                      freeze_bn=True)
    outputs = _classification_head(base.output, cfg)
    return Model(base.input, outputs, name="vgg16")


def build_resnet50(cfg: Config):
    """Modelo C — ResNet50 (ImageNet), fine-tuning do conv5 (conv5_block1..3).

    BN descongelado junto com as convoluções (freeze_bn=False): na ResNet50 o
    BatchNormalization está intercalado dentro dos blocos residuais. Mantê-lo
    congelado enquanto as convoluções são treinadas bloqueia o fluxo do gradiente
    e impede a convergência, como observado experimentalmente neste estudo.
    """
    from tensorflow.keras import Model
    from tensorflow.keras.applications import ResNet50

    base = ResNet50(weights="imagenet", include_top=False, input_shape=cfg.input_shape)
    _apply_finetuning(base, TRANSFER_STRATEGY["resnet50"]["unfreeze_prefixes"],
                      freeze_bn=False)
    outputs = _classification_head(base.output, cfg)
    return Model(base.input, outputs, name="resnet50")


# Registro nome -> (builder, learning_rate)
MODEL_BUILDERS = {
    "baseline": (build_baseline, "lr_baseline"),
    "vgg16": (build_vgg16, "lr_transfer"),
    "resnet50": (build_resnet50, "lr_transfer"),
}


def build_model(name: str, cfg: Config):
    """Constrói e compila o modelo indicado com o otimizador Adam e a LR adequada."""
    from tensorflow.keras.optimizers import Adam

    if name not in MODEL_BUILDERS:
        raise ValueError(f"Modelo desconhecido: {name!r}. Opções: {list(MODEL_BUILDERS)}")

    builder, lr_attr = MODEL_BUILDERS[name]
    model = builder(cfg)
    lr = getattr(cfg, lr_attr)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=cfg.loss,
        metrics=["accuracy"],
    )
    return model
