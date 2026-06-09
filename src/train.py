"""Treinamento das arquiteturas (Etapa 3 do protocolo) e medição de custo.

Treina um modelo com os mesmos hiperparâmetros gerais (Tabela 2), aplicando
EarlyStopping (paciência de 10 épocas em val_loss) e registrando o custo
computacional: número de parâmetros, tempo por época, tempo total e — após o
treino — tempo médio de inferência por imagem (medido em evaluate.py).
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from .config import Config
from .utils import count_params


class EpochTimer:
    """Callback que registra o tempo de cada época de treinamento."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self._t0: Optional[float] = None

    def make_callback(self):
        from tensorflow.keras.callbacks import Callback

        timer = self

        class _Cb(Callback):
            def on_epoch_begin(self, epoch, logs=None):
                timer._t0 = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                timer.times.append(time.perf_counter() - timer._t0)

        return _Cb()


def train_model(
    model,
    cfg: Config,
    train_gen,
    val_gen,
    class_weights: Optional[Dict[int, float]] = None,
) -> Dict:
    """Treina ``model`` e devolve histórico + métricas de custo computacional.

    Returns
    -------
    dict com chaves:
        history          : dict das curvas de aprendizado (loss/accuracy/val_*);
        params           : {trainable, non_trainable, total};
        epochs_trained   : nº de épocas efetivamente executadas;
        time_per_epoch_s : tempo médio por época (s);
        total_train_time_s : tempo total de treinamento (s);
        epoch_times_s    : lista com o tempo de cada época.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    early = EarlyStopping(
        monitor=cfg.early_stopping_monitor,
        patience=cfg.early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )
    timer = EpochTimer()

    t_start = time.perf_counter()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.max_epochs,
        callbacks=[early, timer.make_callback()],
        class_weight=class_weights if cfg.use_class_weights else None,
        verbose=2,
    )
    total_time = time.perf_counter() - t_start

    epoch_times = timer.times
    return {
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "params": count_params(model),
        "epochs_trained": len(epoch_times),
        "time_per_epoch_s": (sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0,
        "total_train_time_s": total_time,
        "epoch_times_s": epoch_times,
    }
