import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from .dataset import denormalize


@dataclass
class Color:
    green = np.array([1, 108, 94]) / 255.
    blue = np.array([57, 99, 175]) / 255.
    pink = np.array([172, 31, 104]) / 255.
    orange = np.array([255, 85, 1]) / 255.
    purple = np.array([114, 62, 148]) / 255.
    yellow = np.array([255, 195, 0]) / 255.


def plot_predictions(input: jnp.ndarray, y_true: jnp.ndarray, y_pred: jnp.ndarray, name: str, foldername: str,
                     normalizer: Optional[Dict] = None) -> None:
    """ Function to plot prediction and results """

    if normalizer is None:
        normalizer = dict(min_close=0, max_close=1)

    plt.style.use('ggplot')
    plt.figure(figsize=(14, 8), visible=False)

    sequence_data = denormalize(input[:, 1], normalizer)
    prediction_data = jnp.append(sequence_data[-1], denormalize(y_pred[0], normalizer))
    real_data = jnp.append(sequence_data[-1], denormalize(y_true[0], normalizer))
    base = jnp.arange(len(sequence_data))

    base_pred = jnp.array([len(sequence_data)-1, len(sequence_data)])
    error = jnp.abs(real_data[-1] - prediction_data[-1])

    plt.plot(base, sequence_data, label='Close Price', color=Color.blue,  linewidth=4, marker='o', markersize=8)
    plt.plot(base_pred, real_data, label='Next Day Real', color=Color.orange, linewidth=4, marker='o', markersize=8)
    plt.plot(base_pred, prediction_data, label='Next Day Pred', color=Color.green, linewidth=4, marker='o', markersize=8)
    plt.title(f'Jaxer Predictor || Error {error:.1f} USD', fontsize=20, fontweight='bold')
    plt.xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
    plt.ylabel('Close Price [$]', fontsize=18, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{foldername}/{name}.png")
    plt.close()