import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass


@dataclass
class Color:
    green = np.array([1, 108, 94]) / 255.
    blue = np.array([57, 99, 175]) / 255.
    pink = np.array([172, 31, 104]) / 255.
    orange = np.array([255, 85, 1]) / 255.
    purple = np.array([114, 62, 148]) / 255.
    yellow = np.array([255, 195, 0]) / 255.


def plot_predictions(input: jnp.ndarray, y_true: jnp.ndarray, y_pred: jnp.ndarray, name: str, foldername: str,
                     scale_factor: float = 1.) -> None:
    """ Function to plot prediction and results """

    plt.style.use('ggplot')
    plt.figure(figsize=(14, 8))
    plt.plot(input[:, 1], label='Close Price', color=Color.blue,  linewidth=3, marker='o', markersize=5)
    plt.scatter(len(input[:, 1]), y_true[0]*scale_factor, label='Next Close Price', color='red')
    plt.scatter(len(input[:, 1]), y_pred[0]*scale_factor, label='Predicted Next Close Price', color='green')
    plt.title('BTC Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{foldername}/{name}.png")
    plt.close()