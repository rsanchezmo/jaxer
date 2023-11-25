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

def plot_predictions(input: jnp.ndarray, y_true: jnp.ndarray, y_pred: jnp.ndarray, name: str, foldername: Optional[str] = None,
                     normalizer: Optional[Dict] = None, initial_date: Optional[str] = None) -> None:
    """ Function to plot prediction and results """

    if normalizer is None:
        normalizer = {key: dict(min_val=0, max_val=1) for key in ["price", "volume"]}

    plt.style.use('ggplot')
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 12), sharex=True)

    sequence_data = denormalize(input[:, 1], normalizer["price"])
    prediction_data = jnp.append(sequence_data[-1], denormalize(y_pred[0], normalizer["price"]))
    real_data = jnp.append(sequence_data[-1], denormalize(y_true[0], normalizer["price"]))
    base = jnp.arange(len(sequence_data))

    base_pred = jnp.array([len(sequence_data)-1, len(sequence_data)])
    error = jnp.abs(real_data[-1] - prediction_data[-1])

    """ Plot close price """
    axs[0, 0].plot(base, sequence_data, label='Close Price', color=Color.blue,  linewidth=4, marker='o', markersize=8)
    axs[0, 0].plot(base_pred, real_data, label='Next Day Real', color=Color.orange, linewidth=4, marker='o', markersize=8)
    axs[0, 0].plot(base_pred, prediction_data, label='Next Day Pred', color=Color.green, linewidth=4, marker='o', markersize=8)
    axs[0, 0].set_ylabel('Close Price [$]', fontsize=18, fontweight='bold')
    axs[0, 0].legend()

    """ Plot open price """
    open_data = denormalize(input[:, 0], normalizer["price"])
    axs[0, 1].plot(base, open_data, label='Open Price', color=Color.green,  linewidth=4, marker='o', markersize=8)
    axs[0, 1].set_ylabel('Open Price [$]', fontsize=18, fontweight='bold')
    axs[0, 1].legend()

    """ Plot high price """
    high_data = denormalize(input[:, 2], normalizer["price"])
    axs[0, 2].plot(base, high_data, label='High Price', color=Color.pink,  linewidth=4, marker='o', markersize=8)
    axs[0, 2].set_ylabel('High Price [$]', fontsize=18, fontweight='bold')
    axs[0, 2].legend()

    """ Plot low price """
    low_data = denormalize(input[:, 3], normalizer["price"])
    axs[1, 0].plot(base, low_data, label='Low Price', color=Color.purple,  linewidth=4, marker='o', markersize=8)
    axs[1, 0].set_ylabel('Low Price [$]', fontsize=18, fontweight='bold')
    axs[1, 0].set_xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
    axs[1, 0].legend()

    """ Plot volume """
    # volume_data = denormalize(input[:, 5], normalizer["volume"])
    # axs[1, 1].plot(base, volume_data, label='Volume', color=Color.yellow,  linewidth=4, marker='o', markersize=8)
    # axs[1, 1].set_ylabel('Volume', fontsize=18, fontweight='bold')
    # axs[1, 1].set_xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
    # axs[1, 1].legend()

    """ Plot adj close price """
    adj_close_data = denormalize(input[:, 4], normalizer["price"])
    axs[1, 2].plot(base, adj_close_data, label='Adj Close Price', color=Color.orange,  linewidth=4, marker='o', markersize=8)
    axs[1, 2].set_ylabel('Adj Close Price [$]', fontsize=18, fontweight='bold')
    axs[1, 2].set_xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
    axs[1, 2].legend()
    
    if initial_date is not None:
        title = f'Jaxer Predictor || Error {error:.2f} $ || Initial Date: {initial_date}' 
    else:
        title = f'Jaxer Predictor || Error {error:.2f} $'
    plt.suptitle(title, fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    
    if foldername is not None:
        plt.savefig(f"{foldername}/{name}.png") 
    else:
        plt.show()
    plt.close()
    
