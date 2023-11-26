import matplotlib.pyplot as plt
from matplotlib import gridspec
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
    red = np.array([205, 92, 92]) / 255.

def plot_predictions(input: jnp.ndarray, y_true: jnp.ndarray, y_pred: jnp.ndarray, variances: jnp.ndarray, name: str, foldername: Optional[str] = None,
                     normalizer: Optional[Dict] = None, initial_date: Optional[str] = None) -> None:
    """ Function to plot prediction and results """

    if normalizer is None:
        normalizer = {key: dict(min_val=0, max_val=1) for key in ["price", "volume"]}

    plt.style.use('ggplot')

    fig = plt.figure(figsize=(20, 12))

    # Crear un GridSpec con dos filas y tres columnas
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 2])

    # Añadir subplots
    ax0 = plt.subplot(gs[0, :])  # El primer gráfico abarca todas las columnas
    ax1 = plt.subplot(gs[1, 0])  # Segundo gráfico (inferior izquierdo)
    ax2 = plt.subplot(gs[1, 1])  # Tercer gráfico (inferior central)
    ax3 = plt.subplot(gs[1, 2])  # Cuarto gráfico (inferior derecho)


    linewidth = 3
    marker_size = 4

    sequence_data = denormalize(input[:, 1], normalizer["price"])
    prediction_data = jnp.append(sequence_data[-1], denormalize(y_pred[0], normalizer["price"]))
    real_data = jnp.append(sequence_data[-1], denormalize(y_true[0], normalizer["price"]))
    base = jnp.arange(len(sequence_data))

    base_pred = jnp.array([len(sequence_data)-1, len(sequence_data)])
    error = jnp.abs(real_data[-1] - prediction_data[-1])

    #open_data = denormalize(input[:, 0], normalizer["price"])

    """ Plot close price """
    ax0.plot(base, sequence_data, label='Close Price', color=Color.blue,  linewidth=linewidth, marker='o', markersize=marker_size)
    ax0.plot(base_pred, real_data, label='Next Day Real', color=Color.orange, linewidth=linewidth, marker='o', markersize=marker_size)
    ax0.plot(base_pred, prediction_data, label='Next Day Pred', color=Color.green, linewidth=linewidth, marker='o', markersize=marker_size)
    #ax0.plot(base, open_data, label='Open Price', color=Color.red,  linewidth=linewidth, marker='o', markersize=marker_size)


    # Add error bars
    variances = denormalize(variances, normalizer=normalizer["price"])
    std_dev = jnp.sqrt(variances)
    upper_bound = [sequence_data[-1], float(prediction_data[-1]) + 1.96 * float(std_dev)]  # 95% confidence interval
    lower_bound = [sequence_data[-1], float(prediction_data[-1]) - 1.96 * float(std_dev)]  # 95% confidence interval

    ax0.errorbar(base_pred[1], prediction_data[1], yerr=std_dev*1.96, color=Color.green, capsize=5, linewidth=2)
    ax0.fill_between(base_pred, upper_bound, lower_bound, alpha=0.2, color=Color.green)


    ax0.set_ylabel('Close Price [$]', fontsize=18, fontweight='bold')
    ax0.legend()


    """ Plot high/low price """
    high_data = denormalize(input[:, 2], normalizer["price"])
    low_data = denormalize(input[:, 3], normalizer["price"])
    ax1.plot(base, high_data, label='High Price', color=Color.pink,  linewidth=linewidth, marker='o', markersize=marker_size)
    ax1.plot(base, low_data, label='Low Price', color=Color.purple,  linewidth=linewidth, marker='o', markersize=marker_size)
    ax1.set_ylabel('High/Low Price [$]', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
    ax1.legend()

    """ Plot volume """
    volume_data = denormalize(input[:, 5], normalizer["volume"])
    ax2.plot(base, volume_data, label='Volume', color=Color.yellow,  linewidth=linewidth, marker='o', markersize=marker_size)
    ax2.set_ylabel('Volume', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
    ax2.legend()

    """ Plot adj close price """
    adj_close_data = denormalize(input[:, 4], normalizer["price"])
    ax3.plot(base, adj_close_data, label='Adj Close Price', color=Color.orange,  linewidth=linewidth, marker='o', markersize=marker_size)
    ax3.set_ylabel('Adj Close Price [$]', fontsize=18, fontweight='bold')
    ax3.set_xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
    ax3.legend()
    
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
    
