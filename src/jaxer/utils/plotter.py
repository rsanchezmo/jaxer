import matplotlib.pyplot as plt
from matplotlib import gridspec
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Union, Tuple, List
from .dataset import denormalize
from jaxer.run.agent import Agent
import torch
from .dataset import jax_collate_fn
import torch.utils.data


@dataclass
class Color:
    green = np.array([1, 108, 94]) / 255.
    blue = np.array([57, 99, 175]) / 255.
    pink = np.array([172, 31, 104]) / 255.
    orange = np.array([255, 85, 1]) / 255.
    purple = np.array([114, 62, 148]) / 255.
    yellow = np.array([255, 195, 0]) / 255.
    red = np.array([205, 92, 92]) / 255.
    black = np.array([0, 0, 0]) / 255.

    def to_list(self):
        return [self.green, self.blue, self.pink, self.orange, self.purple, self.yellow, self.red, self.black]


def predict_entire_dataset(agent: Agent, dataset: torch.utils.data.Dataset, foldername=None, mode='test'):
    """ Predict entire dataset """

    if dataset.output_mode == 'discrete_grid':
        return
    
    # create a dataloader for the entire dataset and infere all at once
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=jax_collate_fn) 

    for batch in dataloader:
        input, label, normalizer, initial_date = batch
        output = agent(input)

    if isinstance(output, tuple):
        y_pred, _ = output
    else:
        y_pred = output

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
    ax0 = plt.subplot(gs[0:2, 0])  
    ax1 = plt.subplot(gs[2, 0]) 

    close_preds = [denormalize(y_pred[i], normalizer[i]["price"]) for i in range(len(y_pred))]
    close_inputs = [denormalize(input[i, 0, 0], normalizer[i]["price"]) for i in range(len(input))] + denormalize(input[-1, 1:, 0], normalizer[-1]["price"]).tolist()
    mean_avg = [denormalize(np.mean(input[i, :, 0]), normalizer[i]["price"]) for i in range(len(input))]

    # close_inputs (batch_size, max_seq_len, 1)
    # close_preds (batch_size, 1, 1)
    batch_size = input.shape[0]
    seq_len = input.shape[1]
    base_input = jnp.arange(len(close_inputs)) 
    base_pred = jnp.arange(batch_size) + seq_len 

    ax0.plot(base_pred, close_preds, label='Close Price Pred', color=Color.green, linewidth=3, marker='o', markersize=4)
    
    # plot close_inputs
    ax0.plot(base_input, close_inputs, label='Close Price Real', color=Color.blue, linewidth=3, marker='o', markersize=4)

    # plot mean
    ax0.plot(base_pred, mean_avg, label='Close Price Avg', color=Color.orange, linewidth=3, marker='o', markersize=4)
    ax0.set_ylabel('Close Price [$]', fontsize=18, fontweight='bold')

    ax0.legend()

    """ plot the errors """
    errors = [100 * (close_inputs[i+seq_len-1] - close_preds[i])/close_inputs[i+seq_len-1] for i in range(len(close_preds))]
    ax1.stem(base_pred, errors, label='Error')
    ax1.legend()
    ax1.set_ylabel('Error [%]', fontsize=18, fontweight='bold')

    plt.suptitle(f"Predictions [{mode}]", fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    if foldername is not None:
        plt.savefig(f"{foldername}/dataset_{mode}_prediction.png") 
    else:
        plt.show()
    plt.close()


def plot_predictions(input: jnp.ndarray, y_true: jnp.ndarray, output: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], name: str, foldername: Optional[str] = None,
                     normalizer: Optional[Dict] = None, initial_date: Optional[str] = None, output_mode: str = 'mean', discrete_grid_levels: Optional[List[float]] = None) -> None:
    """ Function to plot prediction and results """

    if output_mode == 'distribution':
        y_pred, variances = output
        y_pred = y_pred.squeeze(0)
        variances = variances.squeeze(0)
    else:
        y_pred = output.squeeze(0)
        variances = None

    if normalizer is None:
        normalizer = {key: dict(min_val=0, max_val=1, mode="minmax") for key in ["price", "volume", "trades"]}

    plt.style.use('ggplot')

    fig = plt.figure(figsize=(20, 12))

    # Crear un GridSpec con dos filas y tres columnas
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    # Añadir subplots
    ax0 = plt.subplot(gs[0, :])  
    ax1 = plt.subplot(gs[1, 0]) 
    ax2 = plt.subplot(gs[1, 1]) 
    #ax3 = plt.subplot(gs[1, 2])


    linewidth = 3
    marker_size = 4

    sequence_data = denormalize(input[:, 0], normalizer["price"])

    mean_avg = denormalize(np.mean(input[:, 0]), normalizer["price"])

    real_data = jnp.append(sequence_data[-1], denormalize(y_true[0], normalizer["price"]))
    mean_avg_data = jnp.append(sequence_data[-1], mean_avg)

    base = jnp.arange(len(sequence_data))

    base_pred = jnp.array([len(sequence_data)-1, len(sequence_data)])
    if output_mode != 'discrete_grid':
        prediction_data = jnp.append(sequence_data[-1], denormalize(y_pred[0], normalizer["price"]))
        ax0.plot(base_pred, prediction_data, label='Next Day Pred', color=Color.green, linewidth=linewidth, marker='o', markersize=marker_size, linestyle='--')
        error = jnp.abs(real_data[-1] - prediction_data[-1])
        percent_error = 100 * error / real_data[-1]
        ax0.fill_between(base_pred, prediction_data, real_data, alpha=0.2, color=Color.yellow)

    else:
        colors = Color().to_list()
        for idx in range(0,len(discrete_grid_levels)-1):
            lower_bound = [0.01 * discrete_grid_levels[idx] * sequence_data[-1] + sequence_data[-1], 0.01 * discrete_grid_levels[idx] * sequence_data[-1] + sequence_data[-1]]
            upper_bound = [0.01 * discrete_grid_levels[idx+1] * sequence_data[-1] + sequence_data[-1], 0.01 * discrete_grid_levels[idx+1] * sequence_data[-1] + sequence_data[-1]]
            lower_label = discrete_grid_levels[idx] if idx > 0 else 'Min'
            upper_label = discrete_grid_levels[idx+1] if idx < len(discrete_grid_levels)-2 else 'Max'
            label = f'[{lower_label}, {upper_label}) %'
            ax0.fill_between(base_pred, lower_bound, upper_bound, alpha=0.2, color=colors[idx], 
                             label=label)

        last_open = sequence_data[-1]

        lower_idx = jnp.argmax(y_true)
        upper_idx = lower_idx + 1
        lower_bound = [0.01 * discrete_grid_levels[lower_idx] * last_open + last_open, 0.01 * discrete_grid_levels[lower_idx] * last_open + last_open]
        upper_bound = [0.01 * discrete_grid_levels[upper_idx] * last_open + last_open, 0.01 * discrete_grid_levels[upper_idx] * last_open + last_open]
        ax0.fill_between(base_pred, lower_bound, upper_bound, alpha=0.7, color=Color.orange, label='Next Day Real')
        lower_idx = jnp.argmax(y_pred)
        upper_idx = lower_idx + 1
        lower_bound = [0.01 * discrete_grid_levels[lower_idx] * last_open + last_open, 0.01 * discrete_grid_levels[lower_idx] * last_open + last_open]
        upper_bound = [0.01 * discrete_grid_levels[upper_idx] * last_open + last_open, 0.01 *discrete_grid_levels[upper_idx] * last_open + last_open]
        ax0.fill_between(base_pred, lower_bound, upper_bound, alpha=0.7, color=Color.green, label='Next Day Pred')


    open_data = denormalize(input[:, 1], normalizer["price"])

    """ Plot close price """
    ax0.axvline(x=len(sequence_data)-1, color=Color.red, linewidth=2)

    ax0.plot(base, sequence_data, label='Close Price', color=Color.blue,  linewidth=linewidth, marker='o', markersize=marker_size)
    #ax0.plot(base, open_data, label='Open Price', color=Color.pink,  linewidth=linewidth, marker='o', markersize=marker_size)
    if output_mode != 'discrete_grid':
        ax0.plot(base_pred, real_data, label='Next Day Real', color=Color.orange, linewidth=linewidth, marker='o', markersize=marker_size, linestyle='--')
        ax0.plot(base_pred, mean_avg_data, label='Next Day Window Avg', color=Color.purple, linewidth=linewidth, marker='o', markersize=marker_size, linestyle='--')
    #ax0.plot(base, open_data, label='Open Price', color=Color.red,  linewidth=linewidth, marker='o', markersize=marker_size)


    # Add error bars
    if variances is not None:
        variances = denormalize(variances, normalizer=normalizer["price"])
        std_dev = jnp.sqrt(variances)
        upper_bound = [sequence_data[-1], float(prediction_data[-1]) + 1.96 * float(std_dev)]  # 95% confidence interval
        lower_bound = [sequence_data[-1], float(prediction_data[-1]) - 1.96 * float(std_dev)]  # 95% confidence interval

        ax0.errorbar(base_pred[1], prediction_data[1], yerr=std_dev*1.96, color=Color.green, capsize=5, linewidth=2)
        ax0.fill_between(base_pred, upper_bound, lower_bound, alpha=0.2, color=Color.green)

    ax0.set_ylim([np.min(sequence_data) - 0.05 * np.min(sequence_data), np.max(sequence_data) + 0.05 * np.max(sequence_data)])

    ax0.set_ylabel('Close Price [$]', fontsize=18, fontweight='bold')
    ax0.legend(ncol=2)


    """ Plot high/low price """
    high_data = denormalize(input[:, 2], normalizer["price"])
    low_data = denormalize(input[:, 3], normalizer["price"])
    ax1.plot(base, high_data, label='High Price', color=Color.pink,  linewidth=linewidth, marker='o', markersize=marker_size)
    ax1.plot(base, low_data, label='Low Price', color=Color.purple,  linewidth=linewidth, marker='o', markersize=marker_size)
    ax1.set_ylabel('High/Low Price [$]', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
    ax1.legend()

    """ Plot volume """
    volume_data = denormalize(input[:, 4], normalizer["volume"])
    ax2.plot(base, volume_data, label='Volume', color=Color.yellow,  linewidth=linewidth, marker='o', markersize=marker_size)
    ax2.set_ylabel('Volume', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
    ax2.legend()

    """ Plot trades """
    # trades_data = denormalize(input[:, 5], normalizer["trades"])
    # ax3.plot(base, trades_data, label='Trades', color=Color.blue,  linewidth=linewidth, marker='o', markersize=marker_size)
    # ax3.set_ylabel('Trades', fontsize=18, fontweight='bold')
    # ax3.set_xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
    # ax3.legend()
    
    if initial_date is not None:
        title = f'Jaxer Predictor || Initial Date: {initial_date}' 
    else:
        title = f'Jaxer Predictor'
    
    if output_mode != 'discrete_grid':
        title += f' || Error {error:.2f} $ ({percent_error:.1f}) %'
    else:
        title += f' || Right Prediction: {jnp.argmax(y_true) == jnp.argmax(y_pred)}'

    plt.suptitle(title, fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    
    if foldername is not None:
        plt.savefig(f"{foldername}/{name}.png") 
    else:
        plt.show()
    plt.close()
    
