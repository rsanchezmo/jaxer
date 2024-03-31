import os.path
import matplotlib.pyplot as plt
from matplotlib import gridspec
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from jaxer.utils.normalizer import denormalize
from jaxer.utils.losses import acc_dir, mape, mae, acc_dir_discrete
from tbparse import SummaryReader


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


# def predict_entire_dataset(agent: FlaxAgent, dataset: torch.utils.data.Dataset, foldername=None, mode='test'):
#     """ Predict entire dataset """
#
#     if dataset.output_mode == 'discrete_grid':
#         return
#
#     # create a dataloader for the entire dataset and infere all at once
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=jax_collate_fn)
#
#     for batch in dataloader:
#         input, label, normalizer, initial_date = batch
#         output = agent(input)
#
#     if isinstance(output, tuple):
#         y_pred, _ = output
#     else:
#         y_pred = output
#
#     plt.style.use('ggplot')
#     fig = plt.figure(figsize=(20, 12))
#     gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
#     ax0 = plt.subplot(gs[0:2, 0])
#     ax1 = plt.subplot(gs[2, 0])
#
#     close_preds = [denormalize(y_pred[i], normalizer[i]["price"]) for i in range(len(y_pred))]
#     close_inputs = [denormalize(input[i, 0, 0], normalizer[i]["price"]) for i in range(len(input))] + denormalize(
#         input[-1, 1:, 0], normalizer[-1]["price"]).tolist()
#     mean_avg = [denormalize(np.mean(input[i, :, 0]), normalizer[i]["price"]) for i in range(len(input))]
#
#     # close_inputs (batch_size, max_seq_len, 1)
#     # close_preds (batch_size, 1, 1)
#     batch_size = input.shape[0]
#     seq_len = input.shape[1]
#     base_input = jnp.arange(len(close_inputs))
#     base_pred = jnp.arange(batch_size) + seq_len
#
#     ax0.plot(base_pred, close_preds, label='Close Price Pred', color=Color.green, linewidth=3, marker='o', markersize=4)
#
#     # plot close_inputs
#     ax0.plot(base_input, close_inputs, label='Close Price Real', color=Color.blue, linewidth=3, marker='o',
#              markersize=4)
#
#     # plot mean
#     ax0.plot(base_pred, mean_avg, label='Close Price Avg', color=Color.orange, linewidth=3, marker='o', markersize=4)
#     ax0.set_ylabel('Close Price [$]', fontsize=18, fontweight='bold')
#
#     ax0.legend()
#
#     """ plot the errors """
#     errors = [100 * (close_inputs[i + seq_len - 1] - close_preds[i]) / close_inputs[i + seq_len - 1] for i in
#               range(len(close_preds))]
#     ax1.stem(base_pred, errors, label='Error')
#     ax1.legend()
#     ax1.set_ylabel('Error [%]', fontsize=18, fontweight='bold')
#
#     plt.suptitle(f"Predictions [{mode}]", fontsize=20, fontweight='bold')
#     plt.grid(True)
#     plt.tight_layout()
#     if foldername is not None:
#         plt.savefig(f"{foldername}/dataset_{mode}_prediction.png")
#     else:
#         plt.show()
#     plt.close()


def plot_predictions(x: Tuple[jnp.ndarray, jnp.ndarray],
                     y_true: jnp.ndarray,
                     y_pred: jnp.ndarray,
                     normalizer: jnp.ndarray,
                     window_info: Dict,
                     folder_name: Optional[str] = None,
                     image_name: str = 'pred',
                     denormalize_values: bool = True):
    """ Function to plot a window prediction. Batch size must be 1

    :param x: input data (sequence, extra tokens)
    :type x: Tuple[jnp.ndarray, jnp.ndarray]

    :param y_true: true values
    :type y_true: jnp.ndarray

    :param y_pred: predicted values
    :type y_pred: jnp.ndarray

    :param normalizer: normalizer values
    :type normalizer: jnp.ndarray

    :param window_info: window information
    :type window_info: Dict

    :param folder_name: folder name to save the image
    :type folder_name: Optional[str]

    :param image_name: image name
    :type image_name: str

    :param denormalize_values: denormalize the values on the plot
    :type denormalize_values: bool
    """

    x_hist = x[0]
    if denormalize_values:
        normalizer = jnp.tile(normalizer, (x_hist.shape[1], 1))
        x_hist_ohlc = denormalize(x_hist[0, :, 0:4], normalizer[:, 0:4])  # first dimension must be batch of 1
        x_hist_volume = denormalize(x_hist[0, :, 4:5], normalizer[:, 4:8])
        x_hist_trades = denormalize(x_hist[0, :, 5:6], normalizer[:, 8:12])
    else:
        x_hist_ohlc = x_hist[0, :, 0:4]
        x_hist_volume = x_hist[0, :, 4:5]
        x_hist_trades = x_hist[0, :, 5:6]

    ticker_name = window_info.get('ticker', '')
    initial_date = window_info.get('initial_date', '')
    end_date = window_info.get('end_date', '')
    output_mode = window_info.get('output_mode', 'mean')
    window_size = window_info.get('window_size', 0)
    return_mode = window_info.get('return_mode', False)
    discrete_grid_levels = window_info.get('discrete_grid_levels', [])
    resolution = window_info.get('resolution', 1)
    norm_mode = window_info.get('norm_mode', '')

    if denormalize_values:
        y_true = denormalize(y_true, normalizer[[0], 0:4])

    std = None
    if output_mode == 'distribution':
        y_pred, std = y_pred
        if denormalize_values:
            std = denormalize(std, normalizer[[0], 0:4])

    if denormalize_values:
        y_pred = denormalize(y_pred, normalizer[[0], 0:4])

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 0.8])
    ax0 = plt.subplot(gs[0, :])
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[1, 1])
    ax3 = plt.subplot(gs[1, 2])

    window_base = jnp.arange(len(x_hist_ohlc))
    pred_base = jnp.array([len(x_hist_ohlc) - 1, len(x_hist_ohlc)])
    true_base = jnp.array([len(x_hist_ohlc) - 1, len(x_hist_ohlc)])

    if return_mode:
        y_pred_base = [100 * (np.exp(x_hist_ohlc[-1, -1].item()) - 1), 100 * (np.exp(y_pred[0, 0].item()) - 1)]
        y_true_base = [100 * (np.exp(x_hist_ohlc[-1, -1].item()) - 1), 100 * (np.exp(y_true[0, 0].item()) - 1)]
    else:
        y_pred_base = [x_hist_ohlc[-1, -1].item(), y_pred[0, 0].item()]
        y_true_base = [x_hist_ohlc[-1, -1].item(), y_true[0, 0].item()]

    linewidth = 3
    markersize = 4

    if return_mode:
        ax0.plot(window_base, 100 * (np.exp(x_hist_ohlc[:, 3]) - 1), label='Close', color=Color.blue,
                 linewidth=linewidth,
                 marker='o',
                 markersize=markersize)

        ax0.plot(window_base, 100 * (np.exp(x_hist_ohlc[:, 0]) - 1), label='Open', color=Color.red, linewidth=linewidth,
                 marker='o',
                 markersize=markersize)
    else:
        ax0.plot(window_base, x_hist_ohlc[:, 3], label='Close', color=Color.blue, linewidth=linewidth, marker='o',
                 markersize=markersize)

        ax0.plot(window_base, x_hist_ohlc[:, 0], label='Open', color=Color.red, linewidth=linewidth, marker='o',
                 markersize=markersize)

    if output_mode != 'discrete_grid':
        ax0.plot(pred_base, y_pred_base, label='Pred', color=Color.green, linewidth=linewidth,
                 marker='o', markersize=markersize, linestyle='--')
        ax0.fill_between(pred_base, y_pred_base, y_true_base, alpha=0.2, color=Color.yellow)
        ax0.plot(true_base, y_true_base, label='Real', color=Color.orange, linewidth=linewidth,
                 marker='o', markersize=markersize, linestyle='--')

        ax0.vlines(x=pred_base[1], ymin=y_true, ymax=y_pred, color=Color.black, linewidth=2)

        # plot the mean
        mean_base = jnp.array([len(x_hist_ohlc) - 1, len(x_hist_ohlc)])

        if return_mode:
            ax0.plot(mean_base, [100 * (np.exp(x_hist_ohlc[-1, -1].item()) - 1), 100 * (np.mean(np.exp(x_hist_ohlc[:, -1])) - 1)],
                     label='Mean', color=Color.purple, linewidth=linewidth, marker='o',
                     markersize=markersize, linestyle='--')
        else:
            ax0.plot(mean_base, [x_hist_ohlc[-1, 3].item(), jnp.mean(x_hist_ohlc[:, 3]).item()], label='Mean', color=Color.purple,
                     linewidth=linewidth, marker='o',
                     markersize=markersize, linestyle='--')

        if output_mode == 'distribution':
            ax0.errorbar(pred_base[1], y_pred[1], yerr=std * 1.96, color=Color.green, capsize=5, linewidth=2)
            ax0.fill_between(pred_base, y_pred - std * 1.96, y_pred + std * 1.96, alpha=0.2, color=Color.green)

    else:
        pass

    # ax0.set_ylim(
    #     [jnp.min(x_hist_ohlc[:, -1]) - 0.02 * jnp.min(x_hist_ohlc[:, -1]),
    #      jnp.max(x_hist_ohlc[:, -1]) + 0.02 * jnp.max(x_hist_ohlc[:, -1])]
    # )

    if return_mode:
        ax0.set_ylabel('Returns [%]', fontsize=14, fontweight='bold')
    else:
        ax0.set_ylabel('Predictions [$]', fontsize=14, fontweight='bold')
    ax0.legend(loc='upper center', ncol=5)
    ax0.grid(axis='both', color='0.92')

    """ Plot high/low price """
    if return_mode:
        high_data = 100 * (np.exp(x_hist_ohlc[:, 1]) - 1)
        low_data = 100 * (np.exp(x_hist_ohlc[:, 2]) - 1)
    else:
        high_data = x_hist_ohlc[:, 1]
        low_data = x_hist_ohlc[:, 2]
    ax1.plot(window_base, high_data, label='High', color=Color.pink, linewidth=linewidth, marker='o',
             markersize=markersize)
    ax1.plot(window_base, low_data, label='Low', color=Color.purple, linewidth=linewidth, marker='o',
             markersize=markersize)
    if return_mode:
        ax1.set_ylabel('High/Low Price [%]', fontsize=14, fontweight='bold')
    else:
        ax1.set_ylabel('High/Low Price [$]', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper center', ncol=2)
    ax1.grid(axis='both', color='0.92')

    """ Plot volume """
    volume_data = x_hist_volume
    ax2.plot(window_base, volume_data, label='Volume', color=Color.yellow, linewidth=linewidth, marker='o',
             markersize=markersize)
    ax2.set_ylabel('Volume', fontsize=14, fontweight='bold')
    ax2.grid(axis='both', color='0.92')

    """ Plot trades"""
    trades_data = x_hist_trades
    ax3.plot(window_base, trades_data, label='Trades', color=Color.blue, linewidth=linewidth, marker='o',
             markersize=markersize)
    ax3.set_ylabel('Trades', fontsize=14, fontweight='bold')
    ax3.grid(axis='both', color='0.92')

    if return_mode:
        acc_dir_ = acc_dir_discrete(y_pred, y_true)
    else:
        acc_dir_ = acc_dir(y_pred, y_true, x_hist_ohlc[-1, 3])
    mape_ = mape(y_pred, y_true)
    mae_ = mae(y_pred, y_true)

    initial_date_str = initial_date.strftime('%Y/%m/%d') if initial_date is not None else '0'
    end_date_str = end_date.strftime('%Y/%m/%d') if end_date is not None else str(window_size)

    if not return_mode:
        returns = x_hist_ohlc[1:, 3] / x_hist_ohlc[:-1, 3]
        window_variation = (np.max(x_hist_ohlc[:, 3]) - np.min(x_hist_ohlc[:, 3])) / np.min(x_hist_ohlc[:, 3]) * 100

    else:
        returns = np.exp(x_hist_ohlc[:, 3]) - 1
        window_variation = np.abs((np.max(returns) - np.min(returns)) / np.min(returns) * 100)

    volatility = np.std(returns) * 100

    title = (f'{norm_mode} - {ticker_name} - {output_mode} - {resolution} - '
             f'[{initial_date_str}, {end_date_str}] - acc_dir: {int(acc_dir_.item()/100)} - '
             f'mape: {mape_.item():.2f}% - mae: {mae_.item():.2f}$ - volatility (R): {volatility:.2f}% '
             f' (M): {window_variation:.2f}%')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if folder_name is not None:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        plt.savefig(f"{folder_name}/{image_name}.png")
    else:
        plt.show()
    plt.close()

# def plot_predictions(input: jnp.ndarray, y_true: jnp.ndarray,
#                      output: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], name: str,
#                      foldername: Optional[str] = None,
#                      normalizer: Optional[Dict] = None, initial_date: Optional[str] = None, output_mode: str = 'mean',
#                      discrete_grid_levels: Optional[List[float]] = None) -> None:
#     """ Function to plot prediction and results """
#
#     if output_mode == 'distribution':
#         y_pred, variances = output
#         y_pred = y_pred.squeeze(0)
#         variances = variances.squeeze(0)
#     else:
#         y_pred = output.squeeze(0)
#         variances = None
#
#     if normalizer is None:
#         normalizer = {key: dict(min_val=0, max_val=1, mode="minmax") for key in ["price", "volume", "trades"]}
#
#     plt.style.use('ggplot')
#
#     fig = plt.figure(figsize=(20, 12))
#
#     # Crear un GridSpec con dos filas y tres columnas
#     gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
#
#     # Añadir subplots
#     ax0 = plt.subplot(gs[0, :])
#     ax1 = plt.subplot(gs[1, 0])
#     ax2 = plt.subplot(gs[1, 1])
#     # ax3 = plt.subplot(gs[1, 2])
#
#     linewidth = 3
#     marker_size = 4
#
#     sequence_data = denormalize(input[:, 0], normalizer["price"])
#
#     mean_avg = denormalize(np.mean(input[:, 0]), normalizer["price"])
#
#     real_data = jnp.append(sequence_data[-1], denormalize(y_true[0], normalizer["price"]))
#     mean_avg_data = jnp.append(sequence_data[-1], mean_avg)
#
#     base = jnp.arange(len(sequence_data))
#
#     base_pred = jnp.array([len(sequence_data) - 1, len(sequence_data)])
#     if output_mode != 'discrete_grid':
#         prediction_data = jnp.append(sequence_data[-1], denormalize(y_pred[0], normalizer["price"]))
#         ax0.plot(base_pred, prediction_data, label='Next Day Pred', color=Color.green, linewidth=linewidth, marker='o',
#                  markersize=marker_size, linestyle='--')
#         error = jnp.abs(real_data[-1] - prediction_data[-1])
#         percent_error = 100 * error / real_data[-1]
#         ax0.fill_between(base_pred, prediction_data, real_data, alpha=0.2, color=Color.yellow)
#
#     else:
#         colors = Color().to_list()
#         for idx in range(0, len(discrete_grid_levels) - 1):
#             lower_bound = [0.01 * discrete_grid_levels[idx] * sequence_data[-1] + sequence_data[-1],
#                            0.01 * discrete_grid_levels[idx] * sequence_data[-1] + sequence_data[-1]]
#             upper_bound = [0.01 * discrete_grid_levels[idx + 1] * sequence_data[-1] + sequence_data[-1],
#                            0.01 * discrete_grid_levels[idx + 1] * sequence_data[-1] + sequence_data[-1]]
#             lower_label = discrete_grid_levels[idx] if idx > 0 else 'Min'
#             upper_label = discrete_grid_levels[idx + 1] if idx < len(discrete_grid_levels) - 2 else 'Max'
#             label = f'[{lower_label}, {upper_label}) %'
#             ax0.fill_between(base_pred, lower_bound, upper_bound, alpha=0.2, color=colors[idx],
#                              label=label)
#
#         last_open = sequence_data[-1]
#
#         lower_idx = jnp.argmax(y_true)
#         upper_idx = lower_idx + 1
#         lower_bound = [0.01 * discrete_grid_levels[lower_idx] * last_open + last_open,
#                        0.01 * discrete_grid_levels[lower_idx] * last_open + last_open]
#         upper_bound = [0.01 * discrete_grid_levels[upper_idx] * last_open + last_open,
#                        0.01 * discrete_grid_levels[upper_idx] * last_open + last_open]
#         ax0.fill_between(base_pred, lower_bound, upper_bound, alpha=0.7, color=Color.orange, label='Next Day Real')
#         lower_idx = jnp.argmax(y_pred)
#         upper_idx = lower_idx + 1
#         lower_bound = [0.01 * discrete_grid_levels[lower_idx] * last_open + last_open,
#                        0.01 * discrete_grid_levels[lower_idx] * last_open + last_open]
#         upper_bound = [0.01 * discrete_grid_levels[upper_idx] * last_open + last_open,
#                        0.01 * discrete_grid_levels[upper_idx] * last_open + last_open]
#         ax0.fill_between(base_pred, lower_bound, upper_bound, alpha=0.7, color=Color.green, label='Next Day Pred')
#
#     open_data = denormalize(input[:, 1], normalizer["price"])
#
#     """ Plot close price """
#     ax0.axvline(x=len(sequence_data) - 1, color=Color.red, linewidth=2)
#
#     ax0.plot(base, sequence_data, label='Close Price', color=Color.blue, linewidth=linewidth, marker='o',
#              markersize=marker_size)
#     # ax0.plot(base, open_data, label='Open Price', color=Color.pink,  linewidth=linewidth, marker='o', markersize=marker_size)
#     if output_mode != 'discrete_grid':
#         ax0.plot(base_pred, real_data, label='Next Day Real', color=Color.orange, linewidth=linewidth, marker='o',
#                  markersize=marker_size, linestyle='--')
#         ax0.plot(base_pred, mean_avg_data, label='Next Day Window Avg', color=Color.purple, linewidth=linewidth,
#                  marker='o', markersize=marker_size, linestyle='--')
#     # ax0.plot(base, open_data, label='Open Price', color=Color.red,  linewidth=linewidth, marker='o', markersize=marker_size)
#
#     # Add error bars
#     if variances is not None:
#         variances = denormalize(variances, normalizer=normalizer["price"])
#         std_dev = jnp.sqrt(variances)
#         upper_bound = [sequence_data[-1], float(prediction_data[-1]) + 1.96 * float(std_dev)]  # 95% confidence interval
#         lower_bound = [sequence_data[-1], float(prediction_data[-1]) - 1.96 * float(std_dev)]  # 95% confidence interval
#
#         ax0.errorbar(base_pred[1], prediction_data[1], yerr=std_dev * 1.96, color=Color.green, capsize=5, linewidth=2)
#         ax0.fill_between(base_pred, upper_bound, lower_bound, alpha=0.2, color=Color.green)
#
#     ax0.set_ylim(
#         [np.min(sequence_data) - 0.05 * np.min(sequence_data), np.max(sequence_data) + 0.05 * np.max(sequence_data)])
#
#     ax0.set_ylabel('Close Price [$]', fontsize=18, fontweight='bold')
#     ax0.legend(ncol=2)
#
#     """ Plot high/low price """
#     high_data = denormalize(input[:, 2], normalizer["price"])
#     low_data = denormalize(input[:, 3], normalizer["price"])
#     ax1.plot(base, high_data, label='High Price', color=Color.pink, linewidth=linewidth, marker='o',
#              markersize=marker_size)
#     ax1.plot(base, low_data, label='Low Price', color=Color.purple, linewidth=linewidth, marker='o',
#              markersize=marker_size)
#     ax1.set_ylabel('High/Low Price [$]', fontsize=18, fontweight='bold')
#     ax1.set_xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
#     ax1.legend()
#
#     """ Plot volume """
#     volume_data = denormalize(input[:, 4], normalizer["volume"])
#     ax2.plot(base, volume_data, label='Volume', color=Color.yellow, linewidth=linewidth, marker='o',
#              markersize=marker_size)
#     ax2.set_ylabel('Volume', fontsize=18, fontweight='bold')
#     ax2.set_xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
#     ax2.legend()
#
#     """ Plot trades """
#     # trades_data = denormalize(input[:, 5], normalizer["trades"])
#     # ax3.plot(base, trades_data, label='Trades', color=Color.blue,  linewidth=linewidth, marker='o', markersize=marker_size)
#     # ax3.set_ylabel('Trades', fontsize=18, fontweight='bold')
#     # ax3.set_xlabel('Date [Sequence]', fontsize=18, fontweight='bold')
#     # ax3.legend()
#
#     if initial_date is not None:
#         title = f'Jaxer Predictor || Initial Date: {initial_date}'
#     else:
#         title = f'Jaxer Predictor'
#
#     if output_mode != 'discrete_grid':
#         title += f' || Error {error:.2f} $ ({percent_error:.1f}) %'
#     else:
#         title += f' || Right Prediction: {jnp.argmax(y_true) == jnp.argmax(y_pred)}'
#
#     plt.suptitle(title, fontsize=20, fontweight='bold')
#     plt.grid(True)
#     plt.tight_layout()
#
#     if foldername is not None:
#         plt.savefig(f"{foldername}/{name}.png")
#     else:
#         plt.show()
#     plt.close()


def interquartile_mean(window):
    q1, q3 = np.percentile(window, [25, 75])
    iqm_vals = [x for x in window if q1 <= x <= q3]

    if len(iqm_vals) == 0:  # Check for empty slice
        return np.median(window)  # Return median as an alternative measure
    else:
        return np.mean(iqm_vals)


def moving_iqm_and_bounds(data, window_size=20):
    iqm_data = []
    max_data = []
    min_data = []
    n = len(data)

    # We smooth the data using a moving window and Interquartile mean, the last points
    # have a reduced size window to keep the same dimensions as the input vector
    for i in range(n):
        start_idx = i
        end_idx = min(n, i + window_size)
        window = data[start_idx:end_idx]

        iqm = interquartile_mean(window)
        max_val = np.max(window)
        min_val = np.min(window)

        iqm_data.append(iqm)
        max_data.append(max_val)
        min_data.append(min_val)

    return iqm_data, max_data, min_data


def plot_tensorboard_experiment(exp_path: str, save_path: Optional[str] = None, window_size: int = 5):
    """ Plot tensorboard experiment for documentation

    :param exp_path: path to the experiment
    :type exp_path: str

    :param save_path: path to save the plot
    :type save_path: Optional[str]

    :param window_size: window size for the moving average
    :type window_size: int
    """

    # Load the experiment
    reader = SummaryReader(exp_path, pivot=True, extra_columns={'dir_name', 'wall_time'})
    df = reader.scalars

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(10, 8))

    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    # Plot the experiment
    train_acc_dir = df['train/acc_dir']
    test_acc_dir = df['test/acc_dir']
    base = df['step']
    avg_data, max_data, min_data = moving_iqm_and_bounds(train_acc_dir, window_size=window_size)
    ax[0].plot(base, avg_data, label='train', color=Color.green, linewidth=2)
    ax[0].plot(base, max_data, linestyle='--', color=Color.green, linewidth=0.5)
    ax[0].plot(base, min_data, linestyle='--', color=Color.green, linewidth=0.5)
    ax[0].fill_between(base, min_data, max_data, alpha=0.2, color=Color.green)

    avg_data, max_data, min_data = moving_iqm_and_bounds(test_acc_dir, window_size=window_size)
    ax[0].plot(base, avg_data, label='test', color=Color.yellow, linewidth=2)
    ax[0].plot(base, max_data, linestyle='--', color=Color.yellow, linewidth=0.5)
    ax[0].plot(base, min_data, linestyle='--', color=Color.yellow, linewidth=0.5)
    ax[0].fill_between(base, min_data, max_data, alpha=0.2, color=Color.yellow)
    ax[0].legend()
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('acc_dir (%)')

    train_mape = df['train/mape']
    test_mape = df['test/mape']
    base = df['step']
    avg_data, max_data, min_data = moving_iqm_and_bounds(train_mape, window_size=window_size)
    ax[1].plot(base, avg_data, label='train', color=Color.green, linewidth=2)
    ax[1].plot(base, max_data, linestyle='--', color=Color.green, linewidth=0.5)
    ax[1].plot(base, min_data, linestyle='--', color=Color.green, linewidth=0.5)
    ax[1].fill_between(base, min_data, max_data, alpha=0.2, color=Color.green)

    avg_data, max_data, min_data = moving_iqm_and_bounds(test_mape, window_size=window_size)
    ax[1].plot(base, avg_data, label='test', color=Color.yellow, linewidth=2)
    ax[1].plot(base, max_data, linestyle='--', color=Color.yellow, linewidth=0.5)
    ax[1].plot(base, min_data, linestyle='--', color=Color.yellow, linewidth=0.5)
    ax[1].fill_between(base, min_data, max_data, alpha=0.2, color=Color.yellow)
    ax[1].legend()
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('mape (%)')

    train_loss = df['train/loss']
    avg_data, max_data, min_data = moving_iqm_and_bounds(train_loss, window_size=window_size)
    ax[2].plot(base, avg_data, label='train', color=Color.pink, linewidth=2)
    ax[2].plot(base, max_data, linestyle='--', color=Color.pink, linewidth=0.5)
    ax[2].plot(base, min_data, linestyle='--', color=Color.pink, linewidth=0.5)
    ax[2].fill_between(base, min_data, max_data, alpha=0.2, color=Color.pink)
    ax[2].legend()
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('loss')
    # log format
    ax[2].set_yscale('log')

    lr = df['train/lr']
    ax[3].plot(base, lr, label='lr', color=Color.red, linewidth=2)
    ax[3].legend()
    ax[3].set_xlabel('Epochs')
    ax[3].set_ylabel('learning rate')

    # get training time
    times = df["wall_time"].values
    begin = times[0][0]
    end = times[-1][0]
    elapsed_time = (end - begin) / 60.

    plt.suptitle(f'jaxer training metrics | elapsed: {elapsed_time:.2f} min ({elapsed_time/60.:.2f} hours)')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
