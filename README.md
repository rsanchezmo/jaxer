# JAXER
Jax and Flax Time Series Prediction Transformer. The goal of this repo is to learn [**Jax**](https://jax.readthedocs.io/en/latest/) and [**Flax**](https://flax.readthedocs.io/en/latest/) by implementing a deep learning model. I just wanted to test the easyness, speed and robustness of this new libraries compared to PyTorch or Tensorflow. 

In this case, a **transformer** for time series prediction. I have decided to predict BTC due to the availability of the data. However, this could be used for any other time series data, such as market stocks.

Note that this is yet a **work in progress**. Once the works is done, some results and graphs will be added to this README.

The current logo was created by DALLE-3 from OpenAI using the following prompt:
```
Render of a transformer model as a hologram, projecting from a digital device, with a faint BTC logo in the holographic projection, without any text.
```

![Jaxer Logo](/data/btc_transformer.png)

## ROADMAP
- Download BTC dataset ✔️
- Create a dataset class ✔️
- Create the trainer class ✔️
- Train the model:
    - **Encoder** only (next day prediction)
    - **Encoder + Decoder** (N days predictions)
- Create a test script to display the results

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Training
Set the configuration parameters in the `config.py` file. Then, run the training script:
```bash
python train.py
```

### Testing
*TODO*

## Contributors
- [Rodrigo Sánchez Molina](rsanchezm98@gmail.com)
