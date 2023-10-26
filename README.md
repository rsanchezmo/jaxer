# JAXER
Jax and Flax Time Series Prediction Transformer. The goal of this repo is to learn [**Jax**](https://jax.readthedocs.io/en/latest/) and [**Flax**](https://flax.readthedocs.io/en/latest/) by implementing a deep learning model. I just wanted to test the easyness, speed and robustness of this new libraries compared to PyTorch or Tensorflow. 

In this case, a **transformer** for time series prediction. I have decided to predict BTC due to the availability of the data. However, this could be used for any other time series data, such as market stocks.

Note that this is yet a **work in progress**. Once the works is done, some results and graphs will be added to this README.

The current logo was created by DALLE-3 from OpenAI using the following prompt:
```
Render of a transformer model as a hologram, projecting from a digital device, with a faint BTC logo in the holographic projection, without any text.
```

![Jaxer Logo](/data/btc_transformer.png)


## Roadmap
- Download BTC dataset ✔️
- Create a dataset class ✔️
- Create the trainer class ✔️
- Train the model:
    - **Encoder** only (next day prediction) ✔️ Not succesful yet, but the training loop works. Must check data sequences.
    - **Encoder + Decoder** (N days predictions)
- Create an agent that loads the model and act as a predictor. Now built, but too slow. CHECK WHY
- Add a logger to the trainer class
- Add a lr scheduler to the trainer class

## Installation

Clone the repo:
```bash
git clone https://github.com/rsanchezmo/jaxer.git
cd jaxer
```

Create a python venv:
```bash
python -m venv venv
```

Install your desired jax version. For example, for CUDA 12 on Linux:
```bash
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then install the rest of the dependencies:

```bash
pip install .
```

## Usage
### Training
Set the configuration parameters in the `config.py` file. Then, run the training script:
```bash
python src/train.py
```

### Testing
*TODO*

## Contributors
- [Rodrigo Sánchez Molina](rsanchezm98@gmail.com)
