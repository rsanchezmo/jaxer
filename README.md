# Jaxer
Jax and Flax Time Series Prediction Transformer. The goal of this repo is to learn [**Jax**](https://jax.readthedocs.io/en/latest/) and [**Flax**](https://flax.readthedocs.io/en/latest/) by implementing a deep learning model. In this case, a **transformer** for time series prediction. I have decided to predict BTC due to the availability of the data. However, this could be used for any other time series data, such as market stocks.

> [!IMPORTANT]
> This repository is yet under development. However, I am planning to complete it soon. 


The current logo was created by DALLE-3 from OpenAI using the following prompt:
```
Render of a transformer model as a hologram, projecting from a digital device, with a faint BTC logo in the holographic projection, without any text.
```

![Jaxer Logo](/data/btc_transformer.png)


## Roadmap
- Download BTC dataset ✔️
- Create a dataset class ✔️
- Add normalizer as dataset output ✔️
- Create the trainer class ✔️
- Train the model:
    - **Encoder** only (next day prediction) ✔️
    - **Encoder + Decoder** (N days predictions)
- Create an agent that loads the model and act as a predictor ✔️
- Add tensorboard support to the trainer class ✔️
- Add a logger to the trainer class ✔️
- Add a lr scheduler to the trainer class: warmup cosine scheduler ✔️
- Make the prediction head output a distribution instead of a single value to cover uncertainaty ✔️
- Include Time2Vec as time embedding ✔️
- Select output head format [resnet, mlp, etc]
- Select input feature extractor format [resnet, mlp, etc]

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

## Description
### Dataset 
The dataset has been downloaded from [Yahoo Finance](https://es.finance.yahoo.com/quote/BTC-USD?p=BTC-USD&.tsrc=fin-srch). The time resolution is a day to avoid any preprocessing to fill missing timesteps. There is a ```.csv``` with all de BTC-USD data. 

The available information is:
- Low/High price
- Close/Open price
- Volume
- Adjusted close price: I decided not to use this information as in BTC is the same as the close price. Must avoid redundant information.

The dataset class is implemented in PyTorch due to the easyness of creating a dataloader. However, as we are working with jax arrays, it was necessary to pass a function to the dataloader to map the torch tensors to jax.ndarrays.

The test set is not randomly computed across the entire dataset. For better generalization capabilities understanding, the test set is taken from the last dataset components; simulating real world prediction. 

#### Normalization
Data must be normalized to avoid exploding gradients during training. Two normalization methods are implemented. First, a normalization across the entire dataset. Second, a window normalization. The second one seems more suitable to avoid losing too much resolution and also to ensure to work over time and not become obsolete (BTC may surpass the current max BTC price or volume). However, this approach can introduce inconsistencies in data scaling across different windows, potentially affecting the model's performance and its ability to generalize to new data. For each approach, two normalizers are implemented: a min-max scaler and a standard scaler.

### Model

#### Encoder Only
![Encoder Only Model](./data/encoder_only_model.png)

Model is implemented in Flax. Feature extractors and prediction head consist on an MLP with residual blocks and layer normalization. The encoder consists on L x Encoder blocks with self attention. layer norms and feedforwarding. The output of the encoder is flattened to feed the prediction head. Each token of the output sequence contains the "context" of the others related to itself. 

We could get the last token instead of flattening the encoder output, but as we are not masking the attention, each contextual embedding can be valuable for the prediction head. However, this is something I am exploring these days. The need to mask the attention, and if not, the need to use the positional encoding.

The output of the prediction head is a probability distribution. I want to have a better understanding of the uncertainty of the model, so I have decided to use a distribution instead of a single value. The distribution is a normal distribution with mean and variance. By doing so, the loss function is the negative log likelihood of the distribution.

## Usage
### Training
Set the configuration parameters in the `training_config.py` file. The training of the model is made really easy, as simple as creating a trainer and calling the `train_and_evaluate` method:

```python
from jaxer.utils.trainer import FlaxTrainer as Trainer
from training_config import config

trainer = Trainer(config=config)
trainer.train_and_evaluate()
```

### Inference
An agent class has been created so you can load a model and use it to predict to any data you want. 

```python
from jaxer.utils.agent import Agent
from jaxer.utils.config import get_best_model
from jaxer.utils.plotter import predict_entire_dataset
import jax.numpy as jnp

# the experiment in the "results" folder
experiment = "transformer_encoder_only_window"
agent = Agent(experiment=experiment, model_name=get_best_model(experiment))

# a random input
x_test = jnp.ones((1, agent.config.model_config["max_seq_len"], agent.config.model_config["input_features"]))

# predict
pred = agent(x_test)

# plot the test set predictions
dataset = Dataset('./data/BTCUSD.csv', agent.config.model_config["max_seq_len"], 
                    norm_mode=agent.config.normalizer_mode, 
                    initial_date=agent.config.initial_date)

train_ds, test_ds = dataset.get_train_test_split(test_size=agent.config.test_split)

predict_entire_dataset(agent, test_ds, mode='test')
```

## Results
Some of the predictions are shown below. As we are predicting a distribution, the 95% confidence interval is shown in the plots in order to have a better understanding of the uncertainty of the model. The upper and lower bounds are computed as ```[mean + 1.96*std, mean - 1.96*std]``` respectively.

![Jaxer Predictions 1](./data/1.png)

![Jaxer Predictions 2](./data/4.png)

![Jaxer Predictions 3](./data/2.png)

Now, the model is either predicting the window mean or lagging the input sequence. It is related to model size and hp and hope to fix it soon. 
![Jaxer Predictions Test](./data/mean_test.png)
![Jaxer Predictions Train](./data/mean_test_2.png)

## Conclusions
- Jax and Flax are easy to use once you learn the basics. I have tried to make the code as simple as possible so it can be easily understood, encapsulating the complexity of the libraries.
- The model performance is not bad at all considering the input information that feeds the model. However, dataset size is small due to one day resolution. Increasing resolution may improve the model performance. 
- The general idea from this repo is that the transformer can be applied to time series prediction, and can be implemented with state of the art gpu accelerated deep learning libraries such as Jax and Flax. 

## Future Work
- Compare the model against other models such as LSTM or GRU.
- Increase time resolution to predict intraday prices (e.g. 1h).
- Add more variables to the model to improve accuracy such as market sentiment analysis. 
- Compare speed and performance against other libraries such as PyTorch or Tensorflow.

## Contributors
Rodrigo Sánchez Molina
- Email: rsanchezm98@gmail.com
- Linkedin: [rsanchezm98](https://www.linkedin.com/in/rsanchezm98/)
- Github: [rsanchezmo](https://github.com/rsanchezmo)
