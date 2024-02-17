# Jaxer

Jax and Flax Time Series Prediction Transformer. The goal of this repo is to learn [Jax](https://jax.readthedocs.io/en/latest/) and [Flax](https://flax.readthedocs.io/en/latest/) by implementing a deep learning model. In this case, a **transformer** for time series prediction. I have decided to predict BTC due to the availability of the data. However, this could be used for any other time series data, such as market stocks.

**Note:** This repository is yet under development.

![Jaxer Logo](./docs/images/btc_transformer.png)

You can check out the project [documentation](https://jaxer.readthedocs.io/).

## Roadmap

- Download BTC dataset - Done
- Create a dataset class - Done
- Add normalizer as dataset output - Done
- Create the trainer class - Done
- Train the model:
    - **Encoder** only (next day prediction) - Done
    - **Encoder + Decoder** (N days predictions)
- Create an agent that loads the model and acts as a predictor - Done
- Add tensorboard support to the trainer class - Done
- Add a logger to the trainer class - Done
- Add a lr scheduler to the trainer class: warmup cosine scheduler - Done
- Make the prediction head output a distribution instead of a single value to cover uncertainty - Done
- Include Time2Vec as time embedding - Done
- Select output head format [resblocks, mlp, etc] - Done
- Select input feature extractor format [resblocks, mlp, etc] - Done
- Create HP tuner - Done
- Add support for outputting discrete ranges instead of a distribution or the mean - Done
- Add support for multiple tickers dataset (to get more points)
- Refactor the plotter code to be more general
- Make logger a singleton

## Description

### Model

#### Encoder Only

Model is implemented in Flax. Feature extractors and prediction head consist of an MLP with residual blocks and layer normalization. The encoder consists of L x Encoder blocks with self-attention, layer norms, and feedforwarding. The output of the encoder is flattened to feed the prediction head. Each token of the output sequence contains the "context" of the others related to itself.

We could get the last token instead of flattening the encoder output, but as we are not masking the attention, each contextual embedding can be valuable for the prediction head. However, this is something I am exploring these days. The need to mask the attention, and if not, the need to use the positional encoding.

The output of the prediction head is a probability distribution. I want to have a better understanding of the uncertainty of the model, so I have decided to use a distribution instead of a single value. The distribution is a normal distribution with mean and variance. By doing so, the loss function is the negative log likelihood of the distribution.

## Conclusions

- Jax and Flax are easy to use once you learn the basics. I have tried to make the code as simple as possible so it can be easily understood, encapsulating the complexity of the libraries.
- The model performance is not bad at all considering the input information that feeds the model. However, dataset size is small due to one day resolution. Increasing resolution may improve the model performance.
- The general idea from this repo is that the transformer can be applied to time series prediction and can be implemented with state-of-the-art GPU-accelerated deep learning libraries such as Jax and Flax.

## Future Work

- TBD

## Contributors

Rodrigo Sánchez Molina

- Email: rsanchezm98@gmail.com
- Linkedin: [rsanchezm98](https://www.linkedin.com/in/rsanchezm98/)
- Github: [rsanchezmo](https://github.com/rsanchezmo)
