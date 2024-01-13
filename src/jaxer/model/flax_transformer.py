from typing import Any, Optional, Callable
from flax import linen as nn
import jax.numpy as jnp
from flax import struct
import numpy as np


@struct.dataclass
class TransformerConfig:
    d_model: int = 512
    n_heads: int = 8
    num_layers: int = 6
    head_layers: int = 2
    dim_feedforward: int = 2048
    max_seq_len: int = 256
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    input_features: int = 7
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    deterministic: bool = False
    flatten_encoder_output: bool = False
    fe_blocks: int = 2
    use_time2vec: bool = True
    output_distribution: bool = True
    use_resblocks_in_head: bool = True
    use_resblocks_in_fe: bool = True


class FeedForwardBlock(nn.Module):
    config: TransformerConfig
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Applies the feed forward block module """

        x = nn.Dense(
            features=self.config.dim_feedforward,
            dtype=self.config.dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )
        x = nn.Dense(
            features=self.out_dim,
            dtype=self.config.dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )

        return x       


class FeedForwardBlockConv1D(nn.Module):
    config: TransformerConfig
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Applies the feed forward block module """

        x = nn.Conv(
            features=self.config.dim_feedforward,
            dtype=self.config.dtype,
            kernel_size=(1,),
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )
        x = nn.Conv(
            features=self.out_dim,
            dtype=self.config.dtype,
            kernel_size=(1,),
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )

        return x 
    
    
def sinusoidal_init(max_len: int = 2048, min_scale: float = 1.0, max_scale: float = 10000.0) -> Callable:
  """ 1D Sinusoidal Position Embedding Initializer """

  def init(shape):
    """ Sinusoidal init """
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, : d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionalEncoding(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Applies the positional encoding module """

        pos_embed_shape = (1, self.config.max_seq_len, x.shape[-1])
        pos_embedding = sinusoidal_init(self.config.max_seq_len)(
            pos_embed_shape
        )

        pe = pos_embedding[:, :x.shape[1], :]

        return x + pe
    
class Time2Vec(nn.Module):
    dtype: jnp.dtype
    kernel_init: Callable
    bias_init: Callable
    max_seq_len: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        weights_linear = self.param(
            "weights_linear",
            nn.initializers.uniform(),
            (self.max_seq_len,),
            self.dtype,
        )
        weights_periodic = self.param(
            "weights_periodic",
            nn.initializers.uniform(),
            (self.max_seq_len,),
            self.dtype,
        )
        bias_linear = self.param(
            "bias_linear",
            nn.initializers.uniform(),
            (1,),
            self.dtype,
        )
        bias_periodic = self.param(
            "bias_periodic",
            nn.initializers.uniform(),
            (1,),
            self.dtype,
        )

        time_linear = weights_linear * x[:, :, -1] + bias_linear

        time_periodic = jnp.sin(weights_periodic * x[:, :, -1] + bias_periodic)  

        # add a dimension
        time_linear = time_linear[:, :, None]
        time_periodic = time_periodic[:, :, None] 

        return jnp.concatenate([time_linear, time_periodic], axis=-1)
    

class ResidualBlock(nn.Module):
    dtype: jnp.dtype
    feature_dim: int
    kernel_init: Callable
    bias_init: Callable
    norm: bool = True
    norm_prev: bool = False  # Noticed that the normalization should be done after the residual connection, if not, divergence occurs

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        inputs = x
        
        assert x.shape[-1] == self.feature_dim, f"Expected x to have {self.feature_dim} dimensions, got {x.shape[-1]}"

        if self.norm and self.norm_prev:
            x = nn.LayerNorm()(x)

        x = nn.Dense(
            features=self.feature_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        
        x = nn.gelu(x) 

        x = inputs + x

        if self.norm and not self.norm_prev:
            x = nn.LayerNorm()(x)

        return x

    
class FeatureExtractor(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Feature extractor based on residual MLP networks """

        # first part of the network to get to the right dimension
        x = nn.Dense(
            features=self.config.d_model,  # time embeddings will be concatenated later
            dtype=self.config.dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )

        # some residual blocks
        for _ in range(self.config.fe_blocks):
            if self.config.use_resblocks_in_fe:
                x = ResidualBlock(
                    dtype=self.config.dtype,
                    feature_dim=self.config.d_model,  # time embeddings will be concatenated later
                    kernel_init=self.config.kernel_init,
                    bias_init=self.config.bias_init,
                    norm=True,
                    norm_prev=False
                )(x)
            else:
                x = nn.Dense(
                    features=self.config.d_model,  # time embeddings will be concatenated later
                    dtype=self.config.dtype,
                    kernel_init=self.config.kernel_init,
                    bias_init=self.config.bias_init,
                )(x)
                x = nn.gelu(x)

        return x
    
class EncoderBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, encoder_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        assert inputs.ndim == 3, f"Expected x to have 3 dimensions, got {inputs.ndim}"

        """ Self Attention Block"""
        x = nn.LayerNorm(dtype=self.config.dtype)(inputs)

        # nn.SelfAttention because the input is from the same source [self], 
        # nn.MultiHeadDotProductAttention if the input is from different sources
        x = nn.SelfAttention(
            num_heads=self.config.n_heads,
            dtype=self.config.dtype,
            qkv_features=self.config.d_model,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=self.config.dropout,
            deterministic=self.config.deterministic,
        )(x, mask=encoder_mask)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )

        x = x + inputs

        """ Feed Forward Block """
        y = nn.LayerNorm(dtype=self.config.dtype)(x)
        # y = FeedForwardBlockConv1D(
        #     config=self.config,
        #     out_dim=self.config.d_model
        # )(y)

        y = FeedForwardBlock(
            config=self.config,
            out_dim=self.config.d_model
        )(y)

        return x + y

class Encoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Applies the encoder module """

        """ Time2Vec """
        if self.config.use_time2vec:
            time_embeddings = Time2Vec(
                dtype=self.config.dtype,
                kernel_init=self.config.kernel_init,
                bias_init=self.config.bias_init,
                max_seq_len=self.config.max_seq_len
            )(x)

            x = jnp.concatenate([x, time_embeddings], axis=-1)

        """ Feature Embeddings"""
        x = FeatureExtractor(
                config=self.config
        )(x)
    
        # TODO: check if the positional encoding is needed when using time2vec as it already contains the time information relationship
        x = AddPositionalEncoding(
            config=self.config
        )(x)

        """ Encoder Blocks """
        for _ in range(self.config.num_layers):
            x = EncoderBlock(
                config=self.config
            )(x)

        return x
    
class PredictionHead(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        if self.config.flatten_encoder_output:
            """ to adapt the size to be able to use the residual blocks as they expect the same size across the block """
            x = nn.Dense(features=self.config.d_model,
                        dtype=self.config.dtype,
                        kernel_init=self.config.kernel_init,
                        bias_init=self.config.bias_init)(x)
            x = nn.relu(x)
                        
        for _ in range(self.config.head_layers - 1):
            if self.config.use_resblocks_in_head:
                x = ResidualBlock(feature_dim=self.config.d_model,
                                    dtype=self.config.dtype,
                                    kernel_init=self.config.kernel_init,
                                    bias_init=self.config.bias_init,
                                    norm=True,
                                    norm_prev=False)(x)
            else:
                x = nn.Dense(features=self.config.d_model,
                            dtype=self.config.dtype,
                            kernel_init=self.config.kernel_init,
                            bias_init=self.config.bias_init)(x)
                x = nn.relu(x)

        mean = nn.Dense(
            features=1,
            dtype=self.config.dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(x)

        if not self.config.output_distribution:
            return mean
            
        log_variance = nn.Dense(
            features=1,
            dtype=self.config.dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(x)

        return mean, log_variance

class Transformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """ Applies the transformer module """

        """ Encoder """
        x = Encoder(
            config=self.config
        )(x)

        """ Regression Head """
        # x = nn.LayerNorm(dtype=self.config.dtype)(x)

        # flatten the output:
        if self.config.flatten_encoder_output:
            x = x.reshape((x.shape[0], -1))
        else:
            #x = jnp.mean(x, axis=1)  # average over the time dimension [Global Average Pooling 1D]
            x = x[:, -1, :]  # get the last element of the sequence

        x = PredictionHead(config=self.config)(x)

        """ output a probability distribution """
        if self.config.output_distribution:
            mean, log_variance = x
            variance = jnp.exp(log_variance)
            return mean, variance
        
        return x