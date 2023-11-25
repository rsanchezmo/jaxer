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
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    deterministic: bool = False

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
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )

        return x        
    
    
def sinusoidal_init(max_len: int = 2048, min_scale: float = 1.0, max_scale: float = 10000.0) -> Callable:
  """ 1D Sinusoidal Position Embedding Initializer """

  def init(key, shape, dtype=np.float32):
    """ Sinusoidal init """
    del key, dtype
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
            None, pos_embed_shape, None
        )

        pe = pos_embedding[:, :x.shape[1], :]

        return x + pe
    

class ResidualBlock(nn.Module):
    dtype: jnp.dtype
    feature_dim: int
    kernel_init: Callable
    bias_init: Callable

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        inputs = x
        x = nn.Dense(
            features=self.feature_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        
        x = nn.gelu(x) 

        x = inputs + x

        x = nn.LayerNorm()(x)

        return x

    
class FeatureExtractor(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Feature extractor based on residual MLP networks """

        x = ResidualBlock(feature_dim=self.config.d_model,
                          dtype=self.config.dtype,
                          kernel_init=self.config.kernel_init,
                          bias_init=self.config.bias_init)(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )

        x = ResidualBlock(feature_dim=self.config.d_model,
                          dtype=self.config.dtype,
                          kernel_init=self.config.kernel_init,
                          bias_init=self.config.bias_init)(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )

        x = ResidualBlock(feature_dim=self.config.d_model,
                          dtype=self.config.dtype,
                          kernel_init=self.config.kernel_init,
                          bias_init=self.config.bias_init)(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )

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

        """ Feature Embeddings"""
        x = FeatureExtractor(
            config=self.config
        )(x)

        """ Positional Encoding """
        x = AddPositionalEncoding(
            config=self.config
        )(x)

        """ Encoder Blocks """
        for _ in range(self.config.num_layers):
            x = EncoderBlock(
                config=self.config
            )(x)

        return x
    
class Head(nn.module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.config.head_layers):
            x = ResidualBlock(feature_dim=self.config.d_model,
                                dtype=self.config.dtype,
                                kernel_init=self.config.kernel_init,
                                bias_init=self.config.bias_init)(x)
            
        x = nn.Dense(
            features=1,
            dtype=self.config.dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(x)

        return x

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
        x = nn.LayerNorm(dtype=self.config.dtype)(x)
        x = x[:, -1, :]

        x = Head(config=self.config)(x)

        return x