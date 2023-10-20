from typing import Optional, Callable
from flax import linen as nn
import jax.numpy as jnp
from flax import struct
import numpy as np


@struct.dataclass
class TransformerConfig:
    d_model: int = 512
    n_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    deterministic: bool = False
    decode: bool = False


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


class DecoderBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, targets: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """ Applies the decoder block module """

        assert targets.ndim == 3, f"Expected x to have 3 dimensions, got {targets.ndim}"

        """ Self Attention Block"""
        x = nn.LayerNorm(dtype=self.config.dtype)(targets)

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
            decode=self.config.decode
        )(x, mask=mask)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )

        x = x + targets

        """ Feed Forward Block """
        y = nn.LayerNorm(dtype=self.config.dtype)(x)
        y = FeedForwardBlock(
            config=self.config
        )(y)

        return x + y
    
    
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

        length = x.shape[1]
        pos_embed_shape = (1, self.config.max_seq_len, length)
        pos_embedding = sinusoidal_init(self.config.max_seq_len)(
            None, pos_embed_shape, None
        )

        pe = pos_embedding[:, :length, :]

        return x + pe
    

class Decoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, targets: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """ Applies the decoder module """

        """ Positional Encoding """


        """ Decoder Blocks """
        for _ in range(self.config.num_layers):
            targets = DecoderBlock(
                config=self.config
            )(targets, mask=mask)


        """ Output Layer """
        targets = nn.LayerNorm(dtype=self.config.dtype)(targets)
        targets = nn.Dense(
            features=self.config.d_model,
            dtype=self.config.dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(targets)

        return targets


class Transformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """ Applies the transformer module """

        """ Decoder """
        x = Decoder(
            config=self.config
        )(x, mask=mask)

        return x