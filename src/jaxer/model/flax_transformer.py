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
    output_mode: str = "distribution"  # [distribution, mean, discrete_grid]
    discrete_grid_levels: int = 2  # 1 level up, 1 level down
    use_resblocks_in_head: bool = True
    use_resblocks_in_fe: bool = True
    average_encoder_output: bool = False
    norm_encoder_prev: bool = False


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
    d_model: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        weights_linear = self.param(
            "weights_linear",
            nn.initializers.uniform(),
            (self.max_seq_len, 1),
            self.dtype,
        )
        weights_periodic = self.param(
            "weights_periodic",
            nn.initializers.uniform(),
            (self.max_seq_len, self.d_model - 1),
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
            (self.d_model - 1,),
            self.dtype,
        )

        # TIME LINEAR IS THE 1ST COMPONENT OF THE TIME2VEC wich has the shape (batch_size, max_seq_len, d_model)
        # so time_linear has the shape (batch_size, max_seq_len, 1)
        # and time_periodic has the shape (batch_size, max_seq_len, d_model - 1)

        tau = x[:, :, -1]  # the last column of the input is the time which has the shape (batch_size, max_seq_len, 1)
        tau = tau[:, :, None]  # tau has the shape (batch_size, max_seq_len, 1)
        time_linear = weights_linear * tau + bias_linear  # time_linear has the shape (batch_size, max_seq_len)

        time_periodic = jnp.sin(weights_periodic * tau + bias_periodic)  

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
        step_dim = 1
        if self.config.fe_blocks == 0:
            feature_dim = self.config.d_model
        else:
            step_dim = self.config.d_model // (self.config.fe_blocks + 1)
            feature_dim = step_dim
            residual = (self.config.d_model % (self.config.fe_blocks + 1))

        x = nn.Dense(
            features=feature_dim,  # time embeddings will be concatenated later
            dtype=self.config.dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )

        # some residual blocks
        """
        I think the use of RB may not be relevant here, as the input is not very complex at this point
        """
        for i in range(self.config.fe_blocks):
            if self.config.use_resblocks_in_fe:
                x = ResidualBlock(
                    dtype=self.config.dtype,
                    feature_dim=feature_dim,  # time embeddings will be concatenated later
                    kernel_init=self.config.kernel_init,
                    bias_init=self.config.bias_init,
                    norm=True,
                    norm_prev=False
                )(x)
            else:
                feature_dim += step_dim
                if i == self.config.fe_blocks - 1:
                    feature_dim += residual

                x = nn.Dense(
                    features=feature_dim,  # time embeddings will be concatenated later
                    dtype=self.config.dtype,
                    kernel_init=self.config.kernel_init,
                    bias_init=self.config.bias_init,
                )(x)
                x = nn.relu(x)
        
        if self.config.use_resblocks_in_fe: 
            x = nn.Dense(
                features=self.config.d_model,  # time embeddings will be concatenated later
                dtype=self.config.dtype,
                kernel_init=self.config.kernel_init,
                bias_init=self.config.bias_init,
            )(x)
            x = nn.relu(x)

        return x
    
class EncoderBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, encoder_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        assert inputs.ndim == 3, f"Expected x to have 3 dimensions, got {inputs.ndim}"

        """ Self Attention Block"""
        if self.config.norm_encoder_prev:
            x = nn.LayerNorm(dtype=self.config.dtype)(inputs)
        else:
            x = inputs

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

        if not self.config.norm_encoder_prev:
            x = nn.LayerNorm(dtype=self.config.dtype)(x)

        """ Feed Forward Block """
        if self.config.norm_encoder_prev:
            y = nn.LayerNorm(dtype=self.config.dtype)(x)
        else:
            y = x
        # y = FeedForwardBlockConv1D(
        #     config=self.config,
        #     out_dim=self.config.d_model
        # )(y)

        y = FeedForwardBlock(
            config=self.config,
            out_dim=self.config.d_model
        )(y)

        result = x + y

        if not self.config.norm_encoder_prev:
            result = nn.LayerNorm(dtype=self.config.dtype)(result)

        return result

class Encoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Applies the encoder module """

        """ Feature Embeddings"""
        input_embeddings = FeatureExtractor(
                config=self.config
        )(x[:, :, :-1])

        """ Time2Vec """
        if self.config.use_time2vec:
            time2vec = Time2Vec(
                dtype=self.config.dtype,
                kernel_init=self.config.kernel_init,
                bias_init=self.config.bias_init,
                max_seq_len=self.config.max_seq_len,
                d_model=self.config.d_model
            )(x)

            x = input_embeddings + time2vec

        else:
            x = AddPositionalEncoding(
                config=self.config
            )(input_embeddings)

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


        if self.config.output_mode == 'discrete_grid':
            x = nn.Dense(features=self.config.discrete_grid_levels,
                        dtype=self.config.dtype,
                        kernel_init=self.config.kernel_init,
                        bias_init=self.config.bias_init)(x)
            x = nn.softmax(x, axis=-1)
            return x

        mean = nn.Dense(
            features=1,
            dtype=self.config.dtype,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(x)

        if self.config.output_mode == 'mean':
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
            if self.config.average_encoder_output:
                x = jnp.mean(x, axis=1)
            else:
                x = x[:, -1, :]  # get the last element of the sequence

        x = PredictionHead(config=self.config)(x)

        """ output a probability distribution """
        if self.config.output_mode == 'distribution':
            mean, log_variance = x
            variance = jnp.exp(log_variance)
            return mean, variance
        
        return x