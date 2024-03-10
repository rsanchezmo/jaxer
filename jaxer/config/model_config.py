from dataclasses import dataclass


@dataclass
class ModelConfig:
    """ Configuration class for the model

    :param d_model: dimension of the model
    :type d_model: int

    :param num_layers: number of encoder layers in the transformer
    :type num_layers: int

    :param head_layers: number of layers in the prediction head
    :type head_layers: int

    :param n_heads: number of heads in the multihead attention
    :type n_heads: int

    :param dim_feedforward: dimension of the feedforward network
    :type dim_feedforward: int

    :param dropout: dropout rate
    :type dropout: float

    :param max_seq_len: maximum sequence length (context window)
    :type max_seq_len: int

    :param flatten_encoder_output: whether to flatten the encoder output or get the last token
    :type flatten_encoder_output: bool

    :param fe_blocks: number of blocks in the feature extractor
    :type fe_blocks: int

    :param use_time2vec: whether to use time2vec in the feature extractor
    :type use_time2vec: bool

    :param output_mode: output mode of the model (mean, distribution or discrete_grid)
    :type output_mode: str

    :param use_resblocks_in_head: whether to use residual blocks in the head
    :type use_resblocks_in_head: bool

    :param use_resblocks_in_fe: whether to use residual blocks in the feature extractor
    :type use_resblocks_in_fe: bool

    :param average_encoder_output: whether to average the encoder output (if not flattened)
    :type average_encoder_output: bool

    :param norm_encoder_prev: whether to normalize the encoder prev to the attention
    :type norm_encoder_prev: bool

    """
    d_model: int
    num_layers: int
    head_layers: int
    n_heads: int
    dim_feedforward: int
    dropout: float
    max_seq_len: int
    flatten_encoder_output: bool
    fe_blocks: int
    use_time2vec: bool
    output_mode: str
    use_resblocks_in_head: bool
    use_resblocks_in_fe: bool
    average_encoder_output: bool
    norm_encoder_prev: bool

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
