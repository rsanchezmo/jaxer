from jaxer.utils.config import ModelConfig, Config


model_config = ModelConfig(
    d_model=128,
    num_layers=8,
    head_layers=3,
    n_heads=8,
    dim_feedforward=4*128,  # 4 * d_model
    dropout=0.05,
    max_seq_len=30,
    input_features=5,
    flatten_encoder_output=False
)

config = Config(
    model_config=model_config,
    log_dir="results",
    experiment_name="global_minmax_larger_2",
    num_epochs=100,
    learning_rate=1e-4,
    warmup_epochs=10,
    dataset_path="./data/BTCUSD.csv",
    batch_size=128,
    test_split=0.1,
    seed=0,
    normalizer_mode="global_minmax",
)