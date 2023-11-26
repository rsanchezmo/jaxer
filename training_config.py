from jaxer.utils.config import ModelConfig, Config


model_config = ModelConfig(
    d_model=64,
    num_layers=2,
    head_layers=2,
    n_heads=2,
    dim_feedforward=4*64,  # 4 * d_model
    dropout=0.05,
    max_seq_len=40,
    input_features=5,
    flatten_encoder_output=False
)

config = Config(
    model_config=model_config,
    log_dir="results",
    experiment_name="distribution",
    num_epochs=100,
    learning_rate=5e-4,
    warmup_epochs=20,
    dataset_path="./data/BTCUSD.csv",
    batch_size=128,
    test_split=0.2,
    seed=0,
    normalizer_mode="window"
)