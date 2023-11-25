from jaxer.utils.config import ModelConfig, Config


config = ModelConfig(
    d_model=32,
    num_layers=2,
    head_layers=2,
    n_heads=2,
    dim_feedforward=64,
    dropout=0.05,
    max_seq_len=20,
    input_features=6
)

training_config = Config(
    model_config=config,
    log_dir="results",
    experiment_name="fixed_short",
    num_epochs=100,
    learning_rate=5e-4,
    warmup_epochs=20,
    dataset_path="./data/BTCUSD.csv",
    batch_size=128,
    test_split=0.2,
    seed=0,
    normalizer_mode="window"
)