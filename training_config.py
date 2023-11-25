from jaxer.utils.config import ModelConfig, Config


config = ModelConfig(
    d_model=128,
    num_layers=8,
    head_layers=4,
    n_heads=8,
    dim_feedforward=256,
    dropout=0.05,
    max_seq_len=10,
    input_features=6
)

training_config = Config(
    model_config=config,
    log_dir="results",
    experiment_name="fixed_large",
    num_epochs=100,
    learning_rate=1e-4,
    warmup_epochs=10,
    dataset_path="./data/BTCUSD.csv",
    batch_size=128,
    test_split=0.2,
    seed=0,
    normalizer_mode="window"
)