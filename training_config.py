from jaxer.utils.config import ModelConfig, Config


config = ModelConfig(
    d_model=64,
    num_layers=1,
    head_layers=2,
    n_heads=4,
    dim_feedforward=128,
    dropout=0.0,
    max_seq_len=30,
    input_features=6
)

training_config = Config(
    model_config=config,
    log_dir="results",
    experiment_name="v1",
    num_epochs=100,
    learning_rate=1e-4,
    warmup_epochs=10,
    dataset_path="./data/BTCUSD.csv",
    batch_size=128,
    test_split=0.1,
    seed=0,
    normalizer_mode="window"
)