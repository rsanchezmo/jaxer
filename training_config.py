from jaxer.utils.config import ModelConfig, Config


model_config = ModelConfig(
    d_model=128,
    num_layers=4,
    head_layers=1,
    n_heads=8,
    dim_feedforward=4*64,  # 4 * d_model
    dropout=0.05,
    max_seq_len=30,
    input_features=5,
    flatten_encoder_output=False
)

config = Config(
    model_config=model_config,
    log_dir="results",
    experiment_name="distribution_v2_def_global",
    num_epochs=100,
    learning_rate=3e-4,
    warmup_epochs=15,
    dataset_path="./data/BTCUSD.csv",
    batch_size=64,
    test_split=0.2,
    seed=0,
    normalizer_mode="global"
)