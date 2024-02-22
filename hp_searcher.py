import numpy as np
from jaxer import ExperimentConfig, ModelConfig, DatasetConfig
from jaxer.run import FlaxTrainer as Trainer
from jaxer import get_logger

if __name__ == '__main__':
    model_config_ranges = {
        'd_model': [128, 256, 512],
        'num_layers': [1, 2, 3],
        'head_layers': [1, 2, 3],
        'n_heads': [1, 2, 4],
        'dim_feedforward': [2, 4],
        'dropout': [0.0, 0.05, 0.1],
        'max_seq_len': [12, 24, 36, 48],
        'flatten_encoder_output': [False],
        'fe_blocks': [1, 2],
        'use_time2vec': [False, True],
        'output_mode': ['mean'],
        'use_resblocks_in_head': [False, True],
        'use_resblocks_in_fe': [False, True],
        'average_encoder_output': [False, True],
        'norm_encoder_prev': [False, True]
    }

    training_ranges = {
        'learning_rate': [1e-4, 1e-5],
        'lr_mode': ['linear', 'cosine'],
        'warmup_epochs': [5, 10, 20],
        'batch_size': [64, 128],
        'normalizer_mode': ['global_minmax'],
        'resolution': ['4h'],
        'tickers': ['btc_usd'],
    }

    n_trainings = 20
    trained_set = set()

    counter_trains = 0
    logger = get_logger()

    while counter_trains < n_trainings:
        logger.info(f"Training {counter_trains + 1}/{n_trainings}")

        d_model = int(np.random.choice(model_config_ranges['d_model']))
        num_layers = int(np.random.choice(model_config_ranges['num_layers']))
        head_layers = int(np.random.choice(model_config_ranges['head_layers']))
        n_heads = int(np.random.choice(model_config_ranges['n_heads']))
        dim_feedforward = d_model * int(np.random.choice(model_config_ranges['dim_feedforward']))
        dropout = float(np.random.choice(model_config_ranges['dropout']))
        max_seq_len = int(np.random.choice(model_config_ranges['max_seq_len']))
        flatten_encoder_output = bool(np.random.choice(model_config_ranges['flatten_encoder_output']))
        fe_blocks = int(np.random.choice(model_config_ranges['fe_blocks']))
        use_time2vec = bool(np.random.choice(model_config_ranges['use_time2vec']))
        output_mode = str(np.random.choice(model_config_ranges['output_mode']))
        use_resblocks_in_head = bool(np.random.choice(model_config_ranges['use_resblocks_in_head']))
        use_resblocks_in_fe = bool(np.random.choice(model_config_ranges['use_resblocks_in_fe']))
        average_encoder_output = bool(np.random.choice(model_config_ranges['average_encoder_output']))
        norm_encoder_prev = bool(np.random.choice(model_config_ranges['norm_encoder_prev']))

        learning_rate = float(np.random.choice(training_ranges['learning_rate']))
        lr_mode = str(np.random.choice(training_ranges['lr_mode']))
        warmup_epochs = int(np.random.choice(training_ranges['warmup_epochs']))
        batch_size = int(np.random.choice(training_ranges['batch_size']))
        normalizer_mode = str(np.random.choice(training_ranges['normalizer_mode']))
        resolution = str(np.random.choice(training_ranges['resolution']))
        tickers = str(np.random.choice(training_ranges['tickers']))

        params = (d_model, num_layers, head_layers, n_heads, dim_feedforward, dropout, max_seq_len,
                  flatten_encoder_output, fe_blocks, use_time2vec, output_mode, use_resblocks_in_head,
                  use_resblocks_in_fe, learning_rate, lr_mode, warmup_epochs, batch_size, normalizer_mode,
                  resolution, tickers)

        if params in trained_set:
            logger.warning(f"Already trained {params}, skipping...")
            continue

        counter_trains += 1
        trained_set.add(params)

        model_config = ModelConfig(
            d_model=d_model,
            num_layers=num_layers,
            head_layers=head_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            flatten_encoder_output=flatten_encoder_output,
            fe_blocks=fe_blocks,
            use_time2vec=use_time2vec,
            output_mode=output_mode,
            use_resblocks_in_head=use_resblocks_in_head,
            use_resblocks_in_fe=use_resblocks_in_fe,
            average_encoder_output=average_encoder_output,
            norm_encoder_prev=norm_encoder_prev
        )

        dataset_config = DatasetConfig(
            datapath='./data/datasets/data',
            output_mode=output_mode,
            discrete_grid_levels=[],
            initial_date='2018-01-01',
            norm_mode=normalizer_mode,
            resolution=resolution,
            tickers=[tickers],
            indicators=None,
            seq_len=max_seq_len
        )

        config = ExperimentConfig(model_config=model_config,
                                  log_dir="hp_search_report",
                                  experiment_name=output_mode,
                                  num_epochs=500,
                                  learning_rate=learning_rate,
                                  lr_mode=lr_mode,
                                  warmup_epochs=warmup_epochs,
                                  batch_size=batch_size,
                                  test_split=0.1,
                                  seed=0,
                                  save_weights=True,
                                  dataset_config=dataset_config,
                                  early_stopper=10000
                                  )

        trainer = Trainer(config=config)
        trainer.train_and_evaluate()
        del trainer
