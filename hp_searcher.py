import numpy as np
from jaxer.utils.config import Config, ModelConfig
from jaxer.utils.trainer import FlaxTrainer as Trainer
from jaxer.utils.logger import Logger


if __name__ == '__main__':
    model_config_ranges = {
        'd_model': [64, 128, 256, 512],
        'num_layers': [1, 2, 3],
        'head_layers': [1, 2, 3],
        'n_heads': [4, 8, 16],
        'dim_feedforward': [4, 8, 16],
        'dropout': [0.0],
        'max_seq_len': [5, 10, 20, 40],
        'input_features': [7],
        'flatten_encoder_output': [False],
        'fe_blocks': [1, 2, 3],
        'use_time2vec': [False, True],
        'output_distribution': [False, True],
        'use_resblocks_in_head': [False, True],
        'use_resblocks_in_fe': [False, True]
    }

    training_ranges = {
        'learning_rate': [1e-4, 1e-5, 1e-6],
        'lr_mode': ['linear', 'cosine'],
        'warmup_epochs': [5, 10, 20],
        'batch_size': [64, 128, 256],
        'normalizer_mode': ['global_minmax']
    }

    n_trainings = 20
    trained_set = set()

    counter_trains = 0
    logger = Logger(name='HP SEARCHER')

    while counter_trains < n_trainings:
        logger.info(f"Training {counter_trains+1}/{n_trainings}")

        d_model = int(np.random.choice(model_config_ranges['d_model']))
        num_layers = int(np.random.choice(model_config_ranges['num_layers']))
        head_layers = int(np.random.choice(model_config_ranges['head_layers']))
        n_heads = int(np.random.choice(model_config_ranges['n_heads']))
        dim_feedforward = 4 * d_model
        dropout = float(np.random.choice(model_config_ranges['dropout']))
        max_seq_len = int(np.random.choice(model_config_ranges['max_seq_len']))
        input_features = int(np.random.choice(model_config_ranges['input_features']))
        flatten_encoder_output = bool(np.random.choice(model_config_ranges['flatten_encoder_output']))
        fe_blocks = int(np.random.choice(model_config_ranges['fe_blocks']))
        use_time2vec = bool(np.random.choice(model_config_ranges['use_time2vec']))
        output_distribution = bool(np.random.choice(model_config_ranges['output_distribution']))
        use_resblocks_in_head = bool(np.random.choice(model_config_ranges['use_resblocks_in_head']))
        use_resblocks_in_fe = bool(np.random.choice(model_config_ranges['use_resblocks_in_fe']))

        learning_rate = float(np.random.choice(training_ranges['learning_rate']))
        lr_mode = str(np.random.choice(training_ranges['lr_mode']))
        warmup_epochs = int(np.random.choice(training_ranges['warmup_epochs']))
        batch_size = int(np.random.choice(training_ranges['batch_size']))
        normalizer_mode = str(np.random.choice(training_ranges['normalizer_mode']))

        params = (d_model, num_layers, head_layers, n_heads, dim_feedforward, dropout, max_seq_len, input_features,
                    flatten_encoder_output, fe_blocks, use_time2vec, output_distribution, use_resblocks_in_head,
                    use_resblocks_in_fe, learning_rate, lr_mode, warmup_epochs, batch_size, normalizer_mode)
        
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
            input_features=input_features,
            flatten_encoder_output=flatten_encoder_output,
            fe_blocks=fe_blocks,
            use_time2vec=use_time2vec,
            output_distribution=output_distribution,
            use_resblocks_in_head=use_resblocks_in_head,
            use_resblocks_in_fe=use_resblocks_in_fe
        )

        config = Config(model_config=model_config,
                        log_dir="hp_search",
                        experiment_name='output_mean' if not output_distribution else 'output_distribution',
                        num_epochs=1500,
                        learning_rate=learning_rate,
                        lr_mode=lr_mode,
                        warmup_epochs=warmup_epochs,
                        dataset_path="./data/BTCUSD.csv",
                        initial_date='2020-01-01',
                        batch_size=batch_size,
                        test_split=0.1,
                        seed=0,
                        normalizer_mode=normalizer_mode,
                        save_weights=False,
                        early_stopper=100)
                        
        trainer = Trainer(config=config)
        trainer.train_and_evaluate()
        del trainer
