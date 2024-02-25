import jaxer

from torch.utils.data import DataLoader

if __name__ == '__main__':
    """ LOAD THE AGENT """
    experiment = "hp_search_report_all_1h/mean"
    agent = jaxer.run.FlaxAgent(experiment=experiment, model_name=jaxer.utils.get_best_model(experiment))
    # agent = jaxer.run.FlaxAgent(experiment=experiment,
    #                             model_name=('lr_0.0001_cosine_bs_128_ep_400_wmp_5_seed_0_dmodel_256_nlayers_2_ndl_3_nhds_4_dimff_512_drpt_0.1_maxlen_12_flat_False_feblk_2_t2v_False_window_meanstd_ind_False_ds_0.15_2018-01-01_outmd_mean_reshd_True_resfe_True_avout_False_nrmpre_True', '361'))
    #                              #model_name=('lr_0.0001_cosine_bs_64_ep_400_wmp_10_seed_0_dmodel_128_nlayers_3_ndl_2_nhds_2_dimff_512_drpt_0.05_maxlen_12_flat_False_feblk_2_t2v_False_window_minmax_ind_False_ds_0.15_2018-01-01_outmd_mean_reshd_False_resfe_False_avout_False_nrmpre_False',
    #                              #            '376'))
    """ DATALOADERS """
    dataset = jaxer.utils.Dataset(dataset_config=jaxer.config.DatasetConfig.from_dict(agent.config.dataset_config))

    train_ds, test_ds = dataset.get_train_test_split(test_size=agent.config.test_split,
                                                     test_tickers=agent.config.test_tickers)

    """ INFERENCE """

    # infer entire dataset
    plot_entire_dataset = False
    if plot_entire_dataset:
        jaxer.utils.predict_entire_dataset(agent, test_ds, mode='test')
        jaxer.utils.predict_entire_dataset(agent, train_ds, mode='train')

    # infer once
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=jaxer.utils.jax_collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=jaxer.utils.jax_collate_fn)

    infer_test = False

    dataloader = train_dataloader
    if infer_test:
        dataloader = test_dataloader
    for batch in dataloader:
        x, y_true, normalizer, window_info = batch
        y_pred = agent(x)
        jaxer.utils.plot_predictions(x=x, y_true=y_true, y_pred=y_pred, normalizer=normalizer, window_info=window_info)
