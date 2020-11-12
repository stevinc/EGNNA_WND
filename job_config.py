
def set_params(params, id_optim):
    if id_optim is None:
        pass
    else:
        if id_optim == 0:
            params.batch_size = 16
            params.seed = 19
            params.epochs = 20
            params.log_dir = "Journal_Graph/EGNNA/top10spatial/LST-SSM-SRTM/3b_imagenet_1layers_residual"
            params.model = "Resnet"
            params.pretrained = 1
            params.dsize = 224
            params.colorization = 0
            params.path_colorization = "/nas/softechict-nas-2/svincenzi/colorization_resnet/experiments_resnet18_esa/500k_augmentation_dropout05/_batch_16/last.pth.tar"
            params.lr = 0.001
            params.bands = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
            params.RGB = 1
            params.in_channels_bands = 3
            params.in_channels_aux = 3
            params.out_cls = 2
            params.optim = "SGD"
            params.scheduler = 1
            params.sched_step = 25
            params.weighted_loss = 0
            params.use_dropout = 1
            params.drop_rate = 0.2
            params.use_graph = 1
            params.graph_version = 'egnna'
            params.layers_graph = 1
            params.adjacency_type = 'lst-srtm-ssm-separate'
            params.neighbours_labels = 0
            params.num_workers = 4
            params.residual = 1
            params.name_excel = "excel_results/RGB_imagenet_egnna_1L_res_lst_srtm_ssm"
        elif id_optim == 1:
            params.batch_size = 16
            params.seed = 19
            params.epochs = 20
            params.log_dir = "Journal_Graph/EGNNA/top10spatial/LST-SSM-SRTM/4b_color_1layers_residual"
            params.model = "Resnet"
            params.pretrained = 1
            params.dsize = 224
            params.colorization = 1
            params.path_colorization = "/nas/softechict-nas-2/svincenzi/colorization_resnet/experiments_resnet18_esa/500k_augmentation_dropout05/_batch_16/last.pth.tar"
            params.lr = 0.001
            params.bands = [1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            params.RGB = 0
            params.in_channels_bands = 4
            params.in_channels_aux = 3
            params.out_cls = 2
            params.optim = "SGD"
            params.scheduler = 1
            params.sched_step = 25
            params.weighted_loss = 0
            params.use_dropout = 1
            params.drop_rate = 0.2
            params.use_graph = 1
            params.graph_version = 'egnna'
            params.layers_graph = 1
            params.adjacency_type = 'lst-srtm-ssm-separate'
            params.neighbours_labels = 0
            params.num_workers = 4
            params.residual = 1
            params.name_excel = "excel_results/4b_color_egnna_1L_res_lst_srtm_ssm"
        elif id_optim == 2:
            params.batch_size = 16
            params.seed = 19
            params.epochs = 20
            params.log_dir = "Journal_Graph/EGNNA/top10spatial/LST-SSM/4b_color_1layers_residual"
            params.model = "Resnet"
            params.pretrained = 1
            params.dsize = 224
            params.colorization = 1
            params.path_colorization = "/nas/softechict-nas-2/svincenzi/colorization_resnet/experiments_resnet18_esa/500k_augmentation_dropout05/_batch_16/last.pth.tar"
            params.lr = 0.001
            params.bands = [1, 0, 0, 0, 1, 1, 1, 1, 0, 1]
            params.RGB = 0
            params.in_channels_bands = 4
            params.in_channels_aux = 2
            params.out_cls = 2
            params.optim = "SGD"
            params.scheduler = 1
            params.sched_step = 25
            params.weighted_loss = 0
            params.use_dropout = 1
            params.drop_rate = 0.2
            params.use_graph = 1
            params.graph_version = 'egnna'
            params.layers_graph = 1
            params.adjacency_type = 'lst-ssm-separate'
            params.neighbours_labels = 0
            params.num_workers = 4
            params.residual = 1
            params.name_excel = "excel_results/4b_color_egnna_1L_res"
        elif id_optim == 3:
            params.batch_size = 16
            params.seed = 19
            params.epochs = 30
            params.log_dir = "Report_ESA/Graph_neighbours/EGNNA/top5spatial/no_neigh_cls/no_concat/4b_col_adj_lst_srtm_ssm_1layers_res_hav_correct"  # new_norm_base" # top5_spatial/no_neigh_cls/no_concat/4B_col_ds_adj_lst_ssm_pointwise_1024" # no_graph_baseline_correct" # Change the name
            params.model = "Resnet"
            params.pretrained = 0
            params.dsize = 224
            params.colorization = 1
            params.path_colorization = "/nas/softechict-nas-2/svincenzi/colorization_resnet/experiments_resnet18_esa/500k_augmentation_dropout05/_batch_16/last.pth.tar"
            params.lr = 0.001
            params.bands = [1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            params.RGB = 0
            params.in_channels_bands = 4
            params.in_channels_aux = 0
            params.out_cls = 2
            params.optim = "SGD"
            params.scheduler = 1
            params.sched_step = 25
            params.weighted_loss = 0
            params.use_dropout = 1
            params.drop_rate = 0.2
            params.use_graph = 2  # Graph only mode after features extraction
            params.graph_version = 'gat'
            params.layers_graph = 1
            params.adjacency_type = 'lst-srtm-ssm-separate'
            params.neighbours_labels = 0
            params.num_workers = 4
            params.residual = 1

        params.log_dir = params.log_dir + "_batch_" + str(params.batch_size)

    return params
