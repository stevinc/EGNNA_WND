from __future__ import print_function

import argparse
import copy
import logging
import os
import warnings

import numpy as np
import torch.nn as nn
import torch.utils.data
from torch import optim
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

import utils
from dataset.esa_dataset_graph import EsaDatasetGraph
from job_config import set_params
from metrics.metric import metrics_def
from models.Resnet18_v2 import RESNET18_v2
from test import test
from train import train
from dataset.esa_dataset_graph import id_collate

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='West_Nile_classification')
parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
parser.add_argument('--json_config_file', default='config/configuration.json', help='name of the json config file')
parser.add_argument('--id_optim', default=0, type=int, help='id_optim parameter')

os.environ["OMP_NUM_THREADS"] = "1"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def init_worker(id_worker):
    np.random.seed(7 + id_worker)


def main():
    # read the args
    args = parser.parse_args()

    # enable cuda if available
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # READ JSON CONFIG FILE
    assert os.path.isfile(args.json_config_file), "No json configuration file found at {}".format(args.json_config_file)
    params = utils.Params(args.json_config_file)

    # for change params related to job-id
    params = set_params(params, args.id_optim)

    # set the torch seed
    torch.manual_seed(params.seed)

    # initialize summary writer; every folder is saved inside runs
    writer = SummaryWriter(params.path_nas + params.log_dir + '/runs/' + params.log_dir)

    # create dir for log file
    if not os.path.exists(params.path_nas + params.log_dir):
        os.makedirs(params.path_nas + params.log_dir)

    # save the json config file of the model
    params.save(os.path.join(params.path_nas + params.log_dir, "params.json"))

    # Set the logger
    utils.set_logger(os.path.join(params.path_nas + params.log_dir, "log"))

    # DATASET
    esa_train = EsaDatasetGraph(dataset_file=params.dataset, nas_path=params.nas_path_dataset, bands=params.bands,
                                dsize=params.dsize, RGB=params.RGB, mode='train', val=0, neighbours_labels=params.neighbours_labels,
                                adjacency_type=params.adjacency_type)
    esa_test = EsaDatasetGraph(dataset_file=params.dataset, nas_path=params.nas_path_dataset, bands=params.bands,
                                dsize=params.dsize, RGB=params.RGB, mode='test', val=0, neighbours_labels=params.neighbours_labels,
                               adjacency_type=params.adjacency_type)
    # if params.val:
    #     esa_val = EsaDatasetGraph(dataset_file=params.dataset, nas_path=params.nas_path_dataset, bands=params.bands,
    #                                dsize=params.dsize, RGB=params.RGB, mode='val', val=0)
    #     val_sampler = RandomSampler(esa_val)

    # Define the sampler
    train_sampler = RandomSampler(esa_train)
    test_sampler = RandomSampler(esa_test)
    # define the loader
    train_loader = torch.utils.data.DataLoader(esa_train, batch_size=params.batch_size,
                                               sampler=train_sampler, num_workers=params.num_workers, collate_fn=id_collate)
    test_loader = torch.utils.data.DataLoader(esa_test, batch_size=params.batch_size,
                                              sampler=test_sampler, num_workers=params.num_workers, collate_fn=id_collate)
    # MODEL definition
    model = RESNET18_v2(in_channels_bands=params.in_channels_bands, colorization=params.colorization,
                        in_channels_aux=params.in_channels_aux, out_cls=params.out_cls, pretrained=params.pretrained,
                        device=device, use_dropout=params.use_dropout, drop_rate=params.drop_rate, use_graph=params.use_graph,
                        graph_version=params.graph_version, neighbours_labels=params.neighbours_labels, layers_graph=params.layers_graph,
                        residual=params.residual)
    # If Colorization mode load the checkpoint from nas,
    if params.colorization:
        checkpoint = torch.load(params.path_colorization)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    # Set the correct number of channels in the first convolutional layer
    model.set_weights_conv1()

    # CUDA
    model.to(device)

    # LOSSES
    if params.weighted_loss:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
    else:
        loss_fn = nn.CrossEntropyLoss()

    # METRICS
    metrics = metrics_def

    # OPTIMIZER
    optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum)

    # SCHEDULER
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.sched_step, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 50], gamma=0.1)

    # SAVE THE BEST MODEL STATE DICT
    best_model = copy.deepcopy(model.state_dict())
    # BEST F1
    best_f1_score = 0.
    for epoch in range(params.epochs):
        # Training
        logging.info("Starting training for {} epoch(s)".format(epoch + 1))
        logging.info("Epoch {}/{}".format(epoch + 1, params.epochs))
        train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer,
              device=device, params=params, metrics=metrics)
        # test (for the moment I don't use a validation set...)
        if epoch % params.test_step == 0:
            logging.info("Starting test for {} epoch(s)".format(epoch + 1))
            f1_score = test(model=model, test_loader=test_loader, loss_fn=loss_fn,
                            device=device, params=params, metrics=metrics)
            # save best model params based on avg_pr_micro score on validation set
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model = copy.deepcopy(model.state_dict())
                if params.scheduler:
                    state = {'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optim_dict': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict()}
                else:
                    state = {'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optim_dict': optimizer.state_dict()}
                path_to_save_chk = params.path_nas + params.log_dir
                utils.save_checkpoint(state,
                                      is_best=True,  # True if this is the model with best metrics
                                      checkpoint=path_to_save_chk)  # path to folder
        # scheduler step
        if params.scheduler:
            scheduler.step()
        # Save checkpoint, maybe I can avoid this...
        if epoch % params.save_checkpoint == 0:
            if params.scheduler:
                state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict()}
            else:
                state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict()}
            path_to_save_chk = params.path_nas + params.log_dir
            utils.save_checkpoint(state,
                                  is_best=False,  # True if this is the model with best metrics
                                  checkpoint=path_to_save_chk)  # path to folder
    # FINAL TEST
    logging.info("Starting final test...")
    test(model=model, test_loader=test_loader, loss_fn=loss_fn,
         device=device, params=params, metrics=metrics, excel_version="last")
    # FINAL TEST WITH THE BEST MODEL
    logging.info("Starting final test with best model...")
    model.load_state_dict(best_model)
    test(model=model, test_loader=test_loader, loss_fn=loss_fn,
         device=device, params=params, metrics=metrics, excel_version="best")

    # CLOSE THE WRITER
    writer.close()


if __name__ == '__main__':
    main()





