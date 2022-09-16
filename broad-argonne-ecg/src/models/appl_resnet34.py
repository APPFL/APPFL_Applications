import os
import time

import numpy as np
import torch

from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
# from mnist_cnn_model import *
from resnet34 import *

import appfl.run_serial as rs
import appfl.run_mpi as rm
from mpi4py import MPI

import argparse

import pandas as pd
import ecg_dataset

""" read arguments """

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="physionet")

## clients
parser.add_argument("--num_clients", type=int, default=1)
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=1e-3)
parser.add_argument("--num_local_epochs", type=int, default=1)

## server
parser.add_argument("--server", type=str, default="ServerFedAvg")
parser.add_argument("--num_epochs", type=int, default=2)

parser.add_argument("--server_lr", type=float, required=False)
parser.add_argument("--mparam_1", type=float, required=False)
parser.add_argument("--mparam_2", type=float, required=False)
parser.add_argument("--adapt_param", type=float, required=False)


args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"


def appl_fix_loss(pred, y):
    return torch.nn.MSELoss()(pred, y.unsqueeze(-1))
 

def get_data(comm: MPI.Comm):
    meta_data = pd.read_parquet(os.getcwd() + '/../../data/freeze/BROAD_ml4h_klarqvist___physionet__meta_data__graded_splits__84fe7e5e413d4dc8b6de645ed5f06c5d.pq')
    meta_data['Age'] = meta_data['Age'].astype(np.float32)
    meta_data = meta_data[meta_data['n_observations'] >= 5000]
    h5py_path = os.getcwd() + '/../../data/freeze/BROAD_ml4h_klarqvist___physionet__waveforms__409faaa082084ae5aef22838e35dae06__combined.h5'
    meta_data = meta_data[~meta_data.Age.isna()]

    train_data = meta_data[meta_data['is_graded_train'] == True]
    test_data = meta_data[meta_data['is_graded_test'] == True]

    test_ds = ecg_dataset.EcgDataset(h5py_path, test_data.index.values, test_data.Age)
    # validation_ds = ecg_dataset.EcgDataset(h5py_path, validation_data.index.values, validation_data.Age)

    split_train_data_raw = np.array_split(train_data.index.values, args.num_clients)
    train_datasets = []
    for i in range(args.num_clients):
        train_datasets.append(
            ecg_dataset.EcgDataset(h5py_path, split_train_data_raw[i], train_data.Age)
        )
    return train_datasets, test_ds


def get_model(comm: MPI.Comm):
    model = EcgResNet34(num_classes=1)
    return model


## Run
def main():

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    """ Configuration """
    cfg = OmegaConf.structured(Config)

    cfg.device = args.device
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## outputs

    cfg.use_tensorboard = False

    cfg.save_model_state_dict = False

    cfg.output_dirname = "./outputs_%s_%s_%s" % (
        args.dataset,
        args.server,
        args.client_optimizer,
    )
    if args.server_lr != None:
        cfg.fed.args.server_learning_rate = args.server_lr
        cfg.output_dirname += "_ServerLR_%s" % (args.server_lr)

    if args.adapt_param != None:
        cfg.fed.args.server_adapt_param = args.adapt_param
        cfg.output_dirname += "_AdaptParam_%s" % (args.adapt_param)

    if args.mparam_1 != None:
        cfg.fed.args.server_momentum_param_1 = args.mparam_1
        cfg.output_dirname += "_MParam1_%s" % (args.mparam_1)

    if args.mparam_2 != None:
        cfg.fed.args.server_momentum_param_2 = args.mparam_2
        cfg.output_dirname += "_MParam2_%s" % (args.mparam_2)

    cfg.output_filename = "result"

    start_time = time.time()

    """ User-defined model """
    model = get_model(comm)
    loss_fn = appl_fix_loss

    ## loading models
    cfg.load_model = False
    if cfg.load_model == True:
        cfg.load_model_dirname = "./save_models"
        cfg.load_model_filename = "Model"
        model = load_model(cfg)

    """ User-defined data """
    train_datasets, test_dataset = get_data(comm)

    ## Sanity check for the user-defined data
    # if cfg.data_sanity == True:
    #     data_sanity_check(
    #         train_datasets, test_dataset, args.num_channel, args.num_pixel
    #     )

    print(
        "-------Loading_Time=",
        time.time() - start_time,
    )

    """ saving models """
    cfg.save_model = False
    if cfg.save_model == True:
        cfg.save_model_dirname = "./save_models"
        cfg.save_model_filename = "Model"

    """ Running """
    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(
                cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset
            )
        else:
            rm.run_client(
                cfg, comm, model, loss_fn, args.num_clients, train_datasets, test_dataset
            )
        print("------DONE------", comm_rank)
    else:
        rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset, args.dataset)
        


if __name__ == "__main__":
    main()

# To run CUDA-aware MPI:
# mpiexec -np 6 --mca opal_cuda_support 1 python ./appl_resnet34.py --num_epochs 25 --client_lr 1e-4
