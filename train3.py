from share import *
import numpy as np
import argparse, os, sys
from functools import partial
from omegaconf import OmegaConf

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import ConcatDataset
import pytorch_lightning as pl

from cldm.model import create_model, load_state_dict

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

from edit_dataset import EditDataset

from train_util.multi_task_scheduler import BatchSchedulerSampler
import train_util.dataset_collate as dataset_collate
import random

# from cldm.logger import ImageLogger, CheckpointEveryNSteps



def get_parser(**parser_kwargs):

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default='./models/cldm_v15.yaml',
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs_moe",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--only_mid_control",
        action="store_true",
        default=False,
        help="only_mid_control for control net",
    )
    parser.add_argument(
        "--sd_locked",
        action="store_false",
        default=True,
        help="sd_locked for control net",
    )
    # Training
    parser.add_argument(
        
        "--gpus",
        type=int,
        default=1,
        help="number of gpus for training",
    )
    parser.add_argument(
        "--nnode",
        type=int,
        default=1,
        help="number of nodes for training",
    )

    # Prompt Engineering
    parser.add_argument(
        "--data_config",
        type=str,
        help="path to data config files",
        default='./models/dataset_moe.yaml',
    )
    parser.add_argument(
        "--sd_v2",
        action="store_true",
        default=False,
        help="if use stable diffusion 2.0",
    )

    return parser
def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, tasks, train=True, validation=True,
                 num_workers=None, use_worker_init_fn=False,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.tasks=tasks
        # print(self.tasks)
        if train:
            self.train_dataloader = self._train_dataloader
        if validation:
            self.val_dataloader = self._val_dataloader

    def _train_dataloader(self):
        datasets_list1 = []
        for _task in self.tasks:
            datasets_list1.append(EditDataset(path='/home/data_2t/cwl/Prompt-Diffusion2/data/clip-filtered-dataset',
                                              task=_task,
                                              split='train', 
                                              prompt_option='output',
                                              min_resize_res=256,
                                              max_resize_res=256,
                                              crop_res=256,
                                              flip_prob=0.5))
        multi_dataset = ConcatDataset(datasets_list1)
        is_iterable_dataset = isinstance(datasets_list1[0], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None# Construct Training Datasets
        return DataLoader(multi_dataset, 
                          num_workers=self.num_workers,  
                          sampler=BatchSchedulerSampler(dataset=multi_dataset, batch_size=self.batch_size),
                          batch_size=self.batch_size, 
                          persistent_workers=True, 
                          worker_init_fn=init_fn, 
                          collate_fn=dataset_collate.collate_fn,
                          pin_memory=True)

    def _val_dataloader(self, shuffle=False):
        datasets_list2 = []
        for _task in self.tasks:
            datasets_list2.append(EditDataset(path='/home/data_2t/cwl/Prompt-Diffusion2/data/clip-filtered-dataset',
                                              task=_task,
                                              split='val', 
                                              prompt_option='output',
                                              min_resize_res=256,
                                              max_resize_res=256,
                                              crop_res=256))
        multi_dataset = ConcatDataset(datasets_list2)
        if isinstance(datasets_list2[0], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(multi_dataset, 
                          num_workers=self.num_workers,  
                          sampler=BatchSchedulerSampler(dataset=multi_dataset, batch_size=self.batch_size),
                          batch_size=self.batch_size, 
                          persistent_workers=True, 
                          worker_init_fn=init_fn, 
                          collate_fn=dataset_collate.collate_fn,
                          pin_memory=True)

if __name__ == "__main__":
    sys.path.append(os.getcwd())

    opt, _ = get_parser().parse_known_args()

    nowname = f"{opt.name}"
    logdir = os.path.join(opt.logdir, nowname)
    ckptdir = os.path.join(logdir, "checkpoints")

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)

    # Configs
    resume_path = './models/control_sd15_ini_moe_restraint.ckpt' if not opt.sd_v2 else './models/control_sd21_ini.ckpt'
    batch_size = 8
    learning_rate = 1e-4

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(opt.base).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    model.sd_locked = opt.sd_locked
    model.only_mid_control = opt.only_mid_control

    # Data
    data_config = OmegaConf.load(opt.data_config)
    # print(data_config)
    dataloader = instantiate_from_config(data_config.data)
    # print("#### Data #####")
    # for k in dataloader.datasets:
    #     print(f"{k}, {dataloader.datasets[k].__class__.__name__}, {len(dataloader.datasets[k])}")

    # Callbacks
    callbacks_cfg = {
        "checkpoint_callback": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}-{step:09}",
                "verbose": True,
                'save_top_k': -1,
                'every_n_train_steps': 1000,
                'save_weights_only': True,
                "save_last": True,
            }
        },
        "image_logger": {
            "target": "cldm.logger.ImageLogger",
            "params": {
                "batch_frequency": 500,
                "max_images": 16,
                "clamp": True,
                "log_images_kwargs": {'N': 64,
                                      'unconditional_guidance_scale': 9.0}
            }
        },
    }

    callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    # Trainer
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=logdir)
    trainer = pl.Trainer(gpus=opt.gpus, accelerator='ddp', num_nodes=opt.nnode,
                         replace_sampler_ddp=False,
                         max_steps=10000, val_check_interval=0.2, accumulate_grad_batches=4,
                         precision=32, callbacks=callbacks, logger=tb_logger)

    # Train!
    trainer.fit(model, dataloader)

