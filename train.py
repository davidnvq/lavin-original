from math import log
import os
import argparse
import datetime
import json
import time
import wandb
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm.optim.optim_factory as optim_factory

import lavin.utils.misc as misc
from lavin.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from engine import train_one_epoch

from lavin.utils.datasets import ScienceQADataSet, InstrcutDataSet
from lavin.mm_adaptation import LaVIN

# import bitsandbytes as bnb # don't need this if you don't use paged optimizer


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=2, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--bits', default='16bit')

    # Model parameters
    parser.add_argument('--llama_model_path', default='./data/weights/', type=str)
    parser.add_argument('--llm_model', default='100M', type=str)
    parser.add_argument('--use_vicuna', action='store_true')
    parser.add_argument('--cpu_load', action='store_true')

    parser.add_argument('--adapter_type', type=str, default='attn', choices=['block', 'attn'], help='the insert position  of adapter layer')
    # choices=['normal', 'router', 'router_block']
    parser.add_argument('--visual_adapter_type', type=str, default='router', help='the type of adapter layer')
    parser.add_argument('--adapter_dim', type=int, default=8, metavar='LENGTH', help='the dims of adapter layer')
    parser.add_argument('--hidden_proj', type=int, default=128, metavar='LENGTH', help='the visual adapter dim')
    parser.add_argument('--temperature', type=float, default=10., metavar='LENGTH', help='the temperature of router')
    parser.add_argument('--n_prompt', type=int, default=6, metavar='LENGTH', help='the length of visual features')
    parser.add_argument('--adapter_scale', type=float, default=1., metavar='LENGTH', help='the scales of adapter layer')
    parser.add_argument('--drop_path', type=float, default=0., metavar='LENGTH', help='drop path')
    parser.add_argument('--max_seq_len', type=int, default=512, metavar='LENGTH', help='the maximum sequence length')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='clip gradient')
    parser.add_argument('--blr', type=float, default=9e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--gradient_checkpointing', action='store_true', help='saving memory costs via gradient_checkpointing')
    parser.add_argument('--warmup_epochs', type=float, default=2, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/captions.json', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./outputs/debug', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./outputs/debug', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--rank', default=0, type=int)

    #datasets
    parser.add_argument('--prompt_format', type=str, default='QCM-ALE', help='prompt format template')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--caption_file', type=str, default='./data/captions.json')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--wandb_enable', action='store_true', help='to use wandb')
    args = parser.parse_args()
    args.clip_grad = True
    return args


def main(args):

    misc.init_distributed_mode(args)
    if misc.is_main_process() and args.wandb_enable:
        args.output_dir = args.output_dir[:-1] if args.output_dir.endswith('/') else args.output_dir
        wandb.init(project="lavin-original", name=args.output_dir.split("/")[-1], dir=args.output_dir, config=vars(args))
        print('Experiment name: {}'.format(wandb.run.name))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = ScienceQADataSet(args, 'train', args.llama_model_path, args.max_seq_len)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    print("Sampler_train = %s" % str(sampler_train))

    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = LaVIN(args)

    model.to(device)

    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[misc.get_rank()], find_unused_parameters=True)
    model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)

    #following qlora: apply paged optimizer
    # optimizer = bnb.optim.AdamW32bit(param_groups, lr=args.lr, betas=(0.9, 0.95), is_paged=True)  #
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    #mixed precision scaler
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        epoch_time = time.time()
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args)
        epoch_time = time.time() - epoch_time
        print("Epoch time: {}".format(str(datetime.timedelta(seconds=int(epoch_time)))))

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
            print("Saved model and optimizer to {}".format(args.output_dir))
            torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = get_args()
    main(args)
