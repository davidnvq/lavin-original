import os
import json
import time
import datetime
from dataclasses import dataclass, asdict, field

import fire
import wandb
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import timm.optim.optim_factory as optim_factory
from torch.nn.parallel import DistributedDataParallel as DDP

import lavin.utils.misc as misc
from lavin.utils.misc import NativeScalerWithGradNormCount
from lavin.utils.datasets import DHPRDataset, dhpr_collate
from engine import train_one_epoch
from lavin.mm_adaptation import LaVIN


@dataclass
class TrainArgs:
    batch_size: int = 2
    accum_iter: int = 2
    epochs: int = 20
    bits: str = '16bit'

    # Model parameters
    llama_model_path: str = './data/weights/'
    llm_model: str = '100M'
    use_vicuna: bool = False
    cpu_load: bool = False
    visual_adapter_type: str = 'router'  # normal, router_block,
    has_indicator: bool = True
    adapter_type: str = 'attn'  # normal, attn, adapter_box
    num_routers: int = 3
    weight_kind: str = 'box_modality'  # if adapter_type = 'adapter_box', then weight_kind = 'learn' or 'fixed', 'indicator_modality', 'box_modality'
    adapter_dim: int = 8
    hidden_proj: int = 128
    temperature: float = 10.
    n_prompt: int = 6
    adapter_scale: float = 1.
    drop_path: float = 0.
    max_seq_len: int = 512

    # Optimizer parameters
    weight_decay: float = 0.02
    lr: float = None
    clip_grad: float = None
    blr: float = 9e-3
    min_lr: float = 0.

    gradient_checkpointing: bool = False
    warmup_epochs: float = 2

    # Dataset parameters
    dataset: str = 'dhpr'
    debug: bool = False
    has_speed: bool = False
    has_boxes: bool = False
    prompt_id: str = 'A'
    max_words: int = 128
    output_dir: str = './outputs/debug_dhpr'
    device: str = 'cuda'
    seed: int = 0
    resume: str = ''

    start_epoch: int = 0
    num_workers: int = 2
    pin_mem: bool = True

    # Distributed training parameters
    world_size: int = 1
    local_rank: int = 0
    rank: int = 0

    wandb_enable: bool = False


def init_args(**kwargs):
    args = TrainArgs(**kwargs)
    if args.debug:
        args.batch_size = 2
        args.epochs = 2
        args.num_workers = 0
    if kwargs.get('no_indicator', 'False'):
        args.has_indicator = False

    for k, v in asdict(args).items():
        print(f"{k:<20}: {v}")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), mode="w", encoding="utf-8") as f:
        f.write(json.dumps(asdict(args), indent=4, sort_keys=True))
    return args


def main(**kwargs):
    args = init_args(**kwargs)
    # torch.autograd.set_detect_anomaly(True)

    misc.init_distributed_mode(args)
    if misc.is_main_process() and args.wandb_enable:
        args.output_dir = args.output_dir[:-1] if args.output_dir.endswith('/') else args.output_dir
        wandb.init(
            project="lavin-original",
            name=args.output_dir.split("/")[-1],
            dir=args.output_dir,
            config=vars(args),
        )
        print('Experiment name: {}'.format(wandb.run.name))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = DHPRDataset(
        'train',
        debug=args.debug,
        has_speed=args.has_speed,
        prompt_id=args.prompt_id,
        max_words=args.max_words,
        has_boxes=args.has_boxes,
    )

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )

    print("Sampler_train = %s" % str(sampler_train))

    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=dhpr_collate,
    )

    # define the model
    model = LaVIN(args)
    model.to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    model = DDP(model, device_ids=[misc.get_rank()], find_unused_parameters=True)
    _model = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(_model, args.weight_decay)

    #following qlora: apply paged optimizer
    # optimizer = bnb.optim.AdamW32bit(param_groups, lr=args.lr, betas=(0.9, 0.95), is_paged=True)  #
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    #mixed precision scaler
    loss_scaler = NativeScalerWithGradNormCount()

    misc.load_model(args=args, model_without_ddp=_model, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        epoch_time = time.time()
        train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args)
        epoch_time = time.time() - epoch_time
        print("Epoch time: {}".format(str(datetime.timedelta(seconds=int(epoch_time)))))

        if args.output_dir:
            misc.save_model(args=args, model=model, model_without_ddp=_model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
            print("Saved model and optimizer to {}".format(args.output_dir))
            torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    fire.Fire(main)
