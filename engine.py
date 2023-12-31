import time
import datetime
import math
import sys
from typing import Iterable

import torch

import lavin.utils.misc as misc
import lavin.utils.lr_sched as lr_sched
import wandb


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    args=None):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    prefix_img = torch.tensor(data_loader.dataset.tokenizer.encode("Image: ", bos=False, eos=False), dtype=torch.int64)
    # prefix_box = torch.tensor(data_loader.dataset.tokenizer.encode("Entity Boxes: ", bos=False, eos=False), dtype=torch.int64)

    start_time = time.time()
    total_iters = len(data_loader)
    epoch_loss = torch.tensor(0.0).cuda()
    actual_iters = 0
    for data_iter_step, (examples, labels, example_mask, images, indicators, boxes) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        prefix_img = prefix_img.to(examples.device)
        # prefix_box = prefix_box.to(examples.device)
        c_loss = model(examples, labels, images=images, prefix_img=prefix_img, img_indicators=indicators, batch_boxes=boxes)
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()

        losses_from_all_ranks = [torch.zeros_like(loss) for _ in range(misc.get_world_size())]
        torch.distributed.all_gather(losses_from_all_ranks, loss)

        has_nan_loss = False
        for gather_loss in losses_from_all_ranks:
            if torch.isnan(gather_loss) or gather_loss.item() > 100:
                print(f"NaN loss encountered. Skipping this batch. Value: {gather_loss}")
                has_nan_loss = True
        if has_nan_loss:
            continue

        loss = loss / accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0, clip_grad=args.clip_grad)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        actual_iters += 1
        epoch_loss += c_loss_value_reduce

        if misc.is_main_process() and wandb.run is not None:
            wandb.log({"c_loss_iter": c_loss_value_reduce, "lr": lr})

        iter_time = time.time() - start_time
        eta_time = iter_time * (total_iters - data_iter_step - 1)
        eta_time_str = str(datetime.timedelta(seconds=int(eta_time)))

        MB = 1024.0 * 1024.0
        memory = torch.cuda.max_memory_allocated() / MB
        print(f"Epoch: [{epoch:2d}], Iter: [{data_iter_step:4d}/{total_iters:<4d}], "
              f"Eta: {eta_time_str}, Loss: {c_loss_value_reduce:.4f}, lr: {lr:.6f} Mem: {memory:.0f}MB")

        start_time = time.time()

    epoch_loss = epoch_loss / actual_iters
    if misc.is_main_process() and wandb.run is not None:
        wandb.log({"c_loss_epoch": epoch_loss, "epoch": epoch, "memory": memory})
    return None


def val_one_epoch(model: torch.nn.Module,
                  data_loader: Iterable,
                  optimizer: torch.optim.Optimizer,
                  device: torch.device,
                  epoch: int,
                  loss_scaler,
                  log_writer=None,
                  args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        with torch.no_grad():
            c_loss = model(examples, labels)
        loss = c_loss
        loss_value = loss.item()

        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
