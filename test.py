import torch
import torch.distributed as dist
import lavin.utils.misc as misc

dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank()

if rank == 0:
    loss = torch.tensor(float('nan'), device='cuda:0')
else:
    loss = torch.tensor(100, device='cuda:1')

print("Rank:", dist.get_rank(), loss)

x = loss.clone()
losses = []

group = dist.group.WORLD
group_size = torch.distributed.get_world_size(group)
gather_t_tensor = [torch.zeros_like(x) for _ in range(group_size)]

dist.all_gather(gather_t_tensor, x)

if rank == 1:
    print("Rank:", rank, "loss:", loss, "Gathered losses:", gather_t_tensor)
    for l in gather_t_tensor:
        if torch.isnan(l):
            print("NaN loss encountered. Skipping this batch.")
