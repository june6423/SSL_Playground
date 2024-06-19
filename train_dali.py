import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from model import ResNet50, info_nce_loss, SimCLR, save_checkpoint
from dali import create_dali_pipeline, DALIIterator
import wandb
import time

def setup_ddp(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend="nccl", init_method='tcp://localhost:23458', world_size=world_size, rank=rank)

    # Set the GPU to use
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, simclr_model, criterion, optimizer, epochs=5001, batch_size=256):
#     run = wandb.init(project="SimCLR_ResNet50_with_DALI", config={
#     "epochs": epochs,
#     "batch_size": batch_size,
#     # Add any other hyperparameters here
# })
    # Setup DDP
    st = time.time()
    setup_ddp(rank, world_size)

    eii, pipe = create_dali_pipeline(batch_size=batch_size, num_threads=8, device_id=rank, images_dir="/shared/data/dongjun.nam/CIFAR-10-images")
    dali_iterator = DALIIterator(pipe, size=int(len(eii) / batch_size))


    # Wrap the model with DDP
    simclr_model = simclr_model.cuda(rank)
    simclr_model = DDP(simclr_model, device_ids=[rank])

    # Training loop
    simclr_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_idx, (image1,image2, _) in enumerate(dali_iterator):
            image1 = torch.as_tensor(image1.as_tensor())
            image2 = torch.as_tensor(image2.as_tensor())
            # Forward pass
            h_i, h_j, z_i, z_j = simclr_model(image1, image2)

            # Compute the loss
            loss = criterion(z_i, z_j, batch_size=image1.size(0))
            total_loss += loss.item()

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #wandb.log({"loss": total_loss})

        if rank == 0:
            print(f'Rank {rank}, Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dali_iterator):.4f}, Elapsed time: {time.time()-st:.2f}')
            st = time.time()
        if epoch % 20 == 0 and rank == 0:
            save_checkpoint(epoch, simclr_model.module, optimizer, f'./checkpoint_dali_{epoch}.pth')
    #run.finish()
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Number of GPUs available
    #world_size = 1
    simclr_model = SimCLR(ResNet50())
    criterion = info_nce_loss(temperature=0.5)
    optimizer = optim.Adam(simclr_model.parameters(), lr=0.03)
    mp.spawn(train, args=(world_size, simclr_model, criterion, optimizer), nprocs=world_size, join=True)
