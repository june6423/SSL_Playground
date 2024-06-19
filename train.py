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
import wandb
import time

def setup_ddp(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend="nccl", init_method='tcp://localhost:23453', world_size=world_size, rank=rank)

    # Set the GPU to use
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class contrastive_transforms(object):
    def __init__(self, transform, n_view = 2):
        self.transform = transform
        self.n_view = n_view

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.n_view)]

transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    transforms.ToTensor()])

def train(rank, world_size, simclr_model, criterion, optimizer,scheduler, epochs=2001):
    run = wandb.init(project="SimCLR_ResNet_more_aug_50", config={
    "epochs": epochs,
    "batch_size": 256,
    # Add any other hyperparameters here
})
    st = time.time()
    # Setup DDP
    setup_ddp(rank, world_size)

    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='/shared/data/dongjun.nam', train=True, download=True, transform=contrastive_transforms(transform,n_view=2)) 

    # Wrap the model with DDP
    simclr_model = simclr_model.cuda(rank)
    simclr_model = DDP(simclr_model, device_ids=[rank])

    # Setup data loading
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, sampler=train_sampler, num_workers=4)

    # Training loop
    simclr_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        train_sampler.set_epoch(epoch)
        
        for batch_idx, (image_list, _) in enumerate(train_loader):
            image1, image2 = image_list
            image1 = image1.cuda(rank)
            image2 = image2.cuda(rank)

            # Forward pass
            h_i, h_j, z_i, z_j = simclr_model(image1, image2)

            # Compute the loss
            loss = criterion(z_i, z_j, batch_size=image1.size(0))
            total_loss += loss.item()

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        wandb.log({"loss": total_loss})

        if rank == 0:
            print(f'Rank {rank}, Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Elapsed time: {time.time()-st:.2f}')
            st = time.time()
        if epoch % 200 == 0 and rank == 0:
            save_checkpoint(epoch, simclr_model.module, optimizer, f'./checkpoint_more_aug_{epoch}.pth')
    run.finish()
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Number of GPUs available
    #world_size = 7
    simclr_model = SimCLR(ResNet50())
    criterion = info_nce_loss(temperature=0.5)

    optimizer = optim.Adam(simclr_model.parameters(), lr=0.03)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    mp.spawn(train, args=(world_size, simclr_model, criterion, optimizer, scheduler), nprocs=world_size, join=True)
