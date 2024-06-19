import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model import ResNet50, SimCLR, regression, load_checkpoint
import wandb
import time

def eval(simclr_model, fc_model):
    device = 'cuda:7'
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_dataset = CIFAR10(root='/shared/data/dongjun.nam', train=True, download=True, transform = test_transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)

    test_dataset = CIFAR10(root='/shared/data/dongjun.nam', train=False, download=True, transform = test_transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    simclr_model = simclr_model.to(device)
    fc_model = fc_model.to(device)
    simclr_model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fc_model.parameters(), lr=1e-3)

    epochs = 100

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, label) in enumerate(train_loader):
            images = torch.as_tensor(images, device=device)
            label = label.to(device=device)
            h = simclr_model.get_embedding(images)
            y = fc_model(h)

            loss = criterion(y, label)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = 0
        with torch.no_grad():
            for batch_idx, (images, label) in enumerate(test_loader):
                images = torch.as_tensor(images, device=device)
                label = label.to(device=device)
                h = simclr_model.get_embedding(images)
                y = fc_model(h)

                _, predicted = torch.max(y, 1)
                correct += (predicted == label).sum().item()

        print(f"Epoch {epoch} Loss: {total_loss} Accuracy: {correct / len(test_dataset)}")


if __name__ == "__main__":
    simclr_model = SimCLR(ResNet50())
    simclr_model = load_checkpoint(simclr_model, "./checkpoint_2000.pth")
    fc_model = regression(2048, 10)
    eval(simclr_model, fc_model)