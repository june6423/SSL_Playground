import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model import ResNet50, SimCLR, load_checkpoint
import gc

import umap

def get_data(simclr_model):
    device = 'cuda'
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    test_dataset = CIFAR10(root='/shared/data/dongjun.nam', train=False, download=True, transform = test_transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    simclr_model.eval()
    simclr_model.to(device)
    data = []
    labels = []
    with torch.no_grad():
        for batch_idx, (images, label) in enumerate(test_loader):
            images = torch.as_tensor(images, device=device)
            label = label.to(device=device)
            h = simclr_model.get_embedding(images)

            data.append(h.cpu())
            labels.append(label.cpu())

            del images, label, h
            torch.cuda.empty_cache()
            gc.collect()

    data = torch.cat(data, dim=0).detach().numpy()
    labels = torch.cat(labels, dim=0).detach().numpy()
    return data, labels

if __name__ == "__main__":
    simclr_model = SimCLR(ResNet50())
    simclr_model = load_checkpoint(simclr_model, "./checkpoint_2000.pth")
    data, labels = get_data(simclr_model)
    n_components = 2

    reducer = umap.UMAP(n_components=n_components, min_dist=0.0125)
    transformed_data = reducer.fit_transform(data)

    # Create a scatter plot
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, alpha=0.5,cmap='viridis', s=10)
    plt.colorbar()

    plt.savefig('umap_0.0125.png')
    plt.show()
