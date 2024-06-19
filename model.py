import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

class SimCLR(nn.Module):
    def __init__(self, base_encoder, feature_dim=2048, projection_dim=128):
        super(SimCLR, self).__init__()
        self.base_encoder = base_encoder
        self.projection_dim = projection_dim

        # Define the projection head
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )

    def forward(self, x_i, x_j):
        h_i = self.base_encoder(x_i)
        h_j = self.base_encoder(x_j)

        h_i = h_i.view(h_i.size(0), -1)  # Flatten if necessary
        h_j = h_j.view(h_j.size(0), -1)  # Flatten if necessary

        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)

        return h_i, h_j, z_i, z_j

    def get_embedding(self, x):
        h = self.base_encoder(x)
        h = h.view(h.size(0), -1)
        return h
    
class regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(regression, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        y = self.fc(x)
        return self.softmax(y)


class NT_XentLoss(nn.Module):
    def __init__(self, temperature):
        super(NT_XentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j, batch_size):
        # Normalize z_i and z_j
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)

        # Calculate cosine similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature

        # Create a mask to prevent the overlap of samples with themselves
        mask = torch.eye(2 * batch_size, device=sim.device).bool()
        
        # Set diagonal elements to a very large negative value
        sim[mask] = -1e9
        
        # Compute logits
        logits_i_j = torch.cat([sim, sim], dim=1)  # Concatenate sim with sim

        # Target labels
        labels = torch.arange(2 * batch_size, device=logits_i_j.device)
        
        # Compute loss
        loss = self.criterion(logits_i_j, labels)

        return loss / (2 * batch_size)


class info_nce_loss(nn.Module):
    def __init__(self, temperature):
        super(info_nce_loss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, A, B, batch_size):
        assert A.size() == B.size()
        feature = torch.cat([A, B], dim=0)
        feature = F.normalize(feature, dim=1)
        sim = torch.matmul(feature, feature.T)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(feature.device)

        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(feature.device)
        labels = labels[~mask].view(2 * batch_size, -1)
        sim = sim[~mask].view(2 * batch_size, -1)

        positives = sim[labels.bool()].view(2 * batch_size, -1)
        negatives = sim[~labels.bool()].view(2 * batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size).to(feature.device).long()

        logits = logits / self.temperature
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)


# Save checkpoint
def save_checkpoint(epoch, model, optimizer, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']

    return model