from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import sys
from backend.src.models.model_resnet import get_resnet18
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import numpy as np
from backend.src.config import mobile_net_path, TRAIN_FILE

# Transformations for minimum overfitting and better generalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),               # 10° tilt
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Zoom in/out
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# MIXUP and CUTMIX to avoid overfitting

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    y_a, y_b = y, y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


# Data loading

full_dataset = ImageFolder(
    root=TRAIN_FILE,
    transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

num_workers = 1 if sys.platform=='darwin' else 6
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)


# MODEL SETUP
num_classes = len(train_dataset.dataset.classes)
model = get_resnet18(num_classes=num_classes, pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4) #For Resnet
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) #For MobileNet
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) #Patience Resnet:2 MobileNet:1
loss_function = nn.CrossEntropyLoss()
num_epochs = 7
bes_val_loss = float('inf')


# TRAIN and VALIDATION
def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        # Randomly choose MixUp or CutMix
        rand = np.random.rand()
        if rand < 0.3:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            outputs = model(images)
            loss = mixup_criterion(loss_function, outputs, labels_a, labels_b, lam)
        elif rand < 0.6:
            images, labels_a, labels_b, lam = cutmix_data(images, labels)
            outputs = model(images)
            loss = mixup_criterion(loss_function, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        if rand>=0.6:
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def validation(model, data_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


if __name__ == "__main__":
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy = validation(model, val_loader, device)

        #For plotting loss curves
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save model if validation improves
        if val_loss < bes_val_loss:
            bes_val_loss = val_loss
            torch.save(model.state_dict(),mobile_net_path )
            print("Model saved!")

    # Plotting loss curves
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
