
from torch.utils.data import DataLoader
from torchvision import transforms
from backend.src.models.model_cnn import get_model
from  torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import sys
import torch.nn.functional as F
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from backend.src.config import cnn_path, TRAIN_FILE


#Transformations for minimum overfitting and better generalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),               # 10° tilt
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Zoom in/out
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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





num_epochs=7

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    running_loss=0.0
    correct=0
    total=0

    for images, labels in data_loader:
        images= images.to(device)
        labels= labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted= torch.max(outputs, 1)
        correct+=(predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss=running_loss/ total
    accuracy=correct / total
    return epoch_loss, accuracy

def validate(model, data_loader, device):
    model.eval()
    running_loss=0
    correct=0
    total=0

    with torch.no_grad():
        for images,labels in data_loader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            loss=F.cross_entropy(outputs, labels)

            running_loss+= loss.item() * images.size(0)


            _,predicted=torch.max(outputs,1)
            correct+= (predicted == labels).sum().item()
            total+= labels.size(0)

    epoch_loss= running_loss / total
    accuracy= correct / total
    return epoch_loss, accuracy

#Defining the model and training elements
CNN_model = get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CNN_model=CNN_model.to(device)
optimizer = torch.optim.Adam(CNN_model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=1)
best_val_loss=float('inf')


if __name__ == "__main__":
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(CNN_model, train_loader, optimizer, device)
        val_loss, val_accuracy = validate(CNN_model, val_loader, device)

        #For plotting loss curves
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Logging the results
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save model if validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(CNN_model.state_dict(), cnn_path)
            print("Model saved!")

    # Plotting the loss curves
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
