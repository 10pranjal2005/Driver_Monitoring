import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dir = "Data/train"
val_dir = "Data/test"


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


train_data = datasets.ImageFolder(train_dir, transform)
val_data = datasets.ImageFolder(val_dir, transform)


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)


model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)


for param in model.parameters():
    param.requires_grad = False


model.classifier[1] = nn.Linear(model.last_channel, 2)


model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


epochs = 1


for epoch in range(epochs):

    model.train()
    running_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx}")


    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")


torch.save(model.state_dict(), "eye_model.pt")

print("Training complete. Model saved as eye_model.pt")