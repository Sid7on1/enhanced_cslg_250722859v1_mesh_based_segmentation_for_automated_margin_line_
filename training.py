import os
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Tuple, Dict
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up configuration
class Config:
    def __init__(self):
        self.img_size = IMG_SIZE
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.learning_rate = LEARNING_RATE
        self.weight_decay = WEIGHT_DECAY
        self.device = DEVICE

config = Config()

# Set up data loading
class ImageDataset(Dataset):
    def __init__(self, img_paths: List[str], labels: List[int], transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path)
        img = img.resize((config.img_size, config.img_size))
        img = np.array(img)

        if self.transform:
            img = self.transform(img)

        return img, label

# Set up data augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Set up data loading
train_dataset = ImageDataset(img_paths=['path/to/train/img1.jpg', 'path/to/train/img2.jpg'], labels=[0, 1], transform=transform)
test_dataset = ImageDataset(img_paths=['path/to/test/img1.jpg', 'path/to/test/img2.jpg'], labels=[0, 1], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Set up model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
model.to(config.device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# Set up training loop
def train(model, device, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(loader)}')

# Set up testing loop
def test(model, device, loader, criterion):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == target).sum().item()
    accuracy = total_correct / len(loader.dataset)
    logger.info(f'Test Accuracy: {accuracy:.4f}')

# Train model
for epoch in range(config.epochs):
    train(model, config.device, train_loader, criterion, optimizer, epoch)
    test(model, config.device, test_loader, criterion)

# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model.load_state_dict(torch.load('model.pth'))

# Test model
test(model, config.device, test_loader, criterion)