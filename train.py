import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import AttackedDataset, TrafficSignDataset
from transformers import AutoFeatureExtractor
from tqdm import tqdm
from adv_detector import AdvDetectorModel
from torch.utils.data import random_split
import numpy as np
import logging

logging.basicConfig(filename='model.log', # Log file path
                    level=logging.INFO,    # Minimum level of messages to log
                    filemode='w',           # Mode to open the file, 'w' for overwrite, 'a' for append
                    format='%(asctime)s - %(levelname)s - %(message)s') 

feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Define the transformations
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

train_dataset = TrafficSignDataset("./DATA", transform=transform)

train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

batch_size = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# # Setup the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdvDetectorModel().to(device)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(tqdm(train_loader)):
        inputs, labels = batch[0].to(device), torch.tensor(batch[1]).to(device)

        optimizer.zero_grad()

        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)


        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    model.eval() 
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in tqdm(valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            _, detector_outputs = model(inputs)
            output = torch.tensor([0.0 for _ in range(detector_outputs[0].shape[0])]).to(device)
            for detector_output in detector_outputs:
                avg_confidences = detector_output.mean(dim=[2, 3])

                weighted_scores = avg_confidences[:, 1] - avg_confidences[:, 0]

                final_scores = (weighted_scores > 0).int()

                output += final_scores
            
            predicted = (output > 2).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {100 * correct / total}%')

logging.info('Finished Training the model.')


for param in model.parameters():
    param.requires_grad = False

for detector in model.adv_detectors:
    for param in detector.parameters():
        param.requires_grad = True

optimizer = optim.Adam(
    [param for detector in model.adv_detectors for param in detector.parameters() if param.requires_grad], 
    lr=0.0001
)

adv_dataset = AttackedDataset("./TEST", "./attacked_image_labels.csv", transform=transform)

train_size = int(0.8 * len(adv_dataset))
valid_size = len(adv_dataset) - train_size
train_dataset, valid_dataset = random_split(adv_dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# Define your loss function
criterion = nn.CrossEntropyLoss()
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.float().to(device)
        optimizer.zero_grad()

        _, detector_outputs = model(inputs)

        total = torch.tensor([0.0 for _ in range(batch_size)], requires_grad=True).to(device)
        for detector_output in detector_outputs:
            avg_confidences = detector_output.mean(dim=[2, 3])

            weighted_scores = avg_confidences[:, 1] - avg_confidences[:, 0]

            final_scores = (weighted_scores > 0).int()

            total += final_scores
        
        total = (total > 2).float()
        output = total.clone().detach().requires_grad_(True)

        loss = criterion(output, labels)

        loss.backward()
        
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    model.eval() 
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in tqdm(valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            _, detector_outputs = model(inputs)
            output = torch.tensor([0.0 for _ in range(detector_outputs[0].shape[0])]).to(device)
            for detector_output in detector_outputs:
                avg_confidences = detector_output.mean(dim=[2, 3])

                weighted_scores = avg_confidences[:, 1] - avg_confidences[:, 0]

                final_scores = (weighted_scores > 0).int()

                output += final_scores
            
            predicted = (output > 2).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {100 * correct / total}%')

logging.info('Finished Training')
