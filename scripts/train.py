import sys
import os

# Add the project root to the Python path before importing any modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from models.alexNetClassifier import AlexNetSLAMClassifier
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data.kitti_dataset import KITTIDataset


# def train(model, iterator, optimizer, criterion, device):
#     epoch_loss = 0.0
#     model.train()

#     for batch in iterator:
#         images = batch['images'].to(device)
#         labels1, labels2 = batch['rpe'][:, 0], batch['rpe'][:, 1]

#         labels1, labels2 = labels1.to(device), labels2.to(device)


#         # One-hot encode the labels
#         labels1_one_hot = F.one_hot(labels1, num_classes=5).float()
#         labels2_one_hot = F.one_hot(labels2, num_classes=5).float()


#         # Reset gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs1, outputs2 = model(images)
        
#         # Compute loss
#         loss1 = criterion(outputs1, labels1_one_hot)
#         loss2 = criterion(outputs2, labels2_one_hot)
#         loss = loss1 + loss2

#         # Backward pass
#         loss.backward()

#         # Update model params
#         optimizer.step()

#         # Accumulate model params
#         epoch_loss += loss.item() * images.size(0)

#     return epoch_loss / len(iterator)

def train(model, iterator, optimizer, criterion, device, num_classes):
    epoch_loss = 0.0
    model.train()

    for batch in iterator:
        (features, images), labels = batch
        features = features.to(device)
        images = images.to(device)
        labels = labels.long().to(device)

        # One-hot encode the labels (rot, trans)
        labels1 = labels[:, 0]
        labels2 = labels[:, 1]
        labels1_one_hot = F.one_hot(labels1, num_classes=num_classes).float()
        labels2_one_hot = F.one_hot(labels2, num_classes=num_classes).float()

        optimizer.zero_grad()

        outputs1, outputs2 = model((features, images))

        loss1 = criterion(outputs1, labels1_one_hot)
        loss2 = criterion(outputs2, labels2_one_hot)
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * labels.size(0)

    return epoch_loss / len(iterator.dataset)

# def evaluate(model, iterator, criterion, device):
#     epoch_loss = 0.0
#     model.eval()

#     with torch.no_grad():
#         for batch in iterator:
#             images = batch['images'].to(device)
#             labels1, labels2 = batch['rpe'][:, 0], batch['rpe'][:, 1]
#             labels1, labels2 = labels1.to(device), labels2.to(device)

#             # One-hot encode the labels
#             labels1_one_hot = F.one_hot(labels1, num_classes=3).float()
#             labels2_one_hot = F.one_hot(labels2, num_classes=5).float()

#             output1, output2 = model(images)

#             loss1, loss2 = criterion(output1, labels1_one_hot), criterion(output2, labels2_one_hot)
#             loss = loss1 + loss2

#             epoch_loss += loss.item() * images.size(0)

#     return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device, num_classes):
    epoch_loss = 0.0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            (features, images), labels = batch
            features = features.to(device)
            images = images.to(device)
            labels = labels.long().to(device)

            labels1 = labels[:, 0]
            labels2 = labels[:, 1]
            labels1_one_hot = F.one_hot(labels1, num_classes=num_classes).float()
            labels2_one_hot = F.one_hot(labels2, num_classes=num_classes).float()

            outputs1, outputs2 = model((features, images))

            loss1 = criterion(outputs1, labels1_one_hot)
            loss2 = criterion(outputs2, labels2_one_hot)
            loss = loss1 + loss2

            epoch_loss += loss.item() * labels.size(0)

    return epoch_loss / len(iterator.dataset)


# if __name__ == "__main__":
#     args = parse_args()
#     config = load_config(args.config)
    
#     # Initialize wandb
#     #wandb.init(project="ood-slam", entity="udem-mila", mode="offline")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     train_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
    
#     val_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     # train_dir = config['dataset']['train_data_dir']
#     # val_dir = config['dataset']['val_data_dir']

#     images_dir = config['dataset']['images_dir'] #/media/adam/T9/ood_slam_data/datasets/kitti/odometry_gray/sequences/
#     errors_dir = config['dataset']['errors_dir'] #/media/adam/T9/slam_performance_model/data/errors/
    
#     # images_dir = "/media/adam/T9/ood_slam_data/datasets/kitti/odometry_gray/sequences"
#     # errors_dir = "/media/adam/T9/slam_performance_model/data/errors/discretized"

#     batch_size = config['dataset']['batch_size']
#     sequence_length = config['dataset']['sequence_length']
#     learning_rate = config['training']['learning_rate']
#     num_epochs = config['training']['num_epochs']


#     # Combine all the training sequences (00–08)
#     train_sequence_dirs = [f'{images_dir}/{i:02d}' for i in range(9)]
#     train_label_files = [f'{errors_dir}/{i:02d}.csv' for i in range(9)]

#     # Initialize the dataset
#     train_dataset = KITTIDataset(sequence_dirs=train_sequence_dirs, label_files=train_label_files, sequence_length=1, transform=train_transforms)

#     # Create a DataLoader for the dataset
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

#     val_sequence_dirs = [f'{images_dir}/{i:02d}' for i in range(9, 11)]
#     val_label_files = [f'{errors_dir}/{i:02d}.csv' for i in range(9, 11)]
#     val_dataset = KITTIDataset(sequence_dirs=val_sequence_dirs, label_files=val_label_files, sequence_length=1, transform=val_transforms)

#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

#     model = AlexNetSLAMClassifier(config['model']['weights_path'], num_classes=5)

#     output_dir = config['model']['output_dir']
    
#     criterion = EMDSquaredLoss()

#     pretrained_classifier_lr = 1e-4
#     pretrained_conv_layer_lr = 1e-5
#     new_lr = 1e-3

#     params = [
#         {'params': model.features[0].parameters(), 'lr': new_lr},
#         {'params': model.features[10].parameters(), 'lr': pretrained_conv_layer_lr},
#         {'params': model.classifier.parameters(), 'lr': pretrained_classifier_lr},
#         {'params': model.fc1.parameters(), 'lr': new_lr},
#         {'params': model.fc2.parameters(), 'lr': new_lr}
#     ]
#     optimizer = optim.Adam(params)

#     print(f'The model has {count_parameters(model):,} trainable parameters')
    

#     model = model.to(device)

#     # Log hyperparameters
#     # wandb.config.update({
#     #     "learning_rate": learning_rate,
#     #     "batch_size": batch_size,
#     #     "num_epochs": num_epochs,
#     #     "sequence_length": sequence_length
#     # })

#     for epoch in range(num_epochs):
#         train_loss = train(model, train_loader, optimizer, criterion, device)
#         print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {train_loss}")
#         #wandb.log({"train_loss": train_loss, "epoch": epoch + 1})

#         val_loss = evaluate(model, val_loader, criterion, device)
#         print(f"Epoch {epoch + 1}/{num_epochs}, Val loss: {val_loss}")
#         #wandb.log({"val_loss": val_loss, "epoch": epoch + 1})

#     output_dir = config['model']['output_dir']
#     #torch.save(model.state_dict(), f"{output_dir}/fine_tuned_alexnet_weights.pth")



#     mat1, mat2 = create_aggregated_probability_matrix(model, val_loader, 5)

#     visualize_confusion_matrix(mat1, "rotation", output_dir)
#     visualize_confusion_matrix(mat2, "translation", output_dir)