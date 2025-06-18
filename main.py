"""
This is the entry point, the main purpose is to parse command-line arguments,
load configurations and call the appropriate functions to start training or evaluation.
"""

# main.py

import argparse
import yaml
import torch
from data.kitti_dataset import KITTIDataset, MultiDataset
from scripts.train import train
from scripts.train import evaluate
from utils.logging import get_logger
from torchvision import transforms
from utils.losses import get_loss_function, EMDSquaredLoss
import os
from models.get_model import get_model
from scripts.results import compute_metrics, create_aggregated_probability_matrix, visualize_confusion_matrix

def main():
    parser = argparse.ArgumentParser(description='SLAM Performance Model Training')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger = get_logger(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the model based on the config
    model = get_model(config)
    model.to(device)

    images_dir = config['dataset']['images_dir']
    flipped_images_dir = config['dataset']['flipped_images_dir']
    errors_dir = config['dataset']['errors_dir']

    # Define data transforms
    num_channels = config['model']['input_channels']
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Prepare datasets and dataloaders
    train_sequence_dirs = [f'{images_dir}/{i:02d}' for i in range(9)]
    train_flipped_sequence_dirs = [f'{flipped_images_dir}/{i:02d}' for i in range(9)]
    train_label_files = [f'{errors_dir}/{i:02d}.csv' for i in range(9)]

    train_dataset = KITTIDataset(sequence_dirs=train_sequence_dirs, flipped_sequence_dirs=train_flipped_sequence_dirs, label_files=train_label_files,
                                 sequence_length=config['dataset']['sequence_length'], transform=train_transforms)

    val_sequence_dirs = [f'{images_dir}/{i:02d}' for i in range(9, 11)]
    val_flipped_sequence_dirs = [f'{flipped_images_dir}/{i:02d}' for i in range(9, 11)]
    val_label_files = [f'{errors_dir}/{i:02d}.csv' for i in range(9, 11)]
    val_dataset = KITTIDataset(sequence_dirs=val_sequence_dirs, flipped_sequence_dirs=val_flipped_sequence_dirs, label_files=val_label_files,
                               sequence_length=config['dataset']['sequence_length'], transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset'].get('num_workers', 4),
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset'].get('num_workers', 4),
        pin_memory=True
    )

    # Check for different learning rate setups in the config
    if 'learning_rate_groups' in config['training']:
        # Separate learning rates for different parameter groups
        pretrained_lr = config['training']['learning_rate_groups']['pretrained_lr']
        new_lr = config['training']['learning_rate_groups']['new_lr']

        param_groups = [
            {'params': model.pretrained_parameters, 'lr': pretrained_lr},
            {'params': model.non_pretrained_parameters, 'lr': new_lr}
        ]
        optimizer = torch.optim.Adam(param_groups, weight_decay=config['training']['weight_decay'])

    else:
        # Single learning rate for all parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'],
                                     weight_decay=config['training']['weight_decay'])

    # Initialize loss function
    criterion = get_loss_function(config)

    # Start training and evaluation for the specified number of epochs
    num_epochs = config['training']['num_epochs']

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train loss: {train_loss}")

        val_loss = evaluate(model, val_loader, criterion, device)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Val loss: {val_loss}")

    # Save the trained model
    output_dir = os.path.join(config['model']['output_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/final_model_weights.pth")
    logger.info(f"Model saved at {output_dir}/final_model_weights.pth")


    # Generate Aggregated Probability Matrices
    mat1, mat2 = create_aggregated_probability_matrix(model, val_loader, num_classes=5, device=device)

    # Save visualizations
    visualize_confusion_matrix(mat1, "rotation", output_dir)
    visualize_confusion_matrix(mat2, "translation", output_dir)

    logger.info("Aggregated probability matrices and visualizations saved.")


def multi_dataset_test():
    parser = argparse.ArgumentParser(description='SLAM Performance Model Training')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger = get_logger(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the model based on the config
    model = get_model(config)
    model.to(device)

    # Define data transforms
    num_channels = config['model']['input_channels']
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    root_dir = config['dataset']['root_dir']

    dataset_dict = {
        "train":
            {
                "euroc": [
                    "MH_01_easy",
                    "MH_03_medium",
                    "MH_04_difficult",
                    "MH_05_difficult",
                    "V1_01_easy",
                    "V1_03_difficult",
                    "V2_01_easy",
                    "V2_02_medium",
                    "V2_03_difficult"
                ]
            },
        "val":
            {
                "euroc": [
                    "MH_02_easy",
                    "V1_02_medium"
                ]
            }
    }
    
    train_dataset = MultiDataset(
        root_dir=root_dir,
        dataset_dict=dataset_dict['train'],
        context_len=config['dataset']['context_len'],
        horizon=config['dataset']['horizon'],
        normalize=False,
        transform=train_transforms
    )
    
    val_dataset = MultiDataset(
        root_dir=root_dir,
        dataset_dict=dataset_dict['val'],
        context_len=config['dataset']['context_len'],
        horizon=config['dataset']['horizon'],
        normalize=False,
        transform=train_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset'].get('num_workers', 4),
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset'].get('num_workers', 4),
        pin_memory=True
    )

    # Check for different learning rate setups in the config
    if 'learning_rate_groups' in config['training']:
        # Separate learning rates for different parameter groups
        pretrained_lr = config['training']['learning_rate_groups']['pretrained_lr']
        new_lr = config['training']['learning_rate_groups']['new_lr']

        param_groups = [
            {'params': model.pretrained_parameters, 'lr': pretrained_lr},
            {'params': model.non_pretrained_parameters, 'lr': new_lr}
        ]
        optimizer = torch.optim.Adam(param_groups, weight_decay=config['training']['weight_decay'])

    else:
        # Single learning rate for all parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'],
                                     weight_decay=config['training']['weight_decay'])

    # Initialize loss function
    criterion = get_loss_function(config)

    # Start training and evaluation for the specified number of epochs
    num_epochs = config['training']['num_epochs']

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, num_classes=5)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train loss: {train_loss}")

        val_loss = evaluate(model, val_loader, criterion, device, num_classes=5)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Val loss: {val_loss}")

    # Save the trained model
    output_dir = os.path.join(config['model']['output_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/final_model_weights.pth")
    logger.info(f"Model saved at {output_dir}/final_model_weights.pth")


    # Generate Aggregated Probability Matrices
    mat1, mat2 = create_aggregated_probability_matrix(model, val_loader, num_classes=5, device=device)

    # Save visualizations
    visualize_confusion_matrix(mat1, "rotation", output_dir)
    visualize_confusion_matrix(mat2, "translation", output_dir)

    logger.info("Aggregated probability matrices and visualizations saved.")


if __name__ == '__main__':
   # main()
    multi_dataset_test()