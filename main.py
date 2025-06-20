"""
This is the entry point, the main purpose is to parse command-line arguments,
load configurations and call the appropriate functions to start training or evaluation.
"""

# main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

import random

import os
import argparse
import yaml
import time
import tqdm
import wandb

from data.datasets import MultiDataset, get_datasets
from scripts.train import train as train_model
from scripts.train import evaluate
from utils.logging import get_logger
from torchvision import transforms
from utils.losses import get_loss_function, EMDSquaredLoss
from models.get_model import get_model
from scripts.results import compute_metrics, create_aggregated_probability_matrix, visualize_confusion_matrix

def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability
    
    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DummyScheduler:
    """
    Dummy LR Scheduler that supports standard methods like state_dict, load_state_dict, etc.,
    but does nothing to the optimizer or learning rates.
    """
    def __init__(self, optimizer, *args, **kwargs):
        """
        Initialize the DummyScheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer (required to match the API, not used).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.optimizer = optimizer
        self._state = {}

    def step(self, *args, **kwargs):
        """
        Dummy step function that does nothing.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def state_dict(self):
        """
        Return the state of the scheduler as a dictionary.

        Returns:
            dict: A dictionary representing the scheduler's state.
        """
        return self._state

    def load_state_dict(self, state_dict):
        """
        Load the scheduler's state from a dictionary.

        Args:
            state_dict (dict): The state dictionary to load.
        """
        self._state.update(state_dict)

    def get_last_lr(self):
        """
        Get the last computed learning rate(s).

        Returns:
            list: A list of the last learning rates.
        """
        return [group['lr'] for group in self.optimizer.param_groups]
    
    
    
def train(config):
    # Seed the experiment, for repeatability
    seed_experiment(config['seed'])
    
    # Initialize wandb
    if config['misc']['use_wandb']:
        run = wandb.init(
            project=config['misc']['wandb_project'],
            entity=config['misc']['wandb_entity'],
            config=config,
            name=config['misc']['exp_name'],
            dir=config['misc']['output_dir']
        )
    else:
        run = None
        
    # Create a directory to save the experiment results
    base_path = os.path.join(config['log_dir'], config['exp_id'])
    checkpoint_path = base_path
    i = 0
    while os.path.exists(checkpoint_path):
        i += 1
        checkpoint_path = f"{base_path}_{i}"
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Data    
    train_dataset, val_dataset = get_datasets(config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config['dataset']['train_batch_size'], len(train_dataset)),
        shuffle=True,
        num_workers=config['dataset']['num_workers']
    )
    
    train_loader_for_eval = DataLoader(
        train_dataset,
        batch_size=min(config['dataset']['eval_batch_size'], len(train_dataset)),
        shuffle=False,
        num_workers=config['dataset']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(config['dataset']['eval_batch_size'], len(val_dataset)),
        shuffle=False,
        num_workers=config['dataset']['num_workers']
    )
    
    # Model
    model = get_model(config)
    model.to(config['misc']['device'])
    
    # Optimizer
    optimizer = config['optimization']['optimizer']
    lr = config['optimization']['lr']
    weight_decay = config['optimization']['weight_decay']
    if optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, 
                              momentum=config['optimization']['momentum'], 
                              weight_decay=weight_decay)
        
    scheduler = DummyScheduler(optimizer)  # Change later to real scheduler if needed
    # Initialize loss function
    criterion = get_loss_function(config)
    
    all_metrics = train_model(
        model=model, 
        train_loader=train_loader, 
        train_loader_for_eval=train_loader_for_eval, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        num_classes_rot=config['model']['num_classes_rot'],
        num_classes_trans=config['model']['num_classes_trans'],
        scheduler=scheduler,
        device=config['misc']['device'],
        exp_name=config['misc']['exp_name'],
        checkpoint_path=checkpoint_path,
        num_epochs=config['training']['num_epochs'],
        eval_period=config['training']['eval_period'],
        print_period=config['training']['print_period'],
        save_model_period=config['training']['save_model_period'],
        save_statistics_period=config['training']['save_statistics_period'],
        verbose=config['misc']['verbose'],
        wandb_run=run
    )
    
    # close wandb run
    if run is not None:
        run.finish()
    
    return all_metrics, checkpoint_path

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLAM Performance Model Training')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)