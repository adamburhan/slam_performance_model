import os
import wandb
import numpy as np
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

@torch.no_grad()
def eval_model(
    model, 
    loader, 
    num_classes_rot, 
    num_classes_trans,
    criterion, 
    device
    ):
    
    model.eval()
    loss_rot = 0
    loss_trans = 0
    n = 0
    
    all_preds_rot = []
    all_preds_trans = []
    all_labels_rot = []
    all_labels_trans = []
    
    for batch in loader:
        (features, images), labels = batch
        features = features.to(device)
        images = images.to(device)
        labels = labels.long().to(device)
        
        labels_rot = labels[:, 0]
        labels_trans = labels[:, 1]
        
        # one hot encoding
        labels_rot_oh = F.one_hot(labels_rot, num_classes=num_classes_rot).float()
        labels_trans_oh = F.one_hot(labels_trans, num_classes=num_classes_trans).float()
        
        logits_rot, logits_trans = model(features, images)
        batch_loss_rot = criterion(logits_rot, labels_rot_oh)
        batch_loss_trans = criterion(logits_trans, labels_trans_oh)
        
        # accumulate loss
        loss_rot += batch_loss_rot.item() * labels.shape[0]
        loss_trans += batch_loss_trans.item() * labels.shape[0]
        
        # get predictions
        preds_rot = torch.argmax(logits_rot, dim=1)
        preds_trans = torch.argmax(logits_trans, dim=1)
        
        all_preds_rot.append(preds_rot.cpu())
        all_preds_trans.append(preds_trans.cpu())
        all_labels_rot.append(labels_rot.cpu())
        all_labels_trans.append(labels_trans.cpu())

        n += labels.shape[0]
        
    # Concatenate all predictions and labels
    all_preds_rot = torch.cat(all_preds_rot)
    all_preds_trans = torch.cat(all_preds_trans)
    all_labels_rot = torch.cat(all_labels_rot)
    all_labels_trans = torch.cat(all_labels_trans)
    
    # Calculate accuracy
    acc_rot = (all_preds_rot == all_labels_rot).float().mean().item() 
    acc_trans = (all_preds_trans == all_labels_trans).float().mean().item()
        
    avg_loss_rot = loss_rot / n
    avg_loss_trans = loss_trans / n
    avg_loss = avg_loss_rot + avg_loss_trans
    
    return {
        "loss": avg_loss,
        "loss_rot": avg_loss_rot,
        "loss_trans": avg_loss_trans,
        "acc_rot": acc_rot,
        "acc_trans": acc_trans
    }


def train_model(
    model,
    train_loader,
    train_loader_for_eval,
    val_loader,
    optimizer,
    criterion,
    num_classes_rot,
    num_classes_trans,
    scheduler,
    device,
    exp_name,
    checkpoint_path,
    num_epochs=25,
    eval_period=1,
    print_period=10,
    save_model_period=5,
    save_statistics_period=1,
    verbose=True,
    wandb_run=None,
):
    """
    Train a SLAM performance model and evaluate it periodically.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        train_loader_for_eval: DataLoader for evaluating on training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        device: Device to train on ('cpu', 'cuda', etc.)
        exp_name: Experiment name for saving files
        checkpoint_path: Path to save checkpoints
        num_epochs: Number of epochs to train
        eval_period: Frequency of evaluation (in epochs)
        print_period: Frequency of printing status (in iterations)
        save_model_period: Frequency of saving model checkpoints (in epochs)
        save_statistics_period: Frequency of saving statistics (in epochs)
        verbose: Whether to print verbose output
        
    Returns:
        dict: Dictionary containing all metrics during training
    """
    
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Initialize metrics tracking
    all_metrics = defaultdict(lambda: [])
    all_metrics["train"] = defaultdict(lambda: [])
    all_metrics["val"] = defaultdict(lambda: [])
        
    # count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_metrics["num_params"] = num_params
    
    if wandb_run is not None:
        wandb_run.summary["num_params"] = num_params
    
    # Initial evaluation
    if verbose:
        print("Performing initial evaluation")
        
    train_statistics = eval_model(model, 
                                  train_loader_for_eval, 
                                  num_classes_rot, 
                                  num_classes_trans, 
                                  criterion,
                                  device)
    val_statistics = eval_model(model, 
                                val_loader, 
                                num_classes_rot, 
                                num_classes_trans, 
                                criterion,
                                device)
    
    
    for k, v in train_statistics.items():
        all_metrics["train"][k].append(v)
        
    for k, v in val_statistics.items():
        all_metrics["val"][k].append(v)
    
    all_metrics["epoch"].append(0)
    
    # save initial model
    state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, f"{checkpoint_path}/{exp_name}_state_{0}_loss={val_statistics['loss']}.pth")
    
    # Log initial metrics to wandb
    if wandb_run is not None:
        metrics_to_log = {
            "epoch": 0,
            "train_loss": train_statistics["loss"],
            "train_loss_rot": train_statistics["loss_rot"],
            "train_loss_trans": train_statistics["loss_trans"],
            "train_acc_rot": train_statistics["acc_rot"],
            "train_acc_trans": train_statistics["acc_trans"],
            "val_loss": val_statistics["loss"],
            "val_loss_rot": val_statistics["loss_rot"],
            "val_loss_trans": val_statistics["loss_trans"],
            "val_acc_rot": val_statistics["acc_rot"],
            "val_acc_trans": val_statistics["acc_trans"]
        }
        wandb_run.log(metrics_to_log)
        
    for epoch in range(1, num_epochs + 1):
        for i, batch in enumerate(train_loader):
            (features, images), labels = batch
            features = features.to(device)
            images = images.to(device)
            labels = labels.long().to(device)
            
            labels_rot = labels[:, 0]
            labels_trans = labels[:, 1]
            
            # one hot encoding
            labels_rot = F.one_hot(labels_rot, num_classes=num_classes_rot).float()
            labels_trans = F.one_hot(labels_trans, num_classes=num_classes_trans).float()
            
            optimizer.zero_grad(set_to_none=True)
            model.train()                
            
            logits_rot, logits_trans = model(features, images)
            loss_rot = criterion(logits_rot, labels_rot)
            loss_trans = criterion(logits_trans, labels_trans)
            
            loss = loss_rot + loss_trans
            
            loss.backward()
            optimizer.step() 
            
            
            # Print training status
            if verbose and i % print_period == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Loss Rot: {loss_rot.item():.4f}, Loss Trans: {loss_trans.item():.4f}")
                
        if scheduler is not None:
            scheduler.step()
        
        if epoch % eval_period == 0 or epoch == num_epochs:
            if verbose:
                print(f"Evaluating model at epoch {epoch}")
                
            train_statistics = eval_model(model, 
                                          train_loader_for_eval, 
                                          num_classes_rot,
                                          num_classes_trans,
                                          criterion,
                                          device)
            val_statistics = eval_model(model, 
                                        val_loader, 
                                        num_classes_rot,
                                        num_classes_trans,
                                        criterion,
                                        device)
            
            for k, v in train_statistics.items():
                all_metrics["train"][k].append(v)
        
            for k, v in val_statistics.items():
                all_metrics["val"][k].append(v)
            
            all_metrics["epoch"].append(epoch)            
        
            if wandb_run is not None:
                metrics_to_log = {
                    "epoch": epoch,
                    "train_loss": train_statistics["loss"],
                    "train_loss_rot": train_statistics["loss_rot"],
                    "train_loss_trans": train_statistics["loss_trans"],
                    "train_acc_rot": train_statistics["acc_rot"],
                    "train_acc_trans": train_statistics["acc_trans"],
                    "val_loss": val_statistics["loss"],
                    "val_loss_rot": val_statistics["loss_rot"],
                    "val_loss_trans": val_statistics["loss_trans"],
                    "val_acc_rot": val_statistics["acc_rot"],
                    "val_acc_trans": val_statistics["acc_trans"]
                }
                wandb_run.log(metrics_to_log)
                
    # Final evaluation
    train_statistics = eval_model(model, 
                                train_loader_for_eval, 
                                num_classes_rot,
                                num_classes_trans,
                                criterion,
                                device)
    val_statistics = eval_model(model, 
                                val_loader, 
                                num_classes_rot,
                                num_classes_trans,
                                criterion,
                                device)
    
    for k, v in train_statistics.items():
        all_metrics["train"][k].append(v)

    for k, v in val_statistics.items():
        all_metrics["val"][k].append(v)
    
    all_metrics["epoch"].append(num_epochs)            

    if wandb_run is not None:
        metrics_to_log = {
            "epoch": num_epochs,
            "train_loss": train_statistics["loss"],
            "train_loss_rot": train_statistics["loss_rot"],
            "train_loss_trans": train_statistics["loss_trans"],
            "train_acc_rot": train_statistics["acc_rot"],
            "train_acc_trans": train_statistics["acc_trans"],
            "val_loss": val_statistics["loss"],
            "val_loss_rot": val_statistics["loss_rot"],
            "val_loss_trans": val_statistics["loss_trans"],
            "val_acc_rot": val_statistics["acc_rot"],
            "val_acc_trans": val_statistics["acc_trans"]
        }
        wandb_run.log(metrics_to_log)
        
    # Save final model
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, f"{checkpoint_path}/{exp_name}_state_final_loss={val_statistics['loss']}.pth")