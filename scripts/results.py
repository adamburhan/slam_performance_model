# scripts/results.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(predictions, labels):
    """
    Computes classification accuracy and other metrics.

    Args:
        predictions (tensor): Model predictions.
        labels (tensor): Ground truth labels.

    Returns:
        dict: Dictionary of computed metrics.
    """
    accuracy = (predictions == labels).float().mean().item()
    return {'accuracy': accuracy}

def create_aggregated_probability_matrix(model, dataloader, num_classes, device):
    """
    Create a matrix that aggregates the predicted probability distributions
    for each true label across all samples in the validation dataset.

    Args:
        model: The trained PyTorch model.
        dataloader: The DataLoader for the validation dataset.
        num_classes: The number of classes.
        device: Torch device (CPU or CUDA).

    Returns:
        agg_prob_matrix: A 2D numpy array of shape (num_classes, num_classes)
                         representing the aggregated probability distributions.
    """
    model.eval()

    agg_prob_matrix_rotation = np.zeros((3, 3))
    agg_prob_matrix_translation = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            outputs1, outputs2 = model(images)

            prob_dist1 = torch.exp(torch.log_softmax(outputs1, dim=1))
            prob_dist2 = torch.exp(torch.log_softmax(outputs2, dim=1))

            for i in range(prob_dist1.size(0)):
                true_label = batch['rpe'][i, 0].item()
                agg_prob_matrix_rotation[true_label] += prob_dist1[i].cpu().numpy()

            for i in range(prob_dist2.size(0)):
                true_label = batch['rpe'][i, 1].item()
                agg_prob_matrix_translation[true_label] += prob_dist2[i].cpu().numpy()

    return agg_prob_matrix_rotation, agg_prob_matrix_translation

def visualize_confusion_matrix(matrix, component, output_dir):
    """
    Visualizes and saves a confusion matrix.

    Args:
        matrix (ndarray): Aggregated probability matrix.
        component (str): Either 'rotation' or 'translation'.
        output_dir (str): Directory to save the visualization.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = matrix / row_sums

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=np.arange(len(matrix)), yticklabels=np.arange(len(matrix)),
                vmin=0, vmax=1)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"Aggregated Probability Distribution Matrix ({component})")

    plt.savefig(f"{output_dir}/aggregated_probability_matrix_{component}.png")
    plt.close()
