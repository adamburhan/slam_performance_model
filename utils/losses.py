import torch
import torch.nn as nn

# EMD Squared Loss Function
class EMDSquaredLoss(nn.Module):
    def __init__(self):
        super(EMDSquaredLoss, self).__init__()

    def forward(self, input, target):
        # Apply softmax to the input to get probabilities
        input_prob = torch.exp(torch.log_softmax(input, dim=1))
        
        # Calculate the cumulative distribution functions
        input_cdf = torch.cumsum(input_prob, dim=1)
        target_cdf = torch.cumsum(target, dim=1)
        
        # Compute the EMD squared
        emd_squared = torch.mean(torch.sum((input_cdf - target_cdf) ** 2, dim=1))
        
        return emd_squared

def get_loss_function(config):
    """
    Retrieves the loss function based on the configuration.
    
    Args:
        config (dict): Configuration dictionary loaded from the config file.
        
    Returns:
        criterion (nn.Module): PyTorch loss function.
    """

    # Check the type of task and select the corresponding loss function
    if config['training']['task'] == 'classification':
        # Classification Losses
        if config['training']['loss_function'] == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        elif config['training']['loss_function'] == 'NLLLoss':
            return nn.NLLLoss()
        elif config['training']['loss_function'] == 'EMDSquaredLoss':
            return EMDSquaredLoss()
        else:
            raise ValueError(f"Unknown classification loss function: {config['training']['loss_function']}")

    elif config['training']['task'] == 'regression':
        # Regression Losses
        if config['training']['loss_function'] == 'MSELoss':
            return nn.MSELoss()
        elif config['training']['loss_function'] == 'L1Loss':
            return nn.L1Loss()
        elif config['training']['loss_function'] == 'SmoothL1Loss':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown regression loss function: {config['training']['loss_function']}")

    else:
        raise ValueError(f"Unsupported task type: {config['training']['task']}")