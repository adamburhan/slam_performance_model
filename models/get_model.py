from models.alexNetClassifier import AlexNetSLAMClassifier, AlexNetSLAMClassifierBase, AlexNetSLAMClassifierTest
from models.resnet18 import ResNet18SLAMClassifier
import torch

def get_model(config):
    architecture = config['model']['architecture']
    weights_path = config['model'].get('weights_path', None)
    num_classes = config['model']['num_outputs'][1]
    # input_channels = config['model']['input_channels'] * config['dataset']['sequence_length']
    feature_dim = config['dataset']['feature_dim']
    context_len = config['dataset']['context_len']
    input_channels = config['model']['input_channels'] * context_len

    if architecture == 'alexnet':
        return AlexNetSLAMClassifier(weights_path, num_classes, input_channels)
    elif architecture == 'alexnet_base':
        return AlexNetSLAMClassifierBase(weights_path, num_classes)
    elif architecture == 'resnet18':
        return ResNet18SLAMClassifier(weights_path, num_classes, input_channels)
    elif architecture == 'alexnet_test':
        return AlexNetSLAMClassifierTest(
                weights_path=weights_path,
                num_classes=num_classes,
                input_channels=input_channels,
                feature_dim=feature_dim,
                context_len=context_len
            )
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")
