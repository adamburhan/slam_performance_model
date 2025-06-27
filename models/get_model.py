from models.alexNetClassifier import AlexNetSLAMClassifier, AlexNetSLAMClassifierBase, AlexNetSLAMClassifierTest
from models.resnet18 import ResNet18SLAMClassifier
from models.FusionClassifier import FusionClassifier
import torch

def get_model(config):
    architecture = config['model']['architecture']
    weights_path = config['model'].get('weights_path', None)
    num_classes_rot = config['model']['num_classes_rot']
    num_classes_trans = config['model']['num_classes_trans']
    feature_dim = config['dataset']['feature_dim']
    context_len = config['dataset']['context_len']
    input_channels = config['model']['input_channels'] * context_len

    if architecture == 'alexnet':
        return AlexNetSLAMClassifier(weights_path, num_classes_rot, input_channels)
    elif architecture == 'alexnet_base':
        return AlexNetSLAMClassifierBase(weights_path, num_classes_rot)
    elif architecture == 'resnet18':
        return ResNet18SLAMClassifier(weights_path, num_classes_rot, input_channels)
    elif architecture == 'alexnet_test':
        return AlexNetSLAMClassifierTest(
                weights_path=weights_path,
                num_classes=num_classes_rot,
                input_channels=input_channels,
                feature_dim=feature_dim,
                context_len=context_len
            )
    elif architecture == 'fusion_classifier':
        img_encoder = config['model'].get('img_encoder')
        if img_encoder not in ['ImageEncoder', 'TemporalImageEncoder', 'LSTMImageEncoder', 'TransformerImageEncoder']:
            raise ValueError(f"Unsupported image encoder: {img_encoder}")
        
        return FusionClassifier(
            weights_path=weights_path,
            num_classes=num_classes_rot, 
            stat_input_dim=feature_dim,
            stat_output_dim=feature_dim,
            img_encoder=img_encoder,
            context_len=context_len
        )
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")