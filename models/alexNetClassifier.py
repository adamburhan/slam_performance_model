import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class AlexNetSLAMClassifierBase(nn.Module):
    def __init__(self, weights_path, num_classes):
        super().__init__()
        
        # load the pretrained alexnet model
        alexnet = models.alexnet()
        alexnet.load_state_dict(torch.load(weights_path, weights_only=True))

        self.features = alexnet.features

        # Modify the first convolution layer to accept 6 channels instead of 3
        self.features[0] = nn.Conv2d(6, 64, kernel_size=11, stride=4, padding=2)

        # Initialize the modified first conv layer and new layers
        nn.init.kaiming_normal_(self.features[0].weight)
        nn.init.constant_(self.features[0].bias, 0)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
        )

        self.fc1 = nn.Linear(4096, 3) # first RPE component
        self.fc2 = nn.Linear(4096, num_classes) # first RPE component


        self._initialize_weights()

        self.pretrained_parameters = list(self.features.parameters()) + list(self.classifier.parameters())
        self.non_pretrained_parameters = list(self.fc1.parameters()) + list(self.fc2.parameters())


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        out1 = self.fc1(x)
        out2 = self.fc2(x)

        return out1, out2
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)


class AlexNetSLAMClassifier(nn.Module):
    def __init__(self, weights_path, num_classes, input_channels=6):
        super(AlexNetSLAMClassifier, self).__init__()
        
        # Load the pretrained AlexNet model
        alexnet = models.alexnet()
        alexnet.load_state_dict(torch.load(weights_path, weights_only=True))

        self.features = alexnet.features

        # Modify the first convolution layer
        original_weights = self.features[0].weight.data  # Shape: [64, 3, 11, 11]
        if input_channels != 3:
            # Duplicate weights to match new input channels
            new_weights = original_weights.repeat(1, input_channels // 3, 1, 1) / (input_channels // 3)
            self.features[0] = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
            self.features[0].weight.data = new_weights
            self.features[0].bias.data = alexnet.features[0].bias.data

        self.avgpool = alexnet.avgpool

        # Modify the classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        # Output layers for the two RPE components
        self.fc1 = nn.Linear(4096, 3)           # Rotation component
        self.fc2 = nn.Linear(4096, num_classes) # Translation component

        # Initialize the new layers
        self._initialize_weights()

        self.pretrained_parameters = list(self.features.parameters()) + list(self.classifier.parameters())
        self.non_pretrained_parameters = list(self.fc1.parameters()) + list(self.fc2.parameters())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)



class AlexNetSLAMClassifierTest(nn.Module):
    def __init__(self, weights_path, num_classes, context_len, input_channels=3, feature_dim=10):
        super().__init__()

        # Load pretrained AlexNet
        alexnet = models.alexnet()
        alexnet.load_state_dict(torch.load(weights_path, weights_only=True))
        self.features = alexnet.features

        # Adapt input channels for context-length * 3-channel RGB
        if input_channels != 3:
            orig_weights = self.features[0].weight.data
            new_weights = orig_weights.repeat(1, input_channels // 3, 1, 1) / (input_channels // 3)
            self.features[0] = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
            self.features[0].weight.data = new_weights
            self.features[0].bias.data = alexnet.features[0].bias.data

        self.avgpool = alexnet.avgpool
        self.img_fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        # VO feature processing
        self.vo_fc = nn.Sequential(
            nn.Linear(feature_dim *  context_len, 256),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        # Fusion and classification
        self.fusion_fc = nn.Sequential(
            nn.Linear(4096 + 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc1 = nn.Linear(4096, num_classes)  # Rotation
        self.fc2 = nn.Linear(4096, num_classes)  # Translation

        self._initialize_weights()
        self.pretrained_parameters = list(self.features.parameters()) + list(self.img_fc.parameters())
        self.non_pretrained_parameters = list(self.vo_fc.parameters()) + list(self.fusion_fc.parameters()) + list(self.fc1.parameters()) + list(self.fc2.parameters())

    def forward(self, x):
        vo_features, images = x  # Expect [B, seq_len, feat_dim], [B, C, H, W]

        # Image stream
        x_img = self.features(images)
        x_img = self.avgpool(x_img)
        x_img = torch.flatten(x_img, 1)
        x_img = self.img_fc(x_img)

        # VO stream
        x_vo = torch.flatten(vo_features, 1)  # shape [B, seq_len * feat_dim]
        x_vo = self.vo_fc(x_vo)

        # Fuse
        x = torch.cat([x_img, x_vo], dim=1)
        x = self.fusion_fc(x)

        return self.fc1(x), self.fc2(x)

    def _initialize_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
