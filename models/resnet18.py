import torch
import torch.nn as nn
import torchvision.models as models


class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
        # Uncomment the following lines, replace the ? with correct values.
        self.conv1 = nn.Conv2d(
           in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
           self.shortcut = nn.Sequential(
               nn.Conv2d(in_planes, planes,
                         kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(planes)
           )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        # 2. Go through conv2, bn
        # 3. Combine with shortcut output, and go through relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.max(torch.tensor(0.0), out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        return torch.max(torch.tensor(0.0), out)

class ResNet18SLAMClassifier(nn.Module):
    def __init__(self, weights_path, num_classes, input_channels=6):
        super(ResNet18SLAMClassifier, self).__init__()

        # Load the pretrained ResNet18 model
        resnet18 = models.resnet18()
        resnet18.load_state_dict(torch.load(weights_path, weights_only=True))

        # Modify the first convolution layer
        if input_channels != 3:
            original_weights = resnet18.conv1.weight.data  # Shape: [64, 3, 7, 7]
            new_weights = original_weights.repeat(1, input_channels // 3, 1, 1) / (input_channels // 3)
            resnet18.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet18.conv1.weight.data = new_weights

        # **Assign layers to self**
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool

        # Output layers for the two RPE components
        num_ftrs = resnet18.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 3)           # Rotation component
        self.fc2 = nn.Linear(num_ftrs, num_classes) # Translation component

        # Initialize the new layers
        self._initialize_weights()

        # **Track parameters for optimizer**
        # Pretrained parameters
        self.pretrained_parameters = list(self.conv1.parameters()) + \
                                     list(self.bn1.parameters()) + \
                                     list(self.layer1.parameters()) + \
                                     list(self.layer2.parameters()) + \
                                     list(self.layer3.parameters()) + \
                                     list(self.layer4.parameters())

        # Non-pretrained parameters
        self.non_pretrained_parameters = list(self.fc1.parameters()) + list(self.fc2.parameters())

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)