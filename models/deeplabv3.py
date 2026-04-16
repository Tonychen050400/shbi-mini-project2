import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=21, pretrained_backbone=True):
        super().__init__()
        if pretrained_backbone:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            self.model = deeplabv3_resnet50(weights=weights)
        else:
            self.model = deeplabv3_resnet50(weights=None)

        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, 1)

        if self.model.aux_classifier is not None:
            in_channels_aux = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, 1)

    def forward(self, x):
        return self.model(x)["out"]
