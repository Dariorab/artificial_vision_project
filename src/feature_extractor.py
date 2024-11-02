import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Base_Weights


class FeatureExtractor(nn.Module):
    def __init__(self, last_layer_to_train=0):
        super(FeatureExtractor, self).__init__()

        self.backbone = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).features
        self.out_features = 1024
        self.params = list()

        index = len(self.backbone) - last_layer_to_train

        # setting the requirements of the gradients
        for i, param in enumerate(self.backbone):
            if i >= index:
                param.requires_grad_(True)
                self.params.append(param)
            else:
                param.requires_grad_(False)

    def forward(self, x):
        return self.backbone(x)
