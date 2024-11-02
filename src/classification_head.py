import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import CBAM


class ClassificationModule(nn.Module):
    def __init__(self, num_classes, input_features=1024):
        super(ClassificationModule, self).__init__()

        self.num_classes = num_classes
        self.input_features = input_features

        self.fc1 = nn.Linear(self.input_features, 512)  # first dense layer
        self.fc2 = nn.Linear(512, 256)  # second dense layer
        self.fc3 = nn.Linear(256, 128)  # third dense layer
        self.fc4 = nn.Linear(128, num_classes)  # fourth dense layer

        # dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x


class Head(nn.Module):
    def __init__(self, num_classes, input_features=1024):
        super(Head, self).__init__()

        self.num_classes = num_classes
        self.input_features = input_features

        # attention module
        self.attention = CBAM(self.input_features)

        # avg pool
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # classification module
        self.fc = ClassificationModule(num_classes=self.num_classes, input_features=self.input_features)

    def forward(self, x):
        # attention
        x = self.attention(x)

        # avg
        x = self.avgpool(x)

        # flattening tensors
        x = x.view(x.size(0), -1)

        # fc
        x = self.fc(x)

        return x
