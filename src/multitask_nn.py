from feature_extractor import FeatureExtractor
from classification_head import Head
import torch.nn as nn


class MultitaskNN(nn.Module):
    def __init__(self, last_layer_to_train=0):
        
        super(MultitaskNN, self).__init__()
        # defining backbone for feature extraction
        self.backbone = FeatureExtractor(last_layer_to_train=last_layer_to_train)

        # defining one head for each task
        self.upper_color_head = Head(11, self.backbone.out_features)
        self.lower_color_head = Head(11, self.backbone.out_features)
        self.gender_head = Head(1, self.backbone.out_features)
        self.bag_head = Head(1, self.backbone.out_features)
        self.hat_head = Head(1, self.backbone.out_features)

        self.upper_color_loss = nn.CrossEntropyLoss()
        self.lower_color_loss = nn.CrossEntropyLoss()
        self.gender_loss = nn.BCELoss()
        self.bag_loss = nn.BCELoss()
        self.hat_loss = nn.BCELoss()

    def forward(self, x):
        # getting extracted features
        features = self.backbone(x)

        # getting the prediction for each head
        upper_color_pred = self.upper_color_head(features)
        lower_color_pred = self.lower_color_head(features)
        gender_pred = self.gender_head(features)
        bag_pred = self.bag_head(features)
        hat_pred = self.hat_head(features)

        return upper_color_pred, lower_color_pred, gender_pred, bag_pred, hat_pred

    def compute_loss(self, y_pred, y_true):
        # labels order: gender, bag, hat, upper color, lower color
        upper_color_label, lower_color_label, gender_label, bag_label, hat_label = y_true
        upper_color_pred, lower_color_pred, gender_pred, bag_pred, hat_pred = y_pred

        # mask_upper_color = (upper_color_label >= 0).float()
        loss_upper_color = self.upper_color_loss(upper_color_pred, upper_color_label.float())

        # mask_lower_color = (lower_color_label >= 0).float()
        loss_lower_color = self.lower_color_loss(lower_color_pred, lower_color_label.float())

        # mask_gender = (gender_label >= 0).float()
        gender_label = gender_label.unsqueeze(1)
        loss_gender = self.gender_loss(gender_pred, gender_label.float())

        # mask_bag = (bag_label >= 0).float()
        bag_label = bag_label.unsqueeze(1)
        loss_bag = self.bag_loss(bag_pred, bag_label.float())

        # mask_hat = (hat_label >= 0).float()
        hat_label = hat_label.unsqueeze(1)
        loss_hat = self.hat_loss(hat_pred, hat_label.float())

        return [loss_gender, loss_bag, loss_hat, loss_upper_color, loss_lower_color]
