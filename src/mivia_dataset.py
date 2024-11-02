import csv
import os
import torch
from PIL import Image
import cv2


class MiviaDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, csv_file_path, transformations, is_training_set=True):
        super().__init__()

        self.path = csv_file_path
        self.transforms = transformations
        self.is_training_set = is_training_set
        self.image_folder = image_folder
        self.dataset = {}
        self.column_names = None

        # read the csv file into a dictionary
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            self.column_names = next(csv_reader)
            for i, line in enumerate(csv_reader):
                self.dataset[i] = line

    def _format_input(self, input_str, one_hot=False):
        one_hot_tensor = torch.tensor([float(i) for i in input_str])
        if one_hot:
            return one_hot_tensor
        if one_hot_tensor.size(0) > 1:
            return torch.argmax(one_hot_tensor)
        else:
            return one_hot_tensor[0].int()


    def _parse_labels(self, input_str):
        # creating the corresponding labels for each category
        upper_color = self._format_input(input_str[4:15], True)
        genders = self._format_input(input_str[1])
        lower_color = self._format_input(input_str[15:], True)
        hat = self._format_input(input_str[3])
        bag = self._format_input(input_str[2])
        return upper_color, lower_color, genders, bag, hat

    def __getitem__(self, index):
        if self.is_training_set:
            img_path = self.dataset[index][0]
            labels = self._parse_labels(self.dataset[index])
        else:
            #inference phase
            img_path = self.dataset[index][0]
            labels = -1

        with open(os.path.join(self.image_folder, img_path), 'rb') as img_file:
            # since our datasets include png images, we need to make sure
            # we read only 3 channels and not more!
            img = Image.open(img_file).convert('RGB')
            img = self.transforms(img)
            return img, labels

    def __len__(self):
        return len(self.dataset)

    def Label_names(self):
        gender_names = self.column_names[1]
        bag_names = self.column_names[2]
        hat_names = self.column_names[3]
        upper_color_names = self.column_names[4:15]
        lower_color_names = self.column_names[16:]

        return gender_names, bag_names, hat_names, upper_color_names, lower_color_names
