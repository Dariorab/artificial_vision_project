from src.mivia_dataset import MiviaDataset
from src.multitask_nn import MultitaskNN
import torch
from torch.optim import AdamW
from torchvision.transforms import transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.utils import *
import glob
from src.config import args
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt


def argmax_multi_class(y_pred, y_true):

    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()

    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)

    return y_pred, y_true


def binary_class(y_pred, y_true):

    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()

    y_pred = np.where(y_pred > 0.5, 1, 0)
    y_true = np.where(y_true > 0.5, 1, 0)

    return y_pred, y_true

def load_model(network):
    load_path = "models/model.pth"
    network.load_state_dict(torch.load(load_path))
    print(f"Loaded model from {load_path}")
    return network

def save_matrix(matrix, names, path):
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=names)
    disp.plot()
    plt.savefig(path)

def print_metrics(attribute, accuracy, loss, confusion_matrix):
    print(f'{attribute} - Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}')
    print(f'Confusion Matrix - {attribute}:')
    print(confusion_matrix)


def test_model(model, test_dataloader, device):
    """
    Test the PyTorch model on a given test dataset
    and save confusion matrix for each task and other metrics like (F1, Precision, Recall)

    Parameters:
    - model: PyTorch model to be tested
    - test_loader: PyTorch DataLoader for the test dataset
    - device: Device to run the model on (default is 'cuda' if available, else 'cpu' or 'mps')

    """
    accuracies_upper= 0
    accuracies_lower = 0
    accuracies_hat = 0
    accuracies_bag = 0
    accuracies_gender = 0

    test_losses = []
    values_weights = [0.192375, 0.242835, 0.434068, 0.06841, 0.062312]
    weights = torch.ones_like(torch.FloatTensor(values_weights))
    weights = torch.nn.Parameter(weights)

    losses_per_class = [[] for _ in range(5)]

    label_names = ['gender', 'bag', 'hat', 'black', 'blue', 'brown', 'gray', 'green', 'orange',
                   'pink', 'purple', 'red', 'white', 'yellow']

    batch_size = len(test_dataloader)

    multiclass_upper_prd = np.zeros(shape=1)
    multiclass_lower_prd = np.zeros(shape=1)

    multiclass_upper_lbl = np.zeros(shape=1)
    multiclass_lower_lbl = np.zeros(shape=1)

    gender_prd = np.zeros(shape=1)
    gender_lbl = np.zeros(shape=1)

    bag_prd = np.zeros(shape=1)
    bag_lbl = np.zeros(shape=1)

    hat_prd = np.zeros(shape=1)
    hat_lbl = np.zeros(shape=1)

    with torch.no_grad():

        for i, (imgs, labels) in tqdm(enumerate(test_dataloader), total= len(test_dataloader)):
            imgs = imgs.to(device)
            labels = [lbl.to(device) for lbl in labels]
            (lbl_upp, lbl_low, lbl_gdr, lbl_bag, lbl_hat) = labels

            preds = model(imgs)

            loss = model.compute_loss(preds, labels)

            for j, losses in enumerate(loss):
                    losses_per_class[j].append(losses)

            (prd_upp, prd_low, prd_gdr, prd_bag, prd_hat) = preds

            loss_total = sum(loss)
            loss = torch.stack(loss, dim = 0)
            weighted_loss = weights.to(device) @ loss.to(device)

            test_losses.append(weighted_loss)

            # normalize predictions
            prd_upp = F.softmax(prd_upp, dim = 1)
            prd_low = F.softmax(prd_low, dim = 1)

            _, indxs_upp = prd_upp.topk(1, dim=1)
            _, indxs_low = prd_low.topk(1, dim=1)
            _, indxs_upp_lablel = lbl_upp.topk(1,dim=1)
            _, indxs_low_lablel = lbl_low.topk(1, dim=1)
            #print(indxs_upp, indxs_low)

            accuracies_gender += torch.mean(((prd_gdr.detach() > .5) == lbl_gdr).float())
            accuracies_bag += torch.mean(((prd_bag.detach() > .5) == lbl_bag).float())
            accuracies_hat += torch.mean(((prd_hat.detach() > .5) == lbl_hat).float())
            accuracies_upper += torch.mean((indxs_upp == indxs_upp_lablel).float())
            accuracies_lower += torch.mean(((indxs_low == indxs_low_lablel).float()))


            if i % 50 == 0:
                print(f"Accuracies: {accuracies_gender/batch_size:6f}, {accuracies_bag/batch_size:6f}, "
                      f"{accuracies_hat/batch_size:6f}, {accuracies_upper/batch_size:6f}, {accuracies_lower/batch_size:6f}")

            prd_upp, lbl_upp = argmax_multi_class(prd_upp, lbl_upp)
            prd_low, lbl_low = argmax_multi_class(prd_low, lbl_low)
            prd_gdr, lbl_gdr = binary_class(prd_gdr, lbl_gdr)
            prd_bag, lbl_bag = binary_class(prd_bag, lbl_bag)
            prd_hat, lbl_hat = binary_class(prd_hat, lbl_hat)

            if i == 0:
                multiclass_upper_prd = prd_upp
                multiclass_upper_lbl = lbl_upp

                multiclass_lower_prd = prd_low
                multiclass_lower_lbl = lbl_low

                gender_prd = prd_gdr
                gender_lbl = lbl_gdr

                bag_prd = prd_bag
                bag_lbl = lbl_bag

                hat_prd = prd_hat
                hat_lbl = lbl_hat

            multiclass_upper_prd = np.append(multiclass_upper_prd, prd_upp)
            multiclass_upper_lbl = np.append(multiclass_upper_lbl, lbl_upp)

            multiclass_lower_prd = np.append(multiclass_lower_prd, prd_low)
            multiclass_lower_lbl = np.append(multiclass_lower_lbl, lbl_low)

            gender_prd = np.append(gender_prd, prd_gdr)
            gender_lbl = np.append(gender_lbl, lbl_gdr)

            bag_prd = np.append(bag_prd, prd_bag)
            bag_lbl = np.append(bag_lbl, lbl_bag)

            hat_prd = np.append(hat_prd, prd_hat)
            hat_lbl = np.append(hat_lbl, lbl_hat)

    weighted_validation_loss = torch.stack(test_losses).mean().item()
    mean_losses_per_class = [torch.stack(losses).mean().item() for losses in losses_per_class]

    confusion_mat_upper = confusion_matrix(multiclass_upper_lbl, multiclass_upper_prd)

    print_metrics("Upper Color", accuracies_upper/batch_size, mean_losses_per_class[4], confusion_mat_upper)
    save_matrix(confusion_mat_upper, label_names[3:], "upper_colors.png")

    confusion_mat_lower = confusion_matrix(multiclass_lower_lbl, multiclass_lower_prd)

    print_metrics("Lower Color", accuracies_lower / batch_size, mean_losses_per_class[3], confusion_mat_lower)
    save_matrix(confusion_mat_lower, label_names[3:], "lower_colors.png")

    confusion_mat_gender = confusion_matrix(gender_lbl, gender_prd)

    print_metrics("Gender", accuracies_lower / batch_size, mean_losses_per_class[3], confusion_mat_gender)
    save_matrix(confusion_mat_gender, ["Male", "Female"], "gender.png")

    confusion_mat_bag = confusion_matrix(bag_lbl, bag_prd)
    print_metrics("Bag", accuracies_lower / batch_size, mean_losses_per_class[3], confusion_mat_bag)
    save_matrix(confusion_mat_bag, ["Bag", "No Bag"], "bag.png")

    confusion_mat_hat = confusion_matrix(hat_lbl, hat_prd)
    print_metrics("Bag", accuracies_lower / batch_size, mean_losses_per_class[3], confusion_mat_hat)
    save_matrix(confusion_mat_hat, ["Hat", "No Hat"], "hat.png")


def create_dataset():

    dataset_path = 'validation_set'
    transforms_test = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225]),
                                      ])

    mivia_dataset_test = MiviaDataset(image_folder = dataset_path,
                                      csv_file_path ='annotation_test.csv',
                                      transformations=transforms_test)
    num_workers = args.num_workers
    batch_size = args.batch_size
    dataloader_test = DataLoader(mivia_dataset_test, batch_size = batch_size, num_workers=num_workers)

    return dataloader_test

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = MultitaskNN()
    model = load_model(model)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    test_dataloader = create_dataset()
    test_model(model, test_dataloader, device)
