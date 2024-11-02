import os
from src.mivia_dataset import MiviaDataset
from src.multitask_nn import MultitaskNN
import torch
from torch.optim import AdamW
from torchvision.transforms import transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.utils import *
import glob
from src.config import args
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2


min_val_loss = 1e10
early_stopping_patience = args.early_stopping
dir_model = "models"

def init_dataloader(dataset_path, train_csv, validation_csv, batch_size=64, num_workers=0):

    transforms_train = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])
                                           ])
    
    transforms_val = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])
                                         ])

    train_dataset = MiviaDataset(image_folder=dataset_path,
                                 csv_file_path=train_csv,
                                 transformations=transforms_train)

    validation_dataset = MiviaDataset(image_folder=dataset_path,
                                      csv_file_path=validation_csv,
                                      transformations=transforms_val)


    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(validation_dataset)))

    sampler_train = SubsetRandomSampler(train_indices)
    sampler_val = SubsetRandomSampler(val_indices)

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train,
                                  num_workers=num_workers)
    dataloader_val = DataLoader(validation_dataset, batch_size=batch_size, sampler=sampler_val,
                                num_workers=num_workers)

    return dataloader_train, dataloader_val


def test_dataloader(dataloader):
    for img, labels in dataloader:
        print(type(img), type(labels))


def print_model_info(model):
    for name, param in model.named_parameters():
        print(name)
        print("- requires_grad:", param.requires_grad)


def train(model: MultitaskNN, dataloader_train, dataloader_val, optimizer, ckpt_dict,
          device=torch.device("mps"), colab=False):
    
    # gradnorm
    alpha = 0.6
    best_vloss = args.min_loss
    global save_path, dir_model
    
    early_stopping_counter = early_stopping_patience
    num_epoch = args.epochs
    
    csv_file = f'models_{time_str()}.csv'

    # init weights
    weights = None
    weighted_loss = None
    optimizer2 = None
    T = None
    l0 = None
    init_weights = True
    cptk_epochs = 0

    # load from checkpoint file
    if ckpt_dict is not None:
        print("Loading from checkpoint file...")
        print(f"loss cptk {ckpt_dict['val_loss']}")
        cptk_epochs = int(ckpt_dict['epoch'].split(" ")[-1]) + 1
        weights = ckpt_dict['weights']
        best_vloss = ckpt_dict['val_loss']
        l0 = ckpt_dict['l0']
        l0 = l0.to(device)

        print(f"weights {weights}")
        init_weights = False

        # sum of weights
        T = weights.sum().detach() 
        # set optimizer for weights
        optimizer2 = torch.optim.AdamW([weights], lr=0.01)
        print("Loaded from checkpoint file!")

    create_folder('checkpoints')

    for epoch in range(num_epoch - cptk_epochs):  # loop over the dataset multiple times

        print(f'Epoch {epoch + cptk_epochs}/{num_epoch}')
        running_vloss = None
        running_loss_train = None
        reset = False

        log_weights = []
        log_loss = []

        model.train()

        for i, data in tqdm(enumerate(dataloader_train), desc=f"epoch {epoch + cptk_epochs}", total=len(dataloader_train)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = [label.to(device) for label in labels]


            # forward + backward + optimize
            outputs = model(inputs)
            loss = model.compute_loss(outputs, labels)
            running_loss_train = loss
            loss_total = sum(loss)
            loss = torch.stack(loss,dim=0)

            if init_weights:
                # init weights
                print("Init weights for GradNorm...")
                weights = torch.ones_like(loss) / 5
                weights = torch.nn.Parameter(weights)

                T = weights.sum().detach()  # sum of weights
                # set optimizer for weights
                optimizer2 = torch.optim.AdamW([weights], lr=0.01)
                # set L(0)
                l0 = loss.detach()
                init_weights = False
                print("Initialized weights for GradNorm!")


            # compute the weighted loss
            weighted_loss = weights.to(device) @ loss.to(device)

            if i % 50 == 0:
                    print("\nAll  training  losses:", [l.item() for l in running_loss_train])
                    print("Current training loss_total:", loss_total.item())
                    print("Weighted Loss: ", weighted_loss)
                    print(f"Weights {weights}\n")

            # zero the parameter gradients
            optimizer.zero_grad()
            # backward pass for weigthted task loss
            weighted_loss.backward(retain_graph=True)
            # compute the L2 norm of the gradients for each task
            gw = []

            dl = torch.autograd.grad(weights[0] * loss[0], model.gender_head.parameters(), retain_graph=True,
                                     create_graph=True)[0]
            gw.append(torch.norm(dl))
            dl = torch.autograd.grad(weights[1] * loss[1], model.bag_head.parameters(), retain_graph=True,
                                     create_graph=True)[0]
            gw.append(torch.norm(dl))
            dl = torch.autograd.grad(weights[2] * loss[2], model.hat_head.parameters(), retain_graph=True,
                                     create_graph=True)[0]
            gw.append(torch.norm(dl))
            dl = torch.autograd.grad(weights[3] * loss[3], model.upper_color_head.parameters(), retain_graph=True,
                                     create_graph=True)[0]
            gw.append(torch.norm(dl))
            dl = torch.autograd.grad(weights[4] * loss[4], model.lower_color_head.parameters(), retain_graph=True,
                                     create_graph=True)[0]
            gw.append(torch.norm(dl))

            gw = torch.stack(gw)
            # compute loss ratio per task
            loss_ratio = loss.detach() / l0
            # compute the relative inverse training rate per task
            rt = loss_ratio / loss_ratio.mean()
            # compute the average gradient norm
            gw_avg = gw.mean().detach()
            # compute the GradNorm loss
            constant = (gw_avg * rt ** alpha).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()
            # clear gradients of weights
            optimizer2.zero_grad()
            # backward pass for GradNorm
            gradnorm_loss.backward()

            # weight for each task
            log_weights.append(weights.detach().cpu().numpy().copy())
            # task normalized loss
            log_loss.append(loss_ratio.detach().cpu().numpy().copy())

            # update model weights
            optimizer.step()

            # update loss weights
            optimizer2.step()
            #print(weights)
            # renormalize weights
            weights = (weights / weights.sum() * T).detach()
            # print("Pesi: ", weights)
            weights = torch.nn.Parameter(weights)
            optimizer2 = torch.optim.AdamW([weights], lr=0.01)

        dst = ""

        df_log_weight = pd.DataFrame(np.asarray(log_weights))
        df_log_weight.to_csv(f"log_weights.csv", header=False, mode='a')
        df_log_loss = pd.DataFrame(np.asarray(log_loss))
        df_log_loss.to_csv(f"log_loss.csv", header=False, mode='a')

        if colab:

            copy_file_to_drive(os.path.join(os.getcwd()), dst, filename="log_weights.csv")
            copy_file_to_drive(os.path.join(os.getcwd()), dst, filename="log_loss.csv")

        # ------- VALIDATION PHASE -------

        val_losses = []
        losses_per_class = [[] for _ in range(5)]
        batch_cnt = len(dataloader_val)

        model.eval()
        save_path_model = os.path.join(os.getcwd(), args.result)
        with torch.no_grad():

            for i, vdata in tqdm(enumerate(dataloader_val), desc=f"epoch {epoch + cptk_epochs}", total=len(dataloader_val)):

                vinputs, vlabels = vdata

                vinputs = vinputs.to(device)
                vlabels = [vlabels.to(device) for vlabels in vlabels]

                voutputs = model(vinputs)

                vloss = model.compute_loss(voutputs, vlabels)

                for j, loss in enumerate(vloss):
                    losses_per_class[j].append(loss)

                vloss_total = sum(vloss)

                if i % 25 == 0:
                    print("All  valid  losses:", [l.item() for l in vloss])
                    print("Current valid loss_total:", vloss_total.item())

                vloss = torch.stack(vloss, dim=0)
                #running_vloss += vloss.item()
                weighted_validation_loss = weights.to(device) @ vloss.to(device)

                val_losses.append(weighted_validation_loss)


            mean_losses_per_class = [torch.stack(losses).mean().item() for losses in losses_per_class]


            weighted_validation_loss = torch.stack(val_losses).mean().item()

            print("Loss validation: ", weighted_validation_loss)

            if weighted_validation_loss < best_vloss:
                best_vloss = weighted_validation_loss
                print(best_vloss)
                reset = True
                #path_csv = os.path.join(save_path_model, csv_file)
                model_name = 'model_gradnew_augmentation{}.pth'.format(epoch)


                df_csv = {'model_name': model_name,
                          'epoch': epoch,
                          'loss': f'{best_vloss:6f}',
                          'loss_gender': f'{mean_losses_per_class[0]:6f}',
                          'loss_bag': f'{mean_losses_per_class[1]:6f}',
                          'loss_hat': f'{mean_losses_per_class[2]:6f}',
                          'loss_upper': f'{mean_losses_per_class[3]:6f}',
                          'loss_lower': f'{mean_losses_per_class[4]:6f}',
                          "weight_gender": f'{weights[0].item():6f}',
                          "weight_bag": f'{weights[1].item():6f}',
                          "weight_hat": f'{weights[2].item():6f}',
                          "weight_upper": f'{weights[3].item():6f}',
                          "weight_lower": f'{weights[4].item():6f}'
                          }

                path_csv = os.path.join(save_path_model, csv_file)

                if not os.path.exists(path_csv):
                    #save_path_model = create_folder(save_path_model)
                    pd.DataFrame([df_csv.keys()]).to_csv(path_csv, header=False, mode='a')

                #### format file: version_date_time.csv

                pd.DataFrame([df_csv]).to_csv(path_csv, header=False, mode='a')
                save_best_model(model, model_name, save_path_model, csv_file, drive=colab)
                print(f'New model saved: [Validation] epoch: {epoch} loss: {best_vloss:6f}')
                print("Saved csv file: ", csv_file)

        # save checkpoints
        cpkt_file = filename_custom(epoch=epoch)
        cpkt_path = 'checkpoints'
        save_ckpt(model, cpkt_path, cpkt_file, epoch + cptk_epochs, optimizer, best_vloss, weights, l0, colab)
        print("Saved checkpoints model to {}".format(cpkt_file))

        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {weighted_loss / 2000:.3f}')
            print(f'[{epoch + 1}, {i + 1:5d}] val: {weighted_validation_loss / 2000:.3f}')

        if reset:
            early_stopping_counter = early_stopping_patience
        else:
            early_stopping_counter -= 1

        if early_stopping_counter == 0:
            print("Training interrupted")
            break


BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers

def main():

    # colab
    IN_COLAB = False

    try:
        import google.colab
        IN_COLAB = True
    except:
        pass

    dataset_path = ""

    if IN_COLAB:
        dataset_path = ('/content/ConvNext/data/training_set')
    else:
        dataset_path = ('dataset_new/content/dataset_new/data/mivia/data')
    # ----- dataset and dataloader -----

    train_csv = "annotation_training.csv"
    test_csv = "annotation_validation.csv"

    dataloader_train, dataloader_val = init_dataloader(dataset_path,
                                                       train_csv=train_csv,
                                                       validation_csv=test_csv,
                                                       batch_size=BATCH_SIZE,
                                                       num_workers=NUM_WORKERS)

    # ----- defining model -----
    last_layer_to_train = 2  # setting how many last layers of the backbone to train
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = MultitaskNN(last_layer_to_train=last_layer_to_train)
    model.to(device)  # moving model to selected device

    # ----- defining parameters for training -----
    lr_backbone = args.lr
    lr_classificator = args.lr_classificator

    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.upper_color_head.parameters(), "lr": lr_classificator},
            {"params": model.lower_color_head.parameters(), "lr": lr_classificator},
            {"params": model.gender_head.parameters(), "lr": lr_classificator},
            {"params": model.bag_head.parameters(), "lr": lr_classificator},
            {"params": model.hat_head.parameters(), "lr": lr_classificator}
        ]
    )

    # ----- define args -----

    cptk_epochs = 0
    loss_cpkt = 0
    weights_cpkt = torch.empty(5)
    ckpt_dict = None

    if args.checkpoints:
        print(os.getcwd())
        list_files = glob.glob('checkpoints/*.pth')
        latest_file = ""
        if len(list_files) > 0:
            latest_file = max(list_files, key=os.path.getctime)
            print("latest file", latest_file)

        ckpt_file = latest_file
        print(ckpt_file)
        ckpt_dict = load_ckpt(model, ckpt_file, device)

        optimizer.load_state_dict(ckpt_dict['optimizer'])

    save_path = os.path.join(os.getcwd(), args.result)
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    print(f"{path} created!")

    # ----- train and validation -----
    Path("models").mkdir(parents=True, exist_ok=True)
    train(model=model, dataloader_train=dataloader_train, dataloader_val=dataloader_val, optimizer=optimizer,
          ckpt_dict=ckpt_dict, device=device, colab=IN_COLAB)



if __name__ == "__main__":
    main()
