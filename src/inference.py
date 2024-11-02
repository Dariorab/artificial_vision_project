import cv2
import os
import torch
from PIL import Image
from torchvision import transforms
from src.utils import getProjectRoot
from multitask_nn import MultitaskNN

def img_transform(img):
    transforms_inference = transforms.Compose([transforms.Resize((224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])
                                               ])
    image_transformed = transforms_inference(img)
    return image_transformed


def group_label():
    return (['male', 'female'],
            ['no', 'yes'],
            ['no', 'yes'],
            ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']
            )


def load_from_cv(img, device):
    color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted)
    src = pil_image
    src = img_transform(src)
    src = src.unsqueeze(dim=0)
    src = src.to(device)

    return src


def load_model(network):
    load_path = os.path.join(getProjectRoot(),"models", "model.pth")
    network.load_state_dict(torch.load(load_path, map_location=torch.device("cpu")))
    print(f"Loaded model from {load_path}")

    return network


def show_predictions(names, preds):

    (genders, bag, hat, upper_color) = names

    lower_color = upper_color

    preds_correct = []

    for pred in preds[:2]:
        preds_correct.append(torch.softmax(pred, 1))
    for pred in preds[2:]:
        preds_correct.append(torch.sigmoid(pred))

    upp = upper_color[torch.argmax(preds_correct[0]).item()]
    low = lower_color[torch.argmax(preds_correct[1]).item()]

    gdr = 'male' if torch.max(preds_correct[2]).item() < 0.5 else 'female'
    bag = 'yes' if torch.max(preds_correct[3]).item() > 0.6 else 'no'
    hat = 'yes' if torch.max(preds_correct[4]).item() > 0.6 else 'no'

    predictions = {
        'gender': gdr,
        'bag': bag,
        'hat': hat,
        'upper_color': upp,
        'lower_color': low
    }

    return predictions


def inference(img):
    src = load_from_cv(img, device)
    preds = model(src)
    predictions = show_predictions(group_label(), preds)

    return predictions


device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = MultitaskNN()
model = load_model(model)
model.eval()
model.to(device)
