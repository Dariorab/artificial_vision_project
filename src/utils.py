import json
import numpy as np
import supervision as sv
from supervision import Position
import argparse
from collections import Counter
import cv2
import shutil
import glob
import torch, os, time, datetime
from pathlib import Path


def getProjectRoot() -> Path:
    return Path(__file__).parent.parent

### --- UTILITY FUNCTIONS FOR TRAINING ----

DRIVE_PATH = ''
DRIVE_PATH_CHECKPOINTS = ''

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.datetime.today().strftime(fmt)


def copy_file_to_drive(src, dst="", filename=""):
    #copy dir results/model.../model... .pth

    dir_new = src.split('/')[-1]

    src = os.path.join(src, filename)

    dst_path = os.path.join(dst, dir_new)

    dst_file_path = os.path.join(dst_path, filename)

    try:
        shutil.copyfile(src, dst_file_path)
        print(f"copied in {dst_file_path}")
    except:
        create_folder(dst_path)
        print(f"Created folder {dst_path}")
        shutil.copyfile(src, dst_file_path)

    return dst_path

def create_folder(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return os.path.join(*path.parts)

def filename_custom(epoch, extension='.pth'):
    return f'model_{epoch}_{time_str()}{extension}'

def dir_custom(version):
    return f'model_{version}_{time_str()}'


def save_best_model(model, filename, path_dir, csv_file = "", drive=False):
    """
    Note:
        save in the file results and create a directory for
        training directory format: training_{version_model}_{date}

    """
    save_path = os.path.join(path_dir, filename)

    torch.save(model.state_dict(), str(save_path))

    if drive:
        print("Saving best model...")
        dst_file_path = copy_file_to_drive(path_dir, DRIVE_PATH, filename)
        print(f"Saved best model in {dst_file_path}!")
        dst_file_path = copy_file_to_drive(path_dir, DRIVE_PATH, csv_file)
        print(f"Saved {csv_file} in {dst_file_path}!")


def load_ckpt(model, ckpt_file, device, verbose=True):
    ckpt = torch.load(ckpt_file, map_location=torch.device('cpu'))

    model.load_state_dict(ckpt['model'])

    if verbose:
        print("Resume from ckpt {}, \nepoch: {}".format(
            ckpt_file, ckpt['epoch']))

    load_dict = {
        'epoch': ckpt['epoch'],
        'optimizer': ckpt['optimizer'],
        'val_loss': ckpt['loss_total'],
        'weights': ckpt['weights'],
        'l0': ckpt['l0']
    }

    return load_dict

def remove_old_files(drive_path):

    list_of_files = glob.glob(f'{drive_path}/*.pth')
    latest_file = ""
    if len(list_of_files) > 1:
        latest_file = max(list_of_files, key=os.path.getctime)
        print("latest file", latest_file)
        for i in list_of_files:
            if i != latest_file:
                os.remove(i)


def save_ckpt(model, ckpt_path, filename, epoch, optimizer, val_loss, weights, l0, drive=False):

    """
    Note:
        create dire checkpoints and save there
        torch.save() reserves device type and id of tensors to save.
        So when loading ckpt, you have to inform torch.load() to load these tensors
        to cpu or your desired gpu, if you change devices.
    """

    save_dict = {'model': model.state_dict(),
                 'epoch': f'{time_str()} in epoch {epoch}',
                 'optimizer': optimizer.state_dict(),
                 'weights': weights,
                 'loss_total': val_loss,
                 'l0': l0}

    ckpt_path_files = os.path.join(ckpt_path, filename)

    torch.save(save_dict, str(ckpt_path_files))

    if drive:
        drive_path = copy_file_to_drive(ckpt_path, DRIVE_PATH_CHECKPOINTS, filename)
        remove_old_files(drive_path)
    remove_old_files(ckpt_path)




### --- UTILITY FUNCTIONS FOR INFERENCE ---

class NpEncoder(json.JSONEncoder):
    """
    Encoder class used in order to convert some types not compatibles with default type
    for saving a .json file.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def init_roi(conf_path, video_info):
    """
    Initialization function for the rois present in the configuration file.
    It parses the file, creates the rois (represented by a PolygonZone object) and
    returns the list of created rois.
    """
    f = open(conf_path)

    roi_dict = json.load(f)

    f.close()

    roi_list = list()

    for roi_info in roi_dict.values():
        # getting coordinates of the roi
        x = int(roi_info["x"] * video_info.width)
        y = int(roi_info["y"] * video_info.height)
        width = int(roi_info["width"] * video_info.width)
        height = int(roi_info["height"] * video_info.height)

        # tl, tr, bl, br
        polygon = np.array([
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height]
        ])

        zone = sv.PolygonZone(polygon=polygon,
                              frame_resolution_wh=video_info.resolution_wh,
                              triggering_position=Position.CENTER)

        roi_list.append(zone)

    return roi_list


def checkIntersection(xyxyA, xyxyB):
    boxA = [xyxyA[0], xyxyA[1], xyxyA[2] - xyxyA[0], xyxyA[3] - xyxyA[1]]
    boxB = [xyxyB[0], xyxyB[1], xyxyB[2] - xyxyB[0], xyxyB[3] - xyxyB[1]]
    x = max(boxA[0], boxB[0])
    y = max(boxA[1], boxB[1])
    w = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - x
    h = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - y

    # iou = area overlap / area of union
    iou = 0

    foundIntersect = True
    if w < 0 or h < 0:
        foundIntersect = False

    if foundIntersect:
        overlap_area = w*h
        union_area = boxA[2]*boxA[3] + boxB[2]*boxB[3] - overlap_area
        iou = overlap_area / union_area

    return foundIntersect, iou


def init_person(id: int):
    """
    Initialization function for the person identified by the id passed by argument
    """
    print("New person detected with ID:", id)
    person = dict()

    # id
    person["id"] = id

    # gender
    person["gender"] = None
    person["gender_preds"] = list()

    # bag
    person["bag"] = None
    person["bag_prob"] = 0

    # hat
    person["hat"] = None
    person["hat_preds"] = list()

    # upper color
    person["upper_color"] = None
    person["upper_color_preds"] = list()

    # lower color
    person["lower_color"] = None
    person["lower_color_preds"] = list()

    # rois information
    person["roi1_passages"] = 0
    person["roi1_persistence_time"] = 0
    person["roi1_current_state"] = False
    person["roi1_last_state"] = False
    person["roi1_exit_count"] = 0
    person["roi2_passages"] = 0
    person["roi2_persistence_time"] = 0
    person["roi2_current_state"] = False
    person["roi2_last_state"] = False
    person["roi2_exit_count"] = 0

    # inference information
    person["in_scene_count"] = 0
    person["first_inference"] = False
    person["last_frame"] = None
    person["overlap"] = False

    return person


def update_person_info(person: dict, infer_results):
    """
    Updates the attributed of the person given the results of the models used for recognize
    the various attributes.
    For gender, upper_color and lower_color attributes, it updates the
    latest stored value in the person dictionary only if a value was not set or if the confidence of the new attribute
    is higher than the one stored before.
    For bag and hat attributes, it updates the latest stored value if a value was not set or if for that prediction
    the new value is True.
    """
    # ----------- updating gender value -----------
    person["gender_preds"].append(infer_results["gender"].lower())
    gender_counter = Counter(person["gender_preds"])
    person["gender"] = gender_counter.most_common(1)[0][0]

    # ----------- updating bag value -----------
    if person.get("bag") is None or person.get("bag") is False:
        person["bag"] = True if infer_results["bag"] == "yes" else False

    # ----------- updating hat value -----------
    person["hat_preds"].append(True if infer_results["hat"] == "yes" else False)
    hat_counter = Counter(person["hat_preds"])
    person["hat"] = hat_counter.most_common(1)[0][0]

    # ----------- updating upper color value -----------
    person["upper_color_preds"].append(infer_results["upper_color"])
    upper_color_counter = Counter(person["upper_color_preds"])
    person["upper_color"] = upper_color_counter.most_common(1)[0][0]

    # ----------- updating lower color value -----------
    person["lower_color_preds"].append(infer_results["lower_color"])
    lower_color_counter = Counter(person["lower_color_preds"])
    person["lower_color"] = lower_color_counter.most_common(1)[0][0]


def is_person_in_roi(xyxy, roi) -> bool:
    """
    Function that returns True if the person (identified by the bounding box with coordinates xyxy) is in the
    roi passed as argument, otherwise it returns False.
    """
    center_x = int((xyxy[0] + xyxy[2]) / 2)
    center_y = int((xyxy[1] + xyxy[3]) / 2)

    is_in_roi = roi.polygon[0][0] <= center_x <= roi.polygon[2][0] and roi.polygon[0][1] <= center_y <= roi.polygon[2][
        1]

    return is_in_roi


def save_output(output_filename: str, people_dict: dict):
    """
    Saves people detected during the video in a .json format file.
    """
    print(f"Saving results to {output_filename}...", end=" ")

    output_dict = dict()
    output_dict["people"] = list()

    # attributes to maintain for the output
    output_keys = ["id",
                   "gender",
                   "bag",
                   "hat",
                   "upper_color",
                   "lower_color",
                   "roi1_passages",
                   "roi1_persistence_time",
                   "roi2_passages",
                   "roi2_persistence_time"
                   ]

    for value in people_dict.values():
        for key in value.copy().keys():
            if key not in output_keys:
                value.pop(key)

        # casting persistence time to integer value
        value["roi1_persistence_time"] = int(value["roi1_persistence_time"])
        value["roi2_persistence_time"] = int(value["roi2_persistence_time"])

        output_dict["people"].append(value)

    with open(output_filename, 'w') as fp:
        json.dump(output_dict, fp, indent=4, cls=NpEncoder)

    print("done")
    pass


def get_dets(results):
    """
    Function that converts the results obtained from the detector to the format
    needed by the tracker object.
    """
    num_predictions = len(results[0])
    dets = np.zeros([num_predictions, 6], dtype=np.float32)
    for ind, object_prediction in enumerate(results[0].cpu()):
        dets[ind, :4] = np.array(object_prediction.boxes.xyxy, dtype=np.float32)
        dets[ind, 4] = object_prediction.boxes.conf
        dets[ind, 5] = object_prediction.boxes.cls

    return dets


def adjust_gamma(image, gamma=1.0):
    """
    This function takes in input an image and applies on it a gamma correction with gamma
    passed as parameter, then returns the modified image.
    """
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def parse_opt():
    """
    Parsing function useful to get needed paths from command line.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", dest="input_video_path")
    parser.add_argument("--configuration", dest="configuration_path")
    parser.add_argument("--results", dest="output_json_path")

    opt = parser.parse_args()
    return opt
