import os
import pandas as pd
import numpy as np
import scipy
import json, argparse
import pickle
from utils import getProjectRoot


def nan_index(df):
    """
        Sets specific column ranges to -1 for rows marked by NaN indicator columns ('upper_-1', 'lower_-1', 'gender_-1'),
        then drops these indicator columns and returns the updated DataFrame.
    """
    df_nan_upp = df.loc[df['upper_-1'] == True]
    df_nan_low = df.loc[df['lower_-1'] == True]
    df_nan_gender = df.loc[df['gender_-1'] == True]

    df.loc[df_nan_upp.index, 'upper_1':'upper_11'] = -1
    df.loc[df_nan_low.index, 'lower_1':'lower_11'] = -1
    df.loc[df_nan_gender.index, 'gender_0':'gender_1'] = -1

    df.drop(columns=['upper_-1', 'lower_-1', 'gender_-1'], inplace=True)

    return df


def conversion(filename, type="train", without_nan=False):
    """
        Processes a CSV file by filtering rows (if `without_nan` is True), converting columns to categorical types,
        applying one-hot encoding, and handling NaN indicators based on specified parameters.
        Returns the final DataFrame.
    """
    df = pd.read_csv(filename, header=None, names=['filename', 'upper', 'lower', 'gender', 'bag', 'hat'])

    if without_nan:
        all_values = df.loc[(df['upper'] != -1) & (df['lower'] != -1) & (df['gender'] != -1) & (df['bag'] != -1) & (df['hat'] != -1)]
        df = all_values.copy()
        print("without nan")

    df['upper'] = df['upper'].astype('category')
    df['lower'] = df['lower'].astype('category')
    df['gender'] = df['gender'].astype('category')

    one_hot_encoded_color = pd.get_dummies(df, columns=['upper', 'lower', 'gender'])
    print(one_hot_encoded_color.head())
    one_hot_encoded_color = nan_index(one_hot_encoded_color) if without_nan is False and type=="train" else one_hot_encoded_color

    one_hot_encoded_color.replace(True, 1, inplace=True)
    one_hot_encoded_color.replace(False, 0, inplace=True)

    if without_nan is False and type=="train":
        one_hot_encoded_color.replace(-1, 0, inplace=True)

    return one_hot_encoded_color


def create_file_csv(dataframe, name_columns):
    """
        Prepares arrays of attribute names, label values, and image filenames for CSV export.
    """

    color = ["black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"]

    up_color = ["up" + c for c in color]
    lower_color = ["lo" + c for c in color]
    attributes = ["bag", "hat"] + up_color + lower_color + ["male", "female"]

    attributes = pd.DataFrame(attributes)
    attributes = np.array(attributes[0]).reshape(len(attributes), 1)

    labels = np.array(dataframe[name_columns])

    images_name = np.array(dataframe['filename']).reshape(1, len(dataframe))
    images_name = images_name.transpose()

    return attributes, labels, images_name

def create_dict_data(train, val):
    dict_data = dict()
    for i in train['filename']:
        dict_data[i] = (0, 'front')

    for i in val['filename']:
        dict_data[i] = (0, 'front')

    return dict_data

def main():
    """
        0 -> male
        1 -> female
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_nan", default=False)
    parser.add_argument("--data", default=True)
    args = parser.parse_args()

    path = os.path.join(getProjectRoot(), "dataset/annotations")
    path_data = os.path.join(getProjectRoot(), "dataset/data")

    ### Training set
    df_train = conversion(f"{path}/training_set_fixed.csv", without_nan=args.no_nan)

    columns = df_train.columns[1:]

    attributes, train_label, images_name_train = create_file_csv(df_train, columns)

    ### Validation  and Test
    df_validation = conversion(f"{path}/validation_set.csv", type="val", without_nan=args.no_nan)

    if args.data:
        dataset = create_dict_data(df_train, df_validation)
        outfile_dict = 'dict_data_mivia.pkl'
        if args.no_nan:
            outfile_dict = 'dict_data_mivia_no_nan.pkl'
        with open(os.path.join(path, outfile_dict), 'wb+') as f:
            pickle.dump(dataset, f)

    if args.no_nan:
        ### added image to train
        df_added_training = df_validation.sample(n=4000, random_state=4000)
        _, added_train_label, added_images_name_train = create_file_csv(df_added_training, columns)

        train_label = np.concatenate((train_label, added_train_label), axis=0)
        images_name_train = np.concatenate((images_name_train, added_images_name_train), axis=0)

        df_validation = df_validation.drop(df_added_training.index)

    df_test = df_validation.sample(frac=0.3, random_state=200)
    df_validation = df_validation.drop(df_test.index)
    _, val_label, images_name_val = create_file_csv(df_validation, columns)
    print(val_label.shape)
    print(images_name_val.shape)

    _, test_label, images_name_test = create_file_csv(df_test, columns)

    data = {'attributes': attributes,
            'test_images_name': images_name_test,
            'test_label': test_label,
            'train_images_name': images_name_train,
            'train_label': train_label,
            'val_images_name': images_name_val,
            'val_label': val_label
            }

    data_shapes = {
        'attributes': attributes.shape,
        'test_images_name': images_name_test.shape,
        'test_label': test_label.shape,
        'train_images_name': images_name_train.shape,
        'train_label': train_label.shape,
        'val_images_name': images_name_val.shape,
        'val_label': val_label.shape
    }

    json_file = ""

    if args.no_nan:
        json_file = f"{path}/size_dataset_no_nan.json"
    else:
        json_file = f"{path}/size_dataset.json"

    annotations = ['training', 'validation', 'test']

    for element in annotations:
        outfile = ""
        if args.no_nan:
            outfile = os.path.join(path, f"annotation_{element}.csv")
        else:
            outfile = os.path.join(path, f"annotation_fixed_{element}.csv")

        with open(outfile, 'w') as f:
            if element == 'training':
                df_train.to_csv(f, index=False)
            if element == 'validation':
                df_validation.to_csv(f, index=False)
            else:
                df_test.to_csv(f, index=False)

    with open(json_file, 'w') as output:
        json.dump(data_shapes, output, indent=4)


if __name__ == '__main__':
    main()

