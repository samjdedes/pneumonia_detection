import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split


def get_img_paths(base_dir, ignore_filetypes=['.txt']):
    """
    Inputs:
        base_dir (string/ os.path type) relative path to the directory with folders containing images
        *kwargs
        ignore_filetypes (list of strings) filetypes to exclude from output
        type_foldes (dictionary) folders that identify different filetypes and add identifiers to add to output
    Returns:
        paths (list of tuples) relative paths to files with identifiers
    """
    normal_paths = []
    pneumonia_paths = []

    for root, dirs, files in os.walk(base_dir):
        # remove undesired files
        if ignore_filetypes:
            # check each file
            for file in files:
                for file_type in ignore_filetypes:
                    if file_type in file:
                        files.remove(file)
        if files:
            for file in files:
                full_path = os.path.join(root, file)

                if 'NORMAL' in full_path:
                    normal_paths.append((full_path, 0))

                elif 'PNEUMONIA' in full_path:
                    pneumonia_paths.append((full_path, 1))

    #         if type_folders:
    #             for key, value in type_folders.items():
    #                 if key in root:

    #         print('root: ', root)
    #         print('files: ', files[:5], len(files))
    #         print('paths: ', paths)
    normal_paths.extend(pneumonia_paths)
    return normal_paths


def custom_tts(data, train_val_test_percents=(0.8, 0.025, 0.175), random_state=2021):
    """
    This function takes a tuple of data and returns a tuple of train, test, and validation data
    Input:
        data (list/ array like) each entry contians a file path in position [0] and classification in position [1]
        train_val_test_percents (triple) triple containing decimal representations of split percentages used in train test split
        *kwargs

    Returns:
        x_train
        x_val
        x_test

    """
    # Extract percentages used in split from triple
    train_pct = train_val_test_percents[0]
    test_pct = 1 - train_pct
    val_pct = train_val_test_percents[0] / test_pct
    paths = [d[0] for d in data]
    types = [t[1] for t in data]
    x_train, x_test, y_train, y_test = train_test_split(paths, types, train_size=train_pct, random_state=random_state)

    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=train_pct, random_state=random_state)

    return x_train, x_val, x_test, y_train, y_val, y_test


def get_data(x, y, img_size=150):
    """
    This funciton takes an image type and classification with matching index and returns  image data with classification

    Input:
        x (list of path like entries) paths leading to image data to be loaded
        y (list of classification) list of classifications matching index of x data

    Returns:
        data (np.array) image data with classification
    """


    data = []

    for i in range(len(x)):
        path = x[i]
        class_num = y[i]

        #         for img in os.listdir(path):

        try:
            img_arr = cv2.imread(path)
            resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
            data.append([resized_arr, class_num])

        except Exception as e:
            print(e)

    return np.array(data)