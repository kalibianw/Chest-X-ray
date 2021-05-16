from utils import DataModule

import numpy as np
import cv2
import os

dm = DataModule("None")

# ARC or ASA file name(s)
FILE_NAMES = ["splited_Pneumonia_all"]
ARC_ASA = "ASA"
TARGET_WIDTH = 200
TARGET_HEIGHT = 200
# 0: train_test_unsplit, 1: train_test_split, 2: true_false_split
ORIGINAL_FILE_STATUS = 2
AUGMENTED = True
ZERO_PADDING = True

for file_name_count, file_name in enumerate(FILE_NAMES):
    os.system("cls")

    if ORIGINAL_FILE_STATUS == 0:
        nploader = np.load(f"{ARC_ASA}/{file_name}.npz")
        dataset_dir_path = "D:/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"

        img_paths, label = nploader["img_path"], nploader["label"]

        label = np.array(label, dtype=np.int)

        x_data = list()
        y_data = list()

        per = len(img_paths) / 100
        for i, path in enumerate(img_paths):
            img_path = dataset_dir_path + "all/" + path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = dm.image_resize(img, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT, zero_padding=ZERO_PADDING)
            x_data.append(img)
            y_data.append(label[i])
            if AUGMENTED is True:
                flipped_img = cv2.flip(img, flipCode=1)
                x_data.append(flipped_img)
                y_data.append(label[i])
            if i % int(per) == 0:
                print(f"{file_name_count}번째 {file_name} 작업 {i / per}% 완료")

        print(np.shape(x_data), np.shape(y_data))

        np.savez_compressed(f"APP/{file_name}_{int(AUGMENTED)}_{int(ZERO_PADDING)}_({TARGET_HEIGHT}, {TARGET_WIDTH}).npz",
                            x_data=x_data,
                            y_data=y_data
                            )

    elif ORIGINAL_FILE_STATUS == 1:
        nploader = np.load(f"{ARC_ASA}/{file_name[file_name_count]}.npz")

        dataset_dir_path = "D:/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"

        train_img_path, train_label = nploader["train_img_path"], nploader["train_label"]
        test_img_path, test_label = nploader["test_img_path"], nploader["test_label"]

        train_label = np.array(train_label, dtype=np.float)
        test_label = np.array(test_label, dtype=np.float)

        x_train = list()
        x_test = list()
        y_train = list()
        y_test = list()

        for train_img_number, train_path in enumerate(train_img_path):
            per = len(train_img_path) / 100
            img_path = dataset_dir_path + train_path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = dm.image_resize(img, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT, zero_padding=ZERO_PADDING)
            x_train.append(img)
            y_train.append(train_label[train_img_number])
            if AUGMENTED is True:
                flipped_img = cv2.flip(img, flipCode=1)
                x_train.append(flipped_img)
                y_train.append(train_label[train_img_number])

            if train_img_number % int(per) == 0:
                print(f"{file_name_count}번째 train 작업 {train_img_number / per}% 완료")

        for test_img_number, test_path in enumerate(test_img_path):
            per = len(test_img_path) / 100
            img_path = dataset_dir_path + test_path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = dm.image_resize(img, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT, zero_padding=ZERO_PADDING)
            x_test.append(img)
            y_test.append(test_label[test_img_number])
            if AUGMENTED is True:
                flipped_img = cv2.flip(img, flipCode=1)
                x_test.append(flipped_img)
                y_test.append(test_label[test_img_number])

            if test_img_number % int(per) == 0:
                print(f"{file_name_count}번째 test 작업 {test_img_number / per}% 완료")

        print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))
        np.savez_compressed(f"APP/splited_{file_name[file_name_count]}_{int(AUGMENTED)}_{int(ZERO_PADDING)}_({TARGET_HEIGHT}, {TARGET_HEIGHT}).npz",
                            x_train=x_train,
                            x_test=x_test,
                            y_train=y_train,
                            y_test=y_test
                            )

    elif ORIGINAL_FILE_STATUS == 2:
        nploader = np.load(f"{ARC_ASA}/{file_name}.npz")
        dataset_dir_path = "D:/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"

        train_img_paths, train_labels = nploader["train_img_paths"], nploader["train_labels"]
        valid_img_paths, valid_labels = nploader["valid_img_paths"], nploader["valid_labels"]
        test_img_paths, test_labels = nploader["test_img_paths"], nploader["test_labels"]

        train_x_data, train_y_data = list(), list()
        valid_x_data, valid_y_data = list(), list()
        test_x_data, test_y_data = list(), list()

        per = len(train_img_paths) / 100
        for i, path in enumerate(train_img_paths):
            img_path = dataset_dir_path + "all/" + path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = dm.image_resize(img, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT, zero_padding=ZERO_PADDING)
            train_x_data.append(img)
            train_y_data.append(train_labels[i])
            if AUGMENTED is True:
                flipped_img = cv2.flip(img, flipCode=1)
                train_x_data.append(flipped_img)
                train_y_data.append(train_labels[i])
            if i % int(per) == 0:
                print(f"{file_name_count}번째 {file_name} train 작업 {i / per}% 완료")
        print(np.shape(train_x_data), np.shape(train_y_data))

        per = len(valid_img_paths) / 100
        for i, path in enumerate(valid_img_paths):
            img_path = dataset_dir_path + "all/" + path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = dm.image_resize(img, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT, zero_padding=ZERO_PADDING)
            valid_x_data.append(img)
            valid_y_data.append(valid_labels[i])
            if i % int(per) == 0:
                print(f"{file_name_count}번째 {file_name} valid 작업 {i / per}% 완료")
        print(np.shape(valid_x_data), np.shape(valid_y_data))

        per = len(test_img_paths) / 100
        for i, path in enumerate(test_img_paths):
            img_path = dataset_dir_path + "all/" + path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = dm.image_resize(img, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT, zero_padding=ZERO_PADDING)
            test_x_data.append(img)
            test_y_data.append(test_labels[i])
            if i % int(per) == 0:
                print(f"{file_name_count}번째 {file_name} test 작업 {i / per}% 완료")
        print(np.shape(test_x_data), np.shape(test_y_data))

        print(np.shape(train_x_data), np.shape(train_y_data))
        print(np.shape(valid_x_data), np.shape(valid_y_data))
        print(np.shape(test_x_data), np.shape(test_y_data))

        np.savez_compressed(
            f"APP/{file_name}_true_false_split_{int(AUGMENTED)}_{int(ZERO_PADDING)}_({TARGET_HEIGHT}, {TARGET_WIDTH}).npz",
            train_x_data=train_x_data,
            train_y_data=train_y_data,
            valid_x_data=valid_x_data,
            valid_y_data=valid_y_data,
            test_x_data=test_x_data,
            test_y_data=test_y_data
        )
