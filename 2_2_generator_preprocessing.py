import numpy as np
import cv2
import os

from utils import DataModule

test_mod = False

dm = DataModule("None")

file_names = ["Pneumonia_all"]

for file_name_count, file_name in enumerate(file_names):
    os.system("cls")

    splited = True
    augmented = True

    if splited is True:
        nploader = np.load(f"ASA/{file_name}.npz")

        dataset_dir_path = "C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"

        train_img_paths, test_img_paths, train_labels, test_labels = nploader["train_img_path"], nploader["test_img_path"], \
                                                                     nploader["train_label"], nploader["test_label"]

        train_labels, test_labels = np.array(train_labels, dtype=np.int), np.array(test_labels, dtype=np.int)

        per = len(train_img_paths) / 100
        for i, path in enumerate(train_img_paths):
            img_path = dataset_dir_path + path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = dm.image_resize(img)

            if os.path.exists(dataset_dir_path + "for_generator/splited_" + file_name + "/train/" + str(train_labels[i])) is False:
                os.makedirs(dataset_dir_path + "for_generator/splited_" + file_name + "/train/" + str(train_labels[i]))

            if augmented is True:
                flipped_img = cv2.flip(img, flipCode=1)
                flipped_img = np.expand_dims(flipped_img, axis=-1)
                if test_mod is True:
                    print("file write to", dataset_dir_path + "for_generator/splited_" + file_name + "/train/" + str(train_labels[i]) + "/2_" + os.path.basename(path))
                    cv2.imshow("flipped_img", mat=flipped_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                cv2.imwrite((dataset_dir_path + "for_generator/splited_" + file_name + "/train/" + str(train_labels[i]) + "/2_" + os.path.basename(path)), img=flipped_img)

            img = np.expand_dims(img, axis=-1)
            if test_mod is True:
                print("file write to", dataset_dir_path + "for_generator/splited_" + file_name + "/train/" + str(train_labels[i]) + "/" + os.path.basename(path))
                cv2.imshow("img", mat=img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                exit()

            cv2.imwrite((dataset_dir_path + "for_generator/splited_" + file_name + "/train/" + str(train_labels[i]) + "/" + os.path.basename(path)), img=img)

            if i % int(per) == 0:
                print(f"{file_name_count}번째 train 작업 {i / per}% 완료")

        per = len(test_img_paths) / 100
        for i, path in enumerate(test_img_paths):
            img_path = dataset_dir_path + path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = dm.image_resize(img)

            if os.path.exists(dataset_dir_path + "for_generator/splited_" + file_name + "/test/" + str(test_labels[i])) is False:
                os.makedirs(dataset_dir_path + "for_generator/splited_" + file_name + "/test/" + str(test_labels[i]))

            if augmented is True:
                flipped_img = cv2.flip(img, flipCode=1)
                flipped_img = np.expand_dims(flipped_img, axis=-1)
                cv2.imwrite((dataset_dir_path + "for_generator/splited_" + file_name + "/test/" + str(test_labels[i]) + "/2_" + os.path.basename(path)), img=flipped_img)

            img = np.expand_dims(img, axis=-1)
            cv2.imwrite((dataset_dir_path + "for_generator/splited_" + file_name + "/test/" + str(test_labels[i]) + "/" + os.path.basename(path)), img=img)

            if i % int(per) == 0:
                print(f"{file_name_count}번째 test 작업 {i / per}% 완료")

    else:
        nploader = np.load(f"ARC/{file_name}.npz")

        dataset_dir_path = "C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"

        img_path, label = nploader["img_path"], nploader["label"]

        label = np.array(label, dtype=np.int)

        per = len(img_path) / 100
        for i, path in enumerate(img_path):
            img_path = dataset_dir_path + "all/" + path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = dm.image_resize(img)

            if os.path.exists(dataset_dir_path + "for_generator/" + file_name + "/" + str(label[i])) is False:
                os.makedirs(dataset_dir_path + "for_generator/" + file_name + "/" + str(label[i]))

            if augmented is True:
                flipped_img = cv2.flip(img, flipCode=1)
                flipped_img = np.expand_dims(flipped_img, axis=-1)
                cv2.imwrite((dataset_dir_path + "for_generator/" + file_name + "/" + str(label[i]) + "/2_" + path), img=flipped_img)

            img = np.expand_dims(img, axis=-1)
            cv2.imwrite((dataset_dir_path + "for_generator/" + file_name + "/" + str(label[i]) + "/" + path), img=img)

            if i % int(per) == 0:
                print(f"{file_name_count}번째 작업 {i / per}% 완료")
