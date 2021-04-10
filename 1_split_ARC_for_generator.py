from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import time
import os

ARC_files = ["COVID_all", "Pneumonia_all", "Virus_all", "Virus_bacteria_all"]

for i, file_name in enumerate(ARC_files):
    os.system("cls")
    nploader = np.load(f"ARC/{ARC_files[i]}.npz")

    for key in nploader:
        print(key)

    img_path, label = nploader["img_path"], nploader["label"]
    train_img_paths, test_img_paths, train_labels, test_labels = train_test_split(img_path, label, test_size=0.1, shuffle=True)
    train_img_paths_2 = list()
    test_img_paths_2 = list()

    for train_img_path in train_img_paths:
        if os.path.exists(f"C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train_2/{train_img_path}"):
            train_img_paths_2.append(f"train_2/{train_img_path}")
        else:
            shutil.copy(
                src=f"C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/all/{train_img_path}",
                dst=f"C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train_2/{train_img_path}"
            )
            train_img_paths_2.append(f"train_2/{train_img_path}")

    for test_img_path in test_img_paths:
        if os.path.exists(f"C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test_2/{test_img_path}"):
            test_img_paths_2.append(f"test_2/{test_img_path}")
        else:
            shutil.copy(
                src=f"C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/all/{test_img_path}",
                dst=f"C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test_2/{test_img_path}"
            )
            test_img_paths_2.append(f"test_2/{test_img_path}")

    print(np.shape(train_img_paths_2), np.shape(test_img_paths_2), np.shape(train_labels), np.shape(test_labels))
    time.sleep(3)
    np.savez_compressed(f"ASA/{ARC_files[i]}.npz",
                        train_img_path=train_img_paths_2,
                        test_img_path=test_img_paths_2,
                        train_label=train_labels,
                        test_label=test_labels
                        )
