import numpy as np
import cv2

nploader = np.load("ASA/splited_Pneumonia_all.npz")
for key in nploader:
    print(key)

width = 500
height = 400

train_img_paths, valid_img_paths, test_img_paths = nploader["train_img_paths"], nploader["valid_img_paths"], nploader["test_img_paths"]
train_labels, valid_labels, test_labels = nploader["train_labels"], nploader["valid_labels"], nploader["test_labels"]
dir_path = "C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/all/"

train_x_data = list()
train_y_data = list()
per = len(train_img_paths) / 100
for i, img_path in enumerate(train_img_paths):
    img = cv2.imread(dir_path + img_path, flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(width, height))

    train_x_data.append(img)
    train_y_data.append(train_labels[i])

    img_ = cv2.flip(img, flipCode=1)
    train_x_data.append(img_)
    train_y_data.append(train_labels[i])
    if i % int(per) == 0:
        print(f"train 작업 {i/per}% 완료")
print(np.shape(train_x_data), np.shape(train_y_data))

valid_x_data = list()
valid_y_data = list()
per = len(valid_img_paths) / 100
for i, img_path in enumerate(valid_img_paths):
    img = cv2.imread(dir_path + img_path, flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(width, height))

    valid_x_data.append(img)
    valid_y_data.append(valid_labels[i])
    if i % int(per) == 0:
        print(f"valid 작업 {i/per}% 완료")
print(np.shape(valid_x_data), np.shape(valid_y_data))

test_x_data = list()
test_y_data = list()
per = len(test_img_paths) / 100
for i, img_path in enumerate(test_img_paths):
    img = cv2.imread(dir_path + img_path, flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(width, height))

    test_x_data.append(img)
    test_y_data.append(test_labels[i])
    if i % int(per) == 0:
        print(f"test 작업 {i/per}% 완료")
print(np.shape(test_x_data), np.shape(test_y_data))

np.savez_compressed(
    file=f"APP/no_padding_splited_Pneumonia_all_true_false_split_1_({height}, {width}).npz",
    train_x_data=train_x_data, train_y_data=train_y_data,
    valid_x_data=valid_x_data, valid_y_data=valid_y_data,
    test_x_data=test_x_data, test_y_data=test_y_data
)
