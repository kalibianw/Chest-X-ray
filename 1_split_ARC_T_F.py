from sklearn.model_selection import train_test_split
import numpy as np

nploader = np.load("ARC/Pneumonia_all.npz")
img_paths, labels = nploader["img_path"], nploader["label"]

true_img_paths, true_labels = np.array([]), np.array([])
false_img_paths, false_labels = np.array([]), np.array([])

per = len(labels) / 100
for i, label in enumerate(labels):
    if label == 0:
        false_img_paths = np.append(false_img_paths, img_paths[i])
        false_labels = np.append(false_labels, label)
    elif label == 1:
        true_img_paths = np.append(true_img_paths, img_paths[i])
        true_labels = np.append(true_labels, label)

    if i % int(per) == 0:
        print(f"{i / per}% 완료")

print(np.shape(false_img_paths), np.shape(false_labels), np.shape(true_img_paths), np.shape(true_labels))
false_train_img_paths, false_test_img_paths, false_train_labels, false_test_labels = train_test_split(false_img_paths, false_labels, test_size=0.4)
false_train_img_paths, false_valid_img_paths, false_train_labels, false_valid_labels = train_test_split(false_train_img_paths, false_train_labels, test_size=0.2)
true_train_img_paths, true_test_img_paths, true_train_labels, true_test_labels = train_test_split(true_img_paths, true_labels, test_size=0.4)
true_train_img_paths, true_valid_img_paths, true_train_labels, true_valid_labels = train_test_split(true_train_img_paths, true_train_labels, test_size=0.2)

train_img_paths = np.append(false_train_img_paths, true_train_img_paths, axis=0)
train_labels = np.append(false_train_labels, true_train_labels, axis=0)
valid_img_paths = np.append(false_valid_img_paths, true_valid_img_paths, axis=0)
valid_labels = np.append(false_valid_labels, true_valid_labels, axis=0)
test_img_paths = np.append(false_test_img_paths, true_test_img_paths, axis=0)
test_labels = np.append(false_test_labels, true_test_labels, axis=0)

print(
    np.shape(train_img_paths), np.shape(train_labels),
    np.shape(valid_img_paths), np.shape(valid_labels),
    np.shape(test_img_paths), np.shape(test_labels)
)

np.savez_compressed(
    "ASA/splited_Pneumonia_all.npz",
    train_img_paths=train_img_paths,
    train_labels=train_labels,
    valid_img_paths=valid_img_paths,
    valid_labels=valid_labels,
    test_img_paths=test_img_paths,
    test_labels=test_labels
)
