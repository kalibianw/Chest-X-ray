from utils import DataModule
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2

BATCH_SIZE = 32

dm = DataModule(batch_size=BATCH_SIZE, shuffle=True)
nploader = np.load("splited_Pneumonia_all_true_false_split_1_(300, 300).npz")
train_x_data = nploader["train_x_data"] / 255.0
train_x_data = np.expand_dims(train_x_data, axis=1)
print(np.shape(train_x_data))
train_y_data = nploader["train_y_data"].astype(np.long)
categorical_train_y_data = to_categorical(train_y_data)

for x_data, y_data in zip(train_x_data, train_y_data):
    print(x_data)
    print(x_data.max(), x_data.min())
    print(y_data)
    print(categorical_train_y_data[0, :])
    print(np.shape(x_data))
    print(type(x_data[0, 0, 0]))
    print(np.shape(y_data))
    print(type(y_data))
    print(np.shape(categorical_train_y_data))
    print(type(categorical_train_y_data))
    exit()
