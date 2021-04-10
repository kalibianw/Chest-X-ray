from utils import DataModule
import numpy as np
import cv2

BATCH_SIZE = 32

dm = DataModule(batch_size=BATCH_SIZE, shuffle=True)
nploader = np.load("splited_Pneumonia_all_true_false_split_1_(300, 300).npz")
train_x_data = nploader["train_x_data"]
train_x_data = np.expand_dims(train_x_data, axis=1)
print(np.shape(train_x_data))

for x_data in train_x_data:
    print(np.shape(x_data))
    test_arr = np.transpose(x_data, (1, 2, 0))
    print(np.shape(test_arr))
    cv2.imshow("test", test_arr)
    cv2.waitKey(0)
    exit()
