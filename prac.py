import numpy as np
import cv2

np_loader = np.load("generate_to_torch/splited_Pneumonia_all_true_false_split_1_0_(300, 300).npz")
for key in np_loader:
    print(key)

train_x_data = np_loader["train_x_data"]
train_y_data = np_loader["train_y_data"]
valid_x_data = np_loader["valid_x_data"]
valid_y_data = np_loader["valid_y_data"]
test_x_data = np_loader["test_x_data"]
test_y_data = np_loader["test_y_data"]

for x_i, x_data in enumerate(train_x_data[:3]):
    print(np.shape(x_data))
    cv2.imshow(f"test_{x_i}", x_data)

cv2.waitKey(0)
cv2.destroyAllWindows()
