from tensorflow.keras import models
import numpy as np
import cv2
import os

model_path = "C:/Users/admin/Documents/AI/model/coronahack/splited_Pneumonia_(400, 500)_TF_0.h5"
model = models.load_model(model_path)
model.summary()

img_paths = [
    # False:    2_IM-0010-0001.jpeg
    "img/False.jpeg",
    # True:     2_1-s2.0-S1684118220300608-main.pdf-001.jpg
    "img/True.jpg"
]

img_false = cv2.imread(img_paths[0])
img_true = cv2.imread(img_paths[1])

cv2.imshow("False", mat=img_false)
cv2.imshow("True", mat=img_true)
cv2.waitKey(0)

cv2.destroyAllWindows()

img_True = np.expand_dims(np.expand_dims(cv2.imread(img_paths[0], flags=cv2.IMREAD_GRAYSCALE), axis=-1), axis=0)
img_False = np.expand_dims(np.expand_dims(cv2.imread(img_paths[1], flags=cv2.IMREAD_GRAYSCALE), axis=-1), axis=0)

imgs = np.concatenate((img_True, img_False), axis=0)

results = model.predict(imgs)
print()
for i, result in enumerate(results):
    print(f"{os.path.basename(img_paths[i])} predicted result: {result} - {bool(np.argmax(result))}")
