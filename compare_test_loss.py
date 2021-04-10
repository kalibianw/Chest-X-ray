from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

nploader = np.load("APP/splited_Pneumonia_all_augmentation_1_(400, 500).npz")
for key in nploader:
    print(key)

x_test, y_test = np.expand_dims(nploader["x_test"], axis=-1), to_categorical(nploader["y_test"])

normalized_x_test = np.array(x_test, dtype="float16")
normalized_x_test = normalized_x_test / 255.0

os.system("cls")

print("splited_Pneumonia_(400, 500)_1.h5")
model = models.load_model(
    filepath="C:/Users/admin/Documents/AI/model/coronahack/splited_Pneumonia_(400, 500)_1.h5"
)

model.evaluate(
    x=x_test, y=y_test,
    verbose=2
)
print("----------------------------------------------------------------")

print("generator_normalized_splited_Pneumonia_(400, 500).h5")
model = models.load_model(
    filepath="C:/Users/admin/Documents/AI/model/coronahack/generator_normalized_splited_Pneumonia_(400, 500).h5"
)

model.evaluate(
    x=normalized_x_test, y=y_test,
    verbose=2
)
print("----------------------------------------------------------------")

print("normalized_splited_Pneumonia_(400, 500)_1.h5")
model = models.load_model(
    filepath="C:/Users/admin/Documents/AI/model/coronahack/normalized_splited_Pneumonia_(400, 500)_1.h5"
)

model.evaluate(
    x=normalized_x_test, y=y_test,
    verbose=2
)
print("----------------------------------------------------------------")

print("normalized_splited_Pneumonia_(400, 500)_14_1.h5")
model = models.load_model(
    filepath="C:/Users/admin/Documents/AI/model/coronahack/normalized_splited_Pneumonia_(400, 500)_14_1.h5"
)

model.evaluate(
    x=normalized_x_test, y=y_test,
    verbose=2
)
print("----------------------------------------------------------------")

print("normalized_splited_Pneumonia_(400, 500)_12_1.h5")
model = models.load_model(
    filepath="C:/Users/admin/Documents/AI/model/coronahack/normalized_splited_Pneumonia_(400, 500)_12_1.h5"
)

model.evaluate(
    x=normalized_x_test, y=y_test,
    verbose=2
)
print("----------------------------------------------------------------")

print("normalized_splited_Pneumonia_(400, 500)_8_1.h5")
model = models.load_model(
    filepath="C:/Users/admin/Documents/AI/model/coronahack/normalized_splited_Pneumonia_(400, 500)_8_1.h5"
)

model.evaluate(
    x=normalized_x_test, y=y_test,
    verbose=2
)
print("----------------------------------------------------------------")

print("normalized_splited_Pneumonia_(400, 500)_6_1.h5")
model = models.load_model(
    filepath="C:/Users/admin/Documents/AI/model/coronahack/normalized_splited_Pneumonia_(400, 500)_6_1.h5"
)

model.evaluate(
    x=normalized_x_test, y=y_test,
    verbose=2
)
print("----------------------------------------------------------------")

print("normalized_splited_Pneumonia_(400, 500)_4_1.h5")
model = models.load_model(
    filepath="C:/Users/admin/Documents/AI/model/coronahack/normalized_splited_Pneumonia_(400, 500)_4_1.h5"
)

model.evaluate(
    x=normalized_x_test, y=y_test,
    verbose=2
)
print("----------------------------------------------------------------")

print("normalized_splited_Pneumonia_(400, 500)_2_1.h5")
model = models.load_model(
    filepath="C:/Users/admin/Documents/AI/model/coronahack/normalized_splited_Pneumonia_(400, 500)_2_1.h5"
)

model.evaluate(
    x=normalized_x_test, y=y_test,
    verbose=2
)
print("----------------------------------------------------------------")
