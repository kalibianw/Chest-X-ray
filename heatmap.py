import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import os

from utils import DataModule

model_path = "C:/Users/admin/Documents/AI/model/coronahack/splited_Pneumonia_(400, 500)_TF_0.h5"
model = models.load_model(model_path)
model.summary()

dm = DataModule(None)

nploader = np.load("ASA/splited_Pneumonia_all.npz")
for key in nploader:
    print(key)

img_dir_path = "C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/all/"
test_img_paths, test_labels = nploader["test_img_paths"], nploader["test_labels"]

arr = [
    [True, "R"],
    [True, "R"],
    [True, "E"],
    [True, "R"],
    [True, "E"],

    [False, "R"],
    [False, "R"],
    [False, "R"],
    [False, "E"],
    [False, "E"],
]

arr = np.array(arr)
img_paths = np.append(test_img_paths[:5], test_img_paths[-5:], axis=0)

arr = np.append(arr, np.expand_dims(img_paths, axis=-1), axis=1)

for i, element in enumerate(arr):
    img = cv2.imread((img_dir_path + element[2]), flags=cv2.IMREAD_GRAYSCALE)

    x = image.img_to_array(img)
    x = dm.image_resize(x)
    x = np.expand_dims(np.expand_dims(x, axis=-1), axis=0)
    # print(f"shape of x: {np.shape(x)}")

    pred = model.predict(x)
    print(f"predict result: {pred}\t{np.argmax(pred)}")

    layer_names = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5"]
    width = 500
    height = 400
    for layer_name in layer_names:
        with tf.GradientTape() as tape:
            last_conv_layer = model.get_layer(layer_name)
            iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
            model_out, last_conv_layer = iterate(x)
            class_out = model_out[:, np.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            # print(f"shape of grads: {np.shape(grads)}")
            pooled_grads = K.mean(grads, axis=(0, 1, 2))

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        # print(f"max value of heatmap: {np.max(heatmap)}\nmin value of heatmap: {np.min(heatmap)}")
        heatmap /= np.max(heatmap)
        heatmap = heatmap.reshape((math.ceil(height), math.ceil(width)))
        # plt.matshow(heatmap)
        # plt.show()

        INTENSITY = 0.4
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        # print(f"shape of heatmap: {np.shape(heatmap)}")
        # print(f"shape of img: {np.shape(img)}")
        # print(
        #     f"max value of heatmap after resizing: {np.max(heatmap)}\nmin value of heatmap after resizing: {np.min(heatmap)}"
        # )

        img = cv2.imread((img_dir_path + element[2]))
        # print(f"shape of original img: {np.shape(img)}")
        # print(f"max value of img: {np.max(img)}\nmin value of img: {np.min(img)}")
        # cv2.imshow("Original image", img)

        img = heatmap * INTENSITY + img
        # print(f"shape of heatmap * INTENSITY + img: {np.shape(img)}")
        img = img.astype("uint8")
        # print(
        #     f"max value of img after plus with heatmap: {np.max(img)}\nmin value of img after plus with heatmap: {np.min(img)}"
        # )
        # print(np.mean(img))

        # cv2.imshow("Image with Heatmap", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if layer_name[-1].isdigit() is False:
            cv2.imwrite(f"Heatmap/{str(element[0])}_{element[2]}"
                        f"_{os.path.basename(model_path)}_{layer_name}_0's heatmap.png", img)
        else:
            cv2.imwrite(f"Heatmap/{str(element[0])}_{element[2]}"
                        f"_{os.path.basename(model_path)}_{layer_name}'s heatmap.png", img)

        width /= 2
        height /= 2
