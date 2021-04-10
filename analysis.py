from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
import pandas as pd
import numpy as np
import os

nploaders = [
    np.load("APP/splited_Pneumonia_all_true_false_split_1_(400, 500).npz"),
    np.load("APP/splited_Pneumonia_all_true_false_split_1_(300, 375).npz"),
    np.load("APP/splited_Pneumonia_all_true_false_split_1_(200, 250).npz"),
    np.load("APP/splited_Pneumonia_all_true_false_split_1_(100, 125).npz"),
    np.load("APP/no_padding_splited_Pneumonia_all_true_false_split_1_(400, 500).npz"),
    np.load("APP/no_padding_splited_Pneumonia_all_true_false_split_1_(300, 375).npz")
    ]

model_paths = [
    ["C:/Users/admin/Documents/AI/model/coronahack/splited_Pneumonia_(400, 500)_TF_0.h5", "", 0],
    ["C:/Users/admin/Documents/AI/model/coronahack/no_padding_splited_Pneumonia_(400, 500)_TF_0.h5", "", 4],
    ["C:/Users/admin/Documents/AI/model/coronahack/splited_Pneumonia_(300, 375)_TF_0.h5", "", 1],
    ["C:/Users/admin/Documents/AI/model/coronahack/no_padding_splited_Pneumonia_(300, 375)_TF_0.h5", "", 5]
]

df = pd.DataFrame(
    columns=[
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "recall",
        "precision",
        "f1_score",
        "sensitivity",
        "specificity"
    ]
)

for count, model_path in enumerate(model_paths):
    test_x_data, test_y_data = np.expand_dims(nploaders[model_path[2]]["test_x_data"], axis=-1), \
                               to_categorical(nploaders[model_path[2]]["test_y_data"])
    print(f"shape of test_x_data: {np.shape(test_x_data)}\n"
          f"shape of test_y_data: {np.shape(test_y_data)}")
    normalized_x_test = test_x_data.astype(dtype="float16") / 255.0
    model = models.load_model(model_path[0])
    if model_path[1] == "":
        predict_result = model.predict(x=test_x_data, verbose=1)
    else:
        predict_result = model.predict(x=normalized_x_test, verbose=1)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(predict_result)):
        true_label = np.argmax(test_y_data[i])
        predict_label = np.argmax(predict_result[i])

        if predict_label == 1:
            if true_label == 1:
                true_positive += 1
            else:
                false_positive += 1

        else:
            if true_label == 0:
                true_negative += 1
            else:
                false_negative += 1

    print(f"True Positive = {true_positive}\n"
          f"True Negative = {true_negative}\n"
          f"False Positive = {false_positive}\n"
          f"False Negative = {false_negative}\n")

    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)

    print(f"Recall: {recall}\n"
          f"Precision: {precision}\n"
          f"F1 Score: {2 * ((precision * recall) / (precision + recall))}\n"
          f"Sensitivity: {true_positive / (true_positive + false_negative)}\n"
          f"Specificity: {true_negative / (true_negative + false_positive)}")

    new_df = pd.DataFrame(
        index=[os.path.basename(model_path[0])],
        data={
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "recall": recall,
            "precision": precision,
            "f1_score": 2 * ((precision * recall) / (precision + recall)),
            "sensitivity": true_positive / (true_positive + false_negative),
            "specificity": true_negative / (true_negative + false_positive)
        }
    )

    df = df.append(new_df)

if len(model_paths) == 1:
    df.to_csv(f"analysis report for {os.path.basename(model_paths[0][0])}.csv")
    exit()

for i in range(1, 1000):
    if os.path.exists(f"analysis report_{i}.csv"):
        continue
    df.to_csv(f"analysis report_{i}.csv")
    exit()
