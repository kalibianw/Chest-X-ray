import pandas as pd
import numpy as np

csv_path = "C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Chest_xray_Corona_Metadata.csv"

nploader = np.load("ASA/splited_Pneumonia_all.npz")
train_img_paths, train_labels = nploader["train_img_paths"], nploader["train_labels"]
valid_img_paths, valid_labels = nploader["valid_img_paths"], nploader["valid_labels"]
test_img_paths, test_labels = nploader["test_img_paths"], nploader["test_labels"]

print(train_img_paths)
print(train_labels)

csv = pd.read_csv(
    filepath_or_buffer=csv_path
)

# print(csv["Label"])

# new_df = pd.DataFrame(csv[["X_ray_image_name", "Label"]])
# print(new_df)

csv["Label"].to_numpy()

new_df = pd.DataFrame(data=csv["Label"].to_numpy(), index=csv["X_ray_image_name"], columns=["Label"])

print(new_df.loc["person1634_virus_2830.jpeg"]["Label"])

for i, train_img_path in enumerate(train_img_paths):
    result = new_df.loc[train_img_path]["Label"]
    if result == "Pnemonia":
        if train_labels[i] == 1.:
            print(f"Confirmed. {train_img_path}'s label: {train_labels[i]} is {result}")
            continue
        elif train_labels[i] == 1:
            print(f"Confirmed. {train_img_path}'s label: {train_labels[i]} is {result}")
            continue
        else:
            print("Error")
            exit()

    elif result == "Normal":
        if train_labels[i] == 0.:
            print(f"Confirmed. {train_img_path}'s label: {train_labels[i]} is {result}")
            continue
        elif train_labels[i] == 1:
            print(f"Confirmed. {train_img_path}'s label: {train_labels[i]} is {result}")
            continue
        else:
            print("Error")
            exit()

for i, valid_img_path in enumerate(valid_img_paths):
    result = new_df.loc[valid_img_path]["Label"]
    if result == "Pnemonia":
        if valid_labels[i] == 1.:
            print(f"Confirmed. {valid_img_path}'s label: {valid_labels[i]} is {result}")
            continue
        elif valid_labels[i] == 1:
            print(f"Confirmed. {valid_img_path}'s label: {valid_labels[i]} is {result}")
            continue
        else:
            print("Error")
            exit()

    elif result == "Normal":
        if valid_labels[i] == 0.:
            print(f"Confirmed. {valid_img_path}'s label: {valid_labels[i]} is {result}")
            continue
        elif valid_labels[i] == 1:
            print(f"Confirmed. {valid_img_path}'s label: {valid_labels[i]} is {result}")
            continue
        else:
            print("Error")
            exit()

for i, test_img_path in enumerate(test_img_paths):
    result = new_df.loc[test_img_path]["Label"]
    if result == "Pnemonia":
        if test_labels[i] == 1.:
            print(f"Confirmed. {test_img_path}'s label: {test_labels[i]} is {result}")
            continue
        elif test_labels[i] == 1:
            print(f"Confirmed. {test_img_path}'s label: {test_labels[i]} is {result}")
            continue
        else:
            print("Error")
            exit()

    elif result == "Normal":
        if test_labels[i] == 0.:
            print(f"Confirmed. {test_img_path}'s label: {test_labels[i]} is {result}")
            continue
        elif test_labels[i] == 1:
            print(f"Confirmed. {test_img_path}'s label: {test_labels[i]} is {result}")
            continue
        else:
            print("Error")
            exit()

