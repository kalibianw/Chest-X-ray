from utils import DataModule, TestModule
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nploader = np.load("splited_Pneumonia_all_true_false_split_1_0_(300, 300).npz")
test_x_data, test_y_data = nploader["test_x_data"] / 255.0, nploader["test_y_data"]
test_x_data = np.expand_dims(test_x_data, axis=1)

dm = DataModule(32, shuffle=True)
test_loader = dm.np_to_dataloader(test_x_data, test_y_data)

MODEL_PATH = "splited_Pneumonia_all_true_false_split_1_0_(300, 300).pt"
model = torch.load(MODEL_PATH)

model.eval()

model.to(DEVICE)

tm_ = TestModule()
TP, TN, FP, FN = tm_.analysis(model, test_loader)
recall, precision, f1_score, specificity = tm_.evaluation(TP, TN, FP, FN)
