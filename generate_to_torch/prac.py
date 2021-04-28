from utils import NeuralNetwork, DataModule, TestModule
import torch
import numpy as np
from torchsummary import summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nploader = np.load("splited_Pneumonia_all_true_false_split_1_0_(300, 300).npz")
test_x_data, test_y_data = nploader["test_x_data"] / 255.0, nploader["test_y_data"]
test_x_data = np.expand_dims(test_x_data, axis=1)

dm = DataModule(32, shuffle=True)
test_loader = dm.np_to_dataloader(test_x_data, test_y_data)

MODEL_PATH = "splited_Pneumonia_all_true_false_split_1_0_(300, 300)_combined.pt"
network = NeuralNetwork()

# model = network.to(DEVICE)
# model.load_state_dict(torch.load(MODEL_PATH))
model = torch.load(MODEL_PATH)
summary(model, input_size=(1, 300, 300), batch_size=32)

model.eval()

tm_ = TestModule()
TP, TN, FP, FN = tm_.analysis(model, test_loader)
tm_.evaluation(TP, TN, FP, FN)
