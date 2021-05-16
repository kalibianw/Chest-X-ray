from utils import NeuralNetwork, DataModule
from torchsummary import summary
import torch
import numpy as np

MODEL_PATH = "splited_Pneumonia_all_true_false_split_1_1_(300, 300)_swish.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dm = DataModule(32, shuffle=True)
npz_path = "splited_Pneumonia_all_true_false_split_1_1_(300, 300).npz"
nploader = np.load(npz_path)
for key in nploader:
    print(key)

test_x_data, test_y_data = nploader["test_x_data"], nploader["test_y_data"]
test_x_data = np.expand_dims(test_x_data / 255.0, axis=1)
print(np.shape(test_x_data), np.shape(test_y_data))
print(np.max(test_x_data), np.min(test_x_data), np.max(test_y_data), np.min(test_y_data))

test_loader = dm.np_to_dataloader(test_x_data, test_y_data)

nn = NeuralNetwork()
model = nn.to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))

summary(model, input_size=(1, 300, 300), batch_size=32, device=DEVICE.type)

TP, TN, FP, FN = 0, 0, 0, 0
# model.eval()
model.train()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        # print(outputs)
        # print(np.shape(outputs))
        values, indicies = torch.max(outputs.data, 1)
        for label, index in zip(labels, indicies):
            if int(label) == 1:
                if int(index) == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if int(index) == 0:
                    TN += 1
                else:
                    FN += 1

print(f"TP: {TP} TN: {TN} FP: {FP} FN: {FN}")

Recall = TP / (TP + FN)
Precision = TP / (TP + FP)
F1_score = 2 * ((Precision * Recall) / (Precision + Recall))
Specificity = TN / (TP + FP)

print(f"Recall: {Recall}\nPrecision: {Precision}\nF1 Score: {F1_score}\nSpecificity: {Specificity}")
