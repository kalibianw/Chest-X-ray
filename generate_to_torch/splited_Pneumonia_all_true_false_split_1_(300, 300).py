from utils import DataModule, NeuralNetwork, TrainModule
from tensorflow.keras.utils import to_categorical
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import time
import sys
import os

BATCH_SIZE = 32
EPOCHS = 100
MODEL_PATH = "splited_Pneumonia_all_true_false_split_1_(300, 300).pt"
LOCAL_TIME = time.localtime()
LOG_FOLDER_PATH = f"./torch_logs/{LOCAL_TIME[0]}_{LOCAL_TIME[1]}_{LOCAL_TIME[2]}_{LOCAL_TIME[3]}_{LOCAL_TIME[4]}_{LOCAL_TIME[5]}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REDUCE_LR_RATE = 0.6
LOG_INTERVAL = 10
EARLY_STOPPING_CNT = 20

dm = DataModule(batch_size=BATCH_SIZE, shuffle=True)
nploader = np.load("splited_Pneumonia_all_true_false_split_1_(300, 300).npz")
for key in nploader:
    print(key)

train_x_data = nploader["train_x_data"] / 255.0
train_x_data = np.expand_dims(train_x_data, axis=1)
print(np.shape(train_x_data))
train_y_data = nploader["train_y_data"]
train_loader = dm.np_to_dataloader(train_x_data, train_y_data)
del train_x_data
del train_y_data

valid_x_data = nploader["valid_x_data"] / 255.0
valid_x_data = np.expand_dims(valid_x_data, axis=1)
print(np.shape(valid_x_data))
valid_y_data = nploader["valid_y_data"]
valid_loader = dm.np_to_dataloader(valid_x_data, valid_y_data)
del valid_x_data
del valid_y_data

test_x_data = nploader["test_x_data"] / 255.0
test_x_data = np.expand_dims(test_x_data, axis=1)
print(np.shape(test_x_data))
test_y_data = nploader["test_y_data"]
test_loader = dm.np_to_dataloader(test_x_data, test_y_data)
del test_x_data
del test_y_data

neural_network = NeuralNetwork()
model = neural_network.to(DEVICE)
summary(model, input_size=(32, 1, 300, 300))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()
tm = TrainModule(
    DEVICE=DEVICE,
    optimizer=optimizer,
    loss=loss,
    BATCH_SIZE=BATCH_SIZE,
    REDUCE_LR_RATE=REDUCE_LR_RATE
)

os.makedirs(LOG_FOLDER_PATH)

writer = SummaryWriter(log_dir=LOG_FOLDER_PATH)
current_time = time.time()
best_loss = sys.maxsize
not_improve_cnt = 0
for Epoch in range(1, EPOCHS + 1):
    train_acc, train_loss, valid_acc, valid_loss = tm.training(
        model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        log_interval=LOG_INTERVAL
    )
    print("\n[EPOCH: {}], \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.2f}%".format(Epoch, train_loss, train_acc))
    print("[EPOCH: {}], \tValid Loss: {:.4f}, \tValid Accuracy: {:.2f}%\n".format(Epoch, valid_loss, valid_acc))

    test_acc, test_loss = tm.evaluate(model, test_loader)
    print("[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f}%\n".format(Epoch, test_loss, test_acc))

    writer.add_scalar("Loss/train", train_loss, Epoch)
    writer.add_scalar("Loss/valid", valid_loss, Epoch)
    writer.add_scalar("Loss/test", test_loss, Epoch)
    writer.add_scalar("Accuracy/train", train_acc, Epoch)
    writer.add_scalar("Accuracy/valid", valid_acc, Epoch)
    writer.add_scalar("Accuracy/test", test_acc, Epoch)

    if test_loss < best_loss:
        torch.save(model.state_dict(), MODEL_PATH)
        not_improve_cnt = 0
    if test_loss > best_loss:
        if not_improve_cnt > EARLY_STOPPING_CNT:
            break
        not_improve_cnt += 1

print(time.time() - current_time)
