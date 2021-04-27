from utils import DataModule, NeuralNetwork, TrainModule, TestModule
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
EPOCHS = 1000
MODEL_PATH = "splited_Pneumonia_all_true_false_split_1_0_(300, 300)_relu_re.pt"
LOCAL_TIME = time.localtime()
LOG_FOLDER_PATH = f"./torch_logs/" \
                  f"{os.path.splitext(MODEL_PATH)[0]}_" \
                  f"{LOCAL_TIME[0]}_" \
                  f"{LOCAL_TIME[1]}_" \
                  f"{LOCAL_TIME[2]}_" \
                  f"{LOCAL_TIME[3]}_" \
                  f"{LOCAL_TIME[4]}_" \
                  f"{LOCAL_TIME[5]}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNRING_RATE = 1e-3
REDUCE_LR_PATIENCE = 10
REDUCE_LR_RATE = 0.6
LOG_INTERVAL = 32
EARLY_STOPPING_CNT = 31

dm = DataModule(batch_size=BATCH_SIZE, shuffle=True)
nploader = np.load("splited_Pneumonia_all_true_false_split_1_0_(300, 300).npz")
for key in nploader:
    print(key)

train_x_data = nploader["train_x_data"] / 255.0
train_x_data = np.expand_dims(train_x_data, axis=1)
train_y_data = to_categorical(nploader["train_y_data"])
train_loader = dm.np_to_dataloader(train_x_data, train_y_data)
print(np.shape(train_x_data), np.shape(train_y_data))
print(np.max(train_x_data), np.min(train_x_data))
del train_x_data
del train_y_data

valid_x_data = nploader["valid_x_data"] / 255.0
valid_x_data = np.expand_dims(valid_x_data, axis=1)
valid_y_data = to_categorical(nploader["valid_y_data"])
valid_loader = dm.np_to_dataloader(valid_x_data, valid_y_data)
print(np.shape(valid_x_data), np.shape(valid_y_data))
print(np.max(valid_x_data), np.min(valid_x_data))
del valid_x_data
del valid_y_data

test_x_data = nploader["test_x_data"] / 255.0
test_x_data = np.expand_dims(test_x_data, axis=1)
test_y_data = to_categorical(nploader["test_y_data"])
test_loader = dm.np_to_dataloader(test_x_data, test_y_data)
print(np.shape(test_x_data), np.shape(test_x_data))
print(np.max(test_x_data), np.min(test_x_data))
del test_x_data
del test_y_data

neural_network = NeuralNetwork()
model = neural_network.to(DEVICE)
summary(model, input_size=(32, 1, 300, 300))

optimizer = optim.Adam(model.parameters(), lr=LEARNRING_RATE)
loss = nn.BCEWithLogitsLoss()
tm = TrainModule(
    device=DEVICE,
    optimizer=optimizer,
    loss=loss,
    bach_size=BATCH_SIZE,
    reduce_lr_rate=REDUCE_LR_RATE,
    reduce_lr_patience=REDUCE_LR_PATIENCE
)

os.makedirs(LOG_FOLDER_PATH)

writer = SummaryWriter(log_dir=LOG_FOLDER_PATH)
current_time = time.time()
test_best_loss = sys.maxsize  # for Early stopping
valid_best_loss = sys.maxsize  # for Reduce learning rate
not_improve_cnt = 0
for epoch in range(0, EPOCHS):
    train_acc, train_loss, valid_acc, valid_loss, current_lr = tm.training(
        model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        log_interval=LOG_INTERVAL,
        best_loss=valid_best_loss
    )
    print("\n[EPOCH: {}], \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.2f}%\tLearning Rate: {}".format(epoch,
                                                                                                     train_loss,
                                                                                                     train_acc,
                                                                                                     current_lr))
    print("[EPOCH: {}], \tValid Loss: {:.4f}, \tValid Accuracy: {:.2f}%".format(epoch, valid_loss, valid_acc))

    model.eval()

    tm_ = TestModule()
    true_positive, true_negative, false_positive, false_negative = tm_.analysis(model, test_loader)
    recall, precision, f1_score, specificity = \
        tm_.evaluation(true_positive, true_negative, false_positive, false_negative)
    writer.add_scalar("Test/True Positive", true_positive, epoch)
    writer.add_scalar("Test/True Negative", true_negative, epoch)
    writer.add_scalar("Test/False Positive", false_positive, epoch)
    writer.add_scalar("Test/False Negative", false_negative, epoch)
    writer.add_scalar("Test_/Recall", recall, epoch)
    writer.add_scalar("Test_/precision", precision, epoch)
    writer.add_scalar("Test_/f1_score", f1_score, epoch)
    writer.add_scalar("Test_/specificity", specificity, epoch)

    test_acc, test_loss = tm.evaluate(model, test_loader)
    print("[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f}%\n".format(epoch, test_loss, test_acc))

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/valid", valid_loss, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/valid", valid_acc, epoch)
    writer.add_scalar("Accuracy/test", test_acc, epoch)
    writer.add_scalar("Hyperparameter/current_lr", current_lr, epoch)
    writer.add_scalar("Count/not_improve_cnt", not_improve_cnt, epoch)

    if test_loss < test_best_loss:
        torch.save(model, MODEL_PATH)
        test_best_loss = test_loss
        not_improve_cnt = 0
    if test_loss > test_best_loss:
        if not_improve_cnt > EARLY_STOPPING_CNT:
            break
        not_improve_cnt += 1
    if valid_loss < valid_best_loss:
        valid_best_loss = valid_loss

    print(f"Early stopping non_iprove_cnt: {not_improve_cnt}")
    print(f"Test best loss: {test_best_loss}")
    print(f"Valid best loss: {valid_best_loss}")

print(time.time() - current_time)
