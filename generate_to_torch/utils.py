from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import sys


class DataModule:
    def __init__(self, batch_size: int, shuffle: bool):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def np_to_dataloader(self, xArray: np.ndarray, yArray: np.ndarray):
        tensor_x = torch.Tensor(xArray)
        tensor_y = torch.Tensor(yArray)
        tensor_y = tensor_y.long()

        dataset = TensorDataset(tensor_x, tensor_y)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return data_loader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn1_1 = nn.BatchNorm2d(
            num_features=64
        )
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn1_2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn1_3 = nn.BatchNorm2d(num_features=256)
        self.conv5 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.bn1_4 = nn.BatchNorm2d(num_features=256)
        self.conv6 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, padding=(1, 1))
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=6 * 6 * 512,
            out_features=512
        )
        self.bn2_1 = nn.BatchNorm1d(num_features=512)

        self.fc2 = nn.Linear(
            in_features=512,
            out_features=256
        )
        self.bn2_2 = nn.BatchNorm1d(num_features=256)

        self.fc3 = nn.Linear(
            in_features=256,
            out_features=128
        )
        self.bn2_3 = nn.BatchNorm1d(num_features=128)

        self.fc4 = nn.Linear(
            in_features=128,
            out_features=64
        )
        self.bn2_4 = nn.BatchNorm1d(num_features=64)

        self.fc5 = nn.Linear(
            in_features=64,
            out_features=2
        )

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        torch.nn.init.kaiming_normal_(self.conv6.weight)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1_1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn1_2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.bn1_3(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.bn1_4(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn2_1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2_2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn2_3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn2_4(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = F.softmax(x)

        return x


class TrainModule:
    def __init__(self, device, optimizer, loss, bach_size, reduce_lr_rate, reduce_lr_patience):
        self.DEVICE = device
        self.BATCH_SIZE = bach_size
        self.optimizer = optimizer
        self.criterion = loss
        self.epoch = 0

        self.non_improve_cnt = 0

        self.REDUCE_LR_RATE = reduce_lr_rate
        self.REDUCE_LR_PATIENCE = reduce_lr_patience

    def training(self, model, train_loader, valid_loader, log_interval, best_loss):
        model.train()
        train_loss = 0.
        correct = 0.

        for batch_idx, (image, label) in enumerate(train_loader):
            image = image.to(self.DEVICE)
            label = label.to(self.DEVICE)
            self.optimizer.zero_grad()
            output = model(image)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{} / {}({:.0f}%)]\tTrain Loss: {:.6f}".format(
                        self.epoch, batch_idx * len(image),
                        len(train_loader.dataset),
                        (100. * batch_idx / len(train_loader)),
                        loss.item()
                    ))

            train_loss += loss.item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

        train_loss /= (len(train_loader.dataset) / self.BATCH_SIZE)
        train_accuracy = 100. * correct / len(train_loader.dataset)

        print(f"Reduce LR cnt: {self.non_improve_cnt}")
        valid_acc, valid_loss = self.evaluate(model, valid_loader)
        if valid_loss > best_loss:
            if self.non_improve_cnt > self.REDUCE_LR_PATIENCE:
                self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"] * self.REDUCE_LR_RATE
            self.non_improve_cnt += 1
        else:
            self.non_improve_cnt = 0

        self.epoch += 1
        return train_accuracy, train_loss, valid_acc, valid_loss, self.optimizer.param_groups[0]["lr"]

    def evaluate(self, model, test_loader):
        model.eval()
        test_loss = 0.
        correct = 0.
        with torch.no_grad():
            for image, label in test_loader:
                image = image.to(self.DEVICE)
                label = label.to(self.DEVICE)
                output = model(image)
                test_loss += self.criterion(output, label).item()
                prediction = output.max(1, keepdim=True)[1]
                correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= (len(test_loader.dataset) / self.BATCH_SIZE)
        test_accuracy = 100. * correct / len(test_loader.dataset)

        return test_accuracy, test_loss
