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

    def np_to_dataloader(self, x_data: np.ndarray, y_data: np.ndarray):
        tensor_x = torch.Tensor(x_data)
        tensor_y = torch.Tensor(y_data)
        tensor_y = tensor_y.long()

        dataset = TensorDataset(tensor_x, tensor_y)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return data_loader


class NeuralNetwork100(nn.Module):
    def __init__(self):
        super(NeuralNetwork100, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn1_1 = nn.BatchNorm2d(num_features=64)
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

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1), stride=2)
        self.maxpool_ = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=2 * 2 * 512,
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
        x = F.hardswish(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv2(x)
        x = self.bn1_2(x)
        x = F.relu(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv4(x)
        x = self.bn1_3(x)
        x = F.hardswish(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv5(x)
        x = self.bn1_4(x)
        x = F.hardswish(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv6(x)
        x = F.hardswish(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)
        x = self.dropout(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn2_1(x)
        x = F.hardswish(x)

        x = self.fc2(x)
        x = self.bn2_2(x)
        x = F.hardswish(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn2_3(x)
        x = F.hardswish(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn2_4(x)
        x = F.hardswish(x)

        x = self.fc5(x)
        x = F.softmax(x, dim=1)

        return x


class NeuralNetwork200(nn.Module):
    def __init__(self):
        super(NeuralNetwork200, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn1_1 = nn.BatchNorm2d(num_features=64)
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

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1), stride=2)
        self.maxpool_ = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=4 * 4 * 512,
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
        x = F.hardswish(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv2(x)
        x = self.bn1_2(x)
        x = F.relu(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv4(x)
        x = self.bn1_3(x)
        x = F.hardswish(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv5(x)
        x = self.bn1_4(x)
        x = F.hardswish(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv6(x)
        x = F.hardswish(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)
        x = self.dropout(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn2_1(x)
        x = F.hardswish(x)

        x = self.fc2(x)
        x = self.bn2_2(x)
        x = F.hardswish(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn2_3(x)
        x = F.hardswish(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn2_4(x)
        x = F.hardswish(x)

        x = self.fc5(x)
        x = F.softmax(x, dim=1)

        return x


class NeuralNetwork300(nn.Module):
    def __init__(self):
        super(NeuralNetwork300, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn1_1 = nn.BatchNorm2d(num_features=64)
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

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1), stride=2)
        self.maxpool_ = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=5 * 5 * 512,
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
        x = F.hardswish(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv2(x)
        x = self.bn1_2(x)
        x = F.relu(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv4(x)
        x = self.bn1_3(x)
        x = F.hardswish(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv5(x)
        x = self.bn1_4(x)
        x = F.hardswish(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)

        x = self.conv6(x)
        x = F.hardswish(x)
        x = self.maxpool_(x) if (x.shape[-1] % 2 == 0) else self.maxpool(x)
        x = self.dropout(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn2_1(x)
        x = F.hardswish(x)

        x = self.fc2(x)
        x = self.bn2_2(x)
        x = F.hardswish(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn2_3(x)
        x = F.hardswish(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn2_4(x)
        x = F.hardswish(x)

        x = self.fc5(x)
        x = F.softmax(x, dim=1)

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

        model.eval()

        valid_acc, valid_loss = self.evaluate(model, valid_loader)
        if valid_loss > best_loss:
            if self.non_improve_cnt > self.REDUCE_LR_PATIENCE:
                self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"] * self.REDUCE_LR_RATE
                self.non_improve_cnt = 0
            self.non_improve_cnt += 1
        else:
            self.non_improve_cnt = 0
        print(f"Reduce LR cnt: {self.non_improve_cnt}")

        self.epoch += 1

        return train_accuracy, train_loss, valid_acc, valid_loss, self.optimizer.param_groups[0]["lr"]

    def evaluate(self, model, test_loader):
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


class TestModule:
    def __init__(self):
        pass

    def analysis(self, model, test_loader):
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                for label, output in zip(labels, outputs):
                    if int(label) == 1:
                        if int(torch.argmax(output)) == 1:
                            true_positive += 1
                        else:
                            false_positive += 1
                    else:
                        if int(torch.argmax(output)) == 0:
                            true_negative += 1
                        else:
                            false_negative += 1
        print(f"TP: {true_positive} TN: {true_negative} FP: {false_positive} FN: {false_negative}")

        return true_positive, true_negative, false_positive, false_negative

    def evaluation(self, TP, TN, FP, FN):
        try:
            recall = TP / (TP + FN)
        except ZeroDivisionError:
            recall = TP / (TP + FN + 1e-5)
        try:
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = TP / (TP + FP + 1e-5)
        try:
            specificity = TN / (TN + FP)
        except ZeroDivisionError:
            specificity = TN / (TN + FP + 1e-5)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        print(f"Recall: {recall}\nPrecision: {precision}\nF1 Score: {f1_score}\nSpecificity: {specificity}")

        return recall, precision, f1_score, specificity
