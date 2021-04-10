from tensorflow.keras import models, layers, activations, optimizers, losses, callbacks, regularizers, Model
from sklearn.model_selection import KFold, train_test_split

import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import cv2
import os


class DataModule:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def reading_csv(self, which_label, which, filename):
        fhandler = open(file=self.csv_path, mode='r')
        csv_reader = csv.reader(fhandler)

        train_img_path = list()
        train_label = list()
        test_img_path = list()
        test_label = list()
        next(csv_reader)

        for row in csv_reader:
            if row[3] == "TRAIN":
                img_path = "C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/" + \
                           row[1]
                if os.path.exists(img_path) is False:
                    continue
                for i in range(len(which)):
                    if row[which_label] == which[i]:
                        train_img_path.append(row[1])
                        train_label.append(i)

            elif row[3] == "TEST":
                img_path = "C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/" + \
                           row[1]
                if os.path.exists(img_path) is False:
                    continue
                for i in range(len(which)):
                    if row[which_label] == which[i]:
                        test_img_path.append(row[1])
                        test_label.append(i)

        np.savez_compressed(f"ARC/{filename}.npz",
                            train_img_path=train_img_path,
                            train_label=train_label,
                            test_img_path=test_img_path,
                            test_label=test_label
                            )

    def reading_csv_compilation(self, which_label, which, filename, data_augmentation):
        fhandler = open(file=self.csv_path, mode='r')
        csv_reader = csv.reader(fhandler)

        img_path_list = list()
        label = list()
        next(csv_reader)

        for row in csv_reader:
            if row[3] == "TRAIN":
                img_path = "C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/" + \
                           row[1]
                if os.path.exists(img_path) is False:
                    continue
                for i in range(len(which)):
                    if row[which_label] == which[i]:
                        img_path_list.append(row[1])
                        label.append(i)
                        for j in range(data_augmentation):
                            label.append(i)

            elif row[3] == "TEST":
                img_path = "C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/" + \
                           row[1]
                if os.path.exists(img_path) is False:
                    continue
                for i in range(len(which)):
                    if row[which_label] == which[i]:
                        img_path_list.append(row[1])
                        label.append(i)
                        for j in range(data_augmentation):
                            label.append(i)

        print(f"shape of img_path: {np.shape(img_path_list)}\nshape of label: {np.shape(label)}")

        np.savez_compressed(f"ARC/{filename}_all.npz",
                            img_path=img_path_list,
                            label=label
                            )

    def image_padding(self, img, height, width, target_width, target_height):
        if height < target_height:
            pad_value = target_height - height
            front_pad = np.zeros(shape=(int(pad_value / 2), width), dtype=np.uint8)
            if (pad_value / 2).is_integer() is True:
                rear_pad = np.zeros(shape=(int(pad_value / 2), width), dtype=np.uint8)
            else:
                rear_pad = np.zeros(shape=(int(pad_value / 2) + 1, width), dtype=np.uint8)
            img = np.insert(img, 0, front_pad, axis=0)
            img = np.append(img, rear_pad, axis=0)

        elif width < target_width:
            pad_value = target_width - width
            front_pad = np.zeros(shape=(int(pad_value / 2), height), dtype=np.uint8)
            if (pad_value / 2).is_integer() is True:
                rear_pad = np.zeros(shape=(height, int(pad_value / 2)), dtype=np.uint8)
            else:
                rear_pad = np.zeros(shape=(height, int(pad_value / 2) + 1), dtype=np.uint8)
            img = np.insert(img, 0, front_pad, axis=1)
            img = np.append(img, rear_pad, axis=1)

        return img

    def image_resize(self, img, target_width, target_height, zero_padding: bool):
        img_shape = np.shape(img)
        height = img_shape[0]
        width = img_shape[1]

        if zero_padding:
            if (width / target_width) > (height / target_height):
                diff = target_width / width
                width = target_width
                height = int(height * diff)

                img = cv2.resize(img, (width, height))
                img = self.image_padding(img, height, width, target_width=target_width, target_height=target_height)

            elif (width / target_width) <= (height / target_height):
                diff = target_height / height
                height = target_height
                width = int(width * diff)

                img = cv2.resize(img, (width, height))
                img = self.image_padding(img, height, width, target_width=target_width, target_height=target_height)

            return img

        elif zero_padding is False:
            img = cv2.resize(img, (target_width, target_height))

            return img


class TrainModule:
    def __init__(self, ckpt_path, model_save_name, input_shape, result_file_name):
        self.result_file_name = result_file_name
        self.ckpt_path = ckpt_path
        self.model_save_name = model_save_name
        self.input_shape = input_shape
        self.result_path = f"C:/Users/admin/Documents/Study/AI_X/Coronahack-Chest-XRay/Training/{result_file_name}.txt"

    def create_model(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=self.input_shape),
            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(padding="same"),
            layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(padding="same"),
            layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),
            layers.MaxPooling2D(padding="same"),
            layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(padding="same"),
            layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(padding="same"),
            layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer="he_normal"),
            layers.MaxPooling2D(padding="same"),
            layers.Dropout(rate=0.5),

            layers.Flatten(),

            layers.Dense(512, activation=activations.relu, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dense(256, activation=activations.relu, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(rate=0.5),
            layers.Dense(128, activation=activations.relu, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(rate=0.5),
            layers.Dense(64, activation=activations.relu, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dense(3, activation=activations.softmax)
        ])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def create_model_(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=self.input_shape),
            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(padding="same"),

            layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(padding="same"),

            layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer="he_normal"),
            layers.Dropout(rate=0.5),
            layers.MaxPooling2D(padding="same"),

            layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(padding="same"),

            layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(padding="same"),

            layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation=activations.relu,
                          kernel_initializer="he_normal"),
            layers.MaxPooling2D(padding="same"),
            layers.Dropout(rate=0.5),

            layers.Flatten(),

            layers.Dense(512, activation=activations.relu, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dense(256, activation=activations.relu, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(rate=0.5),
            layers.Dense(128, activation=activations.relu, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(rate=0.5),
            layers.Dense(64, activation=activations.relu, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dense(2, activation=activations.softmax)
        ])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def model_KFold_training(self, model, x_train, y_train, x_test, y_test, es_patience, batch_size):
        fhandler = open(self.result_path, 'w')
        kfold = KFold(shuffle=True)

        valid_acc_fold = list()
        test_acc_fold = list()
        valid_loss_fold = list()
        test_loss_fold = list()

        fold_no = 1
        for train, valid in kfold.split(x_train, y_train):
            start_time = time.time()
            # print(f"shape of x_train[train]: {np.shape(x_train[train])}")
            # print(f"type of element of x_train[train]: {type(x_train[train][0, 0, 0, 0])}")
            # print(f"shape of x_train[valid]: {np.shape(x_train[valid])}")
            # print(f"type of element of x_train[valid]: {type(x_train[valid][0, 0, 0, 0])}")

            hist = model.fit(
                x=x_train[train], y=y_train[train],
                batch_size=batch_size,
                epochs=1000,
                callbacks=[
                    callbacks.ReduceLROnPlateau(
                        factor=0.8,
                        patience=3,
                        verbose=2,
                        min_delta=5e-4,
                        min_lr=1e-6
                    ),
                    callbacks.EarlyStopping(
                        min_delta=5e-4,
                        patience=es_patience,
                        verbose=1
                    ),
                    callbacks.ModelCheckpoint(
                        filepath=self.ckpt_path,
                        verbose=2,
                        save_best_only=True,
                        save_weights_only=True
                    )
                ],
                validation_data=(x_train[valid], y_train[valid])
            )

            model.load_weights(filepath=self.ckpt_path)
            model.save(filepath=f"{self.model_save_name}_{fold_no}.h5")

            total_time = time.time() - start_time

            self.training_visualization(hist=hist.history, fold_no=fold_no)

            fhandler.write(f"\nTraining time for fold {fold_no}: {total_time} sec\n")
            valid_score = model.evaluate(x=x_train[valid], y=y_train[valid], verbose=0)
            fhandler.write(
                f"> Validation score for fold {fold_no}: \n"
                f"Score for validation dataset: {model.metrics_names[0]} - {valid_score[0]}; {model.metrics_names[1]} - {valid_score[1] * 100}%"
            )

            test_score = model.evaluate(x=x_test, y=y_test)
            print(
                f"Score for {fold_no} - test set: {model.metrics_names[0]} of {test_score[0]}; {model.metrics_names[1]} of {test_score[1] * 100}%"
            )
            fhandler.write(
                f"\nScore for test set: {model.metrics_names[0]} of {test_score[0]}; {model.metrics_names[1]} of {test_score[1] * 100}%\n"
            )
            fhandler.write("\n--------------------------------------------------------------------------------------\n")

            valid_loss_fold.append(valid_score[0])
            valid_acc_fold.append(valid_score[1] * 100)
            test_loss_fold.append(test_score[0])
            test_acc_fold.append(test_score[1] * 100)

            fold_no += 1

        fhandler.write("\nAverage valid scores for all folds:\n")
        fhandler.write(f"> Accuracy: {np.mean(valid_acc_fold)}% (+- {np.std(valid_acc_fold)})\n")
        fhandler.write(f"> Loss: {np.mean(valid_loss_fold)} (+- {np.std(valid_loss_fold)})\n")

        fhandler.write("\n\nAverage test scores for all folds:\n")
        fhandler.write(f"> Accuracy: {np.mean(test_acc_fold)}% (+- {np.std(test_acc_fold)})\n")
        fhandler.write(f"> Loss: {np.mean(test_loss_fold)} (+- {np.std(test_loss_fold)})\n")

        fhandler.close()

    def model_training(self, model, x_train, y_train, x_valid, y_valid, x_test, y_test, es_patience, batch_size):
        # es_patience = 30
        fhandler = open(self.result_path, 'w')

        fold_no = 0
        start_time = time.time()

        hist = model.fit(
            x=x_train, y=y_train,
            batch_size=batch_size,
            epochs=1000,
            callbacks=[
                callbacks.ReduceLROnPlateau(
                    factor=0.8,
                    patience=3,
                    verbose=2,
                    min_delta=5e-4,
                    min_lr=1e-6
                ),
                callbacks.EarlyStopping(
                    min_delta=5e-4,
                    patience=es_patience,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    filepath=self.ckpt_path,
                    verbose=2,
                    save_best_only=True,
                    save_weights_only=True
                )
            ],
            validation_data=(x_valid, y_valid)
        )

        model.load_weights(filepath=self.ckpt_path)
        model.save(filepath=f"{self.model_save_name}_{fold_no}.h5")

        total_time = time.time() - start_time

        self.training_visualization(hist=hist.history, fold_no=fold_no)

        fhandler.write(f"\nTraining time for fold {fold_no}: {total_time} sec\n")
        valid_score = model.evaluate(x=x_valid, y=y_valid, verbose=0)
        fhandler.write(
            f"> Validation score for fold {fold_no}: \n"
            f"Score for validation dataset: {model.metrics_names[0]} - {valid_score[0]}; {model.metrics_names[1]} - {valid_score[1] * 100}%"
        )

        test_score = model.evaluate(x=x_test, y=y_test)
        print(
            f"Score for {fold_no} - test set: {model.metrics_names[0]} of {test_score[0]}; {model.metrics_names[1]} of {test_score[1] * 100}%"
        )
        fhandler.write(
            f"\nScore for test set: {model.metrics_names[0]} of {test_score[0]}; {model.metrics_names[1]} of {test_score[1] * 100}%\n"
        )
        fhandler.write("\n--------------------------------------------------------------------------------------\n")
        fhandler.close()

    def training_visualization(self, hist, fold_no):
        localtime = time.localtime()
        tst = str(localtime[0]) + "_" + str(localtime[1]) + "_" + str(localtime[2]) + "_" + str(
            localtime[3]) + "_" + str(localtime[4]) + "_" + str(localtime[5])

        plt.subplot(2, 1, 1)
        plt.plot(hist['acc'], 'b')
        plt.plot(hist['val_acc'], 'g')
        plt.ylim([0, 1])
        plt.xlabel("Epoch")
        plt.ylabel("Accuracies")
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        plt.plot(hist['loss'], 'b')
        plt.plot(hist['val_loss'], 'g')
        plt.ylim([0, 5])
        plt.xlabel("Epoch")
        plt.ylabel("Losses")
        plt.tight_layout()

        fig_path = "C:/Users/admin/Documents/Study/AI_X/Coronahack-Chest-XRay/fig/" + tst + "_" + str(
            fold_no) + f"_{self.result_file_name}.png"
        plt.savefig(fname=fig_path, dpi=300)
        plt.clf()
