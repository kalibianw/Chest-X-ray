from tensorflow.keras.utils import to_categorical
import numpy as np

from utils import TrainModule

nploader = np.load("../APP/splited_Pneumonia_all_true_false_split_1_(200, 250).npz")

train_x_data, valid_x_data, test_x_data, train_y_data, valid_y_data, test_y_data = np.expand_dims(nploader["train_x_data"], axis=-1), \
                                                                                   np.expand_dims(nploader["valid_x_data"], axis=-1), \
                                                                                   np.expand_dims(nploader["test_x_data"], axis=-1), \
                                                                                   to_categorical(nploader["train_y_data"]), \
                                                                                   to_categorical(nploader["valid_y_data"]), \
                                                                                   to_categorical(nploader["test_y_data"])

train_x_data, valid_x_data, test_x_data = train_x_data.astype(dtype="float16"), valid_x_data.astype(dtype="float16"), test_x_data.astype(dtype="float16")

train_x_data = train_x_data / 255.0
valid_x_data = valid_x_data / 255.0
test_x_data = test_x_data / 255.0

print(
    np.max(train_x_data), np.min(train_x_data), np.max(valid_x_data), np.min(valid_x_data), np.max(test_x_data), np.min(test_x_data),
    np.shape(train_x_data),
    np.shape(valid_x_data),
    np.shape(test_x_data),
    np.shape(train_y_data),
    np.shape(valid_y_data),
    np.shape(test_y_data)
)

tm = TrainModule(ckpt_path="C:/Users/admin/Documents/AI/ckpt/Coronahack-Chest-XRay/normalized_splited_Pneumonia_(200, 250)_TF.ckpt",
                 model_save_name="C:/Users/admin/Documents/AI/model/coronahack/normalized_splited_Pneumonia_(200, 250)_TF",
                 input_shape=np.shape(train_x_data)[1:],
                 result_file_name="normalized_splited_Pneumonia_(200, 250)_training_result_TF"
                 )

model = tm.create_model_()
model.summary()

tm.model_training(
    model=model,
    x_train=train_x_data,
    y_train=train_y_data,
    x_valid=valid_x_data,
    y_valid=valid_y_data,
    x_test=test_x_data,
    y_test=test_y_data,
    es_patience=20,
    batch_size=18
)
