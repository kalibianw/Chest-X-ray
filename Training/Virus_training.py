from tensorflow.keras.utils import to_categorical
import numpy as np

from utils import TrainModule

nploader = np.load("../APP/splited_Virus_all_augmentation_1_(400, 500).npz")

x_train, x_test, y_train, y_test = \
    np.expand_dims(nploader["x_train"], axis=-1) / 255.0, \
    np.expand_dims(nploader["x_test"], axis=-1) / 255.0, \
    to_categorical(nploader["y_train"]), \
    to_categorical(nploader["y_test"])

print(
    np.shape(x_train),
    np.max(x_train), np.max(x_test), np.min(x_train), np.min(x_test),
    np.shape(x_test),
    np.shape(y_train),
    np.shape(y_test)
)

tm = TrainModule(ckpt_path="C:/Users/admin/Documents/AI/ckpt/Coronahack-Chest-XRay/Virus_(400, 500).ckpt",
                 model_save_name="C:/Users/admin/Documents/AI/model/coronahack/Virus_(400, 500)",
                 input_shape=np.shape(x_train)[1:],
                 result_file_name="Virus_(400, 500)_training_result"
                 )

model = tm.create_model()
model.summary()

# tm.model_training(
#     model=model,
#     x_train=x_train,
#     y_train=y_train,
#     x_test=x_test,
#     y_test=y_test
# )
