from tensorflow.keras.preprocessing import image
from tensorflow.keras import callbacks

from utils import TrainModule

BATCH_SIZE = 8
train_valid_datagen = image.ImageDataGenerator(
    validation_split=0.2,
    rescale=1. / 255,
    dtype="float16"
)
test_datagen = image.ImageDataGenerator(
    rescale=1. / 255,
    dtype="float16"
)

train_generator = train_valid_datagen.flow_from_directory(
    directory="C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/for_generator/splited_Pneumonia_all/train",
    target_size=(400, 500),
    color_mode="grayscale",
    subset="training",
    batch_size=BATCH_SIZE
)
valid_generator = train_valid_datagen.flow_from_directory(
    directory="C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/for_generator/splited_Pneumonia_all/train",
    target_size=(400, 500),
    color_mode="grayscale",
    subset="validation",
    batch_size=BATCH_SIZE
)

test_generator = test_datagen.flow_from_directory(
    directory="C:/Users/admin/Documents/AI/data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/for_generator/splited_Pneumonia_all/test",
    target_size=(400, 500),
    color_mode="grayscale",
    batch_size=BATCH_SIZE
)

tm = TrainModule(ckpt_path="C:/Users/admin/Documents/AI/ckpt/Coronahack-Chest-XRay/normalized_splited_Pneumonia_(400, 500).ckpt",
                 model_save_name="C:/Users/admin/Documents/AI/model/coronahack/normalized_splited_Pneumonia_(400, 500)",
                 input_shape=(400, 500, 1),
                 result_file_name="normalized_splited_Pneumonia_(400, 500)_generator_training_result"
                 )

model = tm.create_model_()
model.summary()

model.fit(
    train_generator,
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
            patience=40,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath="C:/Users/admin/Documents/AI/ckpt/Coronahack-Chest-XRay/normalized_splited_Pneumonia_(400, 500).ckpt",
            verbose=2,
            save_best_only=True,
            save_weights_only=True
        )
    ],
    validation_data=valid_generator
)

model.load_weights(filepath="C:/Users/admin/Documents/AI/ckpt/Coronahack-Chest-XRay/normalized_splited_Pneumonia_(400, 500).ckpt")
model.save(filepath="C:/Users/admin/Documents/AI/model/coronahack/normalized_splited_Pneumonia_(400, 500).h5")

model.evaluate(test_generator)
