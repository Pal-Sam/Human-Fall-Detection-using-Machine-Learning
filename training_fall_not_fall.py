import os
import numpy as np
import math
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D, Concatenate
from keras.optimizers import SGD
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

SEED = 47
np.random.seed(SEED)
import random
random.seed(SEED)

DATA_DIR = "datasettorch"
WEIGHTS_PATH = "weights/weights.hdf5"

def load_image(path, target_size=(224, 224)):
    img = load_img(path, target_size=target_size)
    return img_to_array(img)

def create_base_model(prefix=None, image_size=224):
    base_model = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        alpha=1.0,
        include_top=False,
        weights="imagenet"
    )

    if prefix is not None:
        for layer in base_model.layers:
            layer._name = prefix + "_" + layer.name

    for layer in base_model.layers:
        layer.trainable = False

    out = base_model.output
    out = GlobalAveragePooling2D()(out)
    out = Dense(256)(out)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dropout(0.5)(out)

    return base_model, out


def create_model(weights_path=None, image_size=224, show_summary=False):
    spatial_stream, spatial_output = create_base_model(prefix="spatial", image_size=image_size)
    temporal_stream, temporal_output = create_base_model(prefix="temporal", image_size=image_size)
    out = Concatenate()([spatial_output, temporal_output])

    out = Dense(128)(out)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dropout(0.5)(out)

    out = Dense(128)(out)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dropout(0.5)(out)


    predictions = Dense(1, activation="sigmoid")(out)
    model = Model(inputs=[spatial_stream.input, temporal_stream.input], outputs=predictions)

    if weights_path is not None:
        model.load_weights(weights_path)
    if show_summary:
        model.summary()
    return model


def create_two_inputs_gen(data_dir, subset, batch_size=32, shuffle=True):
    fall_path = os.path.join(data_dir, subset, "fall")
    not_fall_path = os.path.join(data_dir, subset, "not_fall")
    fall_files_rgb = [f for f in os.listdir(fall_path) if f.endswith("_rgb.png")]
    fall_files_mhi = [f for f in os.listdir(fall_path) if f.endswith(".png") and "_rgb" not in f]
    not_fall_files_rgb = [f for f in os.listdir(not_fall_path) if f.endswith("_rgb.png")]
    not_fall_files_mhi = [f for f in os.listdir(not_fall_path) if f.endswith(".png") and "_rgb" not in f]
    num_samples = min(len(fall_files_rgb), len(fall_files_mhi), len(not_fall_files_rgb), len(not_fall_files_mhi))
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    while True:
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size].tolist()  # Convert to list
            X_rgb_batch_fall = [load_image(os.path.join(fall_path, fall_files_rgb[i])) for i in batch_indices]
            X_mhi_batch_fall = [load_image(os.path.join(fall_path, fall_files_mhi[i])) for i in batch_indices]
            X_rgb_batch_not_fall = [load_image(os.path.join(not_fall_path, not_fall_files_rgb[i])) for i in batch_indices]
            X_mhi_batch_not_fall = [load_image(os.path.join(not_fall_path, not_fall_files_mhi[i])) for i in batch_indices]
            X_rgb_batch = np.array(X_rgb_batch_fall + X_rgb_batch_not_fall)
            X_mhi_batch = np.array(X_mhi_batch_fall + X_mhi_batch_not_fall)
            X_rgb_batch = preprocess_input(X_rgb_batch)
            X_mhi_batch = preprocess_input(X_mhi_batch)
            # Generate true labels with the correct shape
            y_true_fall = np.ones(len(X_rgb_batch_fall))  # Set true labels for fall images to 1
            y_true_not_fall = np.zeros(len(X_rgb_batch_not_fall))  # Set true labels for not fall images to 0
            y_true_batch = np.concatenate([y_true_fall, y_true_not_fall])  # Concatenate labels
            y_true_batch = np.expand_dims(y_true_batch, axis=1)  # Expand dimensions to match predictions
            yield [X_rgb_batch, X_mhi_batch], y_true_batch


def train_model(model=None):
    if model is None:
        print("No model supplied!")
        return

    EPOCHS = 100
    LEARNING_RATE = 0.01
    MOMENTUM = 0.95
    BATCH_SIZE = 48

    train_gen = create_two_inputs_gen(DATA_DIR, "train", batch_size=BATCH_SIZE, shuffle=True)
    validation_gen = create_two_inputs_gen(DATA_DIR, "val", batch_size=BATCH_SIZE, shuffle=False)

    checkpointer = ModelCheckpoint(
        filepath="tmp/weights.hdf5",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True
    )
    early_stopper = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.001,
        patience=40,
        verbose=1
    )
    tensorboard = TensorBoard(log_dir="logs/", write_graph=False)

    optimizer = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    num_train_samples = len(os.listdir(os.path.join(DATA_DIR, "train", "fall"))) + len(os.listdir(os.path.join(DATA_DIR, "train", "not_fall")))
    num_val_samples = len(os.listdir(os.path.join(DATA_DIR, "val", "fall"))) + len(os.listdir(os.path.join(DATA_DIR, "val", "not_fall")))

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=math.ceil(num_train_samples / BATCH_SIZE),
        validation_data=validation_gen,
        validation_steps=math.ceil(num_val_samples / BATCH_SIZE),
        verbose=2,
        callbacks=[checkpointer, early_stopper, tensorboard],
        class_weight={0: 1.0, 1: 1.0}
    )

    return model


def evaluate_model(model=None, validation=False):
    if model is None:
        print("Error: No model specified!")
        return

    target = "val" if validation else "test"
    BATCH_SIZE = 32

    data_gen = create_two_inputs_gen(DATA_DIR, target, batch_size=BATCH_SIZE, shuffle=False)
    predictions = model.predict_generator(data_gen, steps=math.ceil(len(os.listdir(os.path.join(DATA_DIR, target, "fall")))/BATCH_SIZE), verbose=1)

   
    num_fall_samples = len(os.listdir(os.path.join(DATA_DIR, target, "fall")))
    num_not_fall_samples = len(os.listdir(os.path.join(DATA_DIR, target, "not_fall")))
    y_target = np.concatenate((np.ones(num_fall_samples), np.zeros(num_not_fall_samples)))

    predictions = np.round(predictions).astype(int)
    predictions1 = np.round(predictions).astype(bool)
    y_target1 = y_target.astype(bool)
    accuracy = np.mean(predictions == y_target)
    sensitivity = np.sum(predictions1 & y_target1) / np.sum(y_target1) # Recall or true pos
    specificity = np.sum(~ (predictions1 | y_target1)) / np.sum(~ y_target1) # True neg
    precision = np.sum(predictions1 & y_target1) / np.sum(predictions1)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    return {"accuracy": accuracy,
            "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1_score
        }


if __name__ == "__main__":
    # trained_model = train_model(create_model())
    print(evaluate_model(create_model(WEIGHTS_PATH)))
