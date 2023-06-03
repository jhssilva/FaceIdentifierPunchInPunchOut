import matplotlib as mpl
import numpy as np
import cv2 as cv
import os
import tensorflow as tf
from voice import read_message
from tensorflow import keras
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

path_train_images_folder = 'datasets/train_images/'


def train_model():
    if (get_number_employees_on_folder() < 2):
        message_output = "There are not enough employees to train the model. Its needed at least two folders. (One employee and one unknown)"
        read_message(message_output)
        print(message_output)
        exit(False)

    read_message("Training the Model!")
    train_dataset = tf.keras.utils.image_dataset_from_directory('datasets/train_images',
                                                                shuffle=True,
                                                                batch_size=8,
                                                                image_size=(224, 224))

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    data_augumentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                                              tf.keras.layers.experimental.preprocessing.RandomRotation(
        0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)])

    vggface_resnet_base = VGGFace(
        model='resnet50', include_top=False, input_shape=(224, 224, 3))

    # Set layers to non-trainable
    for layer in vggface_resnet_base.layers:
        layer.trainable = False
    vggface_resnet_base.trainable = False

    # Build up the new model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augumentation(inputs)
    x = vggface_resnet_base(x)
    x = Flatten(name="flatten")(x)

    out = Dense(num_classes + 1, activation='softmax', name='classifier')(x)
    custom_vgg_model = keras.Model(inputs, out, name="ResNet50Custom")

    # * Train Model
    base_learning_rate = 0.001

    face_classifier = custom_vgg_model
    custom_vgg_model.summary()

    name_to_save = f"models/face_classifier_{face_classifier.name}.h5"

    custom_vgg_model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=base_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])

    # * Safety
    # ModelCheckpoint to save model in case of interrupting the learning process
    checkpoint = ModelCheckpoint(name_to_save,
                                 monitor="val_loss",
                                 mode="min",
                                 save_best_only=True,
                                 verbose=1)

    # EarlyStopping to find best model with a large number of epochs
    earlystop = EarlyStopping(monitor='val_loss',
                              restore_best_weights=True,
                              patience=3,  # number of epochs with no improvement after which training will be stopped
                              verbose=1)

    callbacks = [earlystop, checkpoint]

    epochs = 30

    history = custom_vgg_model.fit(
        train_dataset, callbacks=callbacks, epochs=epochs)

    face_classifier.save(name_to_save)

    read_message("The model has been trained and saved")


def get_number_employees_on_folder():
    # Get the number of all directories on the path
    dirlist = [item for item in os.listdir(path_train_images_folder) if os.path.isdir(
        os.path.join(path_train_images_folder, item))]
    return len(dirlist)
