

import matplotlib as mpl
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

model_name = 'face_classifier_ResNet50Custom.h5'
face_classifier = keras.models.load_model(f'models/{model_name}')

test_dataset = tf.keras.utils.image_dataset_from_directory('datasets/test_images',
                                                           shuffle=False,
                                                           image_size=(224, 224))

test_loss, test_acc = face_classifier.evaluate(test_dataset, verbose=2)
print('\nTest accuracy:', test_acc)
