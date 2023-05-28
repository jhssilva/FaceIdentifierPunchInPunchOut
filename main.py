# Common imports
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
np.set_printoptions(suppress=True)

width, height = 224, 224

# Colors to draw rectangles in BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# * Load model to face classification
model_name = 'face_classifier_ResNet50Custom.h5'
face_classifier = keras.models.load_model(f'models/{model_name}')

face_cascade = cv.CascadeClassifier(cv.data.haarcascades
                                    + 'haarcascade_frontalface_default.xml')

# * Open camera
video_capture = cv.VideoCapture(0)

if not video_capture.isOpened():
    print("Unable to access the camera")
else:
    print("Access to the camera was successfully obtained")


def get_className(classNo):
    if classNo == 0:
        return "Claudia"
    elif classNo == 1:
        return "Hugo"
    else:
        return "Unknown"


def get_color(index):
    color = RED
    if index < 2:
        color = GREEN
    return color


while (True):
    # Take each frame
    _, frame = video_capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Face detector
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Crop the face
        face_image = frame[y:y+h, x:x+w]

        # resize image to match model's expected sizing
        face_image = tf.image.resize(face_image, [224, 224])

        # return the image with shaping that TF wants.
        face_image = np.expand_dims(face_image, axis=0)
        # Execute module prediction
        prediction = face_classifier.predict(face_image)
        # Get the index of the highest confidence score
        index = np.argmax(prediction[0])
        # Get the confidence score
        confidence = prediction[0][index]

        color = get_color(index)

        cv.rectangle(frame,
                     (x, y),  # start_point
                     (x+w, y+h),  # end_point
                     color,  # color in BGR
                     2)  # thickness in px

        cv.putText(frame,
                   # text to put
                   "{:6} - {:.2f}%".format(get_className(index),
                                           confidence*100),
                   (x, y),
                   cv.FONT_HERSHEY_PLAIN,  # font
                   2,  # fontScale
                   color,
                   2)  # thickness in

    # Display Frame
    cv.imshow('Press esc(ape) to Exit!', frame)

    # sair com a tecla Escape
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

video_capture.release()
cv.destroyAllWindows()
