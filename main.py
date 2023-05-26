# Common imports
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


# OpenCV


width, height = 224, 224

# Colors to draw rectangles in BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# Load model to face classification
model_name = 'face_classifier_ResNet50Custom.h5'

face_classifier = keras.models.load_model(f'models/{model_name}')

face_cascade = cv.CascadeClassifier(cv.data.haarcascades
                                    + 'haarcascade_frontalface_default.xml')

# If this one doesn' seem  to work use MTCNN

np.set_printoptions(suppress=True)

video_capture = cv.VideoCapture(0)

if not video_capture.isOpened():
    print("Unable to access the camera")
else:
    print("Access to the camera was successfully obtained")

while (True):
    # Take each frame
    _, frame = video_capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    def get_className(classNo):
        if classNo == 0:
            return "Claudia"
        elif classNo == 1:
            return "Hugo"
        else:
            return "Unknown"

    for (x, y, w, h) in faces:
        # for each face on the image detected by OpenCV
        # draw a rectangle around the face

        face_image = frame[y:y+h, x:x+w]

        # resize image to match model's expected sizing
        face_image = tf.image.resize(face_image, [224, 224])

        # return the image with shaping that TF wants.
        face_image = np.expand_dims(face_image, axis=0)
        prediction = face_classifier.predict(face_image)
        index = np.argmax(prediction[0])
        confidence = prediction[0][index]
        print(prediction)

        if index < 2:
            color = GREEN
        else:
            color = RED

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
    cv.imshow('Esc to Quit!', frame)

    # sair com a tecla Escape
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
