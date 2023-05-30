# Common imports
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
import datetime
import time
import threading

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
np.set_printoptions(suppress=True)

width, height = 224, 224

# Colors to draw rectangles in BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)

unknown_name = "unknown"
path_train_images_folder = "datasets/train_images"
path_to_employees_info = "employees_info"

time_between_punch_in_punch_out = 20  # In seconds


def camera():
    # * Define variables employee

    global employees_list
    employees_list = get_employees_names_from_file_structure()

    if len(employees_list) == 0:
        print("No employees found")
        exit(False)

    global employees_status_has_punched_in
    employees_status_has_punched_in = set()

    global employee_status_is_blocked
    employee_status_is_blocked = set()

    # * Load model to face classification
    model_name = 'face_classifier_ResNet50Custom.h5'
    face_classifier = keras.models.load_model(f'models/{model_name}')

    if face_classifier is None:
        print("Unable to load the model")
        exit(False)

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades
                                        + 'haarcascade_frontalface_default.xml')

    # * Open camera
    video_capture = cv.VideoCapture(0)

    if not video_capture.isOpened():
        print("Unable to access the camera")
    else:
        print("Access to the camera was successfully obtained")

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
            face_image = tf.image.resize(face_image, [width, height])

            # return the image with shaping that TF wants.
            face_image = np.expand_dims(face_image, axis=0)
            # Execute module prediction
            prediction = face_classifier.predict(face_image, verbose=0)
            # Get the index of the highest confidence score
            index = np.argmax(prediction[0])
            # Get the confidence score
            confidence = prediction[0][index]

            employee_name = get_employee_name(int(index))
            handle_punch_in_punch_out(employee_name)
            color = get_color(employee_name)

            cv.rectangle(frame,
                         (x, y),  # start_point
                         (x+w, y+h),  # end_point
                         color,  # color in BGR
                         2)  # thickness in px

            cv.putText(frame,
                       # text to put
                       "{:6} - {:.2f}%".format(employee_name.upper(),
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
    exit(True)


def get_employee_name(index: int) -> str:
    if (index > get_employees_number_with_unknown()):
        return unknown_name
    return employees_list[index]


def get_employees_number_with_unknown() -> int:
    return len(employees_list)


def get_color(employee_name: str) -> tuple:
    color = GREEN
    if (employee_name == unknown_name):
        color = RED
    return color


def get_employees_names_from_file_structure():
    # Get the list of all directories in the path
    dirlist = [item for item in os.listdir(path_train_images_folder) if os.path.isdir(
        os.path.join(path_train_images_folder, item))]
    dir_list_ordered = sorted(dirlist)
    return dir_list_ordered


def handle_punch_in_punch_out(employee_name):
    if (employee_name == unknown_name):
        return

    if (employee_name in employee_status_is_blocked):
        return

    if (employee_name in employees_status_has_punched_in):
        punch_in_or_out(False, employee_name)
    else:
        punch_in_or_out(True, employee_name)


def get_date_time():
    return datetime.datetime.now()


def punch_in_or_out(punch_in: bool, employee_name: str):
    employee_status_is_blocked.add(employee_name)
    create_employee_countdown(employee_name)
    if punch_in:
        employees_status_has_punched_in.add(employee_name)
    else:
        employees_status_has_punched_in.remove(employee_name)
    message_in_or_out = "in" if punch_in else "out"
    message = f"{employee_name.capitalize()} punched {message_in_or_out} at {get_date_time()}"
    write_to_file(employee_name, message)
    print(message)


def write_to_file(employee_name, message):
    file_name = f"{employee_name}.txt"
    file_path = os.path.join(path_to_employees_info, file_name)

    with open(file_path, "a") as file:
        file.write(message + "\n")


def create_employee_countdown(employee_name):
    thread = threading.Thread(target=countdown, args=(
        time_between_punch_in_punch_out, employee_name))
    thread.start()


def countdown(t, employee_name):
    t_tim = t
    while t_tim:
        time.sleep(1)
        t_tim -= 1
    employee_status_is_blocked.remove(employee_name)
