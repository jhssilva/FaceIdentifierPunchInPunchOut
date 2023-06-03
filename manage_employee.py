import cv2 as cv
import os
import shutil

from voice import read_message


training_folder_path = 'datasets/train_images/'
path_to_employees_info = "employees_info"


def add_employee():
    count = 0

    message_input = "Enter the Employee Name: "
    read_message(message_input)
    nameID = str(input(message_input)).lower()

    path = training_folder_path + nameID

    isExist = os.path.exists(path)

    if isExist:
        message_exists = "Employee Already Exists"
        message_enter_again = "Enter the Employee Name Again: "
        read_message(message_exists)
        print(message_exists)
        read_message(message_enter_again)
        nameID = str(input(message_enter_again))
    else:
        read_message("Taking Pictures of Employee!")
        os.makedirs(path)

    video = cv.VideoCapture(0)

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades
                                        + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video.read()
        gray = gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv.CASCADE_SCALE_IMAGE
        )
        for x, y, w, h in faces:
            count = count+1
            name = path + '/' + str(count) + '.jpg'
            print("Creating Images........." + name)
            cv.imwrite(name, frame[y:y+h, x:x+w])
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv.imshow("WindowFrame", frame)
        cv.waitKey(1)
        if count > 500:
            read_message("Employee Created Successfully")
            break
    video.release()
    cv.destroyAllWindows()


def remove_employee():
    message_out = "Enter the Employee Name: "
    read_message(message_out)
    nameID = str(input(message_out)).lower()

    path = training_folder_path + nameID

    isExist = os.path.exists(path)

    if isExist:
        shutil.rmtree(path)
        message_out = "Employee Removed Successfully"
        read_message(message_out)
        print(message_out)
    else:
        message_out = "Employee Doesnt Exist"
        read_message(message_out)
        print(message_out)
    return


def get_punch_in_out_info_employee():
    input_info = "Enter the Employee Name: "
    read_message(input_info)
    employee_name = str(input(input_info)).lower()
    read_employees_punch_in_out_info(employee_name)


def read_employees_punch_in_out_info(employee_name):
    path_to_employees_info_txt = os.path.join(
        path_to_employees_info, employee_name) + '.txt'
    if (os.path.isfile(path_to_employees_info_txt)):
        read_message("Displaying Employee Information")
        f = open(path_to_employees_info_txt, "r")
        lines = f.readlines()
        for line in lines:
            print(line, end="")
    else:
        message_out = "Employee information doesnt exist!"
        read_message(message_out)
        print(message_out)
