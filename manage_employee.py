import cv2 as cv
import os
import shutil


training_folder_path = 'datasets/train_images/'
path_to_employees_info = "employees_info"


def add_employee():
    count = 0

    nameID = str(input("Enter the Employee Name: ")).lower()

    path = training_folder_path + nameID

    isExist = os.path.exists(path)

    if isExist:
        print("Employee Already Exists")
        nameID = str(input("Enter the Employee Name Again: "))
    else:
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
            break
    video.release()
    cv.destroyAllWindows()


def remove_employee():
    nameID = str(input("Enter the Employee Name: ")).lower()

    path = training_folder_path + nameID
    print(path)

    isExist = os.path.exists(path)

    if isExist:
        shutil.rmtree(path)
        print("Employee Removed Successfully")
    else:
        print("Employee doesn't exist!")
    return


def get_punch_in_out_info_employee():
    employee_name = str(input("Enter the Employee Name: ")).lower()
    read_employees_punch_in_out_info(employee_name)


def read_employees_punch_in_out_info(employee_name):
    path_to_employees_info_txt = os.path.join(
        path_to_employees_info, employee_name) + '.txt'
    if (os.path.isfile(path_to_employees_info_txt)):
        f = open(path_to_employees_info_txt, "r")
        lines = f.readlines()
        for line in lines:
            print(line, end="")
    else:
        print("Employee information doesn't exist!")
