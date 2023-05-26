import cv2 as cv
import os

video = cv.VideoCapture(0)

count = 0

nameID = str(input("Enter Your Name: ")).lower()

path = 'datasets/train_images/'+nameID

isExist = os.path.exists(path)

if isExist:
    print("Name Already Taken")
    nameID = str(input("Enter Your Name Again: "))
else:
    os.makedirs(path)

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
