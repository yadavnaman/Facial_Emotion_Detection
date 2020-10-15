import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

print('Enter image name')
img = input()
if img not in os.listdir():
    print('No such file, ploxx be serious')
    exit()
img = cv2.imread(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
i = 0
im = []
for (x, y, w, h) in faces:
    #converts to grayscale
    im.append(gray[y:y + h, x:x + w])
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

for i in im:
    cv2.imshow('img', gray)
    cv2.waitKey(4000)
    cv2.imshow('img', i)
    cv2.waitKey(4000)
    #added line to output face to 48x48
    j = cv2.resize(i, (48, 48))
    cv2.imshow('img', j)
    cv2.waitKey(4000)
    # cv2.destroyAllWindows()
cv2.destroyAllWindows()
