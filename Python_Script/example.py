import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

width = 640
height = 480

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

pickle_in=open("model_trained.p","rb")
model = pickle.load(pickle_in)

def image_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOrg = cap.read()
    img = np.asarray(imgOrg)
    img = cv2.resize(img,(32,32))
    img = image_processing(img)
    img = img.reshape(1,32,32,1)
    classIndex = (model.predict(img))
    result = np.argmax(classIndex)
    proba = np.amax(classIndex)
    if proba >0.95:
        print(result
              )