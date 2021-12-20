import pickle
import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np

pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)
url = "http://192.168.1.67/cam-lo.jpg"


def image_processing(img):
    img = cv2.flip(img, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


while True:
    cap = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(cap.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img = cv2.resize(img, (32, 32))
    img = image_processing(img)
    img = img.reshape(1, 32, 32, 1)
    classIndex = model.predict(img)
    result = np.argmax(classIndex)
    proba = np.amax(classIndex)
    print(result, proba)
