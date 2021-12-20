import pickle
import socket
import urllib.request
from time import sleep

import cv2
import matplotlib.pyplot as plt
import numpy as np

pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)
url = "http://192.168.1.61/cam.jpg"


def image_processing(img):
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    img = img.reshape(1, 32, 32, 1)
    return img


host = "192.168.1.61"
port = 23
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))


while True:
    cap = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(cap.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img = image_processing(img)
    classIndex = model.predict(img)
    result = np.argmax(classIndex)
    accuracy = np.amax(classIndex)
    print(result, accuracy)
    if (result == 40 or result == 33 or result == 34 or result == 14) and accuracy >= 0.7:
        s.sendall(bytes(str(result), "utf-8"))
    else:
        s.sendall(bytes(str(-1), "utf-8"))
    sleep(0.01)
s.close()
