import os
import pickle

import cv2
import matplotlib.pyplot as mpl
import numpy as np
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

data_path = "myData"
sign_images_list = []
class_number = []
sign_list = os.listdir(data_path)
sign_classes_number = len(sign_list)

for sign in range(0, sign_classes_number):
    image_list = os.listdir(data_path + "/" + str(sign))
    for img in image_list:
        current_sign = cv2.imread(data_path + "/" + str(sign) + "/" + img)
        current_sign = cv2.resize(current_sign, (32, 32))
        sign_images_list.append(current_sign)
        class_number.append(sign)

sign_images_list = np.array(sign_images_list)
class_number = np.array(class_number)

# img = sign_images_list[42]
# img = cv2.resize(img, (300, 300))
# cv2.imshow("cd", img)
# cv2.waitKey(0)

print(sign_images_list.shape)
print(class_number)

def image_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

sign_images_list = np.array(list(map(image_processing,sign_images_list)))

sign_images_list = sign_images_list.reshape(sign_images_list.shape[0],
                                            sign_images_list.shape[1],
                                            sign_images_list.shape[2], 1)

# img = sign_images_list[42]
# img = cv2.resize(img, (300, 300))
# cv2.imshow("cd", img)
# cv2.waitKey(0)

X_train, X_test, Y_train, Y_test = train_test_split(
    sign_images_list, class_number, test_size=0.2
)
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X_train, Y_train, test_size=0.2
)

print(X_validation.shape)
print(X_test.shape)
print(X_train.shape)
#
# for sign_class in range(0, sign_classes_number):
#     sample_numbers.append(len(np.where(Y_train == sign_class)[0]))
# print(sample_numbers)


# reshapownie wszystkiego po zprzetworzeniu do koloru binarnego żeby działało w sieci
#
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
# X_validation = X_validation.reshape(
#     X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1
# )
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


genereted_data = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10,
)
genereted_data.fit(X_train)

# img = X_train[2137]
# img = cv2.resize(img, (300, 300))
# cv2.imshow("cd", img)
# cv2.waitKey(0)

Y_train = to_categorical(Y_train, sign_classes_number)
Y_validation = to_categorical(Y_validation, sign_classes_number)
Y_test = to_categorical(Y_test, sign_classes_number)


def cnn_model():
    model = Sequential()
    model.add((Conv2D(32,(5, 5),input_shape=(32, 32, 1),activation="relu",)))
    model.add((Conv2D(32, (5, 5), activation="relu")))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add((Conv2D(64, (5, 5), activation="relu")))
    model.add((Conv2D(64, (5, 5), activation="relu")))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation="softmax"))
    model.summary()
    model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


model = cnn_model()

batch_size = 50
epoch_value = 50
step_per_epoch = 700

history = model.fit_generator(
    genereted_data.flow(X_train, Y_train, batch_size=batch_size),
    steps_per_epoch=step_per_epoch,
    epochs=epoch_value,
    validation_data=(X_validation, Y_validation),
)


mpl.figure(1)
mpl.plot(history.history["loss"])
mpl.plot(history.history["val_loss"])
mpl.legend(["training", "validation"])
mpl.title("Loss")
mpl.xlabel("epoch")

mpl.figure(2)
mpl.plot(history.history["accuracy"])
mpl.plot(history.history["val_accuracy"])
mpl.legend(["training", "validation"])
mpl.title("Accuracy")
mpl.xlabel("epoch")

mpl.show()

score = model.evaluate(X_test, Y_test, verbose=0)
print(score[0])
print(score[1])

pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
