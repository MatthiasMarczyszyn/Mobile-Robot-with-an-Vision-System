import numpy as np
import cv2
import os
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator





data_path = 'myData'
sing_images_list=[]
class_number=[]
sample_numbers =[]
sing_list = os.listdir(data_path)
sing_classes_number = len(sing_list)

for sing in range (0,sing_classes_number):
    image_list = os.listdir(data_path+"/"+str(sing))
    for img in image_list:
        current_sing = cv2.imread(data_path+"/"+str(sing)+"/"+img)
        current_sing = cv2.resize(current_sing,(32,32))
        sing_images_list.append(current_sing)
        class_number.append(sing)
print(len(sing_images_list))

sing_images_list = np.array(sing_images_list)
class_number = np.array(class_number)

print(sing_images_list.shape)
print(class_number)

X_train,X_test,Y_train,Y_test = train_test_split(sing_images_list,class_number,test_size=0.2)
X_train,X_validation, Y_train,Y_validation = train_test_split(X_train,Y_train, test_size=0.2)
print(X_validation.shape)
print(X_test.shape)
print(X_train.shape)

for sing_class in range(0,sing_classes_number):
    sample_numbers.append(len(np.where(Y_train==sing_class)[0]))
print(sample_numbers)

def image_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

X_train = np.array(list(map(image_processing,X_train)))
X_test = np.array(list(map(image_processing,X_test)))
X_validation = np.array(list(map(image_processing,X_validation)))

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
print(X_train.shape)
genereted_data = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    shear_range=0.1,
                                    rotation_range=10)
genereted_data.fit(X_train)

Y_train = to_categorical(Y_train, sing_classes_number)
Y_validation = to_categorical(Y_validation, sing_classes_number)
Y_test = to_categorical(Y_test, sing_classes_number)
print(Y_train.shape)
def cnn_model():
    numbers_of_filetr = 60
    size_of_filter1 = (5,5)
    size_of_filter2 = (3,3)
    pool_size = (2,2)
    numbers_of_node = 500

    model=Sequential()
    model.add((Conv2D(numbers_of_filetr,size_of_filter1,input_shape=(32,32,1), activation='relu')))
    model.add((Conv2D(numbers_of_filetr, size_of_filter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add((Conv2D(numbers_of_filetr//2, size_of_filter2, activation='relu')))
    model.add((Conv2D(numbers_of_filetr//2, size_of_filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(numbers_of_node,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(sing_classes_number,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = cnn_model()

batchSize = 50
epochValue = 10
stepPerEpoch = 500

history = model.fit_generator(genereted_data.flow(X_train,Y_train, batch_size=batchSize), steps_per_epoch=stepPerEpoch, epochs=epochValue,validation_data=(X_validation,Y_validation), shuffle=1)
pickle_out = open("model_trained.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()