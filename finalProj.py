import os
from tqdm import tqdm
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten, MaxPool2D, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


DATADIR = "D:/Dogs/stanford-dogs-dataset/Images" 
x = []
y = []

def get_labels():
    labels = []
    for directory in os.listdir(DATADIR):
        directoryString = str(directory)
        labels.append(directoryString[directoryString.index('-') + 1:])
    #to_categorical(labels)
    return labels

def get_data():

    for directory in os.listdir(DATADIR):
        directoryString = str(directory)
        i = 0
        for image_file in tqdm(os.listdir(os.path.join(DATADIR,directory))):
            path = os.path.join(DATADIR,directory,image_file)
            image = cv2.imread(path,cv2.IMREAD_COLOR)
            image = cv2.resize(image,(150,150))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x.append(np.array(image))
            y.append(directoryString[directoryString.index('-') + 1:])
            i = i + 1
            if (i == 100):
                break

labels = get_labels()
get_data()
num_classes = len(labels)
x = np.array(x)
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.33,random_state=23)
x_test, x_val, y_test, y_val = train_test_split(x_test,y_test,test_size=.2,random_state=25)

train_datagen = ImageDataGenerator(
                rescale=np.float32(1./255),
                horizontal_flip=True,
                rotation_range=5
)
train_generator = train_datagen.flow_from_directory(DATADIR,
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator()

train_datagen.fit(x_train)

y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(150,150,3)))
model.add(MaxPool2D(pool_size=(2,2),padding="same"))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding="same"))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(120, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=10,
                    validation_data=val_datagen.flow(x_val,y_val,batch_size=32))

evaluation = model.evaluate(x_test,y_test)
print(evaluation)
'''
model.fit_generator(train_generator,steps_per_epoch=100,
                    epochs=5)
'''
