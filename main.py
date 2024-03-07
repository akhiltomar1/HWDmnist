import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout



(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape, x_test, y_test)

def plot_ip_img(i):
    plt.imshow(x_train[i], cmap='binary')
    plt.title(y_train[i])
    plt.axes('off')
    plt.show()

# for i in range(10):
#     plot_ip_img(i)

#PREPROCESS DATA

#Normalizing image to [0,1] range
x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255

#expand dimensions to 28,28,1
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

#covert classes to one hot vector
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)



model = Sequential()

#First Convolutional Layer
model.add(Conv2D(32,(3,3), input_shape = (28,28,1), activation = 'relu'))
model.add(MaxPool2D((2,2)))

#ip shape is given only once , increase np. of units
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())
#Prevents overfitting
model.add(Dropout(0.25))
#used for classification
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

#callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=4, verbose=1)

mc = ModelCheckpoint("./bestmodel.h5", monitor='val_accuracy', verbose=1, save_best_only= True)

cb = [es,mc]

#MODEL TRAINING

his = model.fit(x_train,y_train, epochs=10, validation_split=0.3, callbacks=cb)


#Evaluate model
model_s = keras.models.load_model("./bestmodel.h5")
score = model_s.evaluate(x_test,y_test)
print("Model Accuracy ", {score[1]})

