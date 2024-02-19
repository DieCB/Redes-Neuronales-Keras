import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import regularizers

datos=mnist.load_data()
(x_train, y_train), (x_test, y_test)=datos


x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)

x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')

x_trainv = x_trainv/255
x_testv = x_testv/255

num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()

model.add(Dense(100, batch_input_shape=(None, 784)))
model.add(Activation('sigmoid'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

history=model.fit(x_trainv,y_trainc,batch_size=200,epochs=20,verbose=1, 
                  validation_data=(x_testv, y_testc))
