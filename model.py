import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D,concatenate
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint
import keras.backend as K
import fnmatch





training_images is the array of density value
train_label is the array of the class

#########
# set up model for training
########

## set up training layer with layers.dense
## represent neurons for the neural layer.

model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

## compiling model:

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
## training the model 

model.fit(train_images, train_labels, epochs=5)



### evaluate the model accuracy


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)



### make prediction:

predictions = model.predict(test_images)

## the following show the "confidence"
## that the model correctly guess for each classification class
## np.argmax show the prediction class 
## that has highest confidence.


predictions[0]
np.argmax(predictions[0])

