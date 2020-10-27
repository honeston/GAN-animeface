'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import h5py
import os

import numpy as np
from keras.preprocessing.image import load_img,img_to_array,array_to_img


def datagetBase(path,classAattribute):
    for idx in range(10000):
        img = load_img(path + str(idx + 1) + ").jpg",target_size=(270 ,480))
        nparreay = img_to_array(img)
        sagiri = nparreay.astype('float32')
        sagiri /= 255
        classAattribute2 = np.ones(2)
        classAattribute2[1] = 0
        #classAattribute = (0,1)#keras.utils.to_categorical(classAattribute, 2)
        yield (sagiri,classAattribute2)
data = datagetBase("HDanime\sagiri (",1)
x,y = data.__next__()
print(y.shape)
imga = array_to_img(x)
imga.show()
print(y)
