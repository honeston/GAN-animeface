import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
import numpy as np
import math
from PIL import Image


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def getImage():
    X = []
    for picture in list_pictures('./eroimages/'):
        img = img_to_array(load_img(picture, target_size=(128,128,3)))
        X.append(img)

    npX =  np.array(X)
    return npX

X_train = getImage()
#X_train =  0.299 * X_train[:,:, :, 0] + 0.587 * X_train[:,:, :, 1] + 0.114 * X_train[:,:, :, 2]
#X_train = (X_train - 127.5)/127.5

image = combine_images(X_train)
image = image*127.5+127.5
Image.fromarray(X_train[1].astype(np.uint8)).save(
    "generated_image.png")
print(X_train)
print(X_train.shape)
