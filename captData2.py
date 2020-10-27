from PIL import ImageGrab as imgr
import time

#import keras
#from keras.models import load_model
#import numpy as np
#from keras.preprocessing.image import load_img,img_to_array,array_to_img

#model = load_model('testhozon.h5')
i = 0
while True:
    i += 1
    #print("please input path = ")
    #img = load_img(pathStr,target_size=(169,300))
    img = imgr.grab()
    img = img.resize((960,540))

    #img = load_img("sagiri (185).jpg",target_size=(169,300))
    #nparreay = img_to_array(img)
    #sagiri = np.ones((1, 169 ,300 ,3))
    #sagiri[0] = nparreay
    #fl = model.predict(sagiri, batch_size=128, verbose=True)

    strd = 'HDanime\data' + str(i) + '.jpg'
    #print(fl[0])
    #if np.argmax(fl[0]) == 1:

    #    strd = 'data\kari\sensei\hogehoge' + str(i) + '.jpg'
    #else:

    #    strd = 'data\kari\osensei\hogehoge' + str(i) + '.jpg'
    img.save(strd)

    time.sleep(3)
