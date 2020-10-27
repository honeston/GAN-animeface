import numpy as np
from keras.preprocessing.image import load_img,img_to_array,array_to_img

sagiridataNum = 476
nosagiridataNum = 536
testzise = 50
sagiri = np.ones((sagiridataNum, 169 ,300 ,3))
sagiri_ = np.ones((sagiridataNum,1))
for idx in range(sagiridataNum - testzise):
    img = load_img("data\sagiri\sagiri (" + str(idx + 1) + ").jpg",target_size=(169,300))
    nparreay = img_to_array(img)
    sagiri[idx] = nparreay
nosagiri = np.ones((nosagiridataNum, 169 ,300 ,3))
nosagiri_ = np.ones((nosagiridataNum,1))
for idx in range(nosagiridataNum - testzise):
    img = load_img("data\osagiri\osagiri (" + str(idx + 1) + ").jpg",target_size=(169,300))
    nparreay = img_to_array(img)
    nosagiri[idx] = nparreay
alldata =  np.concatenate([sagiri,nosagiri],axis=0)
alldata_ =  np.concatenate([sagiri_,nosagiri_],axis=0)

testsagiri = np.ones((testzise, 169 ,300 ,3))
testsagiri_ = np.ones((testzise,1))
i = 0
for idx in range(sagiridataNum - testzise , sagiridataNum):
    img = load_img("data\sagiri\sagiri (" + str(idx + 1) + ").jpg",target_size=(169,300))
    nparreay = img_to_array(img)
    testsagiri[i] = nparreay
    i+= 1
testnosagiri = np.ones((testzise, 169 ,300 ,3))
testnosagiri_ = np.ones((testzise,1))
i = 0
for idx in range(nosagiridataNum - testzise , nosagiridataNum):
    img = load_img("data\osagiri\osagiri (" + str(idx + 1) + ").jpg",target_size=(169,300))
    nparreay = img_to_array(img)
    testnosagiri[i] = nparreay
    i+=1
testalldata =  np.concatenate([testsagiri,testnosagiri],axis=0)
testalldata_ =  np.concatenate([testsagiri_,testnosagiri_],axis=0)

imga = array_to_img(testalldata[60])
imga.show()
print('shape:', testalldata.shape)
