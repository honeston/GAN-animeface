import cv2
import os  #必ず必要

cascade = cv2.CascadeClassifier("lbpcascade_animeface.xml")
filePathArray = os.listdir("./gazo3/")

for path in filePathArray:
    if not (path.find(".jpg") or path.find(".png")):
        continue
    image = cv2.imread("./gazo3/" + path)
    if image is None:
        continue
    #print(path)
    #iamge = cv2.resize(image,)
    cv2.imwrite('./gazo4/' + path, image)
