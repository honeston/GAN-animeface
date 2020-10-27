import cv2
import os  #必ず必要

cascade = cv2.CascadeClassifier("lbpcascade_animeface.xml")
filePathArray = os.listdir("./face/")

for path in filePathArray:
    if not (path.find(".jpg") or path.find(".png")):
        continue
    image = cv2.imread("./face/" + path)
    if image is None:
        continue
    #print(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, flags=0, minSize = (128, 128))
    for i,(x, y, w, h) in enumerate(faces):
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        image = image[y:y+h,x:x+w,:]
        cv2.imwrite('./save/' +str(i) + path, image)
