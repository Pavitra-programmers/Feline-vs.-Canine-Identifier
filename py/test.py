import cv2
import numpy as np
import pickle

pick = open('Model.sav','rb')
model = pickle.load(pick)
pick.close()

input_img = input('Path of img to be predicted \n')
imgpred = cv2.imread(input_img)
img_res = cv2.resize(imgpred,(32,32))
img_scale = img_res/255
img_reshape = np.reshape(img_scale, [1,32,32,3])
result = model.predict(img_reshape)
labal = np.argmax(result)

if labal == 0:
    print('its a cat')
else:
    print('Its a dog')