import numpy as np
import cv2

img = cv2.imread("../Train.png")
winSize = 20;
numOfSamples = 50
numOfRowsForNum = 5
numbers = 10
imgs = list() #trenovaci sada
i = 0
labels = list() # trenovaci labely
lab = 0


for j in range(numOfSamples-1):
    for k in range(numOfRowsForNum*numbers-1):
        temp = img[k*winSize:(k+1)*winSize, j*winSize:(j+1)*winSize ]
        imgs.append(temp)
        labels.append(lab)
        # print lab
        # cv2.imshow("x", imgs[i])
        # cv2.waitKey(0)
        i += 1
        if(i%((numOfSamples-1)*(numOfRowsForNum) )==0):
            lab += 1

numOfTestSamples = 1
imgsTest = list() #testovaci sada
i = 0
for j in range(numOfSamples, numOfSamples+numOfTestSamples,1):
    for k in range(numOfRowsForNum * numbers - 1):
        temp = img[k * winSize:(k + 1) * winSize, j * winSize:(j + 1) * winSize]
        imgsTest.append(temp)
        # labels.append(lab)
        # print lab
        #cv2.imshow("x", imgsTest[i])
        #cv2.waitKey(0)
        i += 1
        # if (i % ((numOfSamples - 1) * (numOfRowsForNum)) == 0):
        #     lab += 1
######################################################################################################################

for i in range(0, 10):
    cv2.imshow("0", imgs[i])
    cv2.waitKey(0)
labelsNew = ()

for i in range(0, len(labels)):
    for j in range(0)

print labels
cv2.waitKey(0)
cv2.destroyAllWindows()