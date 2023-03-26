import numpy as np
import cv2
from skimage.feature import hog

#nastaveni hogu
ori = 8
ppc = 4
cpb = 2

img = cv2.imread("../Train.png",0)
winSize = 20
numOfSamples = 69
numOfRowsForNum = 5
numbers = 10
imgs = list()
i = 0
labels = list()
lab = 0

for j in range(numOfSamples):
    for k in range(numOfRowsForNum*numbers):
        temp = img[k*winSize:(k+1)*winSize, j*winSize:(j+1)*winSize ]
        imgs.append(temp)
        labels.append(lab)
        i += 1

ind = 0
for i in range(len(labels)):

    if i%numOfRowsForNum == 0:
        if i != 0:
            ind +=1
    if ind == 10:
        ind = 0

    labels[i] = ind

numOfTestSamples = 1
imgsTest = list()
i = 0

for j in range(numOfSamples, numOfSamples+numOfTestSamples,1):
    for k in range(numOfRowsForNum * numbers - 1):
        temp = img[k * winSize:(k + 1) * winSize, j * winSize:(j + 1) * winSize]
        imgsTest.append(temp)
        i += 1

test_data, test_labels = imgsTest, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5,
                                        5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]

#################################################################################



def countHog(data,labels):

    featureVec = list()
    label = list()

    for i in range(len(data)):
        img = data[i]
        feat, hog_img = hog(img, orientations=ori, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb), visualise=True)
        featureVec.append(feat)
        label.append(labels[i])

    return featureVec, label

def train():

    train_fv, train_label = countHog(imgs,labels)

    svm = cv2.ml.SVM_create()
    # my_svm.setType(cv2.ml.SVM_C_SVC)
    # my_svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.train(np.asarray(train_fv, dtype='float32'), cv2.ml.ROW_SAMPLE, np.asarray([train_label], dtype='int32'))

    return svm

def test(svm):

    test_fv, test_label = countHog(test_data, test_labels)

    _, predicted = svm.predict(np.asarray(test_fv, dtype='float32'))

    result = list()

    for i in range(len(predicted)):
        result.append(predicted[i][0])

    print "Chybne klasifikovano:",
    print np.sum(result != np.asarray(test_label, dtype='float32')),
    print "/", len(result)
    err = (result != np.asarray(test_label, dtype='float32')).mean()
    print 'Error: %.2f %%' % (err * 100)
    print "--------------------------------------------------------"
    print "Vysledek: " + str(100 - (err*100)) + " %"
    print list(result)


trained_svm = train()
test(trained_svm)

