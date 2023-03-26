import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
print "---START---"


# Odmazat porovnavani vzdalenosti v tele bow, protoze uz to mam hotove v metode kmeans
# train = list()
# trainCls = list()
# for i in range(3):
#     for j in range(1,10,1):
#         path = "train/%d/000%d.jpg"%(i,j)
#         train.append(cv2.imread(path, 0))
#         trainCls.append(i)
#        # train.append(cv2.imread("/train/%d/000%d.jpg"%(i,j),cv2.COLOR_BGR2GRAY))
# ################################################################################
# test = list()
# res = list()
# for i in range(11,19,1):
#     test.append(cv2.imread(("test/00%d.jpg"%i), 0))
#################################################################################
def distance(p1, p2):
    l = min([len(p1),len(p2)])
    dist = 0
    for i in range(l):
        dist += (p1[i]-p2[i])**2
    dist = dist**(1./2.)

    return dist

def countAlfa(hist1, hist2):
    temp = 0
    size1 = 0
    size2 = 0

    for i in range(len(hist1)):
        temp += hist1[i]*hist2[i]
        size1 += hist1[i]**2
        size2 += hist2[i]**2
    size1 = size1**(.5)
    size2 = size2**(.5)
    alfa = np.arccos((temp/(size2*size1)))
    return alfa

def kmeans(descriptors, numOfWords, l):
    dimension = 128
    # step = l/numOfWords
    # step = 10
    keyWords = np.zeros((numOfWords, dimension))

    for i in range(numOfWords):
        keyWords[i] = descriptors[i]
    numOfIter = 0
    while True:
        labels = np.zeros(len(descriptors))
        dist = np.zeros((len(descriptors), numOfWords))
        for i in range(len(descriptors)):
            for j in range(numOfWords):
                dist[i][j] = distance(descriptors[i], keyWords[j])
        #         remake as fast fast possible
            labels[i] = np.argmin(dist[i])
        temp = np.zeros((numOfWords, dimension))
        tempCount = 0

        for i in range(len(keyWords)):
            for j in range(len(descriptors)):
                if labels[j] == i:
                    temp[i] += descriptors[j]
                    tempCount += 1
            temp[i] = temp[i]/tempCount
            tempCount = 0
        if (temp == keyWords).all():
            break
        else:
            keyWords = temp
            numOfIter += 1
            # print numOfIter
    print "Hurray"

    # for i in range(numOfWords):
    #     dis[i] = np.sum(np.power(np.subtract(keyWords[i],descriptors[i]),2),0)
    # print "HeyYou"
    # print "here"
    # distances = np.zeros((len(descriptors), numOfWords))
    # clusters = list(list())


    # for i in range(numOfWords):
    #     keyWords[i] = descriptors[i]
    #
    # for i in range(len(descriptors)):
    #     for j in range(numOfWords):
    #         distances[i, j] = distance(descriptors[i],keyWords[j])
    #     minId = np.argmin(distances[i])
    #     labels[i] = minId
    # # for i in range(len(descriptors)):
    # #     clusters[labels[i]].append(descriptors[i])
    # # not this wasy ee... saidly tudududum
    # print labels
    return keyWords

def bow(train_data, train_label, test_data, n_visual_words):
    allDescriptors = list()
    allLabels = list()
    # descriptors = list()
    sift = cv2.xfeatures2d.SIFT_create(125)
    numOfDescriptors = 0
    idx = list()
    for i in range(len(train_data)):
        _, d = sift.detectAndCompute(train_data[i], None)
        for j in range(len(d)):
            allDescriptors.append(d[j])
            allLabels.append(train_label[i])
        numOfDescriptors += len(d)
        if not train_label[i] in idx:
            idx.append(train_label[i])
    # print numOfDescriptors

    kWords = kmeans(allDescriptors, n_visual_words, numOfDescriptors)
    histograms = np.zeros((len(idx), n_visual_words))
    histogramsCount = np.zeros(len(idx))

    for i in range(len(allDescriptors)):
        # temp = np.zeros()
        tempDist = np.zeros(n_visual_words)
        for j in range(n_visual_words):
            tempDist[j] = distance(kWords[j], allDescriptors[i])
        # print np.argmin(tempDist)
        # print allLabels[i]
        histograms[allLabels[i]][np.argmin(tempDist)] += 1
        histogramsCount[allLabels[i]] += 1
    for i in range(len(idx)):
        for j in range(n_visual_words):
            histograms[i][j] = histograms[i][j]/histogramsCount[i]
    # plt.hist
    # print len(histograms)
    # for i in range(len(histograms)):
    #     plt.hist(histograms[i])
    #     plt.show()
    #     cv2.waitKey(0)
    print "Let's go and recognize IT!"
    print "now working...."
    result = np.zeros(len(test_data))
    print len(test_data)
    print len(result)
    for i in range(len(test_data)):
        _,desc = sift.detectAndCompute(test_data[i], None)
        tempHist = np.zeros(n_visual_words)
        for j in range(len(desc)):
            tempD = np.zeros((n_visual_words))
            for k in range(len(tempD)):
                tempD[k] = distance(desc[j], kWords[k])
            tempHist[np.argmin(tempD)] += 1
        tempHist = tempHist / len(desc)
        histDist = np.empty(len(histograms))
        for j in range(len(histograms)):
            histDist[j] = countAlfa(histograms[j], tempHist)
        result[i] = np.argmin(histDist)
        print "stil working...."
    print "----DONE---"
    return result

# print bow(train,trainCls,test,5)
# # cv2.imshow(("%d"%trainCls[0]), test[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print "---END--"
# print False==False in [False]
# print 1 in [1,2,3]
# print 5 in [1,2,3]