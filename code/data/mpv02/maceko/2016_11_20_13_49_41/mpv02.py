# coding: utf-8

import cv2
import numpy as np
from numpy.linalg import norm


def computeSift(images):
	sifts = list()
	sift = cv2.xfeatures2d.SIFT_create(125)
	print "SIFT computing: %d/%d" % (0, len(images))
	for i in range(len(images)):
		one_sifts = sift.detectAndCompute(images[i], None)
		sifts.append([one_sifts[0][0:125], one_sifts[1][0:125]])
		if (i + 1) % int(len(images) / 10) == 0:
			print "SIFT computing: %d/%d" % (i + 1, len(images))

	return sifts


def chooseCenters(points, count):
	centers = []
	index = 0
	while len(centers) < count:
		# index = int(rnd() * len(points))
		centers.append(points[index][0])
		index += 1
	return centers


def computeCenters(points, clazz, count):
	centers = np.zeros([count, len(points[0][0])])
	counts = np.zeros(count)
	for p in range(len(points)):
		centers[clazz[p]] += points[p][0]
		counts[clazz[p]] += 1
	centers /= [[float(x)] * len(points[0][0]) for x in counts]
	return centers


def getClassForPoints(points, centers):
	distances = np.zeros(len(points))
	for p in range(len(points)):
		# noinspection PyTypeChecker
		distances[p] = np.argmin(np.sum(np.power(np.subtract(centers, points[p]), 2), 1))
	return distances


def makeHistograms(points, clazz, count):
	hist = np.zeros([count, count])
	cnt = np.zeros(count)
	for p in range(len(points)):
		hist[points[p][1]][clazz[p]] += 1
		cnt[points[p][1]] += 1
	for c in range(count):
		hist[c] /= cnt[c]
	return hist


def kMeans(points, count):
	centers = chooseCenters(points, count)

	iteration = 0
	while True:
		iteration += 1
		print "K-Means iteration: %d" % iteration
		clazz = getClassForPoints(np.array(points)[:, 0], centers)
		# clazz = markNearest(distances)

		old_centers = centers
		centers = computeCenters(points, clazz, count)

		if (np.array(centers) == np.array(old_centers)).all():
			hists = makeHistograms(points, clazz, count)
			return centers, hists


def getSiftDescriptors(sifts, labels):
	descriptors = []
	for s in range(len(sifts)):
		for d in sifts[s][1]:
			descriptors.append([d, labels[s]])
	return descriptors


def makeHistogram(points, clazz, count):
	hist = np.zeros(count)
	for p in range(len(points)):
		hist[clazz[p]] += 1
	hist /= len(points)
	return hist


def computeAngles(hist, centers_hist):
	angles = []
	for i in range(len(centers_hist)):
		u = hist
		v = centers_hist[i]
		c = np.dot(u, v) / norm(u) / norm(v)
		a = np.arccos(np.clip(c, -1, 1))
		angles.append(a)
	return angles


def classify(sifts, centers, histograms):
	labels = []
	for test in sifts:
		clazz = getClassForPoints(test[1], centers)
		hist = makeHistogram(test[1], clazz, len(histograms))
		angles = computeAngles(hist, histograms)
		labels.append(np.argmin(angles))
	return labels


def bow(train_data, train_label, test_data, n_visual_words):
	"""
	:param train_data: List of train images in gray scale.
	:rtype train_data: list of 2D ndarray, uint8
	:param train_label: List of class labels of data images.
	:rtype train_label: list of int
	:param test_data: List of test images in gray scale.
	:rtype test_data: list of 2D ndarray, uint8
	:param n_visual_words: Number of visual words.
	:rtype n_visual_words: int
	:return cls: List of class for test data.
	:rtype cls: list of int
	"""

	assert isinstance(train_data, list)
	assert isinstance(train_data[0], np.ndarray)
	assert isinstance(train_label, list)
	assert isinstance(train_label[0], int)
	assert isinstance(test_data, list)
	assert isinstance(test_data[0], np.ndarray)
	assert isinstance(n_visual_words, int)

	sifts = computeSift(train_data)
	descriptors = getSiftDescriptors(sifts, train_label)
	for d in descriptors:
		print d
	centers, histograms = kMeans(descriptors, n_visual_words)

	test_sifts = computeSift(test_data)
	test_label = classify(test_sifts, centers, histograms)

	return test_label
