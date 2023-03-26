# coding: utf-8

from random import random as rnd

import cv2
import numpy as np
from numpy.linalg import norm


def computeSift(images):
	sifts = list()
	sift = cv2.xfeatures2d.SIFT_create()
	print "SIFT computing: %d/%d" % (0, len(images))
	for i in range(len(images)):
		one_sifts = sift.detectAndCompute(images[i], None)
		sifts.append([one_sifts[0][0:125], one_sifts[1][0:125]])
		if (i + 1) % int(len(images) / 10) == 0:
			print "SIFT computing: %d/%d" % (i + 1, len(images))

	return sifts


def chooseCenters(points, count):
	centers = []
	while len(centers) < count:
		index = int(rnd() * len(points))
		centers.append(points[index][0])
	return centers


def computeCenters(points, clazz):
	centers = []

	for c in range(len(clazz)):
		total = np.zeros(len(points[0][0]))
		for p in range(len(points)):
			total += points[p][0] * clazz[c][p]
		centers.append(total / float(sum(clazz[c])))
	return centers


def distanceCenters(points, centers):
	distances = np.zeros([len(centers), len(points)])
	for c in range(len(centers)):
		for p in range(len(points)):
			# noinspection PyTypeChecker
			distances[c, p] = np.sqrt(np.sum(np.power(np.subtract(centers[c], points[p]), 2), 0))
	return distances


def markNearest(distances):
	clazz = np.zeros([len(distances), len(distances[0])], 'uint8')
	for i in range(len(distances[0])):
		index = np.argmin(distances[[range(len(distances))], i])
		clazz[index, i] = 1
	return clazz


def labelCenters(centers, points, clazz):
	labeled_centers = []
	for c in range(len(clazz)):
		cls = np.zeros(len(clazz))
		for p in [i for i, v in enumerate(clazz[c]) if v == 1]:
			cls[points[p][1]] += 1
		labeled_centers.append([centers[c], np.argmax(cls), cls / np.sum(cls)])
	return labeled_centers


def makeHistograms(points, clazz):
	hist = np.zeros([len(clazz), len(clazz)])
	cnt = np.zeros(len(clazz))
	for p in range(len(points)):
		hist[points[p][1]][np.argmax(clazz[:, p])] += 1
		cnt[points[p][1]] += 1
	for c in range(len(clazz)):
		hist[c] /= cnt[c]
	return hist


def kMeans(points, count):
	centers = chooseCenters(points, count)

	iteration = 0
	while True:
		iteration += 1
		print iteration
		distances = distanceCenters(np.array(points)[:,0], centers)
		clazz = markNearest(distances)
		old_centers = centers

		centers = computeCenters(points, clazz)

		if (np.array(centers) == np.array(old_centers)).all():
			hists = makeHistograms(points, clazz)
			return centers, hists
		# return labelCenters(centers, points, clazz)
		# else:
		# 	print np.subtract(centers, old_centers)


def getSiftDescriptors(sifts, labels):
	descriptors = []
	for s in range(len(sifts)):
		for d in sifts[s][1]:
			descriptors.append([d, labels[s]])
	return descriptors

def makeHistogram(points, clazz):
	hist = np.zeros(len(clazz))
	for c in range(len(clazz)):
		hist[c] = np.sum(clazz[c])
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
		distances = distanceCenters(test[1], centers)
		clazz = markNearest(distances)
		hist = makeHistogram(test[1], clazz)
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
	descriptors = np.array([[1, 2], [2, 2], [3, 1], [2, 1], [5, 6],
							[7, 5], [6, 7], [9, 9], [9, 8], [8, 9],
							[7, 9], [1, 9], [1, 8], [1, 6], [2, 8],
							[3, 9], [4, 9], [6, 9], [2, 7], [2, 6],
							[8, 1], [7, 1], [6, 2], [6, 3], [6, 4]])
	descriptors = getSiftDescriptors(sifts, train_label)
	centers, histograms = kMeans(descriptors, n_visual_words)

	test_sifts = computeSift(test_data)
	test_label = classify(test_sifts, centers, histograms)

	return test_label
