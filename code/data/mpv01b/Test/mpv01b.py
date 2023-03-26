# coding: utf-8

import cv2
import numpy as np


def getInterestPoints(images):
	interestPoints = list()
	sift = cv2.xfeatures2d.SIFT_create()
	for i in range(len(images)):
		gs = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
		interestPoints.append(sift.detectAndCompute(gs, None))
	return interestPoints


def putImageTo(final, image, position):
	x = position[0]
	y = position[1]
	final[x:x + len(image), y:y + len(image[0]),:] = image
	return final


def getImagePosition(image, offset):
	x = -1
	y = -1
	for a in range(min([len(image), len(image[0])])):
		rng = range(a + 1)
		if x == -1:
			for i in rng:
				px = image[i][a]
				if px.all() != 0:
					x = i + offset[0]
					break
		if y == -1:
			for j in rng:
				px = image[a][j]
				if px.all() != 0:
					y = j + offset[1]
					break
		if x != -1 and y != -1:
			return x, y


def placePiece(final, images, location, iPoints, matches, i, j):
	p1 = np.empty((len(matches), 2))
	p2 = np.empty((len(matches), 2))

	for k in range(len(matches)):
		(x, y) = iPoints[i][0][matches[k][0].queryIdx].pt
		(x1, y1) = iPoints[j][0][matches[k][0].trainIdx].pt
		p1[k] = (round(x, 0), round(y, 0))
		p2[k] = (round(x1, 0), round(y1, 0))

	homograph, _ = cv2.findHomography(p2, p1)
	img = cv2.warpPerspective(images[j], homograph, (final.shape[0], final.shape[1]))

	pos = getImagePosition(img, location[i])
	location[j] = pos
	putImageTo(final, images[j], pos)


def proceedMatching(images, iPoints, size):
	location = dict()
	placed = [0]
	toPlace = range(1, len(images))
	location[0] = (0, 0)
	final = putImageTo(np.zeros([size[1], size[0], 3], 'uint8'), images[0], location[0])

	bf = cv2.BFMatcher()
	while True:
		for i in placed:
			for j in toPlace:
				matches = bf.knnMatch(iPoints[i][1], iPoints[j][1], k=2)

				good = list()
				for m, n in matches:
					if m.distance < 0.2 * n.distance:
						good.append([m])
				if len(good) > 0:
					placePiece(final, images, location, iPoints, good, i, j)
					toPlace.remove(j)
					placed.append(j)
		if len(toPlace) == 0:
			break

	return final


def panorama(images, size):
	"""
	Provede tvorbu panoramaticky sestaveneho obrazku ze vstupnich dat.

	:param images: Seznam (list) vstupnich barevnych obrazku.
	:type images: Seznam 2D ndarrays (BGR obrazku).

	:param size: N-tice (tuple) velikosti obrazku, napriklad (1920, 1080).
	:type size: N-tice (tuple) - (uint, uint).

	:return stitched_img: Vystupni barevny obrazek o velikosti target_size.
	:rtype stitched_img: 2D ndarray
	"""
	iPoints = getInterestPoints(images)
	finalImage = proceedMatching(images, iPoints, size)

	return finalImage
