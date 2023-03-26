# coding: utf-8
from __future__ import division

def my_histogram(img):
	h = len(img)
	w = len(img[0])
	hist = list()

	for i in range(256):
		hist.append(0)

	for i in range(w):
		for j in range(h):
			hist[img[i][j]] += 1
	for i in range(len(hist)):
		hist[i] /= float(w * h)
	assert sum(hist) == 1
	return hist


def myMean(hist, begin, end):
	mu = 0
	for i in range(begin, end):
		mu += hist[i] * i
	return mu


def myOmega(hist, begin, end):
	om = 0
	for i in range(begin, end):
		om += hist[i]
	return om


def findThreshold(hist):
	thr = 0
	sig = 0
	muT = myMean(hist, 0, len(hist) - 1)
	for i in range(1, len(hist) - 1):
		mu0 = myMean(hist, 0, i)
		om0 = myOmega(hist, 0, i)
		om1 = myOmega(hist, i, len(hist) - 1)

		if om0 * om1 != 0:
			sig2B = pow(muT * om0 - mu0, 2) / (om0 * om1)
			if sig < sig2B:
				sig = sig2B
				thr = i
	return thr - 1


def otsu(img):
	"""
	Vypocet optimalniho prahu pomoci Otsuovo metody.

	:param img: Vstupni sedotonovy obrazek.
	:type img: 2D ndarray of uint8

	:return threshold: Vystupni hodnota optimalniho prahu.
	:rtype threshold: int
	"""
	hist = my_histogram(img)

	return findThreshold(hist)
