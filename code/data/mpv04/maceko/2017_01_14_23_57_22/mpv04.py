# coding: utf-8

import cv2
import numpy as np


def makeCollage(L, R):
	both = np.zeros((np.max([L.shape[0], R.shape[0]]), L.shape[1] + R.shape[1], 3), dtype=L.dtype)
	both[0:L.shape[0], 0:L.shape[1]] = L
	both[0:R.shape[0], L.shape[1]:L.shape[1] + R.shape[1]] = R

	x = 1250
	y = x / float(L.shape[1] + R.shape[1]) * np.max([L.shape[0], R.shape[0]])
	both = cv2.resize(both, (x, int(y)))
	return both


# noinspection PyUnresolvedReferences
def computeMatchPoints(img_l, img_r):
	print "Compute SIFT points ...",
	s = cv2.xfeatures2d.SIFT_create()
	KP_l, Des_l = s.detectAndCompute(img_l, None)
	KP_r, Des_r = s.detectAndCompute(img_r, None)

	# visualizeMatches(img_l, img_r, KP_l, KP_r)
	print "OK"

	print "Match SIFT and prepare 2D points ...",
	matcher = cv2.BFMatcher()
	matches = matcher.knnMatch(Des_l, Des_r, k=2)
	print "OK"

	print "Lowe's ratio test ...",
	goodMatches = []
	for m, n in matches:
		if m.distance < 0.35 * n.distance:
			goodMatches.append(m)

	# visualizeGoodMatches(img_l, img_r, goodMatches, KP_l, KP_r)

	PT_l = np.float32([KP_l[m.queryIdx].pt for m in goodMatches]).reshape(-1, 2)
	PT_r = np.float32([KP_r[m.trainIdx].pt for m in goodMatches]).reshape(-1, 2)
	print "OK"
	return PT_l, PT_r


def visualizeMatches(img_l, img_r, KP_l, KP_r):
	print "Visualize Key points ...",
	ImgSIFT_l = cv2.drawKeypoints(img_l, KP_l[:], None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
	ImgSIFT_r = cv2.drawKeypoints(img_r, KP_r[:], None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

	cv2.imshow("Matches", makeCollage(ImgSIFT_l, ImgSIFT_r))
	cv2.waitKey(1)
	print "OK"


def visualizeGoodMatches(img_l, img_r, goodMatches, KP_l, KP_r):
	print "Visualize Good matches ...",
	both = makeCollage(cv2.cvtColor(img_l, cv2.COLOR_GRAY2RGB), cv2.cvtColor(img_r, cv2.COLOR_GRAY2RGB))
	for mat in goodMatches:
		(x1, y1) = KP_l[mat.queryIdx].pt
		(x2, y2) = KP_r[mat.trainIdx].pt

		B = np.random.randint(0, 256)
		G = np.random.randint(0, 256)
		R = np.random.randint(0, 256)

		cv2.circle(both, (int(x1), int(y1)), 3, (B, G, R), 4)
		cv2.circle(both, (int(x2) + img_l.shape[1], int(y2)), 3, (B, G, R), 4)
		cv2.line(both, (int(x1), int(y1)), (int(x2) + img_l.shape[1], int(y2)), (B, G, R), 2, cv2.LINE_AA)

	cv2.imshow("Good matches", both)
	cv2.waitKey(1)
	print "OK"


def computeEightPointAlgorithm(img_l, PT_l, PT_r):
	print "Add ones ...",
	l = np.array([[x[0], x[1], 1] for x in PT_l])
	r = np.array([[x[0], x[1], 1] for x in PT_r])
	print "OK"

	print "Normalize ...",
	N = np.array([[2.0 / img_l.shape[1], 0, -1],
				  [0, 2.0 / img_l.shape[0], -1],
				  [0, 0, 1]], dtype=np.float32)
	x_l = np.dot(N, l.T)
	x_r = np.dot(N, r.T)
	print "OK"

	print "Matrix A ...",
	A = np.array([x_l[0] * x_r[0], x_l[1] * x_r[0], x_r[0],
				  x_l[0] * x_r[1], x_l[1] * x_r[1], x_r[1],
				  x_l[0], x_l[1], np.ones(x_l.shape[1])])
	print "OK"

	print "SVD ...",
	U, _, _ = np.linalg.svd(A)
	Fi = -U[:, 8].reshape(3, 3)

	U2, D2, V2 = np.linalg.svd(Fi)
	D2[2] = 0

	F = np.dot(np.dot(U2, np.diag(D2)), V2)
	F2 = np.dot(np.dot(N.T, F), N)
	print "OK"
	return F2


def find_fundamental_matrix(img_left, img_right):
	"""	Compute fundamental matrix
	:param img_left: Left input image (2D ndarray, uint8)
	:param img_right: Right input image (2D ndarray, uint8)
	:return f: Computed fundamental matrix of size 3x3
	:rtype f: 2D ndarray, float
	"""
	assert isinstance(img_left, np.ndarray)
	assert isinstance(img_right, np.ndarray)

	PT_l, PT_r = computeMatchPoints(img_left, img_right)
	F = computeEightPointAlgorithm(img_left, PT_l, PT_r)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return F
