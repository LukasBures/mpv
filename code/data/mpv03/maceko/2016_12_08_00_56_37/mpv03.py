# coding: utf-8

import cv2
import numpy as np
from skimage.feature import hog


def classify(train_data, train_label, test_data):
	"""
	Provede klasifikaci dat pomoci SVM klasifikatoru.

	:param train_data: Vstupni list trenovacich sedotonovych obrazku.
	:rtype train_data: list of 2D ndarray, uint8

	:param train_label: Vstupni list trid, do kterych patri trenovaci sedotonove obrazky.
	:rtype train_label: list of int

	:param test_data: Vstupni list testovacich sedotonovych obrazku.
	:rtype test_data: list of 2D ndarray, uint8

	:return cls: Vystupni list trid odpovidajicich obrazkum v test_data.
	:rtype cls: list of int
	"""
	assert isinstance(train_data, list)
	assert isinstance(train_data[0], np.ndarray)
	assert isinstance(train_label, list)
	assert isinstance(train_label[0], int)
	assert isinstance(test_data, list)
	assert isinstance(test_data[0], np.ndarray)

	orientations = 8
	# pixels_per_cell = (4, 4)
	# cells_per_block = (5, 5)
	pixels_per_cell = (3, 4)
	cells_per_block = (6, 5)
	visualise = False

	train_hogs = []
	for d in train_data:
		feat = hog(d, orientations=orientations,
				   pixels_per_cell=pixels_per_cell,
				   cells_per_block=cells_per_block,
				   visualise=visualise)
		train_hogs.append(feat)

	svm = cv2.ml.SVM_create()
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_LINEAR)
	svm.setC(12)
	train_hogs = np.asarray(train_hogs, dtype='float32')
	train_label = np.asarray([train_label], dtype='int32')
	svm.train(train_hogs, cv2.ml.ROW_SAMPLE, train_label)

	test_hogs = []
	for t in test_data:
		feat = hog(t, orientations=orientations,
				   pixels_per_cell=pixels_per_cell,
				   cells_per_block=cells_per_block,
				   visualise=visualise)
		test_hogs.append(feat)

	_, predicted = svm.predict(np.asarray(test_hogs, dtype='float32'))
	predicted = np.squeeze(predicted)
	predicted = predicted.astype(dtype='int32')
	return predicted
