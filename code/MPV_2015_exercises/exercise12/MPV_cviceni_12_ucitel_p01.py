# -*- coding: utf-8 -*-
"""
Created on 15:40 1.12.15

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: Ing. Miroslav Hlavac
@version: 1.0.0
"""

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join


def align_to(ref_shape, input_shape, n_points):
    # align shape to reference shape
    npoints = input_shape.shape[0] / 2
    output_shape = np.zeros_like(input_shape)

    xx1 = 0
    yy1 = 0
    xx2 = 0
    yy2 = 0
    z = 0
    c1 = 0
    c2 = 0
    w = n_points

    for i in range(0, npoints):
        x1 = ref_shape[i]
        y1 = ref_shape[i + npoints]

        x2 = input_shape[i]
        y2 = input_shape[i + npoints]

        z = z + x2 * x2 + y2 * y2
        xx1 += x1
        yy1 += y1
        xx2 += x2
        yy2 += y2
        c1 = c1 + x1 * x2 + y1 * y2
        c2 = c2 + y1 * x2 - x1 * y2

    aa = np.array(([xx2, -yy2, w, 0], [yy2, xx2, 0, w], [z, 0, xx2, yy2], [0, z, -yy2, xx2]))
    bb = np.array(([xx1], [yy1], [c1], [c2]))

    solution = np.linalg.solve(aa, bb)

    a = solution[0]
    b = solution[1]
    tx = solution[2]
    ty = solution[3]

    for k in range(0, npoints):
        x = input_shape[k]
        y = input_shape[k + npoints]

        output_shape[k] = (a * x + (-b) * y) + tx
        output_shape[k + npoints] = (b * x + a * y) + ty

    return np.array(([output_shape, a, b, tx, ty]))


def nothing(x):
    pass


def reset_pos(x):
    cv2.setTrackbarPos('1', 'shape', 250)
    cv2.setTrackbarPos('2', 'shape', 250)
    cv2.setTrackbarPos('3', 'shape', 250)
    cv2.setTrackbarPos('4', 'shape', 250)
    cv2.setTrackbarPos('5', 'shape', 250)


def prepare_data(path):
    pts_ext = ".pts"

    files = [join(path, f) for f in listdir(path)
             if (isfile(join(path, f)) and f.endswith(pts_ext))]

    n_samples = files.__len__()
    first_file = list(open(files[0]))
    n_points = int(first_file[1][first_file[1].find(':')+1:first_file[1].__len__()])

    points_x = np.zeros((n_points, n_samples))
    points_y = np.zeros((n_points, n_samples))

    for index, f in enumerate(files):
        f_s = open(f)
        shape = list(f_s)
        shape.pop(0)
        shape.pop(0)
        shape.pop(0)
        shape.pop(78)
        shape_num = list()
        for line in shape:
            for strnum in line.split():
                shape_num.append(float(strnum))

        points = np.asarray(shape_num)
        points_x[:, index] = points[::2]
        points_y[:, index] = points[1::2]

    points_norm_x = np.copy(points_x)
    points_norm_y = np.copy(points_y)

    for i in range(0, n_samples):
        x = np.mean(points_x[:, i])
        y = np.mean(points_y[:, i])

        points_norm_x[:, i] -= x
        points_norm_y[:, i] -= y

    points_norm = np.concatenate((points_norm_x, points_norm_y))
    mean_shape_norm = np.mean(points_norm, 1)
    ref_shape_norm = mean_shape_norm

    for i in range(0, 30):
        for j in range(0, n_samples):
            x = align_to(ref_shape_norm, points_norm[:, j], n_points)
            points_norm[:, j] = x[0]

        new_mean_shape_norm = np.sum(points_norm, 1) / n_samples
        new_mean_shape_norm = align_to(mean_shape_norm, new_mean_shape_norm, n_points)[0]
        diff = np.linalg.norm(new_mean_shape_norm - ref_shape_norm)

        if diff <= 0.0001:
            break

        ref_shape_norm = new_mean_shape_norm

    mean_shape_norm = np.mean(points_norm, 1)
    return mean_shape_norm, points_norm, n_points


def create_model(pts_norm):
    """
        PCA and Dimension Reduction
    :param pts_norm:
    :return:
    """
    cov_mat = np.cov(pts_norm, rowvar=1)
    w, v = np.linalg.eig(cov_mat)

    w_proc = np.divide(w, np.sum(w))

    ih = np.where(np.cumsum(w_proc) < 0.99)

    e_vect = v[:, 0:np.max(ih)]
    e_val = w[0:np.max(ih)]

    return e_vect, e_val


def visualization(e_vect, e_val, n_points, mean_shape_norm):

    img = np.zeros((480, 640, 3), np.uint8) + 255

    cv2.namedWindow('shape')

    tr_one_max = 3 * e_val[0] ** (1./2.)
    tr_two_max = 3 * e_val[1] ** (1./2.)
    tr_three_max = 3 * e_val[2] ** (1./2.)
    tr_four_max = 3 * e_val[3] ** (1./2.)
    tr_five_max = 3 * e_val[4] ** (1./2.)

    cv2.createTrackbar('1', 'shape', 250, 500, nothing)
    cv2.createTrackbar('2', 'shape', 250, 500, nothing)
    cv2.createTrackbar('3', 'shape', 250, 500, nothing)
    cv2.createTrackbar('4', 'shape', 250, 500, nothing)
    cv2.createTrackbar('5', 'shape', 250, 500, nothing)

    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'shape', 0, 1, reset_pos)

    c = np.zeros(e_vect.shape[1])

    lines_joints = np.array(([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
                             [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 0], [18, 19], [19, 20],
                             [20, 21], [21, 22], [22, 23], [23, 18], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29],
                             [29, 24], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38],
                             [39, 30], [39, 40], [40, 41], [41, 42], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48],
                             [48, 49], [49, 50], [50, 51], [51, 52], [52, 43], [53, 54], [54, 55], [55, 56], [56, 57],
                             [57, 58], [58, 59], [59, 60], [60, 53], [70, 71], [71, 72], [73, 74], [74, 75]))

    while 1:
        cv2.imshow('shape', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        c[0] = -tr_one_max + cv2.getTrackbarPos('1', 'shape') * tr_one_max/250
        c[1] = -tr_two_max + cv2.getTrackbarPos('2', 'shape') * tr_two_max/250
        c[2] = -tr_three_max + cv2.getTrackbarPos('3', 'shape') * tr_three_max/250
        c[3] = -tr_four_max + cv2.getTrackbarPos('4', 'shape') * tr_four_max/250
        c[4] = -tr_five_max + cv2.getTrackbarPos('5', 'shape') * tr_five_max/250

        s = cv2.getTrackbarPos(switch, 'shape')
        img = np.zeros((480, 640, 3), np.uint8) + 255

        if s == 0:
            shape = mean_shape_norm + np.dot(e_vect, np.zeros(e_vect.shape[1]))
            shape[:78] += 320
            shape[78:] += 240
            for i in range(0, n_points):
                cv2.circle(img, (int(shape[i]), int(shape[i+n_points])), 3, (0, 0, 255), -1)
            for i in range(0, lines_joints.shape[0]):
                cv2.line(img, (int(shape[lines_joints[i, 0]]), int(shape[lines_joints[i, 0]+n_points])),
                         (int(shape[lines_joints[i, 1]]), int(shape[lines_joints[i, 1]+n_points])), (255, 0, 0), 2)
        else:
            shape = mean_shape_norm + np.dot(e_vect, c)
            shape[:78] += 320
            shape[78:] += 240
            for i in range(0, n_points):
                cv2.circle(img, (int(shape[i]), int(shape[i+n_points])), 3, (0, 0, 255), -1)
            for i in range(0, lines_joints.shape[0]):
                cv2.line(img, (int(shape[lines_joints[i, 0]]), int(shape[lines_joints[i, 0]+n_points])),
                         (int(shape[lines_joints[i, 1]]), int(shape[lines_joints[i, 1]+n_points])), (255, 0, 0), 2)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    dataset_path = "./img/"
    mean_shape_n, points_n, n = prepare_data(dataset_path)
    vect, val = create_model(points_n)
    visualization(vect, val, n, mean_shape_n)
