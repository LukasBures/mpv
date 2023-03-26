# -*- coding: utf-8 -*-
"""
Computer Vision - Semester Project 1/b
@author: Bc. Jan Hranicka
@email: jhranick@students.zcu.cz
@version: 1.0
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg
import logging
plt.rcParams['image.cmap'] = 'gray'





def panorama(imgs, target_size):
    """
    Provede tvorbu panoramaticky sestaveneho obrazku ze vstupnich dat.

    :param imgs: Seznam (list) vstupnich barevnych obrazku.
    :type imgs: Seznam 2D ndarrays (BGR obrazku).

    :param target_size: N-tice (tuple) velikosti obrazku, napriklad (1920, 1080).
    :type target_size: N-tice (tuple) - (uint, uint).

    :return stitched_img: Vystupni barevny obrazek o velikosti target_size.
    :rtype stitched_img: 2D ndarray
    """
    # Logging utility initialization
    logger = logging.getLogger('mpv1b')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    # Parameters
    GKERNEL_SIZE = (3, 3)
    RATIO = 0.75

    # Get random base image from the list
    rand_id = np.random.randint(0, len(imgs))
    # rand_id = 0
    img_base = imgs[rand_id]
    imgs.pop(rand_id)

    for i in range(len(imgs)):
        img_base_gs = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
        img_base_gblur = cv2.GaussianBlur(img_base_gs, GKERNEL_SIZE, 0)

        # Create SURF descriptor for getting keypoints and features
        descriptor = cv2.xfeatures2d.SURF_create()
        logger.debug('SUFT descriptor created')

        (img_base_kps, img_base_features) = descriptor.detectAndCompute(img_base_gblur, None)
        bfMatcher = cv2.DescriptorMatcher_create('BruteForce')      # Create BruteForce matcher

        # Find the most appropriate next image to stitch with the base image
        candidates = list()
        for j in range(len(imgs)):
            img_gs = cv2.cvtColor(imgs[j], cv2.COLOR_BGR2GRAY)
            img_gblur = cv2.GaussianBlur(img_gs, GKERNEL_SIZE, 0)
            (img_kps, img_features) = descriptor.detectAndCompute(img_gblur, None)

            img_matches = bfMatcher.knnMatch(img_features, trainDescriptors=img_base_features, k=2)
            logger.debug('[Img:%d] Number of matches: %s' % (j, len(img_matches)))

            filtered_matches = list()
            for m in img_matches:
                if len(m) == 2 and m[0].distance < m[1].distance * RATIO:
                    filtered_matches.append(m[0])
            logger.debug('[Img:%d] Number of filtered matches: %s' % (j, len(filtered_matches)))
            candidates.append({'id': j, 'kps': img_kps, 'features': img_features, 'matches': filtered_matches,
                               'gblur': img_gblur})

        # Find image with the most matches
        img_next = candidates[0]
        for item in candidates:
            if len(item['matches']) > len(img_next['matches']):
                img_next = item

        logger.debug('Best is id=%s with %s matches' % (img_next['id'], len(img_next['matches'])))

        # Keypoints
        kps_base = list()
        kps_next = list()
        for match in img_next['matches']:
            kps_base.append((img_base_kps[match.trainIdx]))
            kps_next.append(img_next['kps'][match.queryIdx])

        pts_base = np.array([kp.pt for kp in kps_base])
        pts_next = np.array([kp.pt for kp in kps_next])

        # Find the homography matrix
        H, status = cv2.findHomography(pts_base, pts_next, cv2.RANSAC, 4.0)

        H /= H[2, 2]
        H_inv = linalg.inv(H)
        extremes = dims(img_next['gblur'], H_inv)
        extremes = np.ceil(extremes)

        # Adjust sizes depending on base image
        extremes[0] = max(extremes[0], img_base_gblur.shape[1])
        extremes[1] = max(extremes[1], img_base_gblur.shape[0])

        H_move = np.matrix(np.identity(3), np.float32)

        if extremes[2] < 0:
            H_move[0, 2] += -extremes[2]
            extremes[0] += -extremes[2]

        if extremes[3] < 0:
            H_move[1, 2] += -extremes[3]
            extremes[1] += -extremes[3]

        H_inv_move = H_move * H_inv

        # Required final image width and height
        (img_width, img_height) = np.int32(extremes[:2])

        if len(imgs) == 1:
            (img_width, img_height) = target_size
        # (img_width, img_height) = target_size

        logger.debug('Image resized to: %sx%s' % (img_width, img_height))

        img_base_warp = cv2.warpPerspective(img_base, H_move, (img_width, img_height))
        logger.debug('Based image warped')

        img_next_warp = cv2.warpPerspective(imgs[img_next['id']], H_inv_move, (img_width, img_height))
        logger.debug('Next image warped')

        # Prepare final image
        img_final_base = cv2.subtract(img_base_warp, img_next_warp)

        # Now add the warped image
        img_stitched = cv2.add(img_final_base, img_next_warp, dtype=cv2.CV_8U)

        # plt.figure(0)
        # plt.imshow(cv2.cvtColor(img_stitched, cv2.COLOR_BGR2RGB), )
        # plt.show()

        imgs.pop(img_next['id'])
        img_base = img_stitched
    logger.debug('Image stitching done')

    # resized_img = cv2.resize(img_stitched, target_size)
    # plt.figure(0)
    # plt.imshow(cv2.cvtColor(img_stitched, cv2.COLOR_BGR2RGB), )
    # plt.show()
    # cv2.imwrite(r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP1\b\sampleData\stitched.png", img_stitched)

    return img_stitched


def dims(img, H):
    base_pts = list()
    [base_pts.append(np.ones(3)) for x in range(4)]
    (y, x) = img.shape

    base_pts[0][:2] = [0, 0]
    base_pts[1][:2] = [x, 0]
    base_pts[2][:2] = [0, y]
    base_pts[3][:2] = [x, y]

    extremes = list()
    [extremes.append(None) for x in range(4)]
    for pt in base_pts:
        # hpt = np.matmul(H, np.transpose(pt))
        hpt = np.matrix(H) * np.matrix(pt).T
        hpt = np.squeeze(np.array(hpt))
        normal_pt = np.array([hpt[0] / hpt[2], hpt[1] / hpt[2]], np.float32)

        for i in range(len(extremes)):
            if i < 2:
                if extremes[i] is None or normal_pt[i] > extremes[i]:
                    extremes[i] = normal_pt[i]
            else:
                if extremes[i] is None or normal_pt[i-2] < extremes[i]:
                    extremes[i] = normal_pt[i-2]

    extremes[2] = extremes[2] if extremes[2] < 0 else 0
    extremes[3] = extremes[3] if extremes[3] < 0 else 0
    return tuple(extremes)


if __name__ == '__main__':
    imgA = r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP1\b\sampleData\a.png"
    imgB = r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP1\b\sampleData\b.png"
    imgC = r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP1\b\sampleData\c.png"

    imageA = cv2.imread(imgA, cv2.IMREAD_COLOR)
    imageB = cv2.imread(imgB, cv2.IMREAD_COLOR)
    imageC = cv2.imread(imgC, cv2.IMREAD_COLOR)
    images = [imageA, imageB, imageC]

    panorama(images, (720, 404))
    exit(9)

    imgOrig = r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP1\b\sampleData\fox.png"
    imgStitched = r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP1\b\sampleData\stitched_fox.png"

    orig = cv2.imread(imgOrig, cv2.IMREAD_GRAYSCALE)
    stitched = cv2.imread(imgStitched, cv2.IMREAD_GRAYSCALE)

    subt = cv2.subtract(orig, stitched)
    plt.figure(0)
    plt.imshow(subt+255, )
    plt.show()

    np.savetxt('test.txt', np.subtract(orig, stitched))
