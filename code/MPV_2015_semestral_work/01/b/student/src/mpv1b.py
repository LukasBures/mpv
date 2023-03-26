# -*- coding: utf-8 -*-
"""
Created on 18:30 7.10.15

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits:
@version: 2.0.0
"""


def segment(gsimg):
    """
    Provede segmentaci pomoci MaxFlow algoritmu.

    :param gsimg: Vstupni sedotonovy obrazek, 255 reprezentuje pozadi a 0 popredi.
    :rtype gsimg: 2D ndarray, uint8

    :return segmentation: Vystupni binarni obrazek, hodnota 1 reprezentuje objekt a 0 pozadi.
    :rtype segmentation: 2D ndarray, dtype=bool
    """



    return segmentation