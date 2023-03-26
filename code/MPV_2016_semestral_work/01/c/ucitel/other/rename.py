# -*- coding: utf-8 -*-
"""
Created on 17:09 8.10.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import os


path = "../../data/src/"

i = 0
for filename in os.listdir(path):

    os.rename(path + filename, path + str(i) + ".jpg")

    i += 1
