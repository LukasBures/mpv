# -*- coding: utf-8 -*-
"""
Created on 15:22 13.10.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""
import hashlib

name = ["PICEKL", "HERBIG", "FILIP", "SMOLIKD", "MULLERL"]

m = hashlib.md5()

for n in name:
    res = hashlib.md5(n).hexdigest()
    result = ''.join([i for i in res if not i.isdigit()])
    print n, "-", result


