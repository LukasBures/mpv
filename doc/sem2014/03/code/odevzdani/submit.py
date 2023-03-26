#! /usr/bin/python
# -*- coding: utf-8 -*-
# Nacteni knihoven 
import SAKo

#-----------------------------------------------------------------------------
# Vyplne svuj login a heslo, nic jineho neni dovoleno menit!
#-----------------------------------------------------------------------------

login = 'lbures'
passwd = 'testik2'

SAKo.submit(login, passwd, 'mpv03', './mpv03_func.py', 'SVMtest')
SAKo.submitFile(login, passwd, 'mpv03', "DigitsSVM.dat")
