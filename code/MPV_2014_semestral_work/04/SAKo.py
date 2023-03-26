#! /usr/bin/python
# -*- coding: utf-8 -*-
# Nacteni knihoven 
import httplib, urllib
import numpy as np
import sys
import cv2
from skimage import io
import os

version = 0 * 1000000 + 6 * 1000 + 3 * 1

# KONSTANTY
DES_MATRIX = 0

def serialize( result ):
  resStr = '';
  for i in xrange(1,len(result)+1):
    if i == 1:
      resStr = resStr + result[i]['name'] + '##'
    else:
      resStr = resStr + '%' + result[i]['name'] + '##'
    if type(result[i]['value']) == str:
      resStr = resStr + result[i]['value']
    elif  ((type(result[i]['value']) == int) or (type(result[i]['value']) == float) or (type(result[i]['value']) == long)):
      resStr = resStr + str(result[i]['value'])
    elif type(result[i]['value']) == np.ndarray:
      rows = result[i]['value'].shape[0]
      cols = result[i]['value'].shape[1]
      for j in range(rows):
        for k in range(cols):
          resStr = resStr + str(result[i]['value'][j,k])
          if k!= cols-1:
            resStr = resStr + ','
        if j!= rows-1:
          resStr = resStr + ';'
  return resStr
  
def deserialize(data, tp):
  # tp 0 == MATRIX
  if tp == 0:
    data_rows = data.split(";")
    data_cols = data_rows[0].split(",")
    matrix = np.zeros([len(data_rows), len(data_cols)], np.double)
    for i in range(len(data_rows)):
      data_cols = data_rows[i].split(",")
      for j in range(len(data_cols)):
        matrix[i,j] = float(data_cols[j])
    return matrix
  else:
    print 'Neznamy dat vystupnich dat'
    return None
  
def getParams(login, passwd, taskStr):
  #Vytvoreni parametru http pozadavku
  params = urllib.urlencode({'login': login,'passwd': passwd, 'taskStr': taskStr, 'version' : version})
  # Hlavicky http pozadavku
  headers = {"Content-type": "application/x-www-form-urlencoded",
             "Accept": "text/plain"}
  # Server pro pripojeni
  conn = httplib.HTTPConnection("neduchal.cz", 80)
  # Konkretni pozadavek 
  conn.request("POST", "/sako/loadParams.php", params, headers)
  # Provedeni pozadavku
  response = conn.getresponse()
  # Zpracovani vysledku
  data = response.read()   
  conn.close()   
  return data.lstrip('\r\n')    
  
def getTestData(login, passwd, taskStr):
  #Vytvoreni parametru http pozadavku
  params = urllib.urlencode({'login': login,'passwd': passwd, 'taskStr': taskStr, 'version' : version})
  # Hlavicky http pozadavku
  headers = {"Content-type": "application/x-www-form-urlencoded",
             "Accept": "text/plain"}
  # Server pro pripojeni
  conn = httplib.HTTPConnection("neduchal.cz", 80)
  # Konkretni pozadavek 
  conn.request("POST", "/sako/loadData.php", params, headers)
  # Provedeni pozadavku
  response = conn.getresponse()
  # Zpracovani vysledku
  data = response.read()   
  conn.close()   
  return data.lstrip('\r\n')  
  
def handleTestData(data):
  data_arr = data.split('##')
  if(data_arr[0] == 'actualize'):
    data = urllib.urlretrieve(data_arr[1], "SAKo.py")
    print "Klient systemu SAKo byl aktualizovan. Prosim provedte novy pokus o odevzdani"
    return ''
  else:
    return data_arr
    
def sendResults(login, passwd, taskStr, resultStr, script):
  #Vytvoreni parametru http pozadavku
  params = urllib.urlencode({'login': login,'passwd': passwd, 'taskStr': taskStr, 'result': resultStr, 'version' : version, 'script': script})
  # Hlavicky http pozadavku
  headers = {"Content-type": "application/x-www-form-urlencoded",
             "Accept": "text/plain"}
  # Server pro pripojeni
  conn = httplib.HTTPConnection("neduchal.cz", 80)
  # Konkretni pozadavek 
  conn.request("POST", "/sako/index.php", params, headers)
  # Provedeni pozadavku
  response = conn.getresponse()  
  # Zpracovani vysledku
  data = response.read()
  # Ukonceni spojeni 
  conn.close() 
  # Vypsani odpovedi
  print 'Stav pripojeni : '
  print response.status, response.reason
  print data.lstrip('\r\n')
  
def imgToImgs(img):
  Imgs = []
  for i in range(img.shape[1]/img.shape[0]):
    Imgs.append(img[:,i*img.shape[0]:(i*img.shape[0])+img.shape[0]])
  return Imgs  
  
def submitFile(login, passwd, taskStr, filename):
  with open(filename, 'r') as content_file:
    content = content_file.read()      
  #Vytvoreni parametru http pozadavku
  params = urllib.urlencode({'login': login,'passwd': passwd, 'taskStr': taskStr, 'content': content})
  # Hlavicky http pozadavku
  headers = {"Content-type": "application/x-www-form-urlencoded",
             "Accept": "text/plain"}
  # Server pro pripojeni
  conn = httplib.HTTPConnection("neduchal.cz", 80)
  # Konkretni pozadavek 
  conn.request("POST", "/sako/submitFile.php", params, headers)
  # Provedeni pozadavku
  response = conn.getresponse()  
  # Zpracovani vysledku
  data = response.read()
  # Ukonceni spojeni 
  conn.close() 
  # Vypsani odpovedi
  print 'Stav pripojeni : '
  print response.status, response.reason
  print data.lstrip('\r\n')     
     
# Submit function  
def submit(login, passwd, taskStr, filename, func_name):  
  # Nastaveni promennych
  result = {}
  m = __import__(os.path.basename(filename)[:-3])
  method = getattr(m, func_name)  
  # Stazeni a zpracovani testovacich dat
  sub_par = getParams(login, passwd, taskStr)  
  
  par_arr = handleTestData(sub_par)
  if(len(par_arr) == 0):
    sys.exit(0)
    
  if(int(sub_par) != 3):
    data = getTestData(login, passwd, taskStr)  
    data_arr = handleTestData(data)  
  else:
    data_arr = []
  
  if(type(data_arr) != str):
    if(int(sub_par) == 0):
      # Ziskani odezev metody na testovaci img
      for i in range(len(data_arr)):
        if i == 0:
          continue;
        data_split = data_arr[i].split('&')  
        if len(data_split) == 1:  
          image = io.imread(data_arr[i])
          if (len(image.shape) == 3):
            if(image.shape[2] == 3):
              image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          res = {}    
          res['value'] = method(image)
          res['name'] = 'r'+ str(i)
        else:
          image = io.imread(data_split[0])
          if (len(image.shape) == 3):
            if(image.shape[2] == 3):
              image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          res = {}    
          res['value'] = method(image, float(data_split[1]))
          res['name'] = 'r'+ str(i)     
        result[i] = res  
    elif (int(sub_par) == 1):
      Imgs = []
      ImgsParams = []
      for i in range(len(data_arr)):
        if i == 0:
          continue;
        if len(data_arr) == 2: 
          image = io.imread(data_arr[i])
          if (len(image.shape) == 3):
            if(image.shape[2] == 3):
              image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
          Imgs = imgToImgs(image)
        else:  
          print 'Nacitam ' + str(i) + '. obrazek'          
          data_split = data_arr[i].split('&')           
          if len(data_split) == 1:  
            image = io.imread(data_arr[i])
            if (len(image.shape) == 3):
              if(image.shape[2] == 3):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)            
            Imgs.append(image)              
          else:
            image = io.imread(data_split[0])
            if (len(image.shape) == 3):
              if(image.shape[2] == 3):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
            Imgs.append(image)        
            ImgsParams.append(float(data_split[1]))
      if len(ImgsParams) == 0:
        res = {}    
        res['value'] = method(Imgs)
        res['name'] = 'r'    
        result[1] = res  
      else:
        res = {}    
        res['value'] = method(Imgs, ImgsParams)
        res['name'] = 'r'    
        result[1] = res    
    elif (int(sub_par) == 2):
      data = deserialize(data_arr[1], DES_MATRIX)
      res = {}    
      res['value'] = method(data)
      res['name'] = 'r'    
      result[1] = res    
    elif (int(sub_par) == 3):
      res = {}    
      res['value'] = method()
      res['name'] = 'r'    
      result[1] = res         
    else : 
      print 'Neznamy typ odevzdavaci funkce'
    # Pridani nekolika promennych do posilanych dat  
    lang = {}
    lang['name']  = 'language'
    lang['value'] = 'python'     
    result[len(result)+1] = lang;
    pack = {}  
    pack['name']  = 'test_package'    
    if(int(sub_par) != 3):    
      pack['value'] = data_arr[0]    
    else:
      pack['value'] = 0   
    result[len(result)+1] = pack;  
      
      
    # Nacteni uzivatelskeho skriptu a pripojeni k posilanym datum
    with open(filename, 'r') as content_file:
      content = content_file.read()     
    # Vytvoreni tzv result stringu  
    resultStr = serialize( result )
    sendResults(login, passwd, taskStr, resultStr, content)    
  pass
