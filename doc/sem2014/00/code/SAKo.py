#! /usr/bin/python
# -*- coding: utf-8 -*-
# Nacteni knihoven 
import httplib, urllib
import numpy as np
import cv2
from skimage import io

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
  
def submit(login, passwd, taskStr, filename, func_name):
  
  m = __import__(taskStr + '_func')
  method = getattr(m, func_name)  
  #Vytvoreni parametru http pozadavku
  params = urllib.urlencode({'login': login,'passwd': passwd, 'taskStr': taskStr})
  # Hlavicky http pozadavku
  headers = {"Content-type": "application/x-www-form-urlencoded",
             "Accept": "text/plain"}
  # Server pro pripojeni
  conn = httplib.HTTPConnection("neduchal.cz", 80)
  # Konkretni pozadavek 
  conn.request("POST", "/sako/loadData.php", params, headers)
  # Provedeni pozadavku
  response = conn.getresponse()
  #print response.status, response.reason
  # Zpracovani vysledku
  data = response.read() 
  print data 
  
  result = {}
  
  test_data_str = data[6:]
  test_data_arr = test_data_str.split('##')
  for i in range(len(test_data_arr)):
    if i == 0:
      continue;
    test_data_split = test_data_arr[i].split('§')  
    if len(test_data_split) == 1:  
      image = io.imread(test_data_arr[i])
      if (len(image.shape) == 3):
        if(image.shape[2] == 3):
          image = cv2.cvtColor(image, cv2.cv.CV_RGB2BGR)
      res = {}    
      res['value'] = method(image)
      res['name'] = 'r'+ str(i)
    else:
      image = io.imread(test_data_split[0])
      if (len(image.shape) == 3):
        if(image.shape[2] == 3):
          image = cv2.cvtColor(image, cv2.cv.CV_RGB2BGR)
      res = {}    
      res['value'] = method(image, test_data_split[1])
      res['name'] = 'r'+ str(i)     
    result[i] = res
    
  lang = {}
  
  lang['name']  = 'language'
  lang['value'] = 'python' 
    
  result[len(result)+1] = lang;
  
  pack = {}
  
  pack['name']  = 'test_package'
  pack['value'] = test_data_arr[0]  
  
  result[len(result)+1] = pack;  
  
  with open(filename, 'r') as content_file:
    content = content_file.read()
    
  cont = {}  
  cont['name']  = 'script'
  cont['value'] = content
  
  result[len(result)+1] = cont;     
  
  resultStr = serialize( result )
  
  #Vytvoreni parametru http pozadavku
  params = urllib.urlencode({'login': login,'passwd': passwd, 'taskStr': taskStr, 'result': resultStr})
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
  # Vypsani delky vracenych dat
  print data
  # Ukonceni spojeni 
  
  conn.close()
  pass
