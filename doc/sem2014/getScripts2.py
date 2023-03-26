# ! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os.path
import argparse
import httplib, urllib


def main():

  path_directory = "./scripts/" 
 
  parser = argparse.ArgumentParser(
      description='Get scripts from server')
  parser.add_argument('-u', '--user', help = 'User/Student login', default = 'test')
  parser.add_argument('-o', '--option', help= 'Download option', default = 0)# 0 jsou bodovane a  1 jsou nebodovane
  parser.add_argument('-t', '--task', help='Task', default = 'test')
  parser.add_argument('-p', '--userPackage', default = 'default')# vytvari slozky
  args = parser.parse_args()

  # Vytvoreni slozky na sem prace
  if(os.path.exists(path_directory) is False):
      os.makedirs(path_directory) 
  if(os.path.exists(path_directory+args.userPackage+'/') is False):
      os.makedirs(path_directory+args.userPackage+'/')      
  if(os.path.exists(path_directory+args.userPackage+'/'+args.user+'/') is False):
      os.makedirs(path_directory+args.userPackage+'/'+args.user+'/')       
  if(os.path.exists(path_directory+args.userPackage+'/'+args.user+'/'+args.task+'/') is False):
      os.makedirs(path_directory+args.userPackage+'/'+args.user+'/'+args.task+'/')       
      
  path_directory = path_directory+args.userPackage+'/'+args.user+'/'+args.task+'/'
  
 #Vytvoreni parametru http pozadavku
  params = urllib.urlencode({'login': args.user, 'taskStr': args.task, 'option': args.option})
  # Hlavicky http pozadavku
  headers = {"Content-type": "application/x-www-form-urlencoded",
             "Accept": "text/plain"}
  # Server pro pripojeni
  conn = httplib.HTTPConnection("neduchal.cz", 80)
  # Konkretni pozadavek 
  conn.request("POST", "/sako/utils/getScripts.php", params, headers)
  # Provedeni pozadavku
  response = conn.getresponse()  
  # Zpracovani vysledku
  data = response.read()
  # Ukonceni spojeni 
  conn.close() 
  # Vypsani odpovedi
  print 'Stav pripojeni : '
  print response.status, response.reason
  data = data.lstrip('\r\n')      
  
  data = data.split("%%")
  data[0] = data[0][1:]
  data[len(data)-1] = data[len(data)-1][0:-1]
  
  if len(data[0]) == 0:
    return 0
      
  for i in range(len(data)):
    f = open(path_directory + args.task + "_" + str(i) + ".py", "w");    
    f.write(data[i]);
    f.close();
  
  pass
  
  
if __name__ == "__main__":
    main()

