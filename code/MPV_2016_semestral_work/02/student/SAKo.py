#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
:platform: Unix, Windows
:synopsis: Klient Systému Automatické Kontroly (SAKo)

.. moduleauthor:: Petr Neduchal <neduchal@ntis.zcu.cz>

Závislosti
----------
* os
* os.path
* urllib
* glob

**Ukázka použití klienta**

Vytvořte si na disku slozku sako a do ni složku src.
Do složky sako nahrajte soubor SAKo.py.
Do složky src nahrajte soubory, které budete odevzdávat.

Následně ve složce sako vytvořte soubor submit.py.
Do něj pak vložte následující kód.


**Význam jednotlivých parametrů**

*"./src/"* - Relativní cesta ke složce s odevzdávanými soubory

*"login"* - Váš login do systému SAKo

*"heslo"* - Vaše heslo do systému SAKo

*"uloha"* - Odevzdávaná úloha.

**Samotný kód**

.. code-block:: python

    #! /usr/bin/python
    # -*- coding: utf-8 -*-
    import SAKo
    SAKo.submit("./src/", "login", "heslo", "uloha")

"""

# Imports
import os
import os.path
import urllib
import glob

version = 1 * 1000000 + 1 * 1000 + 0 * 1
path_to_script = os.path.dirname(os.path.abspath(__file__))


def submit(p_dir_name, p_login, p_passwd, p_task):
    """
        Funkce pro odevzdání úlohy na server.

        :param p_dir_name:  Složka s odevzdávanými soubory.
        :type p_dir_name: str.
        :param p_login:  Login do systému SAKo.
        :type p_login: str.
        :param p_passwd:  Heslo do systému SAKo.
        :type p_passwd: str.
        :param p_task:  Název odevzdávané úlohy.
        :type p_task: str.


    """
    p_f = None
    url = 'http://147.228.124.51/sako/'
    if p_dir_name[-1:] == '/':
        p_dir_name = p_dir_name[:-1]
    files = glob.glob(p_dir_name + '/*.*')

    p_data = dict(login=p_login, password=p_passwd,
                  task=p_task, version=version, ticket='')
    u = urllib.urlopen(url, urllib.urlencode(p_data))
    respond = u.read().strip()

    if respond[0:3] == '0##':
        print 'Nepodarilo se ziskat listek'
    elif len(respond) == 0:
        print 'Server neodpovida'
    elif respond[0:9] == 'actualize':
        urllib.urlretrieve(respond[11:], os.path.abspath(__file__))
        print 'Klient byl aktualizovan. Provedte novy pokus o odevzdani'
    else:
        p_data['ticket'] = respond
        for i in range(len(files)):
            filename = files[i]
            print 'Oteviram soubor : ' + filename
            p_f = open(filename, 'rb')
            file_data = p_f.read()
            p_data['name' + str(i)] = filename
            p_data['file' + str(i)] = file_data
        print 'Komunikace se serverem...'
        u = urllib.urlopen(url, urllib.urlencode(p_data))
        respond = u.read().strip()
        if respond[0:3] == '0##':
            print "Chyba : "
            print respond[3:]
        else:
            print "Vysledek :"
            print respond
        if p_f is not None:
            p_f.close()
    return respond
