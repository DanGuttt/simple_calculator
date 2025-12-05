import requests
from bs4 import BeautifulSoup
import re


def extraire_nombre_bs4(url):

    response = requests.get(url)
    a = response.text
    a = a[1:-1]
    if response.status_code != 200:
        raise Exception(f"Erreur lors de l'accès au site : {response.status_code}")
    return a



def appel(nb1,nb2):
    url1 = f'https://add-api-71kj.onrender.com/calcm?nombre={nb1}'
    url2 = f'https://add-api-71kj.onrender.com/calcp?nombre={nb2}'
    return int(extraire_nombre_bs4(url1)),int(extraire_nombre_bs4(url2))

print(appel(1,2))
