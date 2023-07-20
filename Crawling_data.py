import requests
from bs4 import BeautifulSoup

# 인증키 
apikey = '75546c586b70617338367a5858524c'
url = f'http://openapi.seoul.go.kr:8088/{apikey}/xml/citydata/1/1/POI006'

res = requests.get(url)
soup = BeautifulSoup(res.text, 'lxml')
print(soup.prettify())
