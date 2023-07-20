import requests
from bs4 import BeautifulSoup

# 인증키 
api_key = '75546c586b70617338367a5858524c'
url = f'http://openapi.seoul.go.kr:8088/{api_key}/xml/citydata/1/1/POI006'

# 결과값 출력 
res = requests.get(url)
soup = BeautifulSoup(res.text, 'lxml')
print(soup.prettify()) # 들여쓰기 출력 

# xml형식 필요한 columns만 가져오고 DataFrame으로 만들기 

# DataFrame을 sql로 저장
