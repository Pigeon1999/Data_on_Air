import requests
import pandas as pd 
from bs4 import BeautifulSoup

# 인증키 
api_key = '75546c586b70617338367a5858524c'
url = f'http://openapi.seoul.go.kr:8088/{api_key}/xml/citydata/1/1/POI006'

# api 추출
res = requests.get(url)
soup = BeautifulSoup(res.text, 'lxml-xml')

# 필요한 변수만 Data Frame형식으로 저장 
data_list = []

for item in soup.find_all('CITYDATA'):  # 필요한 변수 설정
    AREA_NM = item.find('AREA_NM').text
    AREA_CD = item.find('AREA_CD').text
    AREA_PPLTN_MIN = item.find('AREA_PPLTN_MIN').text
    AREA_PPLTN_MAX = item.find('AREA_PPLTN_MAX').text
    data_list.append({'AREA_NM': AREA_NM, 'AREA_CD' : AREA_CD, 'AREA_PPLTN_MIN' : AREA_PPLTN_MIN, 'AREA_PPLTN_MAX' : AREA_PPLTN_MAX})   
    
df = pd.DataFrame(data_list)
print(df)
