import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import pandas as pd 

# 인증키 
api_key = '75546c586b70617338367a5858524c'
url = f'http://openapi.seoul.go.kr:8088/{api_key}/xml/citydata/1/1/POI006'

# 결과값 출력 
res = requests.get(url)
data = res.content
root = ET.fromstring(data)

data_list = []

for item in root.findall('CITYDATA'):  # 'item' 태그에 데이터가 있는 경우
    AREA_NM = item.find('AREA_NM').text
    AREA_CD = item.find('AREA_CD').text
    data_list.append({'AREA_NM': AREA_NM, 'AREA_CD' : AREA_CD})   
    
df = pd.DataFrame(data_list)
print(df)