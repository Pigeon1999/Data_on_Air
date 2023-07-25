import requests
import pandas as pd 
from bs4 import BeautifulSoup

# 인증키 
api_key = '75546c586b70617338367a5858524c'
url = f'http://openapi.seoul.go.kr:8088/{api_key}/xml/citydata/0/5/POI006'

# api 추출
res = requests.get(url)
soup = BeautifulSoup(res.text, 'lxml-xml')

print(soup.find_all('list_total_count'))


"""
# 필요한 변수만 Data Frame형식으로 저장 
data_list = []

for item in soup.find_all('CITYDATA'):  # 필요한 변수 설정
    AREA_NM = item.find('AREA_NM').text
    AREA_CD = item.find('AREA_CD').text
    AREA_PPLTN_MIN = item.find('AREA_PPLTN_MIN').text
    AREA_PPLTN_MAX = item.find('AREA_PPLTN_MAX').text
    MALE_PPLTN_RATE = item.find('MALE_PPLTN_RATE').text
    FEMALE_PPLTN_RATE = item.find('FEMALE_PPLTN_RATE').text
    PPLTN_RATE_0 = item.find('PPLTN_RATE_0').text
    PPLTN_RATE_10 = item.find('PPLTN_RATE_10').text
    PPLTN_RATE_20 = item.find('PPLTN_RATE_20').text
    PPLTN_RATE_30 = item.find('PPLTN_RATE_30').text
    PPLTN_RATE_40 = item.find('PPLTN_RATE_40').text
    PPLTN_RATE_50 = item.find('PPLTN_RATE_50').text
    PPLTN_RATE_60 = item.find('PPLTN_RATE_60').text
    PPLTN_RATE_70 = item.find('PPLTN_RATE_70').text
    TEMP = item.find('TEMP').text
    HUMIDITY = item.find('HUMIDITY').text
    WIND_SPD = item.find('WIND_SPD').text
    PRECIPITATION = item.find('PRECIPITATION').text
    PM25 = item.find('PM25').text
    PM10 = item.find('PM10').text
    EVENT_NM = item.find('EVENT_NM')
    EVENT_PERIOD = item.find('EVENT_PERIOD')
    EVENT_PLACE = item.find('EVENT_PLACE')
    data_list.append({'AREA_NM': AREA_NM, 'AREA_CD' : AREA_CD, 'AREA_PPLTN_MIN' : AREA_PPLTN_MIN, 'AREA_PPLTN_MAX' : AREA_PPLTN_MAX, 'MALE_PPLTN_RATE' : MALE_PPLTN_RATE, 'FEMALE_PPLTN_RATE' : FEMALE_PPLTN_RATE, 'PPLTN_RATE_0' : PPLTN_RATE_0, 'PPLTN_RATE_10' : PPLTN_RATE_10, 'PPLTN_RATE_20' : PPLTN_RATE_20, 'PPLTN_RATE_30' : PPLTN_RATE_30, 'PPLTN_RATE_40' : PPLTN_RATE_40,  'PPLTN_RATE_50' : PPLTN_RATE_50, 'PPLTN_RATE_60' : PPLTN_RATE_60, 'PPLTN_RATE_70' : PPLTN_RATE_70, 'TEMP' : TEMP, 'HUMIDITY' : HUMIDITY, 'WIND_SPD' : WIND_SPD, 'PRECIPITATION' : PRECIPITATION, 'PM25' : PM25, 'PM10' : PM10, 'EVENT_NM' : EVENT_NM, 'EVENT_PERIOD' : EVENT_PERIOD, 'EVENT_PLACE' : EVENT_PLACE})   
    
df = pd.DataFrame(data_list)
print(data_list)
"""