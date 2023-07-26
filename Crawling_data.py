# https://data.seoul.go.kr/dataList/OA-15439/S/1/datasetView.do
# 자치구 단위 서울 생활인구(내국인)의 크롤링 코드입니다. 
# 자치구 단위 서울 생활인구(내국인)의 데이터는 2018-02-15부터 2023-07-21까지의 데이터가 존재합니다. (2023-07-26기준)
# 종로구 자치구 코드는 11110입니다. 
# 인증키는 75546c586b70617338367a5858524c입니다.

## ================= 아직 미완성입니다!!!!! ==========================
import requests
import pandas as pd 
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, date

"""
current_date = datetime(2023, 7, 25).date()

while (datetime.now().date() > current_date):

    str_date = current_date.strftime('%x')
    STDR_DE_ID = f'20{str_date[6:8]}{str_date[0:2]}{str_date[3:5]}'
     
    for TMZON_PD_SE in range(25):
        api_key = '75546c586b70617338367a5858524c'
        url = f'http://openapi.seoul.go.kr:8088/{api_key}/xml/SPOP_LOCAL_RESD_JACHI/1/1000/20230726/{TMZON_PD_SE:02d}/11110'

        # api 추출
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'lxml-xml')

        # 필요한 변수만 Data Frame형식으로 저장 
        data_list = []

        for item in soup.find_all('row'):  # 필요한 변수 설정
            STDR_DE_ID = item.find('STDR_DE_ID').text
            TMZON_PD_SE = item.find('TMZON_PD_SE').text
            ADSTRD_CODE_SE = item.find('ADSTRD_CODE_SE').text
            TOT_LVPOP_CO = item.find('TOT_LVPOP_CO').text
            MALE_F0T9_LVPOP_CO = item.find('MALE_F0T9_LVPOP_CO').text
            MALE_F10T14_LVPOP_CO = item.find('MALE_F10T14_LVPOP_CO').text
            MALE_F15T19_LVPOP_CO = item.find('MALE_F15T19_LVPOP_CO').text
            MALE_F20T24_LVPOP_CO = item.find('MALE_F20T24_LVPOP_CO').text
            MALE_F25T29_LVPOP_CO = item.find('MALE_F25T29_LVPOP_CO').text
            MALE_F30T34_LVPOP_CO = item.find('MALE_F30T34_LVPOP_CO').text
            MALE_F35T39_LVPOP_CO = item.find('MALE_F35T39_LVPOP_CO').text
            MALE_F40T44_LVPOP_CO = item.find('MALE_F40T44_LVPOP_CO').text
            MALE_F45T49_LVPOP_CO = item.find('MALE_F45T49_LVPOP_CO').text
            MALE_F50T54_LVPOP_CO = item.find('MALE_F50T54_LVPOP_CO').text
            MALE_F55T59_LVPOP_CO = item.find('MALE_F55T59_LVPOP_CO').text
            MALE_F60T64_LVPOP_CO = item.find('MALE_F60T64_LVPOP_CO').text
            MALE_F65T69_LVPOP_CO = item.find('MALE_F65T69_LVPOP_CO').text
            MALE_F70T74_LVPOP_CO = item.find('MALE_F70T74_LVPOP_CO').text
            FEMALE_F0T9_LVPOP_CO = item.find('FEMALE_F0T9_LVPOP_CO').text
            FEMALE_F10T14_LVPOP_CO = item.find('FEMALE_F10T14_LVPOP_CO').text
            FEMALE_F15T19_LVPOP_CO = item.find('FEMALE_F15T19_LVPOP_CO').text
            FEMALE_F20T24_LVPOP_CO = item.find('FEMALE_F20T24_LVPOP_CO').text
            FEMALE_F25T29_LVPOP_CO = item.find('FEMALE_F25T29_LVPOP_CO').text
            FEMALE_F30T34_LVPOP_CO = item.find('FEMALE_F30T34_LVPOP_CO').text
            FEMALE_F35T39_LVPOP_CO = item.find('FEMALE_F35T39_LVPOP_CO').text
            FEMALE_F40T44_LVPOP_CO = item.find('FEMALE_F40T44_LVPOP_CO').text
            FEMALE_F45T49_LVPOP_CO = item.find('FEMALE_F45T49_LVPOP_CO').text
            FEMALE_F50T54_LVPOP_CO = item.find('FEMALE_F50T54_LVPOP_CO').text
            FEMALE_F55T59_LVPOP_CO = item.find('FEMALE_F55T59_LVPOP_CO').text
            FEMALE_F60T64_LVPOP_CO = item.find('MALE_F60T64_LVPOP_CO').text
            FEMALE_F65T69_LVPOP_CO = item.find('MALE_F65T69_LVPOP_CO').text
            FEMALE_F70T74_LVPOP_CO = item.find('FEMALE_F70T74_LVPOP_CO').text
            data_list.append({'STDR_DE_ID' : STDR_DE_ID, 'TMZON_PD_SE' : TMZON_PD_SE, 'ADSTRD_CODE_SE' : ADSTRD_CODE_SE, 'TOT_LVPOP_CO' : TOT_LVPOP_CO, 
                            'MALE_F0T9_LVPOP_CO' : MALE_F0T9_LVPOP_CO, 'MALE_F10T14_LVPOP_CO' : MALE_F10T14_LVPOP_CO, 'MALE_F15T19_LVPOP_CO' : MALE_F15T19_LVPOP_CO,
                            'MALE_F20T24_LVPOP_CO' : MALE_F20T24_LVPOP_CO, 'MALE_F25T29_LVPOP_CO' : MALE_F25T29_LVPOP_CO, 'MALE_F30T34_LVPOP_CO' : MALE_F30T34_LVPOP_CO,
                            'MALE_F35T39_LVPOP_CO' : MALE_F35T39_LVPOP_CO, 'MALE_F40T44_LVPOP_CO' : MALE_F40T44_LVPOP_CO, 'MALE_F45T49_LVPOP_CO' : MALE_F45T49_LVPOP_CO,
                            'MALE_F50T54_LVPOP_CO' : MALE_F50T54_LVPOP_CO, 'MALE_F55T59_LVPOP_CO' : MALE_F55T59_LVPOP_CO, 'MALE_F60T64_LVPOP_CO' : MALE_F60T64_LVPOP_CO,
                            'MALE_F65T69_LVPOP_CO' : MALE_F65T69_LVPOP_CO, 'MALE_F70T74_LVPOP_CO' : MALE_F70T74_LVPOP_CO, 'FEMALE_F0T9_LVPOP_CO' : FEMALE_F0T9_LVPOP_CO,
                            'FEMALE_F10T14_LVPOP_CO' : FEMALE_F10T14_LVPOP_CO, 'FEMALE_F15T19_LVPOP_CO' : FEMALE_F15T19_LVPOP_CO, 'FEMALE_F20T24_LVPOP_CO' : FEMALE_F20T24_LVPOP_CO,
                            'FEMALE_F25T29_LVPOP_CO' : FEMALE_F25T29_LVPOP_CO, 'FEMALE_F30T34_LVPOP_CO' : FEMALE_F30T34_LVPOP_CO, 'FEMALE_F35T39_LVPOP_CO' : FEMALE_F35T39_LVPOP_CO,
                            'FEMALE_F40T44_LVPOP_CO' : FEMALE_F40T44_LVPOP_CO, 'FEMALE_F45T49_LVPOP_CO' : FEMALE_F45T49_LVPOP_CO, 'FEMALE_F50T54_LVPOP_CO' : FEMALE_F50T54_LVPOP_CO,
                            'FEMALE_F55T59_LVPOP_CO' : FEMALE_F55T59_LVPOP_CO, 'FEMALE_F60T64_LVPOP_CO' : FEMALE_F60T64_LVPOP_CO, 'FEMALE_F65T69_LVPOP_CO' : FEMALE_F65T69_LVPOP_CO,
                            'FEMALE_F70T74_LVPOP_CO' : FEMALE_F70T74_LVPOP_CO })
    print(f'{STDR_DE_ID} : done')

    # 다음 
    current_date = current_date + timedelta(days = 1)
      
df = pd.DataFrame(data_list)
print(df)
"""

api_key = '75546c586b70617338367a5858524c'
url = f'http://openapi.seoul.go.kr:8088/{api_key}/xml/SPOP_LOCAL_RESD_JACHI/1/1000/20230701/00/11110'

# api 추출
res = requests.get(url)
soup = BeautifulSoup(res.text, 'lxml-xml')
print(soup.text)
