# 2023-07-31 22:04 시작 
# https://factcheck.snu.ac.kr/?topic_id=2
# SNU팩트체크 경제 분야 크롤링 

import time
import pandas as pd
from selenium import webdriver  # 셀레니움을 활성화
from selenium.webdriver import ActionChains  # 액션체인 활성화
from selenium.webdriver.common.by import By

url = 'https://factcheck.snu.ac.kr/?topic_id=2'

dr = webdriver.Chrome()  # 크롬 드라이버를 실행하는 명령어를 dr로 지정
dr.get(url)  # 드라이버를 통해 url의 웹 페이지를 오픈
time.sleep(2)

act = ActionChains(dr)  # 드라이버에 동작을 실행시키는 명령어를 act로 지정
elements = dr.find_elements(by = By.CSS_SELECTOR, value = '.fact-check-card-container.jsx-18931043') 

element_texts = []
for element in elements:
    print(element_texts)
    temp = element.text.split('\n')
    element_texts.append(temp)

column = ['주체', '분류', '뉴스 제목', '출처', '사실 여부']

df = pd.DataFrame(element_texts, columns = column)
print(df)