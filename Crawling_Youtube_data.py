from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains  # 액션체인 활성화
from selenium.webdriver.chrome.options import Options # 브라우저 꺼짐 방지
from bs4 import BeautifulSoup as bs
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import time
import ast
from collections import Counter

'''
# ------------------------------------ ↓↓↓ Selenium Firefox 설정 ↓↓↓ ------------------------------------ #
# headless mode 사용
options = webdriver.FirefoxOptions()
options.headless = True

# binary 경로
firefox_binary_path = "/usr/bin/firefox-esr"
options.binary_location = firefox_binary_path

# display port 설정
display_port = os.environ.get("DISPLAY_PORT", "99")
display = f":{display_port}"
os.environ["DISPLAY"] = display

# Xvfb 서버 시작
xvfb_cmd = f"Xvfb {display} -screen 0 1920x1080x24 -nolisten tcp &"
os.system(xvfb_cmd)

# 파이어폭스 드라이브 시작
driver = webdriver.Firefox(options=options)
# ------------------------------------ ↑↑↑ Selenium Firefox 설정 ↑↑↑ ------------------------------------ #
'''
class Youtube_Crawling:
    
    def __init__(self, keyword):
        self.url = 'https://www.youtube.com/results?search_query={}'.format(keyword)

        self.chrome_options = Options()
        self.chrome_options.add_experimental_option('detach', True) # 브라우저 꺼짐 방지
        
        self.driver = webdriver.Chrome(options = self.chrome_options)  # 크롬 드라이버를 실행하는 명령어를 dr로 지정
        self.driver.get(self.url)  # 드라이버를 통해 url의 웹 페이지를 오픈
        time.sleep(4)

        self.act = ActionChains(self.driver)  # 드라이버에 동작을 실행시키는 명령어를 act로 지정

    def getSubtitle(self, ID):
        try:
            # 동영상의 자막 정보 조회
            transcript_list = YouTubeTranscriptApi.list_transcripts(ID)
            # 한국어 (ko) 자막 가져오기
            korean_transcript = transcript_list.find_transcript(['ko']).fetch()
            captions = [entry['text'] for entry in korean_transcript if '[음악]' not in entry['text'] and '[박수]' not in entry['text']]

        except Exception as e:
            return "0"
        
        return captions

    def searchKeywords(self):        
        time.sleep(1)

        self.driver.find_element(by = By.XPATH, value = '/html/body/ytd-app/div[1]/ytd-page-manager/ytd-search/div[1]/div/ytd-search-header-renderer/div[3]/ytd-button-renderer/yt-button-shape/button/yt-touch-feedback-shape/div/div[2]').click()
        
        time.sleep(1)
        
        self.driver.find_element(by = By.XPATH, value = '/html/body/ytd-app/ytd-popup-container/tp-yt-paper-dialog/ytd-search-filter-options-dialog-renderer/div[2]/ytd-search-filter-group-renderer[5]/ytd-search-filter-renderer[3]/a/div/yt-formatted-string').click()   
        
        time.sleep(2)
        
        data_list = []
        column = ['주제', '내용', '상세내용', '주장/검증매체']
        num = 0
        while True:
            link = ''
            title = ''
            live_tag = ''
            claim = ''
            num = num + 1
            
            try:
                live_tag = self.driver.find_element(by = By.XPATH, value = f'/html/body/ytd-app/div[1]/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[{num}]/div[1]/div/ytd-badge-supported-renderer/div[1]/span').text   
                if live_tag == '새 동영상' or live_tag == '실시간':
                    continue                                  
            except:                                                  
                try:
                    if live_tag != '실시간':
                        title = self.driver.find_element(by = By.XPATH, value = f'/html/body/ytd-app/div[1]/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[{num}]/div[1]/div/div[1]/div/h3/a/yt-formatted-string').text
                        href_tag  = self.driver.find_element(by = By.XPATH, value = f'/html/body/ytd-app/div[1]/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[{num}]/div[1]/div/div[1]/div/h3/a')
                        link = href_tag.get_attribute('href')
                        claim = self.driver.find_element(by = By.XPATH, value = f'/html/body/ytd-app/div[1]/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[{num}]/div[1]/div/div[2]/ytd-channel-name/div/div/yt-formatted-string/a').text
                    else:
                        continue
                except:
                    df = pd.DataFrame(data_list, columns = column)
                    break
            
            #print(title, link)

            if link[len("https://www.youtube.com/"):30] == 'shorts':
                continue  # shorts가 포함된 링크인 경우 pass

            temp = self.getSubtitle(link[len("https://www.youtube.com/watch?v="):].split("&")[0])
            text = ""
            for j in temp:
                text = j + text
            
            if text == '' or text == '0':
                continue
            else:
                data = ["경제", title, text[:700], claim]
                
            data_list.append(data)
            print(data)
            if len(data_list) >= 5:
                df = pd.DataFrame(data_list, columns = column)
                break
    
        self.driver.close()
        return df

def Crawling_Youtube_data(df):
    keyword_list = df['상세내용']
    keyword = []
    for row in keyword_list:
        row = ast.literal_eval(row) # 문자열의 리스트화
        frequency_counter = Counter(row)  # 각 요소의 빈도수 계산
        top_n_frequencies = frequency_counter.most_common(3)  # 빈도수 상위 n개 선택
        top_n_elements = [element for element, _ in top_n_frequencies]  # 요소만 추출
        
        text = ''
        for i in top_n_elements:
            text = text + f'{i} '
        keyword.append(text)
    
    new_df = pd.DataFrame()
    for i in keyword:
        new_df = new_df.append(Youtube_Crawling(i).searchKeywords(), ignore_index=True) 
        print(new_df)
    new_df.to_csv('output.csv', index=True, index_label='row_id')

    print("데이터가 'output.csv' 파일로 저장되었습니다.") 
    
    return new_df


'''
[실행코드]
crawling_youtube_data()

[실행결과]
탄소중립, 더 나은 하루를 위한 생활 습관 https://www.youtube.com/watch?v=OzwSDV8hNH8&pp=ygUG7YOE7IaM
 ..... 
이하생략
 .....
, 0, 0, 'https://www.youtube.com/watch?v=g2U9CfOsYn0&pp=ygUG7YOE7IaM', 0, 0]
데이터가 'output.csv' 파일로 저장되었습니다.
'''
