# 2023-07-31 22:04 시작 
# https://factcheck.snu.ac.kr/?topic_id=2
# SNU팩트체크 경제 분야 크롤링 
import time # 시간 지연
import pandas as pd # 데이터 프레임 생성
from selenium import webdriver  # 셀레니움을 활성화
from selenium.webdriver import ActionChains  # 액션체인 활성화
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options # 브라우저 꺼짐 방지

url = 'https://factcheck.snu.ac.kr/?topic_id=2'

chrome_options = Options()
chrome_options.add_experimental_option('detach', True) # 브라우저 꺼짐 방지

dr = webdriver.Chrome(options = chrome_options)  # 크롬 드라이버를 실행하는 명령어를 dr로 지정
dr.get(url)  # 드라이버를 통해 url의 웹 페이지를 오픈
time.sleep(4)

act = ActionChains(dr)  # 드라이버에 동작을 실행시키는 명령어를 act로 지정

# 공지사항 지우기
btn_outline_secondary = dr.find_elements(by = By.CSS_SELECTOR, value = '.btn-outline-secondary')
btn_outline_secondary_click = act.click(on_element = btn_outline_secondary[4]).perform()

row_id = 0
element_text = []
data = []
column = ['row_id', '주제', '내용', '상세내용', '주장/검증 매체', 'label']

for i in range(0, 1): ### 크롤링 페이지 횟수 지정!!! ###
    
    time.sleep(2)
    
    # 뉴스 데이터 크롤링
    
    for element in range(0,10):
        try:
            find_elements = dr.find_elements(by = By.CSS_SELECTOR, value = '.fact-check-card-container.jsx-18931043') 
            temp = find_elements[element].text.split('\n') # '주체', '분류', '뉴스 제목', '출처', '사실 여부'
            
            # 본문
            class_name = find_elements[element].find_element(by = By.TAG_NAME, value = 'hr').get_attribute('class')
            click_page = find_elements[element].find_element(by = By.CSS_SELECTOR, value = f'.fact-check-title.{class_name}')
            
            Scroll_down = dr.execute_script(f'window.scrollTo(0, {400 * element})')
    
            time.sleep(1)
            
            act.click(click_page).perform() # 페이지 넘기기 
            
            time.sleep(5)
            
            React_content = dr.find_element(by = By.XPATH, value = '/html/body/div/div/div[2]/div/div[1]/div/div[3]/div[3]/div/div/div/div')

            texts = ''
            for i in React_content.find_elements(by = By.TAG_NAME, value = 'p'):
                texts = texts + i.text
            
            dr.back()   
            
            time.sleep(2) 
            
            data = [row_id, '경제', temp[2], texts, temp[0], temp[4]] # row_id

            print(data)
            
            if len(data) <= 7:
                element_text.append(data)
                
            row_id = row_id + 1 # row_id
        except:
            continue

    # 웹 스크롤 내리기 : dataframe으로 만들고나서 배치
    Scroll_down = dr.execute_script('window.scrollTo(0, 5000)')
    time.sleep(2)

    # 웹 페이지 넘기기 : dataframe으로 만들고나서 배치
    current_page = dr.find_elements(by = By.CSS_SELECTOR, value = '.btn-secondary')
    click_page = dr.find_elements(by = By.CSS_SELECTOR, value = '.btn-outline-secondary')

    for page in click_page:
        if int(page.text) == int(current_page[1].text) + 1:
            act.click(page).perform()
            break
    
    print('done')
        
df = pd.DataFrame(element_text, columns = column)
print(df)
