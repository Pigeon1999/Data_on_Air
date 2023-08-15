import time # 시간 지연
import pandas as pd # 데이터 프레임 생성
from selenium import webdriver  # 셀레니움을 활성화
from selenium.webdriver import ActionChains  # 액션체인 활성화
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options # 브라우저 꺼짐 방지

class Dynamic_Page:
    
    def __init__(self):
        self.url = 'https://factcheck.snu.ac.kr/?topic_id=2'

        self.chrome_options = Options()
        self.chrome_options.add_experimental_option('detach', True) # 브라우저 꺼짐 방지

        self.dr = webdriver.Chrome(options = self.chrome_options)  # 크롬 드라이버를 실행하는 명령어를 dr로 지정
        self.dr.get(self.url)  # 드라이버를 통해 url의 웹 페이지를 오픈
        time.sleep(4)

        self.act = ActionChains(self.dr)  # 드라이버에 동작을 실행시키는 명령어를 act로 지정

dynamic_page = Dynamic_Page()
dr = dynamic_page.dr
act = dynamic_page.act

def Crawling_data(start, end):

    # 공지사항 지우기
    btn_outline_secondary = dr.find_elements(by = By.CSS_SELECTOR, value = '.btn-outline-secondary')
    btn_outline_secondary_click = act.click(on_element = btn_outline_secondary[4]).perform()

    row_id = 0
    element_text = []
    data = []
    column = ['row_id', '주제', '내용', '상세내용', '주장/검증 매체', 'label']

    for page_num in range(start, end + 1): ### 크롤링 페이지 횟수 지정!!!     
        # 뉴스 데이터 크롤링
        for element in range(0,10): 
            try: 
                Turn_page(page_num)
                print(f'{page_num}페이지, {element}번째')
                
                time.sleep(1)
                
                temp = [0, 0, 0, 0, 0] # '주체', '분류', '뉴스 제목', '출처', '사실 여부'
                temp[0] = dr.find_element(by = By.XPATH,  value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[{element + 1}]/div/div[1]/div[2]/div[1]').text
                temp[1] = dr.find_element(by = By.XPATH,  value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[{element + 1}]/div/div[1]/div[2]/div[2]').text
                temp[2] = dr.find_element(by = By.XPATH,  value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[{element + 1}]/div/div[2]/div[1]/div[1]/div[1]').text
                temp[3] = dr.find_element(by = By.XPATH,  value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[{element + 1}]/div/div[2]/div[1]/div[1]/div[2]').text[4:]
                temp[4] = dr.find_element(by = By.XPATH,  value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[{element + 1}]/div/div[2]/div[1]/div[2]/div[4]').text

                # 본문           
                tmp = dr.find_element(by = By.XPATH,  value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[{element + 1}]/div/div[2]/div[1]/div[1]/div[1]')                   
                Scroll_down = dr.execute_script(f'window.scrollTo(0, {390 * element})')
                
                time.sleep(2)   
                
                act.click(tmp).perform() # 페이지 넘기기    
                
                time.sleep(6) 
                
                React_content = dr.find_element(by = By.XPATH, value = '/html/body/div/div/div[2]/div/div[1]/div/div[3]/div[3]/div/div/div/div')
                texts = ''                                             
                for k in React_content.find_elements(by = By.TAG_NAME, value = 'p'):
                    texts = texts + k.text
                    
                    
                #Summary_content = [f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[3]/ul/li[{i}]',
                #                   f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[3]/p[{i}]']   
                
                #verified_content = [f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[5]/div/div/div/p[{j}]',
                #                    f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[3]/div/div/div/p[{j}]']
                 
                try: 
                    try:
                        try:
                            for i in range(1, 10):
                                Summary_content_1 = dr.find_element(by = By.XPATH, value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[3]/ul/li[{i}]').text
                                texts = texts + Summary_content_1
                        except:
                            for i in range(1, 10):
                                Summary_content_1 = dr.find_element(by = By.XPATH, value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[3]/p[{i}]').text
                                texts = texts + Summary_content_1
                    except:
                        try:
                            for j in range(1, 30):
                                temp_text = dr.find_element(by = By.XPATH, value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[5]/div/div/div/p[{j}]').text
                                if temp_text != '[검증 대상]' and temp_text != '[검증 방법]' and temp_text != '[검증 내용]' and temp_text != '[검증 결과]':
                                    texts = texts + temp_text  
                        except:
                            for j in range(1, 30):
                                temp_text = dr.find_element(by = By.XPATH, value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[3]/div/div/div/p[{j}]').text
                                if temp_text != '[검증 대상]' and temp_text != '[검증 방법]' and temp_text != '[검증 내용]' and temp_text != '[검증 결과]':
                                    texts = texts + temp_text  
                except:
                    pass
                
                '''
                try:
                    try:
                        try:
                            for i in range(1, 10):
                                Summary_content_1 = dr.find_element(by = By.XPATH, value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[3]/ul/li[{i}]').text
                                texts = texts + Summary_content_1
                        except:
                            for j in range(1, 30):
                                temp_text = dr.find_element(by = By.XPATH, value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[5]/div/div/div/p[{j}]').text
                                if temp_text != '[검증 대상]' and temp_text != '[검증 방법]' and temp_text != '[검증 내용]' and temp_text != '[검증 결과]':
                                    texts = texts + temp_text            
                    except:
                        try:
                            for i in range(1, 10):
                                Summary_content_2 = dr.find_element(by = By.XPATH, value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[3]/p[{i}]').text
                                texts = texts + Summary_content_2
                        except:
                            for j in range(1, 30):
                                temp_text = dr.find_element(by = By.XPATH, value = f'/html/body/div/div/div[2]/div/div[2]/div[1]/div[2]/div[5]/div/div/div/p[{j}]').text
                                if temp_text != '[검증 대상]' and temp_text != '[검증 방법]' and temp_text != '[검증 내용]' and temp_text != '[검증 결과]':
                                    texts = texts + temp_text  
                except:
                    pass
                '''
                
                dr.back()   
                time.sleep(2) 
                
                data = [row_id, '경제', temp[2], texts, temp[0], temp[4]] # row_id
                print(data)
                
                if len(data) <= 6:
                    element_text.append(data)   
                row_id = row_id + 1 # row_id      
            except:
                pass
        print('done')
            
    df = pd.DataFrame(element_text, columns = column)
    print(df)
    
    # CSV 파일로 저장
    df.to_csv(f'D:\Download\SNU_factcheck_{start}_{end}.csv', index=False)
    
def Turn_page(page_num):
    current_page = dr.find_elements(by = By.CSS_SELECTOR, value = '.btn-secondary')
    click_page = dr.find_elements(by = By.CSS_SELECTOR, value = '.btn-outline-secondary')
    
    if page_num > 1 and page_num != current_page:
        while page_num > int(current_page[1].text):
            for page in click_page:
                if int(page.text) == int(current_page[1].text) + 1:
                    # 웹 스크롤 내리기 : dataframe으로 만들고나서 배치
                    Scroll_down = dr.execute_script('window.scrollTo(0, 5000)')
                    time.sleep(2)                 
                    act.click(page).perform()               
                    break
                
            current_page = dr.find_elements(by = By.CSS_SELECTOR, value = '.btn-secondary')
            click_page = dr.find_elements(by = By.CSS_SELECTOR, value = '.btn-outline-secondary')
        
Crawling_data(11, 20)

