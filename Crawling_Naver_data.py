from selenium import webdriver
from bs4 import BeautifulSoup as bs
from collections import Counter
import pandas as pd
import time
import os
import ast

# Selenium 설정
def configure_driver():
    options = webdriver.FirefoxOptions()
    options.headless = True
    firefox_binary_path = "/usr/bin/firefox-esr"
    options.binary_location = firefox_binary_path

    display_port = os.environ.get("DISPLAY_PORT", "99")
    display = f":{display_port}"
    os.environ["DISPLAY"] = display

    xvfb_cmd = f"Xvfb {display} -screen 0 1920x1080x24 -nolisten tcp &"
    os.system(xvfb_cmd)

    return webdriver.Firefox(options=options)

# 키워드 빈도수 상위 n개 추출 함수
def get_top_n_frequencies(input_list, n):
    frequency_counter = Counter(input_list)
    top_n_elements = [element for element, _ in frequency_counter.most_common(n)]
    return top_n_elements

# 기사 URL 리스트 가져오는 함수
def get_url_list(driver, keyword, desired_count):
    links = []
    page_number = 0

    while len(links) < desired_count:
        url = f'https://search.naver.com/search.naver?where=news&sm=tab_jum&query={keyword}&start={page_number}'
        print(url)
        driver.get(url)
        time.sleep(1)
        soup = bs(driver.page_source, 'html.parser')
        link_tags = soup.select('a.info')
        for link in link_tags:
            if 'naver' in link['href']:
                links.append(link['href'])
                if len(links) >= desired_count:
                    break

        if page_number == 250:
            print("기사 갯수 : ", len(links))
            return links

        page_number += 10

    return links

# 기사 정보 추출 함수
def get_news_info(driver, url, data_dict):
    driver.get(url)
    time.sleep(1)
    soup = bs(driver.page_source, 'html.parser')
    title = soup.select_one("#title_area > span")
    main = soup.select("#dic_area")
    press = soup.select_one("em.media_end_linked_more_point")
    
    if title is None or main is None or press is None:
        return
    
    title_str = title.get_text().strip()
    main_lst = [m.get_text().strip() for m in main]
    main_str = " ".join(main_lst)
    press_str = press.get_text()
    
    data_dict["주제"].append("경제")
    data_dict["내용"].append(title_str)
    data_dict["상세내용"].append(main_str)
    data_dict["주장/검증매체"].append(press_str)

# CSV로 저장하는 함수
def save_to_csv(data_dict, filename):
    df = pd.DataFrame(data_dict)
    df.to_csv(filename, index=True, index_label='row_data')

# 메인 함수
def main():
    driver = configure_driver()
    
    df_snu = pd.read_csv('./csv/snu.csv', encoding='utf-8')
    content_total_dict = {'주제': [], '내용': [], '상세내용': [], '주장/검증매체': []}

    top_keywords = 3
    news_per_keyword = 5

    for i in df_snu['상세내용']:
        temp = ast.literal_eval(i)
        keyword_list = get_top_n_frequencies(temp, top_keywords)
        combine = ', '.join(keyword_list)
        
        url_list = get_url_list(driver, combine, news_per_keyword)
        
        for url in url_list:
            get_news_info(driver, url, content_total_dict)
        
        save_to_csv(content_total_dict, 'intermediate.csv')
    
    driver.quit()

if __name__ == "__main__":
    main()
