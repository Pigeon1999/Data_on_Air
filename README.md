# Data_on_Air
2023년 데이터 청년 캠퍼스 1분반 5조 프로젝트 입니다.   
상명대학교 빅데이터 분석 기반 금융분야 비즈니스 인사이트 역량제고 과정

# 프로젝트 주제
금융 분야 페이크 뉴스 탐지시스템 구축

# 프로젝트 목표 
1. 딥러닝을 이용하여 페이크 뉴스 탐지 모델 개발
2. 구축한 모델을 기반으로 기능의 웹 개발

# 프로젝트 기간 
- 집체교육 : 2023.06.26 ~ 2023.07.21 (4주)
- 프로젝트 : 2023.07.25 ~ 2023.08.31 (6주)

# 개발 환경 
- colab
- Python 3.10.12

# 설치 방법
Repository를 clone합니다. 
```
git clone
```

해당 리포지드의 [requirements.txt](https://github.com/Pigeon1999/Data_on_Air/blob/main/requirements.txt)를 사용합니다. 
```
!pip install -r requirement.txt
```

# 사용 방법
### 1. Crawling 
#### Crawling_SNU_data.py : snu_factcheck 사이트 경제 분야 크롤링 
```
from Crawling_SNU_data import Crawling_SNU_data

Crawling_SNU_data(start, end)
```
