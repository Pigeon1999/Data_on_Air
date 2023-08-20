# Data_on_Air
2023년 데이터 청년 캠퍼스 1분반 5조 프로젝트 입니다.   
상명대학교 빅데이터 분석 기반 금융분야 비즈니스 인사이트 역량제고 과정

# 프로젝트 주제
페이크 뉴스 탐지시스템 구축

# 프로젝트 목표 
1. 딥러닝을 이용하여 페이크 뉴스 탐지 모델 개발
2. 구축한 모델을 기반으로 기능의 웹 개발

# 프로젝트 기간 
- 집체교육 : 2023.06.26 ~ 2023.07.21 (4주)
- 프로젝트 : 2023.07.25 ~ 2023.08.31 (6주)

# 개발 환경 
#### 이 프로젝트는 해당 환경에서 개발 되었습니다.
- colab
- Python 3.10.12

# 설치 방법
Repository를 clone합니다. 
```
git clone https://github.com/Pigeon1999/Data_on_Air.git
```

Repository의 [requirements.txt](https://github.com/Pigeon1999/Data_on_Air/blob/main/requirements.txt)를 사용합니다. 
```
!pip install -r requirement.txt
```

# 사용 방법
### 1. Crawling 
#### Crawling_SNU_data.py : snu_factcheck 사이트 경제 분야 크롤링 
```
from Data_on_Air.Modeling.Crawling_SNU_data import Crawling_SNU_data

# 경제분야의 페이지를 start부터 end까지 크롤링
Crawling_SNU_data(start, end)
```

#### Crawling_Naver_data.py : 네이버 경제 분야 크롤링 
```
from Data_on_Air.Modeling.Crawling_Naver_data import Crawling_Naver_data

# 토큰화가 완료된 SNU_keyword_data.csv를 인자로 하여
# 연관 키워드관련 기사 크롤링
Crawling_Naver_data(SNU_keyword_data)
```

#### Crawling_Youtube_data.py : 유튜브 경제 분야 크롤링 
```
from Data_on_Air.Modeling.Crawling_Youtube_data import Crawling_Youtube_data

# 토큰화가 완료된 SNU_keyword_data.csv를 인자로 하여
# 연관 키워드관련 영상 크롤링  
Crawling_Youtube_data(SNU_keyword_data)
```

### 2. Modeling 
#### BiLSTM_Modeling.py 
#### ① preprocessing() : 데이터 셋의 전처리 및 토큰화
```
from Data_on_Air.Modeling.BiLSTM_Modeling import preprocessing

# 크롤링한 데이터의 전처리 (0번 : snu_keyword. 1번 : naver_keyword, 2번 : youtube_keyword)
df = preprocessing(SNU_data, num)
```

#### ② make_model(df) : 전처리된 데이터로 Word2Vec와 BiLSTM기법 적용하여 모델 생성 
```
from Data_on_Air.Modeling.BiLSTM_Modeling import make_model

# 동일한 경로에 'trained_BiLSTM_model'라는 이름의 모델 생성
make_model(df)
```

#### ③ predict_model(x_test, y_test) : 생성된 학습 모델로 검증 데이터 예측 
```
from Data_on_Air.Modeling.BiLSTM_Modeling import predict_model

# 코드 고정 
predict_model(pre_process.x_test, pre_process.y_test)
```
