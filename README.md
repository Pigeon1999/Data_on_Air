# Data_on_Air
2023년 데이터 청년 캠퍼스 1분반 5조 프로젝트 입니다.   
상명대학교 빅데이터 분석 기반 금융분야 비즈니스 인사이트 역량제고 과정

# 프로젝트 주제
페이크 뉴스 탐지시스템 구축

# 프로젝트 목표 
1. 딥러닝을 이용하여 페이크 뉴스 탐지 모델 개발
2. 구축한 모델을 기반으로 기능의 웹 개발

# 기대 효과 

# 프로젝트 기간 
- 집체교육 : 2023.06.26 ~ 2023.07.21 (4주)
- 프로젝트 : 2023.07.25 ~ 2023.08.31 (6주)

# 개발 환경 
#### 이 프로젝트는 해당 환경에서 개발 되었습니다.
- Colab
- Visual Studio Code
- Python 3.10.12


# 설치 방법
Repository를 clone합니다. 
```
!git clone https://github.com/Pigeon1999/Data_on_Air.git
```

Repository의 [requirements.txt](https://github.com/Pigeon1999/Data_on_Air/blob/main/requirements.txt)를 사용합니다. 
```
!pip install -r /content/Data_on_Air/requirements.txt
```

# 사용 방법
### 1. Crawling 
#### Colab은 런타임이 제한되어 있어 VSCode환경에서 진행합니다.

#### Crawling_SNU_data.py : snu_factcheck 사이트 경제 분야 크롤링 
```
# 경제분야의 페이지를 start부터 end까지 크롤링

Crawling_SNU_data(start, end)
```

#### Crawling_Naver_data.py : 네이버 경제 분야 크롤링 
```
# 연관 키워드관련 기사 크롤링
# SNU_keyword_data.csv 파일을 DataFrame으로 불러와 인자로 사용합니다.

Crawling_Naver_data(SNU_keyword_data)
```

#### Crawling_Youtube_data.py : 유튜브 경제 분야 크롤링 
```
# 연관 키워드관련 영상 크롤링
# SNU_keyword_data.csv 파일을 DataFrame으로 불러와 인자로 사용합니다.

Crawling_Youtube_data(SNU_keyword_df)
```

### 2. Modeling 
#### VSCode환경에서도 가능하지만, 모델링을 위한 GPU가 필요하므로 Colab환경을 추천합니다.
#### BiLSTM_Modeling.py 
#### ① preprocessing() : 데이터 셋의 전처리 및 토큰화
```
# 크롤링한 데이터의 전처리 (0번 : snu_keyword. 1번 : naver_keyword, 2번 : youtube_keyword)
# SNU_data.csv, Naver_data.csv, Youtube.csv 파일을 인자로 하여 사용합니다.

# 전처리 + 토큰화까지 과정을 처리하고 DataFrame을 리턴받습니다.
# 또한 {SNU/Naver/Youtube}_keyword_data.csv파일을 생성합니다. 

df = preprocessing(df, num)
```

#### ② main함수 : 
```
def main(): 
    snu_df = pd.read_csv('SNU_keyword_data.csv', encoding = 'utf-8')
    snu_df = Model().list_to_str(snu_df) 

    # label비율 1:1을 위한 오버샘플링
    oversampler = RandomOverSampler(random_state=42)
    x = snu_df.drop('label', axis=1)
    y = snu_df['label']
    X_resampled, y_resampled = oversampler.fit_resample(x, y)
    snu_df = pd.DataFrame(X_resampled, columns=x.columns)
    snu_df['label'] = y_resampled

     # label이 있는 데이터를 BiLSTM기법으로 훈련시켜 모델 생성&저장.
    make_model(snu_df)

    # naver, youtube 데이터 결합
    naver_df = pd.read_csv('D:/Github/Data_on_Air/Dataset/Naver_keyword_data.csv', encoding = 'utf-8')
    youtube_df = pd.read_csv('D:/Github/Data_on_Air/Dataset/Youtube_keyword_data.csv', encoding = 'utf-8')
    new_df = naver_df.append(youtube_df)
    new_df = Model().list_to_str(new_df)
    train_df = snu_df

    count = 1
    start = 0
    end = 100
    try:                                                                             
        for count in range(1, len(train_df)): # 100 * n개 labeling을 5회 실행
            for _ in range(0, 4):
                df = naver_df[start:end]
                df = labeling(df) # train_df를 훈련된 모델로 예측하여 labeling.
                
                train_df = train_df.append(df)
                make_model(train_df) # labeling한 train_df와 snu_df를 합쳐 재학습. 
                predict() # label을 알고 있는 snu_df로 모델 성능 확인 

                start = end
                end = end + count * 100
        count = count + 1   
    except:
        pass
```
