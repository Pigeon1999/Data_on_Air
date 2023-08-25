
import re
import ast
import nltk
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from konlpy.tag import Okt
from konlpy.tag import Kkma
from nltk import sent_tokenize
from collections import Counter
from nltk.corpus import stopwords
from hanspell import spell_checker
from gensim.models import Word2Vec
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
nltk.download('punkt')

# 초기 설정
class pre_process:  
    def __init__(self, csv):
        #os.chdir('py-hanspell')
        self.df = csv
    
# 1. label 처리       
class label(pre_process):
    def label_processing(self):
        self.df.dropna(axis=0, inplace=True)
        self.df= self.df.drop(columns=['주제'])
        
        try:
            idx = self.df.loc[self.df['label']=='판단 유보'].index
            self.df = self.df.drop(idx, axis=0)
            idx = self.df.loc[self.df['label']=='논쟁 중'].index
            self.df = self.df.drop(idx, axis=0)
            self.df['label'] = self.df['label'].replace({'전혀 사실 아님': 0, '대체로 사실 아님': 0, '절반의 사실': 0, '대체로 사실': 1, '사실': 1})
            self.df['label'] = self.df['label'].astype(int)
        except:
            pass
        
        self.df['row_id'] = range(0, len(self.df))
        self.df.index = range(0, len(self.df))
        self.df['row_id'] = self.df['row_id'].astype(int)
        
        return self.df
    
# 2. '내용, 상세내용'의 특수문자 제거, 불용어 제거, 맞춤법 조정       
class text(pre_process):
    # 2-1 특수문자 제거
    def remove_special_characters(self, text):
        return re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s]', '', text)
    
    def text_processing(self):
        contents = ['내용', '상세내용']
        
        special = [r'%', r'M&A', r'㎡', r'㎞', r'~', r'㏊', r'CO₂', r'㎢', r'ℓ']
        transform = [r'퍼센트', r'인수합병', r'제곱미터', r'km', r'에서 ', r'ha', r'이산화탄소', r'제곱킬로미터', r'리터']
        for i in range(0, 9):
            self.df['내용'] = self.df['내용'].str.replace(pat=special[i], repl=transform[i], regex=True)

        circle = [r'①', r'②', r' ③']
        for content in contents:
            for cir in circle:
                self.df[content] = self.df[content].str.replace(pat=cir, repl=r'', regex=True)
        
        self.df['내용'] = self.df['내용'].apply(self.remove_special_characters)
        self.df['상세내용'] = self.df['상세내용'].apply(self.remove_special_characters)
        
        # 맞춤법 검사
        for row in range(0, len(self.df['상세내용'])):
            texts = ''
            for i in range(0, int(len(self.df['상세내용'][row]) / 500) + 1):
                data = self.df['상세내용'][row][0 + (500 * i):500 + (500 * i)]
                try:
                    result = spell_checker.check(data).checked
                except:
                    result = data
                texts = texts + result
            self.df['상세내용'][row] = texts
            print(f'done : {row}') 
        
        # 2-3 불용어 제거 
        stop_words = pd.read_csv('D:\GitHub\Data_on_Air\Dataset\stopwords.csv', encoding = 'utf-8')['불용어']
        for row in range(0, len(self.df['상세내용'])):
            word_tokens = self.okt.morphs(self.df['상세내용'][row])
            result = [word for word in word_tokens if not word in stop_words]
            self.df['상세내용'][row] = result
            
        return self.df 

# 3. 토큰화
class token(pre_process):  
    def __init__(self, csv):
        super().__init__(csv)
        self.okt = Okt() 
        
    # '상세내용' 토큰화
    def tokenizer(self, text):
        morph = self.okt.pos(text)
        words = []
        for word, tag in morph:
            if tag in ['Noun', 'Verb', 'Adjective']:
                if len(word) > 1:
                    if word not in ['주장', '사실', '검증']:
                        words.append(word)
        return words

    def token_processing(self):
        self.df= self.df.astype('str')
        self.df['상세내용'] = self.df['상세내용'].apply(self.tokenizer)

        temp_data = []
        most_common_words_data = []
        for text in self.df['상세내용']:
            word_counts = Counter(text)
            most_common_words = word_counts.most_common() # 단어를 빈도수로 정렬 

            # 단어의 빈도수가 2회이하는 제거 
            most_common_words_only = [word for word, count in most_common_words if count > 2]
            temp_data.append(len(most_common_words_only))
            most_common_words_data.append(most_common_words_only)
            
        self.df['temp'] = temp_data
        self.df['빈도순'] = most_common_words_data
        self.df['상세내용'] = most_common_words_data
        
        for row in range(0, len(self.df)):
            temp = self.df['temp']
            if int(temp[row]) == 0:
                self.df.drop(row, inplace = True)

        del self.df['temp']
        self.df['row_id'] = range(0, len(self.df))
        self.df.index = range(0, len(self.df))  
        
        # 키워드가 5개 이하, 70개 이상이면 제거
        for row in range(0, len(self.df)):
            if len(self.df['상세내용'][row]) <= 7 or len(self.df['상세내용'][row]) >= 70:
                self.df = self.df.drop(row)
        self.df['row_id'] = range(0, len(self.df))
        self.df.index = range(0, len(self.df)) 
        
        return self.topic_modeling()

    # 유사한 주제의 뉴스 제거 
    def topic_modeling(self):
        new_df = self.df['빈도순'].copy() 
        for row in range(0, len(new_df)):
            frequency_counter = Counter(new_df[row])
            top_n_frequencies = frequency_counter.most_common(10)
            new_df[row] = [element for element, _ in top_n_frequencies]

        data = new_df.tolist()
        self.df['group_id'] = None

        # 중복 텍스트 묶기
        text_groups = []

        # 중복 텍스트를 그룹에 묶는 함수
        for i in range(0, len(self.df)):
            if data[i] != []:
                group = []
                # 각행마다 유사도 비교
                for row in range(0, len(data)):
                    if data[i] != data[row]:
                        count = 0
                        for j in range(0, 10):
                            try:
                                if data[i][j] in data[row]:
                                    count = count + 1
                                    if count >= 4: # 4개의 키워드가 같을시 유사한 주제로 판단.
                                        group.append(row)  # 인덱스를 추가
                                        data[row] = []
                                        count = 0
                                        break  
                            except:
                                continue

                group.append(i)
                data[i] = []

                for idx in group:
                    self.df.loc[self.df.index == idx, 'group_id'] = i

                text_groups.append(group)

        # 중복된 group_id를 가진 데이터 중 랜덤하게 하나만 남기고 나머지 삭제
        for group in text_groups:
            if group:
                rows_to_keep = random.choice(group)
                group.remove(rows_to_keep)
                self.df.drop(group, inplace=True)
        
        del self.df['빈도순']
        del self.df['group_id']
        
        self.df['row_id'] = range(0, len(self.df))
        self.df.index = range(0, len(self.df))
        
        return self.df

    '''
    # 명사가 아닌 단어 제거 & train data와의 키워드 갯수 범위 조정 
    def noun_filter(self):
        if type(self.df['상세내용'][0]) == str:
            row_list = []
            for row in self.df['상세내용']:
                row = ast.literal_eval(row) # 문자열의 리스트화
                row_list.append(row)         
            self.df['상세내용'] = row_list
        
        
        # 품사 분석기 초기화
        model = Word2Vec(sentences = self.df['상세내용'], vector_size = 280, window = 10, min_count = 3, workers = 4, sg = 1)
        kkma = Kkma()
        result = model.wv.most_similar("경제", topn=15000) # 경제와 관련된 상위 15,000개 가져오기
    
        # 명사 걸러내기
        filtered_result = []
        for word, score in result: # word가 단어, score가 유사 비율(%)
            # 단어 품사 분석
            pos_tags = kkma.pos(word)
            for _, tag in pos_tags:
                if tag == 'NNG':  # 명사인 경우만 추가
                    filtered_result.append(word)
                    break
        
        for row in range(0, len(self.df)):
            df_data = self.df['상세내용'][row]
            result = list(set(df_data) & set(filtered_result))                       
            self.df['상세내용'][row] = result
            
        # 키워드가 7개 이하, 70개 이상이면 제거
        for row in range(0, len(self.df)):
            if len(self.df['상세내용'][row]) <= 7 or len(self.df['상세내용'][row]) >= 70:
                self.df = self.df.drop(row)
        self.df['row_id'] = range(0, len(self.df))
        self.df.index = range(0, len(self.df))
        
        return self.df
        '''

    def match_label(self):
        T = self.df[self.df['label'] == 1]
        F = self.df[self.df['label'] == 0]
        if len(T) >= len(F):
            for _ in range(0, np.abs(len(T) - len(F))):
                T = T[:-1]
        else:
            for _ in range(0, np.abs(len(T) - len(F))):
                F = F[:-1]
        
        new_df = T.append(F)
        new_df = Model().list_to_str(new_df)
        new_df['row_id'] = range(0, len(new_df))
        new_df.index = range(0, len(new_df))
        
        return new_df
        
# 4. 워드 임베딩 & 5. BiLSTM
class Model: 
    def __init__(self):
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []   
        self.vocab_size = 0
        self.embedding_dim = 0 
        self.model = None
    
    def list_to_str(self, df):
        df['row_id'] = range(0, len(df))
        df.index = range(0, len(df))
        
        if type(df['상세내용'][0]) == str:
            row_list = []
            for row in df['상세내용']:
                row = ast.literal_eval(row) # 문자열의 리스트화
                row_list.append(row)         
            df['상세내용'] = row_list   

        return df     
    
    def word_embedding(self, df):
        # Word2Vec 모델로 학습된 임베딩 벡터 가져오기
        model = Word2Vec(sentences = df['상세내용'], vector_size = 280, window = 30, min_count = 3, workers = 4, sg = 1)
        embedding_matrix = model.wv.vectors
        self.vocab_size, self.embedding_dim = embedding_matrix.shape

        x = df['상세내용']
        # label있는 data와 label없는 data구분
        if 'label' in df.columns:
            y = df['label']

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
            print(list(self.y_train).count(1), list(self.y_train).count(0))
            print(list(self.y_test).count(1), list(self.y_test).count(0))

            # label 타입 변경   
            self.y_train = np.array(self.y_train).astype(np.float32)
            self.y_test = np.array(self.y_test).astype(np.float32)   
                    
        else:
            self.x_train = np.array(x)
            tokenizer = Tokenizer(self.vocab_size, oov_token = 'OOV')
            tokenizer.fit_on_texts(self.x_train)
            
        # 정수 인코딩
        tokenizer = Tokenizer(self.vocab_size, oov_token = 'OOV')
        tokenizer.fit_on_texts(self.x_train)
        try:
            self.x_train = tokenizer.texts_to_sequences(self.x_train)
            self.x_test = tokenizer.texts_to_sequences(self.x_test)
        except:
            pass
        
        # 패딩 : 샘플들의 길이를 동일하게 맞춰줌
        print('리뷰의 최대 길이 :',max(len(review) for review in self.x_train))
        print('리뷰의 평균 길이 :',int(sum(map(len, self.x_train))/len(self.x_train)))
        
        padding_len = 70
        while True:
            count = 0
            for sentence in self.x_train:
                if len(sentence) <= padding_len:
                    count = count + 1
                
            rate = (count / len(self.x_train)) * 100
        
            if rate >= 99:
                print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(padding_len, (count / len(self.x_train))*100))
                print('모든 샘플의 길이 %s로 패딩'%(padding_len))
                self.x_train = pad_sequences(self.x_train, padding_len)
                self.x_test = pad_sequences(self.x_test, padding_len) 
                break
            else:
                padding_len = padding_len + 1      

    def make_BiLSTM(self):
        # BiLSTM 모델 구성
        hidden_units = 128  # LSTM 은닉 상태 크기
        output_classes = 2  # 분류할 클래스 수
        max_sequence_length = 3200

        self.model = Sequential()
        self.model.add(Embedding(input_dim = self.vocab_size, output_dim = self.embedding_dim, input_length = None))  # 임베딩 레이어
        self.model.add(Bidirectional(LSTM(hidden_units, return_sequences = True)))  # 양방향 LSTM 레이어
        self.model.add(Bidirectional(LSTM(hidden_units)))  # 양방향 LSTM 레이어
        self.model.add(Dense(output_classes, activation = 'softmax'))  # 출력 레이어

        es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)
        mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

        self.model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        # 모델 학습
        trained_model = self.model.fit(self.x_train, self.y_train, epochs = 7, callbacks=[es, mc], batch_size = 256, validation_split = 0.2)
        self.model.save('trained_BiLSTM_model') 
        
    def evaluate_model(self):
        loaded_model = load_model('trained_BiLSTM_model')  
        print("테스트 정확도: %.4f" % (loaded_model.evaluate(self.x_test, self.y_test)[1]))
        
        # 정밀도ㅡ재현율-F1-Score 계산 초기식
        predicted_labels = loaded_model.predict(self.x_test)
        predicted_labels = np.argmax(predicted_labels, axis=1)
        true_labels = self.y_test

        # 정밀도
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        print("정밀도 테스트: {}".format(precision))

        # 재현율
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        print("재현율 테스트: {}".format(recall))

        # F1 스코어 계산
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        print("F1 테스트: {}".format(f1))
        
        '''
        correct_data = 0
        score = loaded_model.predict(self.x_test) # 예측
        
        for i in range(0, len(self.x_test)):
            if(score[i][0] > 0.5):
                if self.y_test[i] == 1:
                    correct_data = correct_data + 1
            else:
                if self.y_test[i] == 0:
                    correct_data = correct_data + 1
        print(f'정답률 {len(self.y_test)}개중 {correct_data}개 정답.')
        print(f'{correct_data/len(self.y_test) * 100:.2f}%')
        '''
        
    def predict_model(self):
        loaded_model = load_model('trained_BiLSTM_model')  
        score = loaded_model.predict(self.x_train)
        
        correct_data = 0
        for i in range(0, len(self.x_train)):
            if(score[i][0] > 0.5):
                if self.y_train[i] == 1:
                    correct_data = correct_data + 1
            else:
                if self.y_train[i] == 0:
                    correct_data = correct_data + 1
        print(f'정답률 {len(self.y_train)}개중 {correct_data}개 정답.')
        print(f'{correct_data/len(self.y_train) * 100:.2f}%')            

#######################################################################################################################################################

def preprocessing(df, num):
    Pre_process = pre_process(df)
    df = Pre_process.df

    Label = label(df)
    df = Label.label_processing() # 1. 주제 없애고 label까지 처리  

    Text = text(df)
    df = Text.text_processing() # 2. '내용, 상세내용'의 특수문자 제거, 불용어 제거, 맞춤법 조정  

    Token = token(df)
    df = Token.token_processing() # 3. 토큰화
    
    if num == 0:
        df.to_csv('SNU_keyword_data.csv', index = False)
    elif num == 1:
        df.to_csv('Naver_keyword_data.csv', index = False)
    elif num == 2:
        df.to_csv('Youtube_keyword_data.csv', index = False)
    else:
        df.to_csv('sample_data.csv', index = False)
    
    return df

# SNU_keyword_data로 BiLSTM모델 생성 
def make_model(df):
    model = Model()    
    model.word_embedding(df)
    model.make_BiLSTM()  
    model.evaluate_model()
    
    return model

# 예측하고 싶은 Naver/Youtube data중 택하여 자동 labeling
def labeling(df):
    model = Model()
    df = model.list_to_str(df)
    model.word_embedding(df)

    loaded_model = load_model('trained_BiLSTM_model') 
    score = loaded_model.predict(model.x_train)
    #y_test = np.argmax(score, axis=1)
    print(score)
    label = []
    for i in range(0, len(score)):
        if (score[i][0] > 0.15):
            label.append(1)
        else:
            label.append(0)
    
    print(label.count(1))
    print(label.count(0))

    df['label'] = label
    
    return df
    #return model.x_train, y_test

###########################################################################################
# 1. 전처리 + 토큰화까지 과정 (** 오래 걸림!!! 최소 3시간 **)                                                            
#csv = pd.read_csv('D:\GitHub\Data_on_Air\Dataset\SNU_data.csv', encoding = 'cp949')    
#snu_df = preprocessing(csv, 0)                                                            
#csv = pd.read_csv('D:\GitHub\Data_on_Air\Dataset\Naver_data.csv', encoding = 'cp949')  
#naver_df = preprocessing(csv, 1)                                                          
#csv = pd.read_csv('D:\GitHub\Data_on_Air\Dataset\Youtube_data.csv', encoding = 'cp949')
#youtube_df = preprocessing(csv, 2)                                                        
###########################################################################################

###########################################################################################
# 2. 데이터 간의 토큰 길이 조정 
#model = Model()
#
#snu_df = pd.read_csv('D:/GitHub/Data_on_Air/Dataset/SNU_keyword_data.csv', encoding = 'utf-8')
#Token = token(snu_df)
#snu_df = Token.noun_filter()
#model.word_embedding(snu_df)
#snu_df.to_csv('SNU_keyword_data.csv', index = False)
#
#naver_df = pd.read_csv('D:/GitHub/Data_on_Air/Dataset/Naver_keyword_data.csv', encoding = 'utf-8')
#Token = token(naver_df)
#naver_df = Token.noun_filter()
#model.word_embedding(naver_df)
#naver_df.to_csv('Naver_keyword_data.csv', index = False)
#
#youtube_df = pd.read_csv('D:/GitHub/Data_on_Air/Dataset/youtube_keyword_data.csv', encoding = 'utf-8')
#Token = token(youtube_df)
#youtube_df = Token.noun_filter()
#model.word_embedding(youtube_df)
#youtube_df.to_csv('Youtube_keyword_data.csv', index = False)
###########################################################################################

def main():
    ###########################################################################################
    # 3. SNU_keyword_data로 BiLSTM모델 생성 
    #snu_df = pd.read_csv('SNU_keyword_data.csv', encoding = 'utf-8')
    snu_df = pd.read_csv('D:/Github/Data_on_Air/Dataset/SNU_keyword_data.csv', encoding = 'utf-8')
    snu_df = Model().list_to_str(snu_df) 

    oversampler = RandomOverSampler(random_state=42)
    x = snu_df.drop('label', axis=1)
    y = snu_df['label']
    X_resampled, y_resampled = oversampler.fit_resample(x, y)
    snu_df = pd.DataFrame(X_resampled, columns=x.columns)
    snu_df['label'] = y_resampled

    trained_model = make_model(snu_df)
    ###########################################################################################

    ###########################################################################################
    # 4. 네이버 / 유튜브 TF예측 하면서 BiLSTM모델 생성 
    naver_df = pd.read_csv('D:/Github/Data_on_Air/Dataset/Naver_keyword_data.csv', encoding = 'utf-8')
    youtube_df = pd.read_csv('D:/Github/Data_on_Air/Dataset/Youtube_keyword_data.csv', encoding = 'utf-8')
    new_df = naver_df.append(youtube_df)
    new_df = Model().list_to_str(new_df)
    train_df = snu_df

    start = 0
    end = 10
    while True:
        try:
            for count in range(1, len(train_df)):
                for _ in range(0, 4):
                    df = naver_df[start:end]
                    df = labeling(df)
                    
                    train_df = train_df.append(df)
                    make_model(train_df)
                    
                    start = end
                    end = end + count * 10               
        except:
            pass
    ###########################################################################################


#main()


###########################################################################################
# 5. label을 알고 있는 SNU_data로 재검증
model = Model()
snu_df = pd.read_csv('D:/Github/Data_on_Air/Dataset/SNU_keyword_data.csv', encoding = 'utf-8')
snu_df = model.list_to_str(snu_df)
oversampler = RandomOverSampler(random_state=42)
x = snu_df.drop('label', axis=1)
y = snu_df['label']
X_resampled, y_resampled = oversampler.fit_resample(x, y)
snu_df = pd.DataFrame(X_resampled, columns=x.columns)
snu_df['label'] = y_resampled
model.word_embedding(snu_df)
model.predict_model()
###########################################################################################

# 경우 1 : 10개씩 증가하는 라벨링 데이터 학습...  / epochs = 5, label_rate = 0.5
#         테스트정확도 : 0.6217 / snu대입 : 0.5042

# 경우 2 : 5개씩 증가하는 라벨링 데이터 학습... / epochs = 5, label_rate = 0.5
#         테스트정확도 : 0.6386 / snu대입 : 0.4648
# 폐기... 너무 오래걸려

# 경우 3 : 10개씩 증가하는 라벨링 데이터 학습... / epochs = 6, label_rate = 0.3
#         테스트정확도 : 0.6313 / snu대입 : 0.4535

# 경우 4 : 10개씩 증가하는 라벨링 데이터 학습... / epochs = 5, label_rate = 0.3
#         테스트정확도 : 0.6217 / snu대입 : 0.3944
# 12:46 ~ 1:21

# 경우 5 : 10 * n개씩 5번씩, 증가하는 라벨링 데이터 학습... / epochs = 7, label_rate = 0.2
#         테스트정확도 : 0.65xx / snu대입 : 0.5680
# 1:54 ~ 3:15