
import re
import nltk
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from konlpy.tag import Okt
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
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
nltk.download('punkt')

# 초기 설정
class pre_process: 

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    vocab_size = 0
    embedding_dim = 0
    
    def __init__(self, csv):
        #os.chdir('py-hanspell')
        self.okt = Okt()
        self.df = csv

# 1. 주제 없애고 label까지 처리        
class label(pre_process):
    def label_processing(self):
        self.df.dropna(axis=0, inplace=True) # 결측치 제거
        idx = self.df.loc[self.df['label']=='판단 유보'].index
        self.df = self.df.drop(idx, axis=0)
        idx = self.df.loc[self.df['label']=='논쟁 중'].index
        self.df = self.df.drop(idx, axis=0)
        self.df= self.df.drop(columns=['주제'])
        self.df['label'] = self.df['label'].replace({'전혀 사실 아님': 0, '대체로 사실 아님': 0, '절반의 사실': 0, '대체로 사실': 1, '사실': 1})

        self.df['row_id'] = range(0, len(self.df))
        self.df.index = self.df['row_id']
        self.df['row_id'] = self.df['row_id'].astype(int)
        self.df['label'] = self.df['label'].astype(int)

        return self.df
    
# 2. '내용, 상세내용'의 특수문자 제거, 불용어 제거, 맞춤법 조정       
class text(pre_process):
    # 2-1 특수문자 제거
    def remove_special_characters(self, text):
        # 한글, 알파벳, 숫자, 공백을 제외한 문자 제거
        return re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s]', '', text)

    # 2-2 맞춤법 검사
    def spell_check_text(self, text):
        try:
            result = spell_checker.check(text)
            corrected_text = result.checked
            return corrected_text
        except Exception as e:
            print("Error:", e)
            return text

    # 텍스트를 띄어쓰기 구분해 리스트로
    def split_text(self, text):
        word_list = text.split()
        return word_list

    def group_text_items(self, text_list, max_length=500):
        grouped_texts = []
        current_group = []

        current_length = 0
        for item in text_list:
            item_length = len(item) + 1  # 항목 길이 + 띄어쓰기 길이

            # 항목을 더했을 때 최대 글자수를 초과하는 경우 새로운 그룹 시작
            if current_length + item_length > max_length:
                grouped_texts.append(' '.join(current_group))
                current_group = []
                current_length = 0

            current_group.append(item)
            current_length += item_length

        # 남은 항목이 있는 경우 마지막 그룹 추가
        if current_group:
            grouped_texts.append(' '.join(current_group))

        return grouped_texts
    
    # 2. '내용, 상세내용'의 특수문자 제거, 불용어 제거, 맞춤법 조정
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
        
        for content in contents:
        # 내용 맞춤법 검사
            text = self.split_text(self.df[content][0])

            grouped_text_list = self.group_text_items(text)

            # 맞춤법 검사 및 수정된 결과로 다시 합치기
            corrected_grouped_texts = []
            for group in grouped_text_list:
                corrected_group = self.spell_check_text(group)
                corrected_grouped_texts.append(corrected_group)

            corrected_text = ' '.join(corrected_grouped_texts)

            # 모든 행에 대해 작업 수행
            corrected_texts = []
            for index, row in self.df.iterrows():
                text = self.split_text(row[content])
                grouped_text_list = self.group_text_items(text)

                corrected_grouped_texts = []
                for group in grouped_text_list:
                    corrected_group = self.spell_check_text(group)
                    corrected_grouped_texts.append(corrected_group)

                corrected_text = ' '.join(corrected_grouped_texts)
                corrected_texts.append(corrected_text)

            # 수정된 결과로 변경
            self.df[content] = corrected_texts
            
        return self.df 

# 3. 토큰화
class token(pre_process):  
    # '상세내용' 토큰화
    def tokenizer(self, text):
        morph = self.okt.pos(text)
        words = []
        for word, tag in morph:
            if tag in ['Noun']:
                if len(word) > 1:
                    words.append(word)
        return words

    # 3. 토큰화
    def token_processing(self):
        self.df= self.df.astype('str')
        self.df['상세내용'] = self.df['상세내용'].apply(self.tokenizer)

        for list in self.df['상세내용']:
            word_counts = Counter(list)
            most_common_word = word_counts.most_common(1)[0][0]

        temp_data = []
        most_common_words_data = []
        for text in self.df['상세내용']:
            word_counts = Counter(text)
            most_common_words = word_counts.most_common()

            most_common_words_only = [word for word, count in most_common_words if count > 2] # 20230816 나중에 수정하세요
            temp_data.append(len(most_common_words_only))
            most_common_words_data.append(most_common_words_only)
            
        self.df['temp'] = temp_data
        self.df['빈도순'] = most_common_words_data

        for row in range(0, len(self.df), 1):
            temp = self.df['temp']
            if int(temp[row]) == 0:
                self.df.drop(row, inplace = True)

        del self.df['temp']

        return self.topic_modeling()

    #
    def topic_modeling(self):
        new_df = self.df['빈도순'].copy()  # .copy() 메서드를 사용하여 복사본 생성
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
                                    if count >= 4: # n개의 키워드가 같을시 유사한 주제로 판단.
                                        group.append(row)  # 인덱스를 추가
                                        data[row] = []
                                        count = 0
                                        break  # 더이상 반복할 필요 없음
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
        
        return self.df

# 4. 워드 임베딩 
class Word_Embedding(pre_process):      
    def word_embedding(self):
        # Word2Vec 모델로 학습된 임베딩 벡터 가져오기
        model = Word2Vec(sentences = self.df['상세내용'], vector_size = 280, window = 10, min_count = 3, workers = 4, sg = 1)
        embedding_matrix = model.wv.vectors
        pre_process.vocab_size, pre_process.embedding_dim = embedding_matrix.shape

        x = self.df['상세내용']
        y = self.df['label']

        pre_process.x_train, pre_process.x_test, pre_process.y_train, pre_process.y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
        print(f'TF갯수 : {int(len(pre_process.y_train) / 2)}개씩.. 1:1비율')

        # 정수 인코딩
        tokenizer = Tokenizer(pre_process.vocab_size, oov_token = 'OOV')
        tokenizer.fit_on_texts(pre_process.x_train)
        pre_process.x_train = tokenizer.texts_to_sequences(pre_process.x_train)
        pre_process.x_test = tokenizer.texts_to_sequences(pre_process.x_test)
        
        # 패딩 : 샘플들의 길이를 동일하게 맞춰줌
        print('리뷰의 최대 길이 :',max(len(review) for review in pre_process.x_train))
        print('리뷰의 평균 길이 :',sum(map(len, pre_process.x_train))/len(pre_process.x_train))
        
        max_len = ((max(len(review) for review in pre_process.x_train) // 100) - 1) * 100
        while True:
            count = 0
            for sentence in pre_process.x_train:
                if len(sentence) <= max_len:
                    count = count + 1
                
            rate = (count / len(pre_process.x_train)) * 100
            if rate >= 99:
                print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(pre_process.x_train))*100))
                print('모든 샘플의 길이 %s로 패딩'%(max_len))
                pre_process.x_train = pad_sequences(pre_process.x_train, maxlen=max_len)
                pre_process.x_test = pad_sequences(pre_process.x_test, maxlen=max_len)

                # label 타입 변경   
                pre_process.y_train = np.array(pre_process.y_train).astype(np.float32)
                pre_process.y_test = np.array(pre_process.y_test).astype(np.float32)    
                break
            else:
                max_len = max_len + 1

# 5. BiLSTM 모델 학습 
class BiLSTM(pre_process):
    def make_BiLSTM(self):
        # BiLSTM 모델 구성
        hidden_units = 128  # LSTM 은닉 상태 크기
        output_classes = 2  # 분류할 클래스 수
        max_sequence_length = 3200

        model = Sequential()
        model.add(Embedding(input_dim = pre_process.vocab_size, output_dim = pre_process.embedding_dim, input_length = None))  # 임베딩 레이어
        model.add(Bidirectional(LSTM(hidden_units, return_sequences = True)))  # 양방향 LSTM 레이어
        model.add(Bidirectional(LSTM(hidden_units)))  # 양방향 LSTM 레이어
        model.add(Dense(output_classes, activation = 'softmax'))  # 출력 레이어

        es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)
        mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

        model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        # 모델 학습
        trained_model = model.fit(pre_process.x_train, pre_process.y_train, epochs = 15, callbacks=[es, mc], batch_size = 256, validation_split = 0.2)
        model.save('trained_BiLSTM_model') 
        loaded_model = load_model('trained_BiLSTM_model')   
        print("테스트 정확도: %.4f" % (loaded_model.evaluate(pre_process.x_test, pre_process.y_test)[1]))
    
def preprocessing(csv):
    Pre_process = pre_process(csv)
    df = Pre_process.df
    
    Label = label(df)
    df = Label.label_processing() # 1. 주제 없애고 label까지 처리  

    Text = text(df)
    df = Text.text_processing() # 2. '내용, 상세내용'의 특수문자 제거, 불용어 제거, 맞춤법 조정  

    Token = token(df)
    df = Token.token_processing() # 3. 토큰화

    df.to_csv('SNU_token_data.csv')
            
    return df

def make_model(df):
    word_embedding = Word_Embedding(df)
    word_embedding.word_embedding()

    bilstm = BiLSTM(df)
    bilstm.make_BiLSTM()

def predict_model(x_test, y_test):
    loaded_model = load_model('trained_BiLSTM_model')  
    correct_data = 0
    score = loaded_model.predict(x_test) # 예측

    # 정밀도ㅡ재현율-F1-Score 계산 초기식
    predicted_labels = loaded_model.predict(pre_process.x_test)
    predicted_labels = np.argmax(predicted_labels, axis=1)
    true_labels = pre_process.y_test

    # 정밀도
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    print("정밀도 테스트: {}".format(precision))

    # 재현율
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    print("재현율 테스트: {}".format(recall))

    # F1 스코어 계산
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print("F1 테스트: {}".format(f1))

    for i in range(0, len(x_test)):
        if(score[i][1] > 0.5):
            print(f"{score[i][1] * 100:.2f}% 확률로 참입니다. <실제 판단 여부 : {y_test[i]}>")
            if y_test[i] == 1:
                correct_data = correct_data + 1
        else:
            print(f"{(1 - score[i][1]) * 100:.2f}% 확률로 거짓입니다. <실제 판단 여부 : {y_test[i]}>")
            if y_test[i] == 0:
                correct_data = correct_data + 1
    print(f'정답률 {len(y_test)}개중 {correct_data}개 정답.')
    print(f'{correct_data/len(y_test) * 100:.2f}%')

df = pd.read_csv("D:\GitHub\Data_on_Air\Dataset\SNU_keyword_data.csv", encoding = 'cp949')
df = preprocessing(df)
print(df)
#df.to_csv("D:\Download\SNU_factcheck_keyword_sample.csv")

''' 
<1. preprocessig 함수>
csv = pd.read_csv('파일 주소', encoding = 'cp949')
df = preprocessing(csv)

** 토큰 빈도수 0으로 했으니 나중에 2로 수정하세요  (185번째 줄)

---------------------------------------------------------------------------------------------------------------------------------------------
<2. make_model 함수>
[실행코드]
make_model(df)

[결과]
   row_id	주제	내용	                                                        상세내용	                                                               주장/검증 매체	          label
0	0.0	    경제	직원 16만 명인데 임원 14만 명 새마을금고 조직구조 희한하다	        [한국, 신문, 새마을금고, 부실, 사태, 관련, 방만, 경영, 관리, 부실, 사태...	   언론보도	                 전혀 사실 아님
1	1.0	    경제	탄소 포집 기술은 재생에너지보다 경제성이 뛰어난 탄소중립 대안이다	 [탄소, 중립, 녹색, 성장, 위원회, 지난, 탄소, 중립, 녹색, 성장, 계획, ...	    조원동	                  전혀 사실 아님
2	2.0	    경제	탄소 포집 기술은 2030년 이전에 상용화돼 탄소중립에 기여할 수 있다	 [확정, 정부, 탄소, 중립, 녹색, 성장, 계획, 활용, 연간, 온실가스, 감축,...	    언론사 자체 문제제기	   대체로 사실 아님

리뷰의 최대 길이 : 372
리뷰의 평균 길이 : 351.5
전체 샘플 중 길이가 380 이하인 샘플의 비율: 100.0
모든 샘플의 길이 380로 패딩
Epoch 1/15
1/1 [==============================] - ETA: 0s - loss: 0.7009 - accuracy: 0.0000e+00WARNING:tensorflow:Can save best model only with val_acc available, skipping.
1/1 [==============================] - 12s 12s/step - loss: 0.7009 - accuracy: 0.0000e+00 - val_loss: 0.6153 - val_accuracy: 1.0000

...

Epoch 15/15
1/1 [==============================] - ETA: 0s - loss: 0.0000e+00 - accuracy: 1.0000WARNING:tensorflow:Can save best model only with val_acc available, skipping.
1/1 [==============================] - 0s 90ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
WARNING:absl:Found untraced functions such as lstm_cell_31_layer_call_fn, lstm_cell_31_layer_call_and_return_conditional_losses, lstm_cell_32_layer_call_fn, lstm_cell_32_layer_call_and_return_conditional_losses, lstm_cell_34_layer_call_fn while saving (showing 5 of 8). These functions will not be directly callable after loading.
1/1 [==============================] - 2s 2s/step - loss: 0.0000e+00 - accuracy: 1.0000
테스트 정확도: 1.0000

-----------------------------------------------------------------------------------------------------------------------------------------------
<3. predict_model 함수>
[실행코드]
predict_model(pre_process.x_test, pre_process.y_test) --> 테스트 데이터 x, y

[결과]
1/1 [==============================] - 2s 2s/step
100.00% 확률로 거짓입니다. <실제 판단 여부 : 0.0>
정답률 1개중 1개 정답.
100.00%

20230816 23:59
- label_processing() : TF비율 1:1로 맞추는 기능 추가.

'''
