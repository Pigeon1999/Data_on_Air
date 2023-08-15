#!pip install konlpy
#!pip install tensorflow==2.11
#!pip install nltk
#!pip install gensim
#!pip install --only-binary :all: scikit-learn
#!git clone https://github.com/ssut/py-hanspell
#%cd py-hanspell
#!python setup.py install


import pandas as pd
import re
import nltk
import os
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from konlpy.tag import Okt
from hanspell import spell_checker
from collections import Counter
from konlpy.tag import Okt
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
nltk.download('punkt')

# 초기 설정
class pre_process:
    def __init__(self, df):
        #os.chdir('py-hanspell')
        self.okt = Okt()
        self.df = df

# 1. 주제 없애고 label까지 처리        
class label(pre_process):
    def __init__(self, df):
        super().__init__(df)
    
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
    def __init__(self, df):
        super().__init__(df)
        
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
    
class token(pre_process):   
    def __init__(self, df):
        super().__init__(df)
        
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
        for text in self.df['상세내용']:
            word_counts = Counter(text)
            most_common_words = word_counts.most_common()

            most_common_words_only = [word for word, count in most_common_words if count > 2]
            temp_data.append(len(most_common_words_only))

        self.df['temp'] = temp_data

        for row in range(0, len(self.df), 1):
            temp = self.df['temp']
            if int(temp[row]) == 0:
                self.df.drop(row, inplace = True)

        del self.df['temp']

        return self.df

def main(df):
    Label = label(df)
    Text = text(df)
    Token = token(df)

    df = Label.label_processing()
    df = Text.text_processing()
    df = Token.token_processing()
    
    return df

''' 
[실행코드]
df = pd.read_csv('D:\DownLoad\SNU_factcheck_sample.csv', encoding = 'cp949')
df = main(df)
print(df)

[결과]
   row_id	주제	내용	                                                        상세내용	                                                               주장/검증 매체	          label
0	0.0	    경제	직원 16만 명인데 임원 14만 명 새마을금고 조직구조 희한하다	        [한국, 신문, 새마을금고, 부실, 사태, 관련, 방만, 경영, 관리, 부실, 사태...	   언론보도	                 전혀 사실 아님
1	1.0	    경제	탄소 포집 기술은 재생에너지보다 경제성이 뛰어난 탄소중립 대안이다	 [탄소, 중립, 녹색, 성장, 위원회, 지난, 탄소, 중립, 녹색, 성장, 계획, ...	    조원동	                  전혀 사실 아님
2	2.0	    경제	탄소 포집 기술은 2030년 이전에 상용화돼 탄소중립에 기여할 수 있다	 [확정, 정부, 탄소, 중립, 녹색, 성장, 계획, 활용, 연간, 온실가스, 감축,...	    언론사 자체 문제제기	   대체로 사실 아님
'''