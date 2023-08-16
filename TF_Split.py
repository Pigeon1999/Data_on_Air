import pandas as pd

def label_processing(df):
    df.dropna(axis=0, inplace=True) # 결측치 제거
    idx = df.loc[df['label']=='판단 유보'].index
    df = df.drop(idx, axis=0)
    idx = df.loc[df['label']=='논쟁 중'].index
    df = df.drop(idx, axis=0)
    df= df.drop(columns=['주제'])
    df['label'] = df['label'].replace({'전혀 사실 아님': 0, '대체로 사실 아님': 0, '절반의 사실': 0, '대체로 사실': 1, '사실': 1})
    df['row_id'] = range(0, len(df))
    df.index = df['row_id']
    df['row_id'] = df['row_id'].astype(int)
    df['label'] = df['label'].astype(int)
    
    return df

df = pd.read_csv("D:\Download\SNU_factcheck_20230816.csv", encoding = 'cp949')
df = label_processing(df)
print(len(df[df['label'] == 1]))


