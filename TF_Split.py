import pandas as pd

def Split_TF(df):
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

    T = len(df[df['label'] == 1])
    F = len(df[df['label'] == 0])

    max_num = abs(T - F)
    df = df.drop(df[df['label'] == 0].tail(max_num).index)
    
    T = len(df[df['label'] == 1])
    F = len(df[df['label'] == 0])
    print(df)
    print(T, F)

    return df

'''
[실행 코드]
df = pd.read_csv("D:\Download\SNU_factcheck_20230816.csv", encoding = 'cp949')
df = Split_TF(df)

[실행 결과]
        row_id                                             내용                                               상세내용      주장/검증 매체  label
row_id
0            0                               한국에서는 팁 문화가 불법이다  국내 최대 모빌리티 플랫폼 카카오T가 지난달부터 택시 기사에게 팁을 주는 서비스를 ...  온라인 커뮤니티 게시물      0
1            1                     7만 원짜리 잼버리 텐트를 25만 원에 납품했다  2023 새만금 세계스카우트 잼버리 조직위원회에서 각국 대표단에게 제공한 2~3인용...      인터넷 커뮤니티      0
2            2           직원 1.6만명인데 임원 1.4만명, 새마을금고 조직구조 희한하다   <한국경제신문>은 7월 5일 새마을금고 부실 사태와 관련해 "방만경영과 관리부실이...          언론보도      0
250        250             선진국에선 줄고 있는 청년실업이 우리나라에서만 역주행하고 있다  하태경 국민의힘 대선후보는 지난 8월 5일 공약 발표 기자회견에서 “문재인 정부의 ...           하태경      1     
252        252                         재난지원금으로 최신 전자기기 살 수 있다  편의점에서 신종 코로나바이러스 감염증(코로나19) 상생 국민지원금(5차 재난지원금)...  다수의 온라인 커뮤니티      1        
253        253  대선후보의 주택 공급 대책 250만 호, 다 문재인 정부의 주택 공급 합친 것이다  이낙연 더불어민주당 대선후보는 지난 26일 박용진X이낙연 끝장토론에서 “대선후보가 ...           이낙연      1  
258        258            급증하는 ‘채용형 인턴’, 취업 준비생 보호할 법적 안전망 없다  하반기 채용이 본격화되고 있는 가운데 최근 취업 시장에서는 ‘채용연계형(채용형) 인...   언론사 자체 문제제기      1 

[164 rows x 5 columns]
82 82
'''
