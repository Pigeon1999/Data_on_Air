import requests
import json

# 인증키 
apikey = '75546c586b70617338367a5858524c'
url = f'http://openapi.seoul.go.kr:8088/{apikey}/json/citydata_ppltn/1/10/강남 MICE 관광특구'

# request 결과 json으로 출력 
response = requests.get(url).json()
print(json.dumps(response, indent = 2, ensure_ascii = False))
