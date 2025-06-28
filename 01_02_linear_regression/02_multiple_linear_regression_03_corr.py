import numpy as np # 수학 계산 라이브러리
import matplotlib.pyplot as plt # 시각화 인터페이스
import pandas as pd # 표 데이터 쉽게 다루도록 함

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url) # df : 데이터프레임, pandas로 csv 읽어옴

df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1) # 특정 열 제거(drop)
print(df.corr()) # 변수 사이의 상관계수 계산해서 출력