import numpy as np # 수학 계산 라이브러리
import matplotlib.pyplot as plt # 시각화 인터페이스
import pandas as pd # 표 데이터 쉽게 다루도록 함
from sklearn import preprocessing

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url) # df : 데이터프레임, pandas로 csv 읽어옴

df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1) # 특정 열 제거(drop)
df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)

X = df.iloc[:,[0,1]].to_numpy() # iloc > 행/열을 번호로 인덱싱
y = df.iloc[:,[2]].to_numpy()

std_scaler = preprocessing.StandardScaler() # 표준화 도구 선언(각 변수의 값을 평균 0, 표준편차 1로 변환)
X_std = std_scaler.fit_transform(X) # 각 열에 대해 평균, 표준편차 계산 및 계산된 값으로 표준화

print(pd.DataFrame(X_std).describe().round(2)) # 표준화된 데이터의 통계 요약