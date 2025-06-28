import numpy as np # 수학 계산 라이브러리
import matplotlib.pyplot as plt # 시각화 인터페이스
import pandas as pd # 표 데이터 쉽게 다루도록 함
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url) # df : 데이터프레임, pandas로 csv 읽어옴

df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1) # 특정 열 제거(drop)
df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)

X = df.iloc[:,[0,1]].to_numpy() # iloc > 행/열을 번호로 인덱싱
y = df.iloc[:,[2]].to_numpy()

std_scaler = preprocessing.StandardScaler() # 표준화 도구 선언(각 변수의 값을 평균 0, 표준편차 1로 변환)
X_std = std_scaler.fit_transform(X) # 각 열에 대해 평균, 표준편차 계산 및 계산된 값으로 표준화

X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.2,random_state=42)

regressor = linear_model.LinearRegression()

regressor.fit(X_train, y_train)

coef_ =  regressor.coef_
intercept_ = regressor.intercept_

means_ = std_scaler.mean_ # 평균값 배열
std_devs_ = np.sqrt(std_scaler.var_)# 표준 편차 배열(var_은 분산)

coef_original = coef_ / std_devs_ # 표준화된 스케일의 계수이므로, 표춘편차로 나누어준다
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_) # 마찬가지로 원래 스케일의 절편으로 변환
# 원래 절편 = 표준화 절편 - Σ (각 평균 * 계수 / 표준편차)

plt.scatter(X_train[:,0], y_train,  color='blue')
plt.plot(X_train[:,0], coef_[0,0] * X_train[:,0] + intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()