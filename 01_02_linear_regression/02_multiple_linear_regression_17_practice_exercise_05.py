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

# Exercise 4
# Repeat the same modeling but use FUELCONSUMPTION_COMB_MPG as the independent variable instead. Display the model coefficients including the intercept.
# X의 1이 FUELCONSUMPTION_COMB_MPG, y가 CO2 emissions임
X_train_2 = X_train[:,1] # FUELCONSUMPTION_COMB_MPG만 가져옴
regressor_2 = linear_model.LinearRegression()
regressor_2.fit(X_train_2.reshape(-1, 1), y_train)
coef_2 =  regressor_2.coef_
intercept_2 = regressor_2.intercept_

# Exercise 5
# Generate a scatter plot showing the results as before on the test data. Consider well the model fits, and what you might be able to do to improve it. We'll revisit this later in the course.
X_test_2 = X_test[:,1]
plt.scatter(X_test_2, y_test,  color='blue')
plt.plot(X_test_2, coef_2[0] * X_test_2 + intercept_2, '-r')
plt.xlabel("combined Fuel Consumption (MPG)")
plt.ylabel("Emission")
plt.show()