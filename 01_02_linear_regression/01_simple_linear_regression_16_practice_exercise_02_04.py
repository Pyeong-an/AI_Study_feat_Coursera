import numpy as np # 수학 계산 라이브러리
import matplotlib.pyplot as plt # 시각화 인터페이스
import pandas as pd # 표 데이터 쉽게 다루도록 함
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 자동차 연비, CO2 배출량 데이터 csv 링크
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url) # df : 데이터프레임, pandas로 csv 읽어옴

# 컬럼명 상수 4개 한 줄로 정의
COL_ENGINE_SIZE, COL_CYLINDERS, COL_FUEL_CONS, COL_CO2 = 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS'

cdf = df[[COL_ENGINE_SIZE, COL_CYLINDERS, COL_FUEL_CONS, COL_CO2]] # df에서 원하는 column만 뽑아옴

X = cdf[COL_FUEL_CONS].to_numpy() # NumPy 배열로 변환
y = cdf[COL_CO2].to_numpy() # NumPy 배열로 변환

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

regr = linear_model.LinearRegression() # 단순 선형 회귀 모델 선언(변수 선언 같은 선언)
regr.fit(X_train.reshape(-1, 1), y_train) # 모델에 학습 / 1차원 배열(벡터) → 2차원 배열(열 벡터)로 변환, 목표값 배열
# y = ax + b 형태로 절편과 기울기를 찾음!

# 4. Use the model to make test predictions on the fuel consumption testing data.
y_test_ = regr.predict(X_test.reshape(-1,1)) # 테스트용 입력 데이터를 2차원 배열로