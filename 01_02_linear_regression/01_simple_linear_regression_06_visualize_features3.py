import numpy as np # 수학 계산 라이브러리
import matplotlib.pyplot as plt # 시각화 인터페이스
import pandas as pd # 표 데이터 쉽게 다루도록 함

# 자동차 연비, CO2 배출량 데이터 csv 링크
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url) # df : 데이터프레임, pandas로 csv 읽어옴

# 컬럼명 상수 4개 한 줄로 정의
COL_ENGINE_SIZE, COL_CYLINDERS, COL_FUEL_CONS, COL_CO2 = 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS'

cdf = df[[COL_ENGINE_SIZE, COL_CYLINDERS, COL_FUEL_CONS, COL_CO2]] # df에서 원하는 column만 뽑아옴

plt.scatter(cdf[COL_ENGINE_SIZE], cdf[COL_CO2],  color='blue') # scatter plot(산점도) 그리기
plt.xlabel(COL_ENGINE_SIZE) # 라벨 지정
plt.ylabel(COL_CO2) # 라벨 지정
plt.xlim(0,27) # x축 구간 강제 지정
plt.show() # 화면에 띄우기