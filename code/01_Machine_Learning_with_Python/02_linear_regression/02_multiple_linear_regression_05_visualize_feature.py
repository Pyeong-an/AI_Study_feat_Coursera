import numpy as np # 수학 계산 라이브러리
import matplotlib.pyplot as plt # 시각화 인터페이스
import pandas as pd # 표 데이터 쉽게 다루도록 함

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url) # df : 데이터프레임, pandas로 csv 읽어옴

df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1) # 특정 열 제거(drop)
df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)

axes = pd.plotting.scatter_matrix(df, alpha=0.2) # 산점도가 행렬 형태로 만들어짐, axds에담음
# need to rotate axis labels so we can read them
for ax in axes.flatten(): # 2차원 배열인 axes를 1차원으로
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout() #플롯 간 간격 조정
plt.gcf().subplots_adjust(wspace=0, hspace=0) # plt.gcf > 전체 그림 객체 / subplots_adjust > 서브 플롯 간 간격 조정
plt.show()