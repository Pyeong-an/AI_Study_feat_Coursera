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

X1 = X_test[:, 0] if X_test.ndim > 1 else X_test # nidm은 차원수
X2 = X_test[:, 1] if X_test.ndim > 1 else np.zeros_like(X1) # 주어진 배열과 똑같은 모양으로 0으로 채운 새 배열을 만듬

x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100),
                               np.linspace(X2.min(), X2.max(), 100)) # 격자 좌표 생성

y_surf = intercept_ +  coef_[0,0] * x1_surf  +  coef_[0,1] * x2_surf # 평면의 높이 계산

y_pred = regressor.predict(X_test.reshape(-1, 1)) if X_test.ndim == 1 else regressor.predict(X_test) #예측값 생성
above_plane = y_test >= y_pred # 실제 값이 예측평면보다 위인지 여부
below_plane = y_test < y_pred # 실제 값이 예측평면보다 아래인지 여부
above_plane = above_plane[:,0] # 플로팅 할 때 불리언 마스크로 구분
below_plane = below_plane[:,0]

fig = plt.figure(figsize=(20, 8)) # 그림 생성
ax = fig.add_subplot(111, projection='3d') # 3d 플롯 생성

ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane],  label="Above Plane",s=70,alpha=.7,ec='k') # 위에 있는 점 크고 진하게(70, .7)
ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane],  label="Below Plane",s=50,alpha=.3,ec='k') # 아래에 있는 점 작고 연하게(50, .3)

ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21,label='plane') # 회귀 평면 그림

# Set view and labels
ax.view_init(elev=10) # 시점 높이 설정

ax.legend(fontsize='x-large',loc='upper center') # 범례
ax.set_xticks([]) # 눈금 삭제
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(None, zoom=0.75) # 박스 크기 조정(비율 맞춤)
ax.set_xlabel('ENGINESIZE', fontsize='xx-large') # 엔진 크기
ax.set_ylabel('FUELCONSUMPTION', fontsize='xx-large') # 연료 소비량
ax.set_zlabel('CO2 Emissions', fontsize='xx-large') # CO2 배출량
ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large') # 제목
plt.tight_layout() # 여백 조정
plt.show()
