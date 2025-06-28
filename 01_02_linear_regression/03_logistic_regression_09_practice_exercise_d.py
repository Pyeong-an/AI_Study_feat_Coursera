import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)

# c. What happens to the log loss value if we add both "callcard" and "wireless" to the input features?
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'churn']] # 필요열만 선택
churn_df['churn'] = churn_df['churn'].astype('int') # 값들을 int로 변환(정수)

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ']]) # 필요열 선택 후 array로 변환
y = np.asarray(churn_df['churn']) # 필요열 선택 후 array로 변환

X_norm = StandardScaler().fit(X).transform(X) # 표준화

X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)

LR = LogisticRegression().fit(X_train,y_train) # 로지스틱 회귀 모델 생성 및 학습

yhat = LR.predict(X_test) # X_test의 예측 실행
yhat_prob = LR.predict_proba(X_test) # X_test의 확률 반환

print(log_loss(y_test, yhat_prob)) # 실제 정답과 얼마나 가까운지 측정하는 로그 손실, 낮을수록 잘 만들어진 것