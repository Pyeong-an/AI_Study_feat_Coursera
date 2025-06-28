import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)

continuous_columns = data.select_dtypes(include=['float64']).columns.tolist() # float인 column만

scaler = StandardScaler() # 표준화 도구 정의
scaled_features = scaler.fit_transform(data[continuous_columns]) # 표준화

scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
# 데이터 프레임으로 변환

scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)
# 스케일링 데이터와 나머지 데이터 결합

categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')

encoder = OneHotEncoder(sparse_output=False, drop='first') # 범주형 변수를 0/1로 변환하는 도구
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
# 데이터 프레임으로 변환

prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
# 인코딩 전 범주형 컬럼은 삭제, 인코딩 된 데이터 합침

prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
# 카테고리형으로 변환하고 숫자로 다시 변환

X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000)) # One-vs-One
# ovo란?
# a와 b, a와 c, b와 c... 등으로 전체 비교
# 문제 풀때 모든 비교를 해서 이긴 분류에게 1점씩, 최고 점수가 정답
# 장점 : 세밀하게 구별하기 좋음, 문제를 쪼개 나눌 수 있음
# 단점 : 너무 많으면 쌍 비교가 많아짐 n(n-1), 점수가 똑같을 경우 애매함

model_ovo.fit(X_train, y_train)

y_pred_ovo = model_ovo.predict(X_test)

print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")