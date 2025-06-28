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

# Q1. Experiment with different test sizes in the train_test_split method (e.g., 0.1, 0.3) and observe the impact on model performance.
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))

X_train_01, X_test_01, y_train_01, y_test_01 = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
model_ovo.fit(X_train_01, y_train_01)
y_pred_ovo_01 = model_ovo.predict(X_test_01)

X_train_02, X_test_02, y_train_02, y_test_02 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model_ovo.fit(X_train_02, y_train_02)
y_pred_ovo_02 = model_ovo.predict(X_test_02)

X_train_03, X_test_03, y_train_03, y_test_03 = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model_ovo.fit(X_train_03, y_train_03)
y_pred_ovo_03 = model_ovo.predict(X_test_03)

print("One-vs-One (OvO) Strategy")
print(f"Accuracy 0.1: {np.round(100*accuracy_score(y_test_01, y_pred_ovo_01),2)}%")
print(f"Accuracy 0.2: {np.round(100*accuracy_score(y_test_02, y_pred_ovo_02),2)}%")
print(f"Accuracy 0.3: {np.round(100*accuracy_score(y_test_03, y_pred_ovo_03),2)}%")