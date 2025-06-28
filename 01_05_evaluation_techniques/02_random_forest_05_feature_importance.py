import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import skew

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict on test set
y_pred_test = rf_regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = root_mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# Feature importances
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)[::-1]
features = data.feature_names

# Plot feature importances
plt.bar(range(X.shape[1]), importances[indices],  align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances in Random Forest Regression")
plt.show()

# - **랜덤 포레스트가 왜 좋을까?**
#   - 랜덤 포레스트 회귀(Random Forest Regression)는 **데이터가 삐뚤어지거나(왜도, skewness) 이상치가 있어도 잘 작동**해요.
#   - 선형 회귀는 데이터가 **종 모양(정규분포)**일 때 가장 잘 맞는데, 랜덤 포레스트는 그런 가정을 안 해요.
#   - 그래서 skewness(왜도)가 있어도 성능이 크게 나빠지지 않아요.
#
# ---
#
# - **데이터를 표준화해야 할까?**
#   - 랜덤 포레스트는 **표준화(standardizing)가 꼭 필요하지 않아요.**
#   - 표준화가 중요한 건 **거리 기반 알고리즘**일 때예요. 예를 들어:
#     - KNN (가장 가까운 이웃 찾기)
#     - SVM (서포트 벡터 머신)
#   - 랜덤 포레스트는 **결정 나무를 기반**으로 하기 때문에, 변수의 스케일이 달라도 잘 처리해요.
#
# ---
#
# - **집값이 \$500,000 이상으로 잘린(clipped) 게 문제일까?**
#   - 네! 잘린 값들은 **변화가 없어서 모델이 그 구간을 잘 설명 못해요.**
#   - 모델이 “\$500,000 초과는 전부 똑같다”고 배우면 → 실제 비싼 집값 예측을 잘 못할 수 있어요.
#   - 이런 잘린 값들은 **평가 지표도 속일 수 있어요.**
#     - 예측이 잘 된 것처럼 보일 수 있지만 실제로는 못 맞히는 영역이 생길 수 있어요.
#
# ---
#
# - **어떻게 하면 좋을까?**
#   - \$500,000 이상으로 잘린 데이터를 **제거하거나 따로 처리**하는 게 좋아요.
#   - **결과를 꼭 시각화**해야 해요.
#     - 예측값 vs 실제값을 그림으로 보면 → 모델이 어디서 잘 못 맞추는지 바로 알 수 있어요.