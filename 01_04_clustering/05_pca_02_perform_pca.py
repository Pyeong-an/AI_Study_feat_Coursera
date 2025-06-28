import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 2]]
X = np.random.multivariate_normal(mean=mean, cov=cov, size=200) # 랜덤 뽑기

pca = PCA(n_components=2) # 데이터의 주요 변동성 찾기
X_pca = pca.fit_transform(X)

components = pca.components_
print(components) # 주 성분 벡터
print(pca.explained_variance_ratio_) #분산 정도