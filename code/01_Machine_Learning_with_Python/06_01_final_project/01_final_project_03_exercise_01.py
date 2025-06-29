import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 타이타닉 데이터셋 로드
titanic = sns.load_dataset('titanic')

# feature
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']

# target
target = 'survived'

# 분배
X = titanic[features]
y = titanic[target]

# Exercise 1. How balanced are the classes?
# “survived” 컬럼의 0과 1의 개수와 비율을 확인해보고, 균형이 맞는지 살펴보라는 뜻
print(y.value_counts())