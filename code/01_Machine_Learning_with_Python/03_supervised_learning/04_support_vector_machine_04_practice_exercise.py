# Import the libraries we need to use in this lab
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
raw_data=pd.read_csv(url)

#Q1. Currently, we have used all 30 features of the dataset for training the models. Use the corr() function to find the top 6 features of the dataset to train the models on.
correlation_values = raw_data.corr()['Class'].drop('Class')
top_feature_list = abs(correlation_values).sort_values(ascending=False)[:6].index.tolist()
selected_columns = top_feature_list + ['Class']
print("selected_columns :", selected_columns)

#Q2. Using only these 6 features, modify the input variable for training.
raw_data = raw_data[selected_columns]

raw_data.iloc[:, :6] = StandardScaler().fit_transform(raw_data.iloc[:, :6])
data_matrix = raw_data.values

X = data_matrix[:, :6]
y = data_matrix[:, 6]

X = normalize(X, norm="l1")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

w_train = compute_sample_weight('balanced', y_train)

# for reproducible output across multiple function calls, set random_state to a given integer value
dt = DecisionTreeClassifier(max_depth=4, random_state=35)

dt.fit(X_train, y_train, sample_weight=w_train)

# for reproducible output across multiple function calls, set random_state to a given integer value
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

svm.fit(X_train, y_train)


# Q3. Execute the Decision Tree model for this modified input variable. How does the value of ROC-AUC metric change?
y_pred_dt = dt.predict_proba(X_test)[:,1]
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt)) # 0.939 > 0.952


# Q4. Execute the SVM model for this modified input variable. How does the value of ROC-AUC metric change?
y_pred_svm = svm.decision_function(X_test)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm)) # 0.986 > 0.937

# Q5. What are the inferences you can draw about Decision Trees and SVMs with what you have learnt in this lab?
# 결정 트리는 특징적인 것만 있는 편이 성능 좋음
# svm은 값이 많을수록 좋음(대신 오래 걸림)