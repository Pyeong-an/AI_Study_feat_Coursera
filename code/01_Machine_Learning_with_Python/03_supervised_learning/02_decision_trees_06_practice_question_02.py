import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)

label_encoder = LabelEncoder() #  Scikit-Learn 도구 : 범주형을 숫자로 바꿔준다
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex'])
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])

custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)

y = my_data['Drug']
X = my_data.drop(['Drug','Drug_num'], axis=1)

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4) # 모델 선언
drugTree.fit(X_trainset,y_trainset) # 트레이닝

tree_predictions = drugTree.predict(X_testset) # 예측

print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))

plot_tree(drugTree,
          feature_names=X.columns,
          class_names=drugTree.classes_,  # 여기에 클래스 이름!
          filled=True)
plt.show()

# Along similar lines, identify the decision criteria for all other classes.
# drug Y : Na_to_K > 14.627
# drug A : Na_to_K <= 14.627, BP = High, AGE <= 50.5
# drug B : Na_to_K <= 14.627, BP = High, AGE > 50.5
# drug X : Na_to_K <= 14.627, BP = Low
# drug C : Na_to_K <= 14.627, BP = Normal, Cholesterol = High
