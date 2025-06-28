from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

import warnings
from sklearn.tree import DecisionTreeRegressor
warnings.filterwarnings('ignore')

# read the input data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)

y = raw_data[['tip_amount']].values.astype('float32')
proc_data = raw_data.drop(['tip_amount'], axis=1)
X = proc_data.values
X = normalize(X, axis=1, norm='l1', copy=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Q4. Check the effect of decreasing the max_depth parameter to 4 on the mse and r2 values.
dt_reg = DecisionTreeRegressor(criterion = 'squared_error',
                               max_depth=4,
                               random_state=35)
dt_reg.fit(X_train, y_train)

y_pred = dt_reg.predict(X_test)

mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score)) #24.555 > 24.412

r2_score = dt_reg.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score)) # 0.028 > 0.034

# more suited