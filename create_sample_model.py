# import dataset
from sklearn.datasets import load_boston
# load data
data = load_boston()
import pandas as pd
# load into a dataframe
df = pd.DataFrame(data['data'], columns=data['feature_names'].tolist())
df['label'] = data['target']
# shuffle data
shuffled_df = df.sample(frac=1)
# split data
split = int(len(shuffled_df) * .7)
train_df = shuffled_df[:split]
test_df = shuffled_df[split:]
# stock function for extracting X,y columns from df
def X_and_y_from_df(df, y_column, X_columns = []):
    X = {}
    for feature in X_columns:
        X[feature] = df[feature].tolist()
    y = df[y_column].tolist()
    return X, y
# extract X and y
X_train, y_train = X_and_y_from_df(train_df, 'label', ['RM'])
X_test, y_test = X_and_y_from_df(test_df, 'label', ['RM'])
# reshape
import numpy as np
X_train_arr = np.array(X_train['RM'])
X_train_arr = X_train_arr.reshape(X_train_arr.shape[0],1)
X_test_arr = np.array(X_test['RM'])
X_test_arr = X_test_arr.reshape(X_test_arr.shape[0],1)
# train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_arr, y_train)
# predict results
y_pred = model.predict(X_test_arr)

import pickle
pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict(np.array([4, 5, 6]).reshape(-1, 1)))