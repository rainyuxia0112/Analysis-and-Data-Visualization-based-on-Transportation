import pandas as pd
data=pd.read_csv('/Users/rain/Desktop/datapreprocessing.csv')

#missing data
data.drop(data.index[[33,38]],inplace=True)

Y=data.iloc[:,1].values

X=data.iloc[:,4:10].values

# time
import numpy as np
for i in range(len(X)):
    X[i,2]=float(X[i,2].replace(':','.'))
    if X[i,2]<=10:
        X[i,2]=1
    else:
        X[i,2]=0

#encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])

X=X[:,[0,1,2,3,5]]

onehotencoder = OneHotEncoder(sparse=False,categorical_features=[1,3])
X = onehotencoder.fit_transform(X)
X=np.delete(X,15,axis=1)
X=np.delete(X,0,axis=1)


X = np.append(arr=np.ones((70,1)).astype(int), values=X, axis=1)   # 把b0 也变成 b0*x0
from sklearn.model_selection import train_test_split
#x为数据集的feature熟悉，y为label.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)


# backward method  (找最后使用参数！）
import statsmodels.formula.api as sm
X_opt = x_train[:,:]
regressor_OLS = sm.OLS(endog=y_train,exog=X_opt.astype(float)).fit()
regressor_OLS.summary()


X_opt = X_opt[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17]]
regressor_OLS = sm.OLS(endog=y_train,exog=X_opt.astype(float)).fit()
regressor_OLS.summary()
X_opt = X_opt[:,[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16]]
regressor_OLS = sm.OLS(endog=y_train,exog=X_opt.astype(float)).fit()
regressor_OLS.summary()
X_opt = X_opt[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15]]
regressor_OLS = sm.OLS(endog=y_train,exog=X_opt.astype(float)).fit()
regressor_OLS.summary()
X_opt = X_opt[:,[0,1,2,4,5,6,7,8]]



# prediction
regressor_OLS.params
y_prediction=regressor_OLS.predict(x_test[:,[0,1,6,11,12,13,14,15]])
from sklearn.metrics import r2_score
r2_score(y_true=y_test,y_pred=y_prediction)


#0.8949873051103879