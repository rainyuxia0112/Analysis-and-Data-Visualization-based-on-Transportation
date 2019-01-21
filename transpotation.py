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
# backward method  (找最后使用参数！）
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
X= backwardElimination(X,0.1)


from sklearn.model_selection import train_test_split
#x为数据集的feature熟悉，y为label.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

#model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pre=reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_true=y_test,y_pred=y_pre)



# using SVR to predict

# using PCA to choose peramater
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_ = sc_x.fit_transform(X)
sc_y = StandardScaler()
Y_ = sc_y.fit_transform(Y.reshape(-1,1))
from  sklearn.decomposition import PCA
pca=PCA(n_components=10)
X_new=pca.fit_transform(X)

from sklearn.model_selection import train_test_split
#x为数据集的feature熟悉，y为label.
x_train, x_test, y_train, y_test = train_test_split(X_new, Y_, test_size = 0.3)


from sklearn.svm import SVR
reg=SVR(kernel='rbf',degree=16,C=6)
reg.fit(x_train,y_train)
y_pre=reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_true=y_test,y_pred=y_pre)



