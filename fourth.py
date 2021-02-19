import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

df = pd.read_csv("bitcoin.csv")
df.drop(['Date'], 1, inplace=True)
predictionDays = 30
df['Prediction'] = df[['Price']].shift(-predictionDays)

x = np.array(df.drop(['Prediction'], 1))
x = x[:len(df)-predictionDays]


y = np.array(df['Prediction'])
y= y[:-predictionDays]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)
predictionDays_array = np.array(df.drop(['Prediction'], 1))[-predictionDays:]
svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma = 0.00001)
svr_rbf.fit(xtrain, ytrain)
svr_rbf_confidence = svr_rbf.score(xtest, ytest)
print('SRV_RBF accuracy: ', svr_rbf_confidence)

svm_prediction = svr_rbf.predict(xtest)
print(svm_prediction)
print()
svm_prediction = svr_rbf.predict(predictionDays_array)
print(df.tail(predictionDays))
