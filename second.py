#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



#Working with data
df = pd.read_csv("churn.csv")
sns.countplot(df['Churn'])

numRetained = df[df.Churn == 'No'].shape[0]
numChurned = df[df.Churn == 'Yes'].shape[0]

print(numRetained/(numChurned + numRetained)*100," % of customers stayed in the company")

print(numChurned/(numChurned + numRetained)*100," % of customers stayed in the company")

sns.countplot(x='gender', hue ='Churn', data = df)
sns.countplot(x='InternetService', hue = 'Churn', data = df)
numericFeatures = ['tenure', 'MonthlyCharges']
fig, axis = plt.subplots(2,1,figsize=(28,8))
df[df.Churn == 'No'][numericFeatures].hist(bins = 20, color='blue', alpha=0.5, ax=axis)
df[df.Churn == 'Yes'][numericFeatures].hist(bins = 20, color='orange', alpha=0.5, ax=axis)

cleanDF = df.drop('customerID', axis =1)

for column in cleanDF.columns:
    if cleanDF[column].dtype == np.number:
        continue
    cleanDF[column] = LabelEncoder().fit_transform(cleanDF[column])

x = cleanDF.drop('Churn', axis=1)
y = cleanDF['Churn']
x = StandardScaler().fit_transform(x)



#creating model and implementing it
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2, random_state =42)

model = LogisticRegression()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)
print(classification_report(ytest, predictions))
 