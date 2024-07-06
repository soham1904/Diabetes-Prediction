import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


for dirname, _, filenames in os.walk('diabetes.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('diabetes.csv')
df.columns
df.info()
df.head()
df.tail()
df.describe()
df.isnull().sum()
df.duplicated().sum()
sns.displot(df['Age'])
df['Outcome'].value_counts()
pd.crosstab(df.Age, df.Outcome).plot(kind="bar", figsize=(20, 6), color=['yellow', 'blue'])
plt.title('Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
df
X = df.drop('Outcome', axis=1)
y = df['Outcome']
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_pred
y_test
print('LogisticRegression Model Accuracy Score: {0:0.2f}'.format(accuracy_score(y_test, lr_pred) * 100) + "%")
input_data_reshaped = np.array([[4, 122, 52, 29, 175, 23.9, 0.489, 52]]).reshape(1, -1)
disease_prediction = lr_model.predict(input_data_reshaped)
print(disease_prediction)
if disease_prediction[0] == 0:
    print('The person is NOT DIABETIC')
else:
    print('The person is DIABETIC')
