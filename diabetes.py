import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Loading the dataset
df = pd.read_csv('diabetes.csv')

# Display dataset information and basic statistics
print(df.columns)
print(df.info())
print(df.head())
print(df.tail())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

# Age distribution plot using plotly
fig_age_dist = px.histogram(df, x='Age', title='Age Distribution')
fig_age_dist.show()

# Disease frequency for ages
age_outcome_ct = pd.crosstab(df.Age, df.Outcome).reset_index()
fig_age_outcome = px.bar(age_outcome_ct, x='Age', y=[0, 1], 
                         labels={'value': 'Frequency', 'variable': 'Outcome'}, 
                         title='Disease Frequency for Ages',
                         barmode='group')
fig_age_outcome.update_layout(
    xaxis_title='Age',
    yaxis_title='Frequency',
    legend_title_text='Outcome'
)
fig_age_outcome.show()

# Splitting the dataset
X = df.drop('Outcome', axis=1)
y = df['Outcome']
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Standardizing the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)

# Training the Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Making predictions
lr_pred = lr_model.predict(X_test)
print(lr_pred)
print(y_test)

# Evaluating the model
accuracy = accuracy_score(y_test, lr_pred) * 100
print(f'LogisticRegression Model Accuracy Score: {accuracy:.2f}%')

# Making a prediction for a single input
input_data_reshaped = np.array([[4, 122, 52, 29, 175, 23.9, 0.489, 52]]).reshape(1, -1)
disease_prediction = lr_model.predict(input_data_reshaped)
print(disease_prediction)
if disease_prediction[0] == 0:
    print('The person is NOT DIABETIC')
else:
    print('The person is DIABETIC')
