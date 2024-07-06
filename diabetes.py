import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Streamlit configuration
st.title('Diabetes Prediction')
st.write('Exploring the dataset and building a Logistic Regression model to predict diabetes.')

# Loading the dataset
df = pd.read_csv('diabetes.csv')

# Display dataset information and basic statistics
st.write('### Dataset Information')
st.write(df.info())
st.write('### Head of Dataset')
st.write(df.head())
st.write('### Tail of Dataset')
st.write(df.tail())
st.write('### Dataset Description')
st.write(df.describe())
st.write('### Null Values in Dataset')
st.write(df.isnull().sum())
st.write('### Duplicated Values in Dataset')
st.write(df.duplicated().sum())

# Age distribution plot using seaborn
st.write('### Age Distribution')
fig_age_dist = sns.displot(df['Age'], kde=True)
st.pyplot(fig_age_dist)

# Disease frequency for ages
st.write('### Disease Frequency for Ages')
fig, ax = plt.subplots(figsize=(20, 6))
pd.crosstab(df.Age, df.Outcome).plot(kind="bar", ax=ax, color=['yellow', 'blue'])
plt.title('Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
st.pyplot(fig)

# Splitting the dataset
X = df.drop('Outcome', axis=1)
y = df['Outcome']
st.write('### Features and Target Shapes')
st.write(f'X shape: {X.shape}')
st.write(f'y shape: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)
st.write(f'X_train shape: {X_train.shape}')
st.write(f'X_test shape: {X_test.shape}')
st.write(f'y_train shape: {y_train.shape}')
st.write(f'y_test shape: {y_test.shape}')

# Standardizing the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
st.write('### Standardized Training Features')
st.write(X_train)

# Training the Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Making predictions
lr_pred = lr_model.predict(X_test)
st.write('### Predictions on Test Data')
st.write(lr_pred)
st.write(y_test)

# Evaluating the model
accuracy = accuracy_score(y_test, lr_pred) * 100
st.write(f'LogisticRegression Model Accuracy Score: {accuracy:.2f}%')

# Making a prediction for a single input
input_data_reshaped = np.array([[4, 122, 52, 29, 175, 23.9, 0.489, 52]]).reshape(1, -1)
disease_prediction = lr_model.predict(input_data_reshaped)
st.write('### Single Input Prediction')
st.write(disease_prediction)
if disease_prediction[0] == 0:
    st.write('The person is NOT DIABETIC')
else:
    st.write('The person is DIABETIC')
