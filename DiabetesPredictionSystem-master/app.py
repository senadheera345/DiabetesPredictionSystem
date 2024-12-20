import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# FUNCTION
def user_report():
    age = st.sidebar.slider('Age', 3, 88, 33)
    bmi = st.sidebar.slider('BMI', 0.0, 67.0, 20.0)
    gender = st.sidebar.radio('Gender', ('Male', 'Female'))
    hypertension = st.sidebar.radio('Hypertension', ('Yes', 'No'))
    heart_disease = st.sidebar.radio('Heart Disease', ('Yes', 'No'))
    smoking_history = st.sidebar.radio('Smoking History', ('Never', 'Current', 'Former'))
    hba1c_level = st.sidebar.slider('HbA1c Level', 0.0, 10.0, 5.0)
    blood_glucose_level = st.sidebar.slider('Blood Glucose Level', 0, 300, 100)

    user_report_data = {
        'age': age,
        'bmi': bmi,
        'gender': gender,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'HbA1c_level': hba1c_level,
        'blood_glucose_level': blood_glucose_level,
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Load your dataset
dataset = pd.read_csv('diabetes_prediction_dataset.csv')


# Preprocessing
dataset = dataset.drop_duplicates()
dataset.drop(dataset[dataset['smoking_history'] == 'No Info'].index, inplace=True)
dataset = dataset[dataset['gender'] != 'Other']
dataset.reset_index(drop=True, inplace=True)

# Define a mapping for Gender
gender_mapping = {'Male': 0, 'Female': 1}
# Use the map function to convert Gender column
dataset['gender'] = dataset['gender'].map(gender_mapping)

# Define a mapping for smoking history
smoking_history_mapping = {'never': 0, 'former': 1, 'current': 2, 'not current': 3, "ever": 4}
# Use the map function to convert Gender column
dataset['smoking_history'] = dataset['smoking_history'].map(smoking_history_mapping)


# Horizontal line function
def horizontal_line():
    st.markdown('<hr style="border-top: 4px solid #ff4b4b;">', unsafe_allow_html=True)
# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
horizontal_line()
st.subheader('Training Data Stats')
st.write(dataset.describe())

# X AND Y DATA
X = dataset.drop('diabetes', axis='columns')
y = dataset['diabetes']

# Convert categorical variables to numerical using one-hot encoding
X_encoded = pd.get_dummies(X, columns=['gender', 'hypertension', 'heart_disease', 'smoking_history'], drop_first=True)

# Ensure that the columns in the user report data match those used during model training
user_data = user_report()
user_data = user_data.reindex(columns=X_encoded.columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=5)

# PATIENT DATA
horizontal_line()
st.subheader('Patient Data')
st.write(user_data)

# MODEL SELECTION
model_selection = st.sidebar.selectbox('Choose a model', ('Random Forest', 'SVM', 'Decision Tree', 'Logistic Regression'))

if model_selection == 'Random Forest':
    model = RandomForestClassifier()
elif model_selection == 'SVM':
    from sklearn.svm import SVC
    model = SVC()
elif model_selection == 'Decision Tree':
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
elif model_selection == 'Logistic Regression':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
else:
    st.error('Please select a valid model.')

# MODEL TRAINING
model.fit(X_train, y_train)
user_result = model.predict(user_data)

# VISUALISATIONS
horizontal_line()
st.subheader('Visualised Patient Report')

# Color Function
if user_result[0] == 0:
    color = 'blue'
else:
    color = 'red'

# Add more visualizations based on your dataset features here.

# OUTPUT
st.write('Your Result: ')
output = 'Diabetes Detected' if user_result[0] == 1 else 'No Diabetes Detected'
st.markdown(f'<p style="color: #ff4b4b; font-size: 24px;">{output}</p>', unsafe_allow_html=True)
accuracy_percentage = str(accuracy_score(y_test, model.predict(X_test)) * 100)
st.write('Accuracy :   ', f'<span style="color: white;">{accuracy_percentage}%</span>', unsafe_allow_html=True)


horizontal_line()
st.subheader('Your Report: ')
# CONFUSION MATRIX
st.write('Confusion Matrix:')
cm = confusion_matrix(y_test, model.predict(X_test))

# Heatmap
plt.figure(figsize=(10, 5))
sn.heatmap(cm, annot=True, fmt=".2f")
plt.xlabel('Predicted')
plt.ylabel('Truth')
st.pyplot(plt)

# Bar Graph
st.write('Confusion Matrix - Bar Graph:')
fig, ax = plt.subplots()
width = 0.35

rects1 = ax.bar(np.arange(2) - width/2, cm[0], width, label='Not Diabetic')
rects2 = ax.bar(np.arange(2) + width/2, cm[1], width, label='Diabetic')

ax.set_ylabel('Count')
ax.set_title('Confusion Matrix - Bar Graph')
ax.set_xticks(np.arange(2))
ax.set_xticklabels(['True Negative', 'True Positive'])
ax.legend()

st.pyplot(fig)
