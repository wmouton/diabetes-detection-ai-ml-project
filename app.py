# This AI & ML project is a diabetes detection program that can detect if someone has diabetes

# Library Imports
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# Title And Subtitle
st.write("""
# AI & ML Diabetes Detection
Detect If Someone Has Diabetes Using This Artificial Intelligence Web Application. - by WMouton
""")

# Image To Display
image = Image.open('./img/image.jpg')
st.image(image, caption='ARTIFICIAL INTELLIGENCE & MACHINE LEARNING DIABETES DETECTION', use_column_width=True)

# Load Data
df = pd.read_csv('./data/diabetes.csv')
# Subheader
st.subheader('Data Information:')
# Data Table
st.dataframe(df)
# Data Statistics
st.write(df.describe())
# Data Chart
chart = st.bar_chart(df)

# Split Data - Independent 'X' and Dependent 'Y' Variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

# Split data - 75% Training And 25% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# Get Feature Input From User
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.0)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    # Store Dictionary Into Variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'BMI': bmi,
                 'DPF': diabetes_pedigree_function,
                 'age': age
                 }

    # Transform Data Into Data Frame
    features = pd.DataFrame(user_data, index=[0])


# Store User Input In Variable
user_input = get_user_input()
# get_user_input()


# Set Subheader And Display User Input
st.subheader('User Input: Unavailable')
st.subheader('USER INPUT RESULTS WILL BE AVAILABLE SOON')

st.write(user_input)

# Create And Train Model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show Model Metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(metrics.accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Store Models Predictions In Variable
# prediction = RandomForestClassifier.predict([[user_input]])
prediction = RandomForestClassifier.predict(X)

# Set Subheader - Display Classification
st.subheader('Classification: ')
st.write(prediction)
