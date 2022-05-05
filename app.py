# This AI & ML project is a diabetes detection program that can detect if someone has diabetes

# Library Imports
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# Title And Subtitle
st.write("""
# DIABETES DETECTION
### Artificial Intelligence & Machine Learning
Detect If Someone Has Diabetes Using This Artificial Intelligence Web Application.\n- by WMouton
""")

# Image To Display
image = Image.open('./img/image.jpg')
st.image(image, caption='ARTIFICIAL INTELLIGENCE DIABETES DETECTION', use_column_width=True)

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
    return features


# Store User Input In Variable
user_input = get_user_input()


# Set Subheader And Display User Input
st.subheader('User Input: ')
st.write(user_input)

# Create And Train Model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show Model Metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Store Models Predictions In Variable
prediction = RandomForestClassifier.predict(user_input)

# Set Subheader - Display Classification
st.subheader('Classification: ')
st.write('##### 0 = no diabetes')
st.write('##### 1 = diabetes')
st.write('###### YOU CAN SEE YOUR RESULTS IN THE BOX BELOW')
st.write(prediction)

st.write('###### Artificial Intelligence Web Application.')
st.write('###### - created by WMouton')