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
# AI & ML Diabetes Detection Web Application
Detect If Someone Has Diabetes Using This Web Appliaction.
""")

# Image To Display
image = Image.open('./img/image.jpg')
st.image(image, caption='ARTIFICIAL INTELLIGENCE & MACHINE LEARNING', use_column_width=True)

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