import streamlit as st
import pandas as pd
from matplotlib import image
import plotly.express as px
import altair as alt
import os

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")

IMAGE_PATH = os.path.join(dir_of_interest, "images", "medical_insurance.jpg")
DATA_PATH = os.path.join(dir_of_interest, "data", "insurance.csv")
st.title("Dashboard - Insurance Premium Prediction")

img = image.imread(IMAGE_PATH)
st.image(img)

st.write("Lets look into the data")
#st.write("\n")
insurance_data=pd.read_csv(DATA_PATH)
st.dataframe(insurance_data)

st.write("The Dataset consist of",insurance_data.shape[0]," rows and",insurance_data.shape[1],"columns")

st.write("No of duplicate record in the dataset",insurance_data.duplicated().sum())

insurance_data=insurance_data.drop_duplicates()
st.write("The Dataset consist of",insurance_data.shape[0]," rows and",insurance_data.shape[1],"columns after removing duplicated record")

st.table(insurance_data.head())
#st.text("Age VS Charges")
#st.line_chart(insurance_data,x="age",y="charges")

st.text("Age VS Charges")
st.area_chart(insurance_data,x="age",y="charges")

st.text("insurance charges based upon Gender")
st.bar_chart(insurance_data,x="sex",y="charges")

st.text("insurance charges based upon region")
st.bar_chart(insurance_data,x="region",y="charges")



numerical_column_data=['age', 'bmi', 'children', 'charges']
color_data=["aliceblue","papayawhip", "peachpuff","pink"]
for i,j in zip(numerical_column_data,color_data):
    fig = px.histogram(insurance_data, x=i,title=f"{i}")
    st.plotly_chart(fig)

#if st.button("hey there. good morning bro")

