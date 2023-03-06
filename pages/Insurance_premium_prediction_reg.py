import streamlit as st
import pandas as pd
from matplotlib import image
import plotly.express as px
import altair as alt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

st.write("Lets look into the data")
#st.write("\n")
insurance_data=pd.read_csv(DATA_PATH)
insurance_data=insurance_data.drop_duplicates()

nav=st.sidebar.radio("Navigator",["About","Predict"])

if nav=="About":
    st.title("health Insurance premium Predictor")
    st.dataframe(insurance_data)
    st.text(" ")
    st.text(" ")
    st.image(img,width=600)
    st.text(" The Dataset has been taken from Kaggle")
    st.markdown("https://www.kaggle.com/datasets/simranjain17/insurance")
insurance_data.replace({"sex":{"male":0,"female":1}},inplace=True)
insurance_data.replace({"smoker":{"yes":0,"no":1}},inplace=True)
insurance_data.replace({"region":{"southeast":0,"southwest":1,"northeast":2,"northwest":3}},inplace=True)
x=insurance_data.drop(columns="charges",axis=1)
y=insurance_data["charges"]
rf=RandomForestRegressor()
rf.fit(x,y)

if nav=="Predict":
    st.title("Enter the details")
    st.text(" ")
    age=st.number_input("Age:",step=1,min_value=0)

    sex=st.radio("sex",["Male","Female"])
    if sex=="Male":
        s=0
    if sex=="Female":
        s=1

    bmi=st.number_input("BMI:",min_value=0)
    
    children=st.number_input("No of childerns:",step=1,min_value=0)
    
    smoke=st.radio("Do you smoke",("Yes","No"))
    if smoke=="Yes":
        sm=0
    if smoke=="No":
        sm=1

    region=st.selectbox("Region",("SouthEast","SouthWest","NorthEast","NorthWest"))
    if region=="SouthEast":
        reg=0
    if region=="SouthWest":
        reg=1
    if region=="NorthEast":
        reg=2
    if region=="NorthWest":
        reg=3
    if st.button("Predict"):
        st.subheader("Predicted Premium")
        st.text(rf.predict([[age,s,bmi,children,sm,reg]]))

    
