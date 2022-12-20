import numpy as np
import pandas as pd
import streamlit as st
import pickle

html_temp="""
<div style="background-color:lightblue;padding:16px">
<h2 style="color:back;text-algin:center;">Heart Failure Prediction Using ML</h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

Model=pickle.load(open('RFCModel.pkl','rb'))
df=pickle.load(open('Dataset.pkl','rb'))

# Age
p1=st.slider('Age',1,100)
#Gender
p2=st.selectbox('Gender',['Male','Female'])

#chest pain
p3=st.selectbox('Cheast Pain',df['ChestPainType'].unique())

#Blood Pressure
p4=st.number_input('Blood Pressure')

#Cholesterol
p5=st.number_input('Cholesterol')

#Blood Sugar
p6=st.selectbox('Boold Sugar Level',[0,1])

#electrocardiography
p7=st.selectbox('electrocardiography',df['RestingECG'].unique())

#Heart Rate
p8=st.number_input('Heart Rate (Numeric value between 60 and 202)')

#Exercise Angina
p9=st.selectbox('Person is active in sports and exercise or not',df['ExerciseAngina'].unique())

#Oldpeak
p10=st.number_input('Oldpeak')

#heart rate slope
p11=st.selectbox('heart rate slope',df['ST_Slope'].unique())

if p2=='Male':
    p2='M'
else:
    p2='F'

if st.button('predict'):
    query=np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11],dtype=object).reshape(1,11)
    if(Model.predict(query)==1):
        st.title('Heart Disease')
    else:
        st.title('Normal')