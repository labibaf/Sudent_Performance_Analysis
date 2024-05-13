import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import data_preprocessing, encoder_Application_mode, encoder_Debtor, encoder_Gender, encoder_Scholarship_holder, encoder_Tuition_fees_up_to_date
from prediction import prediction

st.set_page_config(layout="wide")

col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png", width=130)
with col2:
    st.header('Student Performance Analysis')

data =  pd.DataFrame()

col1,col2,col3,col4,col5 = st.columns(5)

with col1:
    Application_mode=st.selectbox('Application mode', encoder_Application_mode.classes_, index=1)
    data['Application_mode'] = [Application_mode]

with col2:
    Debtor=st.selectbox('Debtor', encoder_Debtor.classes_, index=1)
    data['Debtor'] = [Debtor]

with col3:
    Tuition_fees_up_to_date=st.selectbox('Tuition fees up to date', encoder_Tuition_fees_up_to_date.classes_, index=1)
    data['Tuition_fees_up_to_date'] = [Tuition_fees_up_to_date]

with col4:
    Gender=st.selectbox('Gender', encoder_Gender.classes_, index=1)
    data['Gender'] = [Gender]

with col5:
    Scholarship_holder=st.selectbox('Scholarship holder', encoder_Scholarship_holder.classes_, index=1)
    data['Scholarship_holder'] = [Scholarship_holder]

col1,col2,col3,col4,col5 = st.columns(5)

with col1:
    Age_at_enrollment=st.number_input('Age at enrollment', min_value=15, max_value=70, value=15)
    data['Age_at_enrollment'] = [Age_at_enrollment]

with col2:
    Curricular_units_1st_sem_approved=st.number_input('Curricular units 1st sem approved', min_value=0, max_value=60, value=0)
    data['Curricular_units_1st_sem_approved'] = [Curricular_units_1st_sem_approved]

with col3:
    Curricular_units_1st_sem_grade=st.number_input('Curricular units 1st sem grade', min_value=0, max_value=60, value=0)
    data['Curricular_units_1st_sem_grade'] = [Curricular_units_1st_sem_grade]

with col4:
    Curricular_units_2nd_sem_approved=st.number_input('Curricular units 2nd sem approved', min_value=0, max_value=60, value=0)
    data['Curricular_units_2nd_sem_approved'] = [Curricular_units_2nd_sem_approved]

with col5:
    Curricular_units_2nd_sem_grade=st.number_input('Curricular units 2nd sem grade', min_value=0, max_value=60, value=0)
    data['Curricular_units_2nd_sem_grade'] = [Curricular_units_2nd_sem_grade]

if st.button('Predict'):
    new_data = data_preprocessing(data=data)
    st.write("Credit Scoring: {}".format(prediction(new_data)))

