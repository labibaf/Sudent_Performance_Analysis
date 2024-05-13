import pandas as pd
import numpy as np
import joblib

encoder_Application_mode = joblib.load("model/encoder_Application_mode.joblib")
encoder_Debtor = joblib.load("model/encoder_Debtor.joblib")
encoder_Gender = joblib.load("model/encoder_Gender.joblib")
encoder_Scholarship_holder = joblib.load("model/encoder_Scholarship_holder.joblib")
encoder_Tuition_fees_up_to_date = joblib.load("model/encoder_Tuition_fees_up_to_date.joblib")
scaler_Age_at_enrollment = joblib.load("model/scaler_Age_at_enrollment.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("model/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("model/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("model/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("model/scaler_Curricular_units_2nd_sem_grade.joblib")

def data_preprocessing(data):

    data=data.copy()
    df=pd.DataFrame()

    df['Application_mode'] = encoder_Application_mode.transform(data['Application_mode'])
    df['Debtor'] = encoder_Debtor.transform(data['Debtor'])
    df['Tuition_fees_up_to_date'] = encoder_Tuition_fees_up_to_date.transform(data['Tuition_fees_up_to_date'])
    df['Gender'] = encoder_Gender.transform(data['Gender'])
    df['Scholarship_holder'] = encoder_Scholarship_holder.transform(data['Scholarship_holder'])
    
    df['Age_at_enrollment'] = scaler_Age_at_enrollment.transform(np.asarray(data['Age_at_enrollment']).reshape(-1,1))
    df['Curricular_units_1st_sem_approved'] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(data['Curricular_units_1st_sem_approved']).reshape(-1,1))
    df['Curricular_units_1st_sem_grade'] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(data['Curricular_units_1st_sem_grade']).reshape(-1,1))
    df['Curricular_units_2nd_sem_approved'] = scaler_Curricular_units_2nd_sem_approved.transform(np.asarray(data['Curricular_units_2nd_sem_approved']).reshape(-1,1))
    df['Curricular_units_2nd_sem_grade'] = scaler_Curricular_units_2nd_sem_grade.transform(np.asarray(data['Curricular_units_2nd_sem_grade']).reshape(-1,1))

 
    return df