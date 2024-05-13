# %% [markdown]
# # Submission Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

# %% [markdown]
# - Nama: Labib Ammar Fadhali
# - Email: labibfadhali12@gmail.com
# - Id Dicoding: labibaf

# %% [markdown]
# ## Persiapan

# %% [markdown]
# ### Menyiapkan library yang dibutuhkan

# %%
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import joblib

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ### Menyiapkan data yang akan diguankan

# %%
df=pd.read_csv('https://raw.githubusercontent.com/labibaf/Sudent_Performance_Analysis/main/dataset/data.csv',delimiter=';')
df.shape

# %%
df.head()

# %% [markdown]
# ## Data Understanding

# %%
df.info()

# %%
df.describe(include='all')

# %%
df.duplicated().sum()

# %% [markdown]
# ## Data Preparation / Preprocessing

# %%
def decode_dataframe(df):
    df = df.copy() 
    marital_status_mapping = {
        1: 'single',
        2: 'married',
        3: 'widower',
        4: 'divorced',
        5: 'facto union',
        6: 'legally separated'
    }
    df['Marital_status']=df['Marital_status'].map(marital_status_mapping)

    application_mode_mapping = {
        1: '1st phase - general contingent',
        2: 'Ordinance No. 612/93',
        5: '1st phase - special contingent (Azores Island)',
        7: 'Holders of other higher courses',
        10: 'Ordinance No. 854-B/99',
        15: 'International student (bachelor)',
        16: '1st phase - special contingent (Madeira Island)',
        17: '2nd phase - general contingent',
        18: '3rd phase - general contingent',
        26: 'Ordinance No. 533-A/99, item b2) (Different Plan)',
        27: 'Ordinance No. 533-A/99, item b3 (Other Institution)',
        39: 'Over 23 years old',
        42: 'Transfer',
        43: 'Change of course',
        44: 'Technological specialization diploma holders',
        51: 'Change of institution/course',
        53: 'Short cycle diploma holders',
        57: 'Change of institution/course (International)'
    }
    df['Application_mode']=df['Application_mode'].map(application_mode_mapping)

    course_mapping = {
        33: 'Biofuel Production Technologies',
        171: 'Animation and Multimedia Design',
        8014: 'Social Service (evening attendance)',
        9003: 'Agronomy',
        9070: 'Communication Design',
        9085: 'Veterinary Nursing',
        9119: 'Informatics Engineering',
        9130: 'Equinculture',
        9147: 'Management',
        9238: 'Social Service',
        9254: 'Tourism',
        9500: 'Nursing',
        9556: 'Oral Hygiene',
        9670: 'Advertising and Marketing Management',
        9773: 'Journalism and Communication',
        9853: 'Basic Education',
        9991: 'Management (evening attendance)'
    }
    df['Course']=df['Course'].map(course_mapping)

    daytime_evening_mapping = {
        1: 'daytime',
        0: 'evening'
    }
    df['Daytime_evening_attendance']=df['Daytime_evening_attendance'].map(daytime_evening_mapping)

    previous_qualification_mapping = {
        1: 'Secondary education',
        2: "Higher education - bachelor's degree",
        3: 'Higher education - degree',
        4: 'Higher education - master\'s',
        5: 'Higher education - doctorate',
        6: 'Frequency of higher education',
        9: '12th year of schooling - not completed',
        10: '11th year of schooling - not completed',
        12: 'Other - 11th year of schooling',
        14: '10th year of schooling',
        15: '10th year of schooling - not completed',
        19: 'Basic education 3rd cycle (9th/10th/11th year) or equiv.',
        38: 'Basic education 2nd cycle (6th/7th/8th year) or equiv.',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        42: 'Professional higher technical course',
        43: 'Higher education - master (2nd cycle)'
    }
    df['Previous_qualification']=df['Previous_qualification'].map(previous_qualification_mapping)

    nationality_mapping = {
        1: 'Portuguese',
        2: 'German',
        6: 'Spanish',
        11: 'Italian',
        13: 'Dutch',
        14: 'English',
        17: 'Lithuanian',
        21: 'Angolan',
        22: 'Cape Verdean',
        24: 'Guinean',
        25: 'Mozambican',
        26: 'Santomean',
        32: 'Turkish',
        41: 'Brazilian',
        62: 'Romanian',
        100: 'Moldova (Republic of)',
        101: 'Mexican',
        103: 'Ukrainian',
        105: 'Russian',
        108: 'Cuban',
        109: 'Colombian'
    }
    df['Nacionality']=df['Nacionality'].map(nationality_mapping)

    qualification_mapping = {
        1: 'Secondary Education - 12th Year of Schooling or Eq.',
        2: 'Higher Education - Bachelor\'s Degree',
        3: 'Higher Education - Degree',
        4: 'Higher Education - Master\'s',
        5: 'Higher Education - Doctorate',
        6: 'Frequency of Higher Education',
        9: '12th Year of Schooling - Not Completed',
        10: '11th Year of Schooling - Not Completed',
        11: '7th Year (Old)',
        12: 'Other - 11th Year of Schooling',
        14: '10th Year of Schooling',
        18: 'General Commerce Course',
        19: 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
        22: 'Technical-Professional Course',
        26: '7th Year of Schooling',
        27: '2nd Cycle of the General High School Course',
        29: '9th Year of Schooling - Not Completed',
        30: '8th Year of Schooling',
        34: 'Unknown',
        35: 'Can\'t Read or Write',
        36: 'Can Read Without Having a 4th Year of Schooling',
        37: 'Basic Education 1st Cycle (4th/5th Year) or Equiv.',
        38: 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
        39: 'Technological Specialization Course',
        40: 'Higher Education - Degree (1st Cycle)',
        41: 'Specialized Higher Studies Course',
        42: 'Professional Higher Technical Course',
        43: 'Higher Education - Master (2nd Cycle)',
        44: 'Higher Education - Doctorate (3rd Cycle)'
    }
    df['Mothers_qualification']=df['Mothers_qualification'].map(qualification_mapping)
    df['Fathers_qualification']=df['Fathers_qualification'].map(qualification_mapping)

    occupation_mapping = {
        0: 'Student',
        1: 'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
        2: 'Specialists in Intellectual and Scientific Activities',
        3: 'Intermediate Level Technicians and Professions',
        4: 'Administrative staff',
        5: 'Personal Services, Security and Safety Workers and Sellers',
        6: 'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
        7: 'Skilled Workers in Industry, Construction and Craftsmen',
        8: 'Installation and Machine Operators and Assembly Workers',
        9: 'Unskilled Workers',
        10: 'Armed Forces Professions',
        90: 'Other Situation',
        99: '(blank)',
        101: 'Armed Forces Officers',
        102: 'Armed Forces Sergeants',
        103: 'Other Armed Forces personnel',
        112: 'Directors of administrative and commercial services',
        114: 'Hotel, catering, trade and other services directors',
        121: 'Specialists in the physical sciences, mathematics, engineering and related techniques',
        122: 'Health professionals',
        123: 'Teachers',
        124: 'Specialists in finance, accounting, administrative organization, public and commercial relations',
        131: 'Intermediate level science and engineering technicians and professions',
        132: 'Technicians and professionals, of intermediate level of health',
        134: 'Intermediate level technicians from legal, social, sports, cultural and similar services',
        135: 'Information and communication technology technicians',
        141: 'Office workers, secretaries in general and data processing operators',
        143: 'Data, accounting, statistical, financial services and registry-related operators',
        144: 'Other administrative support staff',
        151: 'Personal service workers',
        152: 'Sellers',
        153: 'Personal care workers and the like',
        154: 'Protection and security services personnel',
        161: 'Market-oriented farmers and skilled agricultural and animal production workers',
        163: 'Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence',
        171: 'Skilled construction workers and the like, except electricians',
        172: 'Skilled workers in metallurgy, metalworking and similar',
        174: 'Skilled workers in electricity and electronics',
        175: 'Workers in food processing, woodworking, clothing and other industries and crafts',
        181: 'Fixed plant and machine operators',
        182: 'Assembly workers',
        183: 'Vehicle drivers and mobile equipment operators',
        192: 'Unskilled workers in agriculture, animal production, fisheries and forestry',
        193: 'Unskilled workers in extractive industry, construction, manufacturing and transport',
        194: 'Meal preparation assistants',
        195: 'Street vendors (except food) and street service providers'
    }
    df['Mothers_occupation']=df['Mothers_occupation'].map(occupation_mapping)
    df['Fathers_occupation']=df['Fathers_occupation'].map(occupation_mapping)

    yes_no_mapping = {
        1: 'yes',
        0: 'no'
    }
    df['Displaced']=df['Displaced'].map(yes_no_mapping)
    df['Educational_special_needs']=df['Educational_special_needs'].map(yes_no_mapping)
    df['Debtor']=df['Debtor'].map(yes_no_mapping)
    df['Tuition_fees_up_to_date']=df['Tuition_fees_up_to_date'].map(yes_no_mapping)
    df['Scholarship_holder']=df['Scholarship_holder'].map(yes_no_mapping)
    df['International']=df['International'].map(yes_no_mapping)
    
    gender_mapping = {
        1: 'male',
        0: 'female'
    } 
    df['Gender']=df['Gender'].map(gender_mapping)

    return df

# %%
decoded_df=decode_dataframe(df)
decoded_df.head()

# %%
df['Status'].value_counts()

# %%
df=df.drop(df[df['Status']=='Enrolled'].index)
df['Status'].value_counts()

# %%
df['Status']=df['Status'].map({'Graduate':0,'Dropout':1})
df.sample(3)

# %% [markdown]
# Export Decoded Data untuk membuat dashboard

# %%
# decoded_df.to_csv('./dataset/decoded_data.csv',index=False)

# %% [markdown]
# ## EDA

# %%
status_counts = decoded_df['Status'].value_counts().reset_index()
status_counts.columns = ['Status', 'Count']
fig = px.pie(status_counts, values='Count', 
             names='Status',
             title='Distribution of Status',
             color_discrete_sequence=px.colors.qualitative.Set3)
fig.show()

# %% [markdown]
# Berdasarkan Pie Chart diatas, terlihat bahwa presentasi siswa yang dropout cukup besar di angka 32,1%, sedangkan siswa yang lulus sebesar 49,9% dan siswa yang masih terdaftar sebesar 17,9%. 

# %%
df_corr=df.corr(method='pearson')
fig = px.imshow(df_corr)
fig.update_layout(width=1000, height=1000) 
fig.show()

# %%
df.corr()['Status']

# %%
status_corr = df.corrwith(df['Status'])
high_corr_columns = status_corr[abs(status_corr) >= 0.2]

print("Kolom dengan korelasi tinggi terhadap 'Status':")
print(high_corr_columns)

# %% [markdown]
# Berdasarkan analisis korelasi, Faktor-faktor yang memiliki dampak signifikan terhadap status mahasiswa ('Status') adalah jumlah unit kurikuler yang disetujui pada dua semester pertama serta status pembayaran uang sekolah. Mahasiswa dengan catatan jumlah unit kurikuler yang minim dan tunggakan pembayaran uang sekolah cenderung lebih mungkin untuk mengalami drop out.

# %%
decoded_df.groupby('Status').agg({
    'Curricular_units_1st_sem_approved': 'mean',
    'Curricular_units_2nd_sem_approved': 'mean'
})

# %%
decoded_df.groupby('Status').agg({
    'Curricular_units_1st_sem_grade': 'mean',
    'Curricular_units_2nd_sem_grade': 'mean'
})

# %%
def categorical_plot(features, df, segment_feature=None):
    num_plot = len(features)
    fig = None
    for i, feature in enumerate(features):
        if segment_feature:
            fig = px.histogram(df, x=segment_feature, color=feature, barmode='group', category_orders={feature: df[feature].unique()})
        else:
            fig = px.histogram(df, x=feature, barmode='group', category_orders={feature: df[feature].unique()})
        
        fig.update_layout(title_text=f'Distribution of {feature}')
        fig.show()

# %%
categorical_columns = ['Debtor','Tuition_fees_up_to_date','Gender','Scholarship_holder']
categorical_plot(
   features=categorical_columns,
   df=decoded_df,
   segment_feature='Status'
)

# %% [markdown]
# ### Data Preprocessing

# %% [markdown]
# Memisahkan fitur yang digunakan

# %%
used_cols = [
    'Application_mode', 'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'Age_at_enrollment', 'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade', 'Status'
]

clean_df = decoded_df[used_cols]
clean_df = clean_df[clean_df['Status'] != 'Enrolled']

# %%
def scaling(features,df):
    df = df.copy()
    for feature in features:
        scaler = MinMaxScaler()
        X = np.asanyarray(df[[feature]])
        X = X.reshape(-1,1)
        scaler.fit(X)
        df['{}'.format(feature)] = scaler.transform(X)
        joblib.dump(scaler, './model/scaler_{}.joblib'.format(feature))
    return df

def encoding(features,df):
    df = df.copy()
    for feature in features:
        encoder = LabelEncoder()
        df['{}'.format(feature)] = encoder.fit_transform(df[feature])
        joblib.dump(encoder, './model/encoder_{}.joblib'.format(feature))
    return df

# %%
numerical_columns = clean_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = clean_df.select_dtypes(include=['object']).drop(columns=['Status']).columns.tolist()

# %%
new_clean_df=scaling(numerical_columns,clean_df)
new_clean_df=encoding(categorical_columns,new_clean_df)

# %%
sns.countplot(x='Status',data=new_clean_df)

# %%
X = new_clean_df.drop(columns=['Status'])
y = new_clean_df['Status']

# %% [markdown]
# Penanganan imbalance data menggunakan metode oversampling

# %%
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

# %%
encoder = LabelEncoder()
encoder.fit(y_train)
new_y_train = encoder.transform(y_train)
joblib.dump(encoder, './model/status_encoder.joblib')

new_y_test = encoder.transform(y_test)

# %% [markdown]
# ## Modeling

# %% [markdown]
# Decission Tree

# %%
tree_model = DecisionTreeClassifier(random_state=123)
param_grid = { 
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [5, 6, 7, 8],
    'criterion' :['gini', 'entropy']
}
 
CV_tree = GridSearchCV(estimator=tree_model, param_grid=param_grid, cv=5, n_jobs=-1)
CV_tree.fit(X_train, new_y_train)

# %%
print("best parameters: ", CV_tree.best_params_)

# %%
tree_model = DecisionTreeClassifier(
    random_state=123,
    criterion='entropy', 
    max_depth=7, 
    max_features='sqrt'
)
 
tree_model.fit(X_train, new_y_train)
joblib.dump(tree_model, "model/tree_model.joblib")

# %% [markdown]
# Random Forest

# %%
rdf_model = RandomForestClassifier(random_state=123)
 
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [6, 7, 8],
    'criterion' :['gini', 'entropy']
}
 
CV_rdf = GridSearchCV(estimator=rdf_model, param_grid=param_grid, cv=5, n_jobs=-1)
CV_rdf.fit(X_train, new_y_train)

# %%
print("best parameters: ", CV_rdf.best_params_)

# %%
rdf_model = RandomForestClassifier(
    random_state=123, 
    max_depth=8, 
    n_estimators=500, 
    max_features='log2', 
    criterion='gini', 
    n_jobs=-1
)
rdf_model.fit(X_train, new_y_train)
joblib.dump(rdf_model, "model/rdf_model.joblib")

# %% [markdown]
# Gradien Boosting

# %%
gboost_model = GradientBoostingClassifier(random_state=123)
 
param_grid = {
    'max_depth': [5, 8],
    'n_estimators': [200, 300],
    'learning_rate': [0.01, 0.1],
    'max_features': ['auto', 'sqrt', 'log2']
}
 
CV_gboost = GridSearchCV(estimator=gboost_model, param_grid=param_grid, cv=5, n_jobs=-1)
CV_gboost.fit(X_train, new_y_train)

# %%
print("best parameters: ", CV_gboost.best_params_)

# %%
gboost_model = GradientBoostingClassifier(
    random_state=123,
    learning_rate=0.1, 
    max_depth=8, 
    max_features='sqrt',
    n_estimators=300
)
gboost_model.fit(X_train, new_y_train)
joblib.dump(gboost_model, "model/gboost_model.joblib")

# %% [markdown]
# ## Evaluation

# %%
models = {
    "Decision Tree": tree_model,
    "Random Forest": rdf_model,
    "Gradient Boosting": gboost_model
}

best_accuracy = 0
best_model = None

for name, model in models.items():

    y_pred = model.predict(X_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_test)
    y_test_numeric = label_encoder.transform(y_test)

    accuracy = accuracy_score(y_test_numeric, y_pred)
    print(f"Akurasi model {name}: {accuracy}")

    print(f"Classification Report model {name}:")
    print(classification_report(y_test_numeric, y_pred))

    cm = confusion_matrix(y_test_numeric, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f"Best model: {best_model}, dengan accuracy: {best_accuracy}")


# %%
file_names = os.listdir('model')

# Cetak nama file
for file_name in file_names:
    print(file_name)


