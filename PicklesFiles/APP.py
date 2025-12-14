import streamlit as st
import pandas as pd
import joblib 

model = joblib.load('LogisticRegression_Model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart Disease Prediction App By Tanmay")
st.markdown("provide the following details")

age = st.slider("Age",18,100,40)
se = st.selectbox("SEX",['M','F'])

chest_pain = st.selectbox("Chest Pain Type",['typical angina','atypical angina','non-anginal pain','asymptomatic'])
resting_bp = st.number_input("Resting Blood Pressure",80,200,120)
cholesterol = st.number_input("Cholesterol(mg/dL)",100,600,200)
Fasting_bs = st.selectbox("fasting blood sugar > 120 mg/dL",[0,1])
RestingECG = st.selectbox("Resting ECG",['Normal','ST','LVH'])
Max_hr = st.slider("Maximum Hear Rate",60,220,150)
Exercise_angina = st.selectbox("Exercise-Induced Angina",['Y','N'])
oldpeak = st.slider("Oldpeak",0.0,6.0,1.0)
ST_slope = st.selectbox("ST Slope",['Up','Flat','Down'])

if st.button("Predict"):
    raw_data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': Fasting_bs,
        'MaxHR': Max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + se: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + RestingECG: 1,
        'ExerciseAngina_' + Exercise_angina: 1,
        'ST_Slope_' + ST_slope: 1
    }
    input_df = pd.DataFrame([raw_data])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
        
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("The person is likely to have Heart Disease.")
    else:
        st.success("The person is unlikely to have Heart Disease.")