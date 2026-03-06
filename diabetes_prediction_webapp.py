import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st

@st.cache_resource
def train_model():
    df = pd.read_csv('diabetes.csv')
    X = df.drop(columns='Outcome', axis=1)
    Y = df['Outcome']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2)
    model = svm.SVC(kernel='linear')
    model.fit(X_train, Y_train)
    return model, scaler

def diabetes_prediction(input_data, model, scaler):
    arr = np.asarray(input_data, dtype=float).reshape(1, -1)
    std = scaler.transform(arr)
    prediction = model.predict(std)
    if prediction[0] == 0:
        return '✅ The person is NOT diabetic'
    else:
        return '⚠️ The person IS diabetic'

def main():
    st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
    st.title('🩺 Diabetes Prediction Web App')
    st.markdown("Enter patient details below:")

    model, scaler = train_model()

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        BloodPressure = st.text_input('Blood Pressure (mm Hg)')
        Insulin = st.text_input('Insulin Level')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    with col2:
        Glucose = st.text_input('Glucose Level')
        SkinThickness = st.text_input('Skin Thickness (mm)')
        BMI = st.text_input('BMI Value')
        Age = st.text_input('Age')

    if st.button('🔍 Get Diabetes Test Result', use_container_width=True):
        if all([Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age]):
            result = diabetes_prediction(
                [Pregnancies, Glucose, BloodPressure, SkinThickness,
                 Insulin, BMI, DiabetesPedigreeFunction, Age],
                model, scaler)
            st.success(result)
        else:
            st.warning('⚠️ Please fill in all fields.')

if __name__ == '__main__':
    main()

