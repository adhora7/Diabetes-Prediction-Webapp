# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load trained model
loaded_model = pickle.load(open(r'C:\MachineLearningModel\trained_model.sav', 'rb'))

# 2️⃣ Define feature names (order must match training)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# 3️⃣ Input data
input_data = (5,166,72,19,175,25.8,0.587,51)

# 4️⃣ Convert to DataFrame
input_df = pd.DataFrame([input_data], columns=feature_names)

# 5️⃣ Hardcode scaler parameters (mean & std from training dataset)
# Replace these values with the actual training data mean & std
scaler = StandardScaler()
scaler.mean_ = np.array([3.8, 120.9, 69.1, 20.5, 79.8, 32.0, 0.47, 33.2])
scaler.scale_ = np.array([3.3, 31.9, 19.4, 15.9, 115.2, 7.9, 0.33, 11.8])

# 6️⃣ Scale input
std_data = (input_df - scaler.mean_) / scaler.scale_

# 7️⃣ Make prediction
prediction = loaded_model.predict(std_data)

# 8️⃣ Output result
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')