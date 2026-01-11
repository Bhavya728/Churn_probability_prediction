import streamlit as st
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

## Load the trained model
model = tf.keras.models.load_model('model.h5')

#load the encoder and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)
with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder_geography = pickle.load(f)

## stramlit app
st.title('Customer Churn Prediction')

## User input
geography=st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender=st.selectbox('Gender', le_gender.classes_)
age=st.number_input('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

## Prepare the input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[le_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

# One hot encode Geography
geo_encoded=onehot_encoder_geography.transform([[geography]])
geo_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

#combine input data with geo_df
input_data=pd.concat([input_data.reset_index(drop=True),geo_df],axis=1)

## Scale the input data
input_data_scaled = scaler.transform(input_data)

#Predict churn
prediction=model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write(f'The customer is likely to churn with a probability of {prediction_proba:.2f}')
else:
    st.write(f'The customer is unlikely to churn with a probability of {1 - prediction_proba:.2f}')
  