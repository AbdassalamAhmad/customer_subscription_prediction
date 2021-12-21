# -*- coding: utf-8 -*-

#importing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import xgboost as xgb
import streamlit as st 

model_file = 'model_1.bin'

with open (model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


def welcome():
    return "Welcome All"


def predict_customer_subscription(age, job, martial, education, 
default, balance_logs, housing, loan, contact, day, month, duration, 
campaign, pdays, previous, poutcome):
    customer = {}
    customer = {   
    "age" : age,
    "job" : job,
    "martial" : martial,
    "education" : education,
    "default" : default,
    "balance_logs" : balance_logs,#np.log1p()
    "housing" : housing,
    "loan" : loan,
    "contact" : contact,
    "day" : day,
    "month" : month,
    "duration" : duration,
    "campaign" : campaign,
    "pdays" : pdays,
    "previous" : previous,
    "poutcome" : poutcome,
        }
    X=dv.transform([customer])

    x=xgb.DMatrix(X, label=([0]), feature_names=dv.get_feature_names())
    pred=model.predict(x)
    result = float(pred[0])
    if result >= 0.51:
        return 'This customer will subscribe'
    else:
        return 'This customer will NOT subscribe'




def main():
    st.title("Customer Subscription Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Customer Subscription Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    df = pd.read_csv('bank-full.csv',sep=';')
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    
    st.text("This is the test dataset to confirm some results")
    st.dataframe(df_test)  # Same as st.write(df)

    age = st.number_input("Age")
    job = st.selectbox(
    'Type of yout job',
    ("admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"))
    st.write('You selected:', job)
    martial = st.selectbox(
    'Marital status',
    ("married","divorced","single"))
    st.write('You selected:', martial)    
    education = st.selectbox(
    'Highest education degree you got',
    ("unknown","secondary","primary","tertiary"))
    st.write('You selected:', education)
    default = st.selectbox(
    'Do you have credit in default?',
    ("yes","no"))
    st.write('You selected:', default)
    balance_logs = np.log1p(st.number_input("balance_logs"))
    housing = st.selectbox(
    'Do you have housing loan?',
    ("yes","no"))
    st.write('You selected:', housing)
    loan = st.selectbox(
    'Do you have personal loan?',
    ("yes","no"))
    st.write('You selected:', loan)
    contact = st.selectbox(
    'Prefered contact communication type',
    ("unknown","telephone","cellular"))
    st.write('You selected:', contact)
    day = st.number_input("day")
    month = st.selectbox(
    'last month you got contacted of the year',
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))
    st.write('You selected:', month)
    duration = st.number_input("duration")
    campaign = st.number_input("campaign")
    pdays = st.number_input("pdays")
    previous = st.number_input("previous")
    poutcome = st.selectbox(
    'outcome of the previous marketing campaign',
    ("unknown","other","failure","success"))
    st.write('You selected:', poutcome)
    result=""
    if st.button("Predict"):
        result=""
        result=predict_customer_subscription(age, job, martial, education, default,
                                            balance_logs, housing, loan, contact, day, month, duration, campaign,
                                            pdays, previous, poutcome)
    #st.success('The output is {}'.format(result))
    st.success(result)


if __name__=='__main__':
    main()