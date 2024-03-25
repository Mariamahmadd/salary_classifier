import streamlit as st
import requests
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit_lottie as st_lottie
import joblib
import numpy as np
import PIL as Image
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title='Salary Classifier',
    page_icon=':gem:',
    initial_sidebar_state='collapsed'  # Collapsed sidebar
)

def load_lottie(url): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
model = joblib.load(open("salary_classifier_model", 'rb'))

def predict(age,education_num,gender,capital_gain,capital_loss,hours_per_week):

    features = np.array([age,education_num,gender,capital_gain,capital_loss,hours_per_week]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction
with st.sidebar:
    choose = option_menu(None, ["Home", "Graphs", "About", "Contact"],
                         icons=['house', 'kanban', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": '#E0E0EF', "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose=='Home':
       st.write('# Salary Classifier')
       st.write('---')
       st.subheader('Enter your details to classify your salary')
       # User input
       age = st.number_input("Enter the age: ",min_value=0)
       education_num = st.slider("Enter your education number (1-16):", min_value=1, max_value=16, value=8, step=1)
       gender = st.radio("Select the gender:", ('Male', 'Female'))
       gender_encoding = 1 if gender == 'Male' else 0
       capital_gain = st.number_input("Enter your capital gain value:", min_value=0)
       capital_loss = st.number_input("Enter your capital loss value:", min_value=0)
       hours_per_week = st.number_input("Enter your working hours per week: ",min_value=0)
       # Predict the cluster
       sample_prediction = predict(age,education_num,gender_encoding,capital_gain,capital_loss,hours_per_week)

       if st.button("Predict"):
        if sample_prediction == 0:
            st.warning("Predicted Salary: Low")
            st.write("This indicates a low salary.")
        elif sample_prediction == 1:
            st.success("Predicted Salary: High")
            st.write("This indicates a high salary.")
            st.balloons()
              
 


elif choose=='About':
    st.write('# About Page')
    st.write('---')
    st.write("🎯💡 Welcome to Salary Classification Deployment! We specialize in providing advanced salary classification solutions that help individuals understand their income better. Our data-driven approach combines analytics, machine learning, and financial expertise to create customized salary classification models tailored to your needs. By implementing salary classification, you can gain insights into your income level, plan your finances effectively, and make informed decisions about your career and lifestyle. ✨🚀 Partner with us to unlock the power of salary classification and take control of your financial future. Contact us today to learn more. 📞📧")
    st.image("5355919-removebg-preview.png")
 

elif choose == "Contact":
    st.write('# Contact Us')
    st.write('---')
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        st.write('## Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') 
        Email=st.text_input(label='Please Enter Email')
        Message=st.text_input(label='Please Enter Your Message') 
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')



elif choose == 'Graphs':
    st.write('# Salary Classifier Graphs')
    st.write('---')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("## Female Vs Male workers Graph:")
    st.image("output.png")
    st.write("## Race vs Income Graph:")
    st.image("output2.png")
    st.write("## Income Vs Gender Graph:")
    st.image("output3.png")
    st.write("## Age Period Graph")
    st.image("output4.png")
    st.write("## Age period Vs Gender Graph")
    st.image("output5.png")
    st.write("## Age Period Vs Income Graph")
    st.image("output6.png")
    st.write("## Workclass Vs Income Graph")
    st.image("output7.png")
    st.write("## Education Vs Income Graph")
    st.image("output8.png")
    st.write("## Occupation Vs Income Graph")
    st.image("output9.png")
    st.write("## Working Hours Period Graph")
    st.image("output10.png")
    st.write("## Age Period Vs Working Hours Period Graph")
    st.image("output11.png")
    
    data = pd.read_csv('adult.csv')
    # Create a DataFrame
    df = pd.DataFrame(data)
    

