# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:51:34 2024

@author: SRI LAVANYA
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud

diabetes_model = pickle.load(open("C:/Users/SRI LAVANYA/Desktop/Multiple disease prediction system/saved models/diabetes_model.sav",'rb'))

heart_disease_model = pickle.load(open('C:/Users/SRI LAVANYA/Desktop/Multiple disease prediction system/saved models/heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('C:/Users/SRI LAVANYA/Desktop/Multiple disease prediction system/saved models/parkinsons_model.sav','rb'))

titanic_model = pickle.load(open('C:/Users/SRI LAVANYA/Desktop/Multiple disease prediction system/saved models/titanic_model.sav','rb'))


with st.sidebar:
    selected = option_menu('Multiple diseae prediction',
                           ['Home','Diabetes prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Titanic Survival Prediction',
                            'Data Visualiser'],
                           icons = ['house','activity','heart','person',
                                    'person-arms-up','bar-chart-line-fill'],
                           default_index = 0)
    
if 'input_values' not in st.session_state:
    st.session_state.input_values = {}   
    
if (selected == 'Home'):
    col1 , col2 , col3 = st.columns([1,2,1])
    
    col1.markdown(" # Welcome Health Assistant... ")
    col2.markdown(" # to my app  ")
    #col1.markdown(" Here is some info on the app ")
    
#Diabetes Prediction Page
if (selected == 'Diabetes prediction'):
    st.title('Diabetes Prediction using ML')
    
    col1,col2,col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies',st.session_state.input_values.get('Pregnancies', ''))
    with col2:
        Glucose = st.text_input('Glucose Level',st.session_state.input_values.get('Glucose', ''))
    with col3:
        BloodPressure = st.text_input('Blood Pressure value',st.session_state.input_values.get('BloodPressure', ''))
    with col1:
        SkinThickness = st.text_input('Skin Thickness value',st.session_state.input_values.get('SkinThickness', ''))
    with col2:
        Insulin = st.text_input('Insulin Level',st.session_state.input_values.get('Insulin', ''))
    with col3:
        BMI = st.text_input('BMI value',st.session_state.input_values.get('BMI', ''))
    with col1:
        DPF = st.text_input('Diabetes Pedigree Function',st.session_state.input_values.get('DPF', ''))
    with col2:
        Age = st.text_input('Age',st.session_state.input_values.get('Age', ''))
    
    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age]
        
        st.session_state.input_values['Pregnancies'] = Pregnancies
        st.session_state.input_values['Glucose'] = Glucose
        st.session_state.input_values['BloodPressure'] = BloodPressure
        st.session_state.input_values['SkinThickness'] = SkinThickness
        st.session_state.input_values['Insulin'] = Insulin
        st.session_state.input_values['BMI'] = BMI
        st.session_state.input_values['DPF'] = DPF
        st.session_state.input_values['Age'] = Age
    
        user_input = [float(x) for x in user_input]
        
        diab_prediction = diabetes_model.predict([user_input]) 
        
        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person is Diabetic'
        else:
            diab_diagnosis = 'The person is not Diabetic'
    st.success(diab_diagnosis)
            
if (selected == 'Heart Disease Prediction'):
    st.title('Heart Disease Prediction')
    
    col1,col2,col3 = st.columns(3)
    with col1:
        age1 = st.text_input('Age',st.session_state.input_values.get('age1', ''))
    with col2:
        sex1 = st.text_input('Sex',st.session_state.input_values.get('sex1', ''))
    with col3:
        cp = st.text_input('Chest pain types',st.session_state.input_values.get('cp', ''))
    with col1:
        trestbps = st.text_input('Resting Blood pressure',st.session_state.input_values.get('trestbps', ''))
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl',st.session_state.input_values.get('chol', ''))
    with col3:
        fbs  = st.text_input('Fasting Blood Sugar > 120mg/dl',st.session_state.input_values.get('fbs', ''))
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results',st.session_state.input_values.get('restecg', ''))
    with col2:
        thalach = st.text_input('Maximum HeartRate achieved',st.session_state.input_values.get('thalach', ''))
    with col3:
        exang = st.text_input('Excercise induced Angina',st.session_state.input_values.get('exang', ''))
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise',st.session_state.input_values.get('oldpeak', ''))
    with col2:
        slope = st.text_input('slope of the peak exercise ST segment',st.session_state.input_values.get('slope', ''))
    with col3:
        ca = st.text_input('Major vessel colored by flourosopy',st.session_state.input_values.get('ca', ''))
    with col1:
        thal = st.text_input('thal: 0=normal,1=fixed defect,2=reversable defect',st.session_state.input_values.get('thal', ''))
            
    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        user_input  = [age1,sex1,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        
        user_input = [float(x) for x in user_input]
        
        st.session_state.input_values['age1'] = age1
        st.session_state.input_values['sex1'] = sex1
        st.session_state.input_values['cp'] = cp
        st.session_state.input_values['trestbps'] = trestbps
        st.session_state.input_values['chol'] = chol
        st.session_state.input_values['fbs'] = fbs
        st.session_state.input_values['restecg'] = restecg
        st.session_state.input_values['thalach'] = thalach
        st.session_state.input_values['exang'] = exang
        st.session_state.input_values['oldpeak'] = oldpeak
        st.session_state.input_values['ca'] = ca
        st.session_state.input_values['thal'] = thal
        
        heart_prediction = heart_disease_model.predict([user_input])
        
        if (heart_prediction[0] == 1):
            heart_diagnosis = 'The person having heart disease'
        else:
            heart_diagnosis = 'The person not having heart disease'
    st.success(heart_diagnosis)
       
if (selected == 'Parkinsons Prediction'):
    st.title('Parkinsons Prediction')
    
    col1,col2,col3,col4,col5 = st.columns(5)
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)',st.session_state.input_values.get('fo', ''))

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)',st.session_state.input_values.get('fhi', ''))

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)',st.session_state.input_values.get('flo', ''))

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)',st.session_state.input_values.get('Jitter_percent', ''))

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)',st.session_state.input_values.get('Jitter_Abs', ''))

    with col1:
        RAP = st.text_input('MDVP:RAP',st.session_state.input_values.get('RAP', ''))

    with col2:
        PPQ = st.text_input('MDVP:PPQ',st.session_state.input_values.get('PPQ', ''))

    with col3:
        DDP = st.text_input('Jitter:DDP',st.session_state.input_values.get('DDP', ''))

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer',st.session_state.input_values.get('Shimmer', ''))

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)',st.session_state.input_values.get('Shimmer_dB', ''))

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3',st.session_state.input_values.get('APQ3', ''))

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5',st.session_state.input_values.get('APQ5', ''))

    with col3:
        APQ = st.text_input('MDVP:APQ',st.session_state.input_values.get('APQ', ''))

    with col4:
        DDA = st.text_input('Shimmer:DDA',st.session_state.input_values.get('DDA', ''))

    with col5:
        NHR = st.text_input('NHR',st.session_state.input_values.get('NHR', ''))

    with col1:
        HNR = st.text_input('HNR',st.session_state.input_values.get('HNR', ''))

    with col2:
        RPDE = st.text_input('RPDE',st.session_state.input_values.get('RPDE', ''))

    with col3:
        DFA = st.text_input('DFA',st.session_state.input_values.get('DFA', ''))

    with col4:
        spread1 = st.text_input('spread1',st.session_state.input_values.get('spread1', ''))

    with col5:
        spread2 = st.text_input('spread2',st.session_state.input_values.get('spread2', ''))

    with col1:
        D2 = st.text_input('D2',st.session_state.input_values.get('D2', ''))

    with col2:
        PPE = st.text_input('PPE',st.session_state.input_values.get('PPE', ''))

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]
        
        st.session_state.input_values['fo'] = fo
        st.session_state.input_values['fhi'] = fhi
        st.session_state.input_values['flo'] = flo
        st.session_state.input_values['Jitter_Abs'] = Jitter_Abs
        st.session_state.input_values['RAP'] = RAP
        st.session_state.input_values['PPQ'] = PPQ
        st.session_state.input_values['DDP'] = DDP
        st.session_state.input_values['Shimmer'] = Shimmer
        st.session_state.input_values['Shimmer_dB'] = Shimmer_dB
        st.session_state.input_values['APQ3'] = APQ3
        st.session_state.input_values['APQ5'] = APQ5
        st.session_state.input_values['APQ'] = APQ
        st.session_state.input_values['DDA'] = DDA
        st.session_state.input_values['NHR'] = NHR
        st.session_state.input_values['RPDE'] = RPDE
        st.session_state.input_values['DFA'] = DFA
        st.session_state.input_values['spread1'] = spread1
        st.session_state.input_values['spread2'] = spread2
        st.session_state.input_values['D2'] = D2
        st.session_state.input_values['PPE'] = PPE
        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
   

if (selected == 'Titanic Survival Prediction'):
    st.title('Titanic Survival Prediction Prediction')  
    
    
    col1,col2,col3 = st.columns(3)
    with col1:
        passengername = st.text_input('PassengerId',st.session_state.input_values.get('passengername', ''))
    with col2:
        Cp = st.text_input('Pclass',st.session_state.input_values.get('Cp', ''))
    with col3:
        Sex = st.text_input('Sex',st.session_state.input_values.get('Sex', ''))
    with col1:
        age = st.text_input('Age',st.session_state.input_values.get('age', ''))
    with col2:
        name = st.text_input('SibSp',st.session_state.input_values.get('name', ''))
    with col3:
        parch = st.text_input('Parch',st.session_state.input_values.get('parch', ''))
    with col1:
        Fare = st.text_input('fare',st.session_state.input_values.get('Fare', ''))
    with col2:
        embark = st.text_input('Embarked',st.session_state.input_values.get('embark', ''))
        
    titanic_diagnosis = ''
    if st.button('Titanic Survival Result'):
        user_input = [passengername,Cp,Sex,age,name,parch,Fare,embark]
        
        user_input = [float(x) for x in user_input]
        
        st.session_state.input_values['passengername'] = passengername
        st.session_state.input_values['Cp'] = Cp
        st.session_state.input_values['Sex'] = Sex
        st.session_state.input_values['age'] = age
        st.session_state.input_values['name'] = name
        st.session_state.input_values['parch'] = parch
        st.session_state.input_values['Fare'] = Fare
        st.session_state.input_values['embark'] = embark
        
        titanic_prediction =titanic_model.predict([user_input])
          
        if (titanic_prediction[0] == 1):
            titanic_diagnosis = 'The person is survived'
        else:
            titanic_diagnosis = 'The person is not survived'
    st.success(titanic_diagnosis)
     
    
if selected=='Data Visualiser':
    #st.set_page_config(page_title = 'Data Visualizer',
                   #layout = 'centered',
                  # page_icon ='📊')
    st.title('📊 Data Visualizer')
    
    working_dir = os.path.dirname(os.path.abspath(__file__))

    folder_path  = f'{working_dir}/Data'

    files_list = files = [ f for f in os.listdir(folder_path) if f.endswith('.csv')]

    selected_file = st.selectbox('Select a file',files_list,index=None)
    if selected_file:
        file_path = os.path.join(folder_path,selected_file)
        df = pd.read_csv(file_path)

        col1,col2 = st.columns(2)

        columns = df.columns.tolist()

        with col1:
            st.write("")
            st.write(df.head())
        with col2:
            x_axis = st.selectbox('Select the x-axis',options=columns+['None'],index=None) 
            y_axis = st.selectbox('Select the Y-axis',options=columns+['None'],index=None)
        plot_list = ['Line Plot','Bar Chart','Scatter Plot','Distribution Plot','Count Plot','Box Plot','Word Cloud','Violin Plot','Histogram','Heatmap']
        selected_plot = st.selectbox('Select a Plot',options=plot_list,index=None)

        if st.button('Generate Plot'):
            fig,ax = plt.subplots(figsize=(6,4))

        if selected_plot  == 'Line Plot':
            sns.lineplot(x=df[x_axis],y=df[y_axis],ax=ax)

        elif selected_plot == 'Bar Chart':
            sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)

        elif selected_plot == 'Scatter Plot':
            sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)

        elif selected_plot == 'Distribution Plot':
            sns.histplot(df[x_axis],kde=True,ax = ax)

        elif selected_plot == 'Count Plot':
            sns.countplot(x = df[x_axis],ax=ax)
            
        elif selected_plot == 'Word Cloud':
            wordcloud = WordCloud(width=800,height=400,background_color='white').generate(' '.join(df[x_axis].dropna()))
            plt.imshow(wordcloud,interpolation='bilinear')
            plt.axis('off')

        elif selected_plot == 'Box Plot':
            sns.boxplot(x=df[x_axis], y= df[y_axis],ax=ax) 
            plt.xticks(rotation=90)
            
        elif selected_plot == 'Violin Plot':
            sns.violinplot(x=df[x_axis], y=df[y_axis], ax=ax) 
            plt.xticks(rotation=90) 
        elif selected_plot == 'Heatmap':
            if selected_file == 'Titanic-Dataset.csv':
                value_for_heatmap = 'Fare' 
            elif selected_file == 'diabetes_S.csv': 
                value_for_heatmap = 'insulin' 
            elif selected_file == 'farmersMarkets.csv':
                value_for_heatmap = None
            elif selected_file == 'parkinsons.csv': 
                value_for_heatmap = 'PPE'  
            elif selected_file == 'tips.csv': 
                value_for_heatmap = 'total_bill'
            else:
                value_for_heatmap = None 
            if value_for_heatmap:
                heatmap_data = df.pivot_table(index=x_axis, columns=y_axis, values=value_for_heatmap,
                                              aggfunc='mean').fillna(0) 
                sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".2f", linewidths=.5,
                            cbar_kws={"label": f"{value_for_heatmap} (Heatmap Label)"}) 
            else: 
                st.error("Please provide a suitable dataset for Heatmap") 
        elif selected_plot == 'Histogram':
            sns.histplot(df[x_axis], kde=True, ax=ax)
        ax.tick_params(axis='x',labelsize=10) 
        ax.tick_params(axis='y',labelsize=10) 
        plt.title(f'{selected_plot} of {y_axis} vs {x_axis}', fontsize=10) 
        plt.xlabel(x_axis, fontsize = 10) 
        plt.ylabel(y_axis, fontsize=10) 
        st.pyplot(fig)


      
        
        
        
        
        
        
        
        
        
        
        
   
    

