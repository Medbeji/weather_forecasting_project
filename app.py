import pickle 
import pandas as pd 
import streamlit as st
import base64



data = pd.read_csv("data/weatherAus.csv", header=0, squeeze=True)


filename = 'models/model_13_08.sav'
loaded_model = pickle.load(open(filename, 'rb'))

FEATURES = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']


# result = loaded_model.predict(data)
st.set_page_config(page_title="Will it rain tommorow?", page_icon="🐞", layout="centered")


st.sidebar.write(
    f"This app is the result of our experimentation done during the project for DataScientest program "
)


pages = ['Classification Model', 'Data Visualisation', 'Sydney Time-series forecasting']

page = st.sidebar.radio("Go to",pages)


if page == pages[0] : 
    st.title("Classification Model")
    
    form = st.form(key="annotation")

    with form:
        cols = st.columns((1, 1))
        
        Location = cols[0].selectbox(
        'What location ?',
        ( 'Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
        'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
        'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
        'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
        'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
        'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
        'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
        'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
        'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
        'AliceSprings', 'Darwin', 'Katherine', 'Uluru'))

        Rainfall = cols[1].number_input('Rainfall (mm)')
        cols = st.columns((1, 1))
        MinTemp = cols[0].number_input("MinTemp:")
        MaxTemp = cols[1].number_input("MaxTemp:")

        cols = st.columns((1, 1))
        Evaporation = cols[0].number_input('Evaporation (mm) ')
        Sunshine = cols[1].number_input('Sunshine (# of hours)')
        

        st.subheader('Collected information last 24h')

        cols = st.columns((1, 1))
        WindGustDir = cols[0].selectbox(
        'Wind Direction? (strongest wind last 24h)',
        ( 'NNW', 'N', 'NNE', 'W', 'ENE', 'WNW', 'NW', 'SE', 'NE', 'SW',
       'ESE', 'SSE', 'E', 'SSW', 'WSW','S'))
        WindGustSpeed = cols[1].number_input('Wind speed last 24h')
        
        st.subheader('Collected information at/to 9am')
        cols = st.columns((1, 1, 1))
        WindDir9am = cols[0].selectbox(
        'Wind Direction at 9am? ',
        ( 'NNW', 'N', 'NNE', 'W', 'ENE', 'WNW', 'NW', 'SE', 'NE', 'SW',
       'ESE', 'SSE', 'E', 'SSW', 'WSW','S'))
        WindSpeed9am = cols[1].number_input('Wind speed at 9am ')
        Humidity9am = cols[2].number_input('Humidity at 9am ')
        cols = st.columns((1, 1, 1))
        Pressure9am = cols[0].number_input('Pressure at 9am ')
        Cloud9am = cols[1].number_input('Cloud at 9am ')
        Temp9am = cols[2].number_input('Temperature at 9am ')


        st.subheader('Collected information at/to 3pm')

        cols = st.columns((1, 1, 1))
        WindDir3pm = cols[0].selectbox(
        'Wind Direction at 3pm? ',
        ( 'NNW', 'N', 'NNE', 'W', 'ENE', 'WNW', 'NW', 'SE', 'NE', 'SW',
       'ESE', 'SSE', 'E', 'SSW', 'WSW','S'))
        WindSpeed3pm = cols[1].number_input('Wind speed at 3pm ')
        Humidity3pm = cols[2].number_input('Humidity at 3pm ')
        cols = st.columns((1, 1, 1))

        Pressure3pm = cols[0].number_input('Pressure at 3pm ')
        Cloud3pm = cols[1].number_input('Cloud at 3pm ')
        Temp3pm = cols[2].number_input('Temperature at 3pm ')

        RainToday = st.checkbox('Did it rain today ?')
        submitted = st.form_submit_button(label="Will it rain tommorow ?")


        if submitted:
            myvars = locals()
            submitted_features = { k: myvars[k] for k in FEATURES }
            submitted_features['RainToday'] = int(submitted_features['RainToday'])
            #st.write(submitted_features)
                
            data = pd.DataFrame(data = [submitted_features])
            predicted_weather = loaded_model.predict(data)[0]
            
            message = 'Will' if predicted_weather == 1 else 'Will not'
            filename = 'data/sunny.gif' if predicted_weather == 0 else 'data/raining.gif'
            file_ = open(filename, "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            cols = st.columns((0.5, 3, 0.5))
            cols[1].write('Prediction : ' +message + ' be raining tommorow.. ')

            cols = st.columns((0.5, 3, 0.5))

            cols[1].markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                unsafe_allow_html=True)

            st.write('\n')


                



    

    