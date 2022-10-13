import pickle 
import pandas as pd 
import streamlit as st
import base64

import random
import time

data = pd.read_csv("data/weatherAus.csv", header=0, squeeze=True)
    

filename = 'models/model_13_08.sav'
loaded_model = pickle.load(open(filename, 'rb'))

FEATURES = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']


def find_geographic_code(city): 
    """ Find geographic information givin a city name """
    coordinates = coordinates_df[coordinates_df['index'] == city].iloc[0] 
    return coordinates[1], coordinates[2]

data = pd.read_csv('data/weatherAus.csv', parse_dates=["Date"])
data.dropna(inplace=True)

# result = loaded_model.predict(data)
st.set_page_config(page_title="Will it rain tommorow?", page_icon="🐞", layout="centered")


st.sidebar.write(
    f"This app is the result of our experimentation done during the project for DataScientest program "
)


pages = ['Classification Model', 'Data Visualisation']

page = st.sidebar.radio("Go to",pages)


if page == pages[0] : 

    col1, col2 = st.columns((3, 1))
    with col1:
        st.title("Classification Model") 
    
    with col2: 
        if st.button('random raining data'):
            tmp = data[data['RainTomorrow'] == 'Yes']
            index = random.randint(0, len(tmp)-1)
            value = tmp[FEATURES].iloc[index].to_dict()
            for key, value in value.items(): 
                if key == 'RainToday': 
                    value = False if value == 'No' else True
                st.session_state[key] =  value
        if st.button('random sunny data'):
            tmp = data[data['RainTomorrow'] == 'No']
            index = random.randint(0, len(tmp)-1)
            value = tmp[FEATURES].iloc[index].to_dict()
            for key, value in value.items(): 
                if key == 'RainToday': 
                    value = False if value == 'No' else True
                st.session_state[key] =  value
    
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
        'AliceSprings', 'Darwin', 'Katherine', 'Uluru'), key='Location')

        Rainfall = cols[1].number_input('Rainfall (mm)', key='Rainfall')
        cols = st.columns((1, 1))
        MinTemp = cols[0].number_input("MinTemp:", key='MinTemp')
        MaxTemp = cols[1].number_input("MaxTemp:", key='MaxTemp')

        cols = st.columns((1, 1))
        Evaporation = cols[0].number_input('Evaporation (mm) ', key='Evaporation')
        Sunshine = cols[1].number_input('Sunshine (# of hours)', key='Sunshine')
        

        st.subheader('Collected information last 24h')

        cols = st.columns((1, 1))
        WindGustDir = cols[0].selectbox(
        'Wind Direction? (strongest wind last 24h)',
        ( 'NNW', 'N', 'NNE', 'W', 'ENE', 'WNW', 'NW', 'SE', 'NE', 'SW',
       'ESE', 'SSE', 'E', 'SSW', 'WSW','S'), key='WindGustDir')
        WindGustSpeed = cols[1].number_input('Wind speed last 24h', key='WindGustSpeed')
        
        st.subheader('Collected information at/to 9am')
        cols = st.columns((1, 1, 1))
        WindDir9am = cols[0].selectbox(
        'Wind Direction at 9am? ',
        ( 'NNW', 'N', 'NNE', 'W', 'ENE', 'WNW', 'NW', 'SE', 'NE', 'SW',
       'ESE', 'SSE', 'E', 'SSW', 'WSW','S'), key='WindDir9am')
        WindSpeed9am = cols[1].number_input('Wind speed at 9am ', key='WindSpeed9am')
        Humidity9am = cols[2].number_input('Humidity at 9am ', key='Humidity9am')
        cols = st.columns((1, 1, 1))
        Pressure9am = cols[0].number_input('Pressure at 9am ', key='Pressure9am')
        Cloud9am = cols[1].number_input('Cloud at 9am ', key='Cloud9am')
        Temp9am = cols[2].number_input('Temperature at 9am ', key='Temp9am')


        st.subheader('Collected information at/to 3pm')

        cols = st.columns((1, 1, 1))
        WindDir3pm = cols[0].selectbox(
        'Wind Direction at 3pm? ',
        ( 'NNW', 'N', 'NNE', 'W', 'ENE', 'WNW', 'NW', 'SE', 'NE', 'SW',
       'ESE', 'SSE', 'E', 'SSW', 'WSW','S'), key='WindDir3pm')
        WindSpeed3pm = cols[1].number_input('Wind speed at 3pm ', key='WindSpeed3pm')
        Humidity3pm = cols[2].number_input('Humidity at 3pm ', key='Humidity3pm')
        cols = st.columns((1, 1, 1))

        Pressure3pm = cols[0].number_input('Pressure at 3pm ', key='Pressure3pm')
        Cloud3pm = cols[1].number_input('Cloud at 3pm ', key='Cloud3pm')
        Temp3pm = cols[2].number_input('Temperature at 3pm ', key='Temp3pm')

        RainToday = st.checkbox('Did it rain today ?', key='RainToday')
        submitted = st.form_submit_button(label="Will it rain tommorow ?")


        if submitted:
            myvars = locals()
            submitted_features = { k: myvars[k] for k in FEATURES }
            submitted_features['RainToday'] = int(submitted_features['RainToday'])
            #st.write(submitted_features)
                
            data = pd.DataFrame(data = [submitted_features])
            predicted_weather = loaded_model.predict(data)[0]
            probas = loaded_model.predict_proba(data)[0]
            print(probas)
            print(predicted_weather)
            
            message = 'will' if predicted_weather == 1 else 'will not'
            filename = 'data/sunny.gif' if predicted_weather == 0 else 'data/raining.gif'
            file_ = open(filename, "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            cols = st.columns((0.5, 3, 0.5))
            cols[1].write('Prediction : We are {:.2f} % sure that it {} be raining tommorow.. '.format(100*probas[int(predicted_weather)], message))

            cols = st.columns((0.5, 3, 0.5))

            cols[1].markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                unsafe_allow_html=True)

            st.write('\n')


if page == pages[1] : 
    st.write('Welcome to data visualization ')
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt # basic plotting
    import seaborn as sns # for prettier plots
    import plotly.graph_objects as go

    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import datetime as dt

    coordinates_df = pd.read_csv('data/address_lat_long.csv', index_col=0)
    coordinates_df = coordinates_df.reset_index()

    data['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
    data['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)


    st.plotly_chart(px.histogram(data, 
             x='RainTomorrow', 
             color='RainToday', 
             title='Rain Tomorrow vs. Rain Today'), use_container_width=True)
    
    all_locations = data['Location'].unique()
    locations_lat_long = {k: list(find_geographic_code(k)) for k in all_locations }

    st.write('Temperature variation across cities (avg)')

    with st.spinner('Loading data..'):
        data[['latitude','longitude']] = data['Location'].apply(lambda x: pd.Series(locations_lat_long[x]))
        data['AvgTemp']=(data['MinTemp']+data['MaxTemp'])/2
        data['mont_of_year'] = data['Date'].apply(lambda x: dt.datetime.strftime(x,'%b-%Y'))
        data['mont_of_year_formatted'] = pd.to_datetime(data['mont_of_year'])

        temperature_data = data.groupby(["Location","longitude",'latitude','mont_of_year','mont_of_year_formatted'])['AvgTemp'].mean().reset_index()
        temperature_data.sort_values(by=['mont_of_year_formatted'], inplace=True)
        temperature_data = temperature_data[~temperature_data['AvgTemp'].isnull()]
        temperature_data['AvgTemp'] = temperature_data['AvgTemp'].apply(lambda x: 0 if x < 0 else x )

        fig = px.scatter_mapbox(temperature_data, lat="latitude", lon="longitude", hover_name="Location", color="AvgTemp",
                            size="AvgTemp", color_continuous_scale=px.colors.sequential.matter, size_max=20,
                            zoom=3, height=700, mapbox_style="open-street-map",animation_frame="mont_of_year")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)