#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import streamlit as st
import sklearn 


# In[2]:


pickle_in = open('regressor.pkl','rb')
regressor = pickle.load(pickle_in)


# In[3]:


def welcome():
    return 'Welcome all'


# In[4]:


def prediction(room_bed, room_bath, ceil, coast, sight, condition, quality, ceil_measure, 
               basement, lat, longi, living_measure15, furnished, yr_built_code, total_area15, distance):
    sample = pd.DataFrame({'room_bed':[room_bed], 'room_bath':[room_bath],'ceil':[ceil],
                           'coast':[coast], 'sight':[sight], 'condition':[condition],
                           'quality':[quality], 'ceil_measure':[ceil_measure], 'basement':[basement],
                           'lat':[lat], 'long':[longi], 'living_measure15':[living_measure15],
                           'furnished':[furnished], 'yr_built_code':[yr_built_code], 'total_area15':[total_area15],
                           'distance':[distance] })
    prediction = regressor.predict(sample)
    return prediction 


# In[5]:


def main():
    html_title="""
    <h1 style="color:blue">
    House Price Predictor
    </h1>
    """
    st.markdown(html_title, unsafe_allow_html=True)
    
    st.markdown('This web app can predict the house price in King County (USA) based on multiple attributes.')
    #html_temp = """
    #<div style ="background-color:yellow;padding:13px">
    #<h1 style ="color:black;text-align:center;">Streamlit House Price Prediction App </h1>
    #</div>
    #"""
    html_description = """ 
    <h4 style="color:green">
    Model Used: XGB Regressor
    </h4>
     <table>
     <tr>
    <th>R2 score</th>
    <th>RMSE</th>
    <th>MAE</th>
    </tr>
  <tr>
    <td>0.869</td>
    <td>74273.931</td>
    <td>52068.071</td>
  </tr>
</table> 
    """
    st.markdown(html_description, unsafe_allow_html=True)
    
    st.sidebar.title('Attributes')
    room_bed = st.sidebar.slider('No. of bed rooms',0,11)
    room_bath = st.sidebar.slider('No. of bath rooms',min_value=0.0,max_value=8.0,step=0.25)
    ceil = st.sidebar.slider('No. of floors',min_value=1.0,max_value=3.5,step=0.5)
    sight = st.sidebar.slider('No. of times the house is viewed',0,4)
    condition = st.sidebar.slider('Condition Rating',1,5)
    quality = st.sidebar.slider('Quality Rating',1,13)
    yr_built_code = st.sidebar.slider('Year built code',1,11)
    distance = st.sidebar.slider('Distance from Boston in km',0,50)
    lat = st.sidebar.slider('Latitude',min_value=47.0,max_value=48.0,step=0.05)
    longi = st.sidebar.slider('Longitude',min_value=-123.0,max_value=-121.0,step=0.05)
    
    ceil_measure = st.sidebar.text_input('Square footage of house apart from basement','Type in sqft')
    living_measure15 = st.sidebar.text_input('Living room area','Type in sqft')
    total_area15 = st.sidebar.text_input('Measure of both living and lot area','Type in sqft')
    
    

    coast = st.sidebar.checkbox('Access to a water body',value=0)
    basement = st.sidebar.checkbox('Basement present',value=0)
    furnished = st.sidebar.checkbox('House furnished',value=0)
      
    result = ''

    st.markdown(' ')
    
    if st.button('Predict Price'):
        result = prediction(room_bed, room_bath, ceil, coast, sight, condition, quality, ceil_measure, 
               basement, lat, longi, living_measure15, furnished, yr_built_code, total_area15, distance)
        st.success('The predicted house price (in USD) is {}'.format(result))

if __name__=='__main__':
    main()


# In[ ]:




