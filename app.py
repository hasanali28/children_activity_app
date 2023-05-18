import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta,date

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
from scipy.fftpack import fft
from glob import glob 
import pandas as pd
import numpy as np
import json
import os 

cwd = os.getcwd()
SNS_DATA = "Visualize Sensor Data"
MOTION = "Study Motion"
st.header("Visualisation Analytics")

@st.cache
def foo():
    with open('KeyFrameDensityData_1.json','r') as f:
        js = json.load(f)
    return js
js = foo()
IDS = [i['id'] for i in js ]
TIMES = [i['data']['Created'] for i in js ]
df = pd.DataFrame({'date':TIMES,'sessions':IDS })
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', utc = True).dt.date

opt = st.sidebar.radio("Choose",[SNS_DATA, MOTION])
if opt == MOTION:
    with st.form(MOTION):
        start_date,  viz = st.columns(2)
        with viz:
            st.title(f"#{len(TIMES)} sessions ")
        with start_date:
            dt = st.date_input('Start date', 
                        df.date.min(),
                        min_value= df.date.min(), 
                        max_value= df.date.max())
        

        keyframe_ids = st.multiselect("Choose Session",df[df.date == dt ]['sessions'].values )
        submit = st.form_submit_button("Submit")
else:

    filepath  = st.selectbox("Select Folder",
            [i for i in  os.listdir( cwd ) if ( ( os.path.isdir(i) ) and ('.' not in i))],
            index = 1)
    ls = sorted( glob(f"{filepath}/[!T]*") )
    ls = [ i for i in ls if 'iotool' in i.lower() ]
    kid = st.selectbox('Select Kid', ls)
    fr = st.checkbox("Apply Fourier Transform !")
    if fr:
        col =  st.selectbox("Choose to column to apply Fourier Transform !", 
    ['AccelerometerAbsolute','GyroscopeX','GyroscopeZ','GyroscopeY'] )

    submit = st.button("Submit")

    

    ## Go Buttton 
    # processs = st.button("Go!")
    if submit:
        # loading the Parquet
        with st.spinner("In Progress"):
            st.text(os.path.join(kid,'final.parquet'))
            final = pd.read_parquet(os.path.join(kid,'final.parquet'))
            final['GyroscopeAbsolute'] = np.sqrt(final[["GyroscopeX",	"GyroscopeY",	"GyroscopeZ"]].sum(axis=1)**2)

        # Plotting STD and AR of sensor data
        final = final[["AccelerometerAbsolute","GyroscopeAbsolute"]].diff()
        final.dropna(inplace=True, how='any')
        with st.spinner("Plotting"):
            f,[ax1,ax2] = plt.subplots(2,1, figsize = (20,10))
            final.plot(alpha = 0.6, ax =  ax1)
            df = final.resample('100ms').agg({'AccelerometerAbsolute': [np.var, np.std] , 'GyroscopeAbsolute': [np.var, np.std] })
            df = df.T.reset_index(drop=True).T
            df.columns = ['AccAbs_var','AccAbs_std','GyroAbs_var','GyroAbs_std']
            df.plot(alpha = 0.4,ax = ax2)
            ax1.legend(bbox_to_anchor=(1.1, 1.05))
            ax2.legend(bbox_to_anchor=(1.1, 1.05))
            st.pyplot(f)

        
    # if anime:
    #     with open(js,'r') as f:
    #         js = json.load(f)
    #     X = [i['Position']['X'] for i in js[0]['data']['SpatialData']['DataPoints']]
    #     Y = [i['Position']['Y'] for i in js[0]['data']['SpatialData']['DataPoints']]
    #     Z = [i['Position']['Z'] for i in js[0]['data']['SpatialData']['DataPoints']]
    #     t = [i['DeltaTime'] for i in js[0]['data']['SpatialData']['DataPoints']]
    #     df = pd.DataFrame({"time": t ,"x" : X, "y" : Y, "z" : Z})
    #     df['time'] = df['time'].cumsum()
    #     # data = pd.DataFrame(np.repeat(df.values, [i for i in range(1, df.shape[0] + 1)], axis=0), columns = ['time','x','y','z'])
    #     # data['grp'] = [i for i in range(1,df.shape[0]+1) for j in range( i )]

    #     fig = px.scatter_3d(
    #         df, 
    #         x='x',
    #         y='y',
    #         z='z',
    #         size=[0.3*i for i in range(1,49)],
    #         # color=[0.7*i for i in range(1,49)],
    #         # animation_frame='grp',
    #         # animation_group = 'time'
    #         )    

    #     fig.update_layout(width = 800, height = 800)    
    #     st.plotly_chart(fig)
