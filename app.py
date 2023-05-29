import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta,date
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
from scipy.fftpack import fft
from utils import data_preprocessing
from glob import glob 
import pandas as pd
import numpy as np
import json
import os 

dis = MinMaxScaler()

cwd = os.getcwd()
SNS_DATA = "Visualize Sensor Data"
MOTION = "Study Motion"
st.header("Visualisation Analytics")

@st.cache
def LoadingData():
    with open('KeyFrameDensityData_1.json','r') as f:
        js = json.load(f)
    return js

opt = st.sidebar.radio("Choose",[SNS_DATA, MOTION])
if opt == MOTION:
    js = LoadingData()
    IDS = [i['id'] for i in js ]
    TIMES = [i['data']['Created'] for i in js ]
    df = pd.DataFrame({'date':TIMES,'sessions':IDS })

    with st.form(MOTION):

        st.title(f"#{len(TIMES)} sessions ")

        options = df['sessions'].values
        keyframe_ids = st.selectbox("Choose Session", range(len(options)), format_func=lambda x: options[x] )
        submit = st.form_submit_button("Submit")

    if submit:
        x = [i['Position']['X'] for i in js[keyframe_ids]['data']['SpatialData']['DataPoints']]
        y = [i['Position']['Y'] for i in js[keyframe_ids]['data']['SpatialData']['DataPoints']]
        z = [i['Position']['Z'] for i in js[keyframe_ids]['data']['SpatialData']['DataPoints']]
        X = [i['Rotation']['X'] for i in js[keyframe_ids]['data']['SpatialData']['DataPoints']]
        Y = [i['Rotation']['Y'] for i in js[keyframe_ids]['data']['SpatialData']['DataPoints']]
        Z = [i['Rotation']['Z'] for i in js[keyframe_ids]['data']['SpatialData']['DataPoints']]
        W = [i['Rotation']['W'] for i in js[keyframe_ids]['data']['SpatialData']['DataPoints']]
        t = [i['DeltaTime'] for i in js[keyframe_ids]['data']['SpatialData']['DataPoints']]
        df = pd.DataFrame({ "time": t ,"x" : x, "y" : y, "z" : z, 
                            "rotation_X" : X, "rotation_Y" : Y, "rotation_Z" : Z, "orientaion_W": W })
        df['time'] = df['time'].cumsum()
        df['distance'] = df[["x",  "y",	"z"]].apply(lambda x : (sum([i**2 for i in x.values]))**0.5, axis = 1 )
        df['distance'] = dis.fit_transform(  df[['distance']] )
        fig1 = px.bar(data_frame = df, y = 'distance',title='Distance from origin')
        st.plotly_chart(fig1)
        fig = px.line_3d(
                    data_frame = df, 
                    x = df['x'].values, 
                    y = df['y'].values, 
                    z = df['z'].values,
                    title = "3d line plot of movements"
                    )
        # fig.update_layout(px.scatter_3d(pd.DataFrame("x":[0])))
        st.plotly_chart(fig)
else:

    filepath  = st.selectbox("Select Folder",
            [i for i in  glob( cwd+"/S*" ) if ( ( os.path.isdir(i) ) and ('.' not in i))],
            index = 1)
    ls = sorted( glob(f"{filepath}/[!T]*") )
    ls = [ i for i in ls if 'iotool' in i.lower() ]
    kid = st.selectbox('Select Kid', ls)
    fr = st.checkbox("Apply Fourier Transform !")
    if fr:
        col =  st.selectbox("Choose to column to apply Fourier Transform !", 
                        [   'AccelerometerX',
                            'AccelerometerZ',
                            'AccelerometerY',            
                            'GyroscopeX',
                            'GyroscopeZ',
                            'GyroscopeY'
                        ] )

    submit = st.button("Submit")

    ## Go Buttton 
    # processs = st.button("Go!")
    if submit:
        # loading the Parquet
        with st.spinner("Preprocessing Data !"):
            st.text(os.path.join(kid,'final.parquet'))
            final = pd.read_parquet(os.path.join(kid,'final.parquet'))
            # Data Preprocessing 
            final = data_preprocessing( final )
            final['GyroscopeAbsolute'] = np.sqrt(final[["GyroscopeX",	"GyroscopeY",	"GyroscopeZ"]].sum(axis=1)**2)
        if fr:
            # Fourier Plot
            with st.spinner("Plotting"):
                x = final[col].values
                sr = 1/0.001
                X = fft(x)
                N = len(X)
                n = np.arange(N)
                T = N/sr
                freq = n/T 
                f,ax = plt.subplots(figsize = (20,5))
                ax.stem(freq, np.abs(X),'b', markerfmt=" ", basefmt="-b")
                ax.set_xlabel('Freq (Hz)')
                ax.set_ylabel('FFT Amplitude |X(freq)|')
                st.pyplot(f)

        # Plotting STD and AR of sensor data
        final = final[["AccelerometerAbsolute","GyroscopeAbsolute"]].diff()
        final.dropna(inplace=True, how='any')
        with st.spinner("Plotting"):
            # f,[ax1,ax2] = plt.subplots(2,1, figsize = (20,10))
            # final.plot(alpha = 0.6, ax =  ax1)
            fig1 = px.line(final)
            df = final.resample('100ms').agg({'AccelerometerAbsolute': [np.var, np.std] , 'GyroscopeAbsolute': [np.var, np.std] })
            df = df.T.reset_index(drop=True).T
            df.columns = ['AccAbs_var','AccAbs_std','GyroAbs_var','GyroAbs_std']
            # df.plot(alpha = 0.4,ax = ax2)
            fig2 = px.line(df, title = "Standard Deviation and Variance")
            # ax1.legend(bbox_to_anchor=(1.1, 1.05))
            # ax2.legend(bbox_to_anchor=(1.1, 1.05))
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)
        
