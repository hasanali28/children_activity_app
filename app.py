import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
from scipy.fftpack import fft
from glob import glob 
import pandas as pd
import numpy as np
import json
import os 

def create_dr(df):
    minn = df.index.values.min()
    maxx = df.index.values.max()
    date_range =pd.date_range(start=minn, end=maxx, freq='ms')

    return date_range

st.header("Visualisation Analytics")
filepath  = st.selectbox("Select Folder",
            [i for i in  os.listdir(os.getcwd()) if ((os.path.isdir(i)) and ('.' not in i))],
            index = 1)
ls = sorted( glob(f"{filepath}/[!T]*") )
ls = [ i for i in ls if 'iotool' in i.lower() ]
kid = st.selectbox('Select Kid', ls)
st.text(glob(f"{kid}/f*.csv"))
fr = st.checkbox('Apply Fourier Transform !')
if fr:
    col =  st.selectbox("Choose to column to apply Fourier Transform !", 
    ['AccelerometerAbsolute','GyroscopeX','GyroscopeZ','GyroscopeY'] )


processs = st.button("Go!")

if processs:
    with st.spinner("In Progress"):
        # folder = ls[kid-1]
        set_dt = False
        for csv in glob(f"{kid}/f*.parquet"):
            st.text(csv)
            final = pd.read_parquet(csv)
        # final = final.reindex(sorted(final.columns), axis=1)
        # final.set_index(keys=['date'], inplace=True)
        final['GyroscopeAbsolute'] = np.sqrt(final[["GyroscopeX",	"GyroscopeY",	"GyroscopeZ"]].sum(axis=1)**2)

    if fr:
        with st.spinner("Plotting"):
            X = final[col].values
            X = fft(X)
            f,ax = plt.subplots(figsize = (20,5))
            ax.plot(X)
            st.pyplot(f)

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
        # fig = px.line(final.reset_index(), x ='date' , y = ['AccelerometerAbsolute','GyroscopeX','GyroscopeZ','GyroscopeY'])
        st.pyplot(f)
        # st.plotly_chart(fig,   theme="streamlit")
   
        
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
