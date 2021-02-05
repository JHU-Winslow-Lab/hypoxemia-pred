def get_time_series_smooth(ID,vitals):
    #pull the patients data
    vitals=vitals[vitals['patientunitstayid']==ID]
    vitals=vitals[['observationoffset','sao2']]
    vitals=vitals.fillna(method='pad')
    vitals=vitals.dropna()
    vital_times=np.array(vitals['observationoffset'])

    #smooth the necessary signals
    #from scipy.signal import savgol_filter
    sao2=vitals['sao2']
    sao2_transformed=1-np.exp((sao2-100)/10)
    sao2_smooth = np.concatenate((sao2_transformed[0:4],np.convolve(sao2_transformed, np.ones(5)/5,mode='valid')))

    
    #the patient time series
    #SaO2 only
    pt_series=np.transpose(np.vstack(sao2_smooth))

    return pt_series,vital_times

def time_data30(pt_series,vital_times):
    ys=[]
    xs=[]
    t=[]
    ts=2
    inputshape=0
    #for SaO2 model only
    for i in range(0,len(pt_series[0])-(ts+1)):
        y=pt_series[0][i+ts]
        t.append(vital_times[i+ts])
        #all
        x=pt_series[0][i:i+ts]
        ys.append(y)
        xs.append(x)
    ys=np.transpose(ys)
    n=0
    #downsampling
    d=6
    xs=xs[::d]
    ys=ys[::d]
    t=t[::d]
    return xs,ys,ts,t,n

def time_data5(pt_series,vital_times):
    ys=[]
    xs=[]
    t=[]
    ts=2
    inputshape=0
    #for SaO2 model only
    for i in range(0,len(pt_series[0])-(ts+1)):
        y=pt_series[0][i+ts]
        t.append(vital_times[i+ts])
        #all
        x=pt_series[0][i:i+ts]
        ys.append(y)
        xs.append(x)
    ys=np.transpose(ys)
    n=0
    
    return xs,ys,ts,t,n

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor



import pandas as pd
import numpy as np
vitals=pd.read_csv('vitals.csv')
TrainIDs=pd.read_csv('TrainEICU.csv',header=None)
TrainIDs=np.array(TrainIDs[0].astype(int))
VentTest=pd.read_csv('TestVent.csv',header=None)
VentTest=np.array(VentTest[0].astype(int))
NoVentTest=pd.read_csv('Test_NoVent.csv',header=None)
NoVentTest=np.array(NoVentTest[0].astype(int))

ID = TrainIDs[0]
pt_series,vital_times=get_time_series_smooth(ID,vitals)
full_x_train,full_y_train,ts,t,n = time_data5(pt_series,vital_times)

#Have to change which time_data fxn to use depending on 30 min vs. 5 min
True_TrainIDs=[]
for r in range(1,len(TrainIDs)):
    ID = TrainIDs[r]
    v=vitals[vitals['patientunitstayid']==ID]
    if len(v)>60:
        if ~(np.sum(v.isna()['sao2'])==len(v)):
            True_TrainIDs.append(ID)
            #need to pass ID, vents, vitals
            pt_series,vital_times=get_time_series_smooth(ID,vitals)
            xtrain,ytrain,ts,t,n = time_data5(pt_series,vital_times)
            full_x_train=np.vstack((full_x_train,xtrain))
            full_y_train=np.hstack((full_y_train,ytrain))

s=np.shape(full_x_train)[0]
xtrain=full_x_train.reshape(s,2,1)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
lstm=tf.keras.Sequential()
inputshape=1
lstm.add(tf.keras.layers.BatchNormalization(input_shape=(ts,inputshape)))
lstm.add(tf.keras.layers.LSTM(256,return_sequences=True,input_shape=(ts,inputshape)))
lstm.add(tf.keras.layers.Dropout(.1))
lstm.add(tf.keras.layers.LSTM(16,return_sequences=False))
lstm.add(tf.keras.layers.Dense(units=1))
opt = keras.optimizers.Adam(learning_rate=0.001)
lstm.compile(loss='mse',optimizer=opt)

filepath='FullEICU_trained_model_5min.h5'

# Keep only a single checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')
lstm.fit(xtrain,full_y_train,epochs=100,validation_split=.1,shuffle=True,callbacks=[checkpoint])

