import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Bidirectional, Dense,Dropout,LSTM,Activation, RepeatVector, SimpleRNN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta
import time
from tensorflow import keras


path = 'C:/Users/jbae42/Desktop/솔라미/전력데이터/data'

### parameters
n_features = 7
period = 'week'
num_days = 1
start_date = '2017-06-21 00:15:00' #update this every period
start_date = pd.to_datetime(start_date)
end_date = start_date + timedelta(days=num_days) - timedelta(minutes=15)
saving_rate = 10 #in %
usage_rate = (100 - saving_rate) / 100 
period_len = num_days * 24 * 4
seq_len = 10
n_step = period_len * 2

def split_seq(sequence, n_step):
    X, y = list(), list()
    start = sequence.index.start
    for i in range(start, len(sequence)+start):
        end_ix = i + n_step
        if end_ix > start + len(sequence)-1:
            break
        a = i
        b = end_ix
        seq_x, seq_y = sequence[a:b].values, sequence[8][b]  
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def split_data(sequence, n_step):
    X = list()
    start = sequence.index.start
    for i in range(start, len(sequence)+start):
        end_ix = i + n_step
        if end_ix > start + len(sequence):
            break
        a = i
        b = end_ix
        seq_x = sequence.loc[a:b].values 
        X.append(seq_x)
    return np.array(X)

def make_dataset(var_data, seq_len):
    X_train = []
    for i in range(seq_len, len(var_data)+1):
        X_train.append(var_data[i - seq_len: i])
    return X_train

def add_data(arr,new_data,n_step,seq_len,var=None):
    new_data = np.array(new_data)
    if var == None:
        arr = np.concatenate((arr,new_data[:,None]),axis=1)
        arr = arr[0][-n_step:]
        arr = arr.reshape(1,n_step,9)
        return arr
    arr = np.append(arr,new_data[0,var])[-seq_len:]
    arr = arr.reshape(1,seq_len,1)
    return arr

#compile all data into a single dataset
def build_dataset(path):
    os.chdir(path)
    folders = os.listdir()
    df = pd.DataFrame()
    for folder in folders:
        os.chdir(path)
        if not os.path.isdir(folder): continue
        os.chdir(folder)
        filenames = os.listdir()
        for file in filenames:
            df = pd.concat([df,pd.read_csv(file, parse_dates=True)])
    return df.reset_index(drop=True)

df = build_dataset(path)
df_time = pd.to_datetime(df['times'])

codes = df['buildingcode'].unique()
code = codes[0]

#df_result = pd.DataFrame(index=codes,columns=['RMSE_Train_asos','RMSE_Test_asos','RMSE_Train_aws','RMSE_Test_aws'])

def train_model(df,code,n_step,n_features):
    df = df.sort_values(by='times').reset_index(drop=True)
    df = df.drop_duplicates(subset=['times'])

    df_time = pd.to_datetime(df['times'])
    df = df.drop(columns=['lat','lon','buildingcode'],axis=1)
    df = df.reindex(columns=['ws_asos','temp_asos','ws_aws','temp_aws','pm10','CA','elect'])

    dataset = df.values
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset = scaler.fit_transform(dataset)

    X_train = []
    y_train = []

    for i in range(seq_len, len(dataset)):
        X_train.append(dataset[i - seq_len: i, ])
        y_train.append(dataset[i, -1])

    X_test = X_train[:period_len]
    y_test = y_train[:period_len]

    X_train = X_train[period_len:]
    y_train = y_train[period_len:]

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = X_train.reshape(X_train.shape[0], seq_len, -1)
    X_test = X_test.reshape(X_test.shape[0], seq_len, -1)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(LSTM(40, return_sequences=True, input_shape=(X_train.shape[1],7))))
    model.add(Bidirectional(LSTM(20,return_sequences= False, activation= 'linear')))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(Dense(1))

    model.compile(loss='MSE', optimizer = 'adam')
    model.fit(X_train, y_train, epochs=10, validation_split=0.3, batch_size=1000)
    
    return model
    

### update this when the start_date is updated    
last_start = start_date - timedelta(days=num_days)
last_end = end_date - timedelta(days=num_days)
cur_timestamp = start_date

trained_model = train_model(df,code,n_step,n_features) #initial train of the model
df_pred = pd.DataFrame(columns=['times','current','previous'])

os.chdir('C:/Users/jbae42/Desktop/솔라미/전력데이터/elect_reduction/multioutput/pickled_models')

ws_asos_model = keras.models.load_model("ws_asos_model")
temp_asos_model = keras.models.load_model("temp_asos_model")
ws_aws_model = keras.models.load_model("ws_aws_model")
temp_aws_model = keras.models.load_model("temp_aws_model")
pm10_model = keras.models.load_model("pm10_model")
CA_model = keras.models.load_model("CA_model")

#while datetime.now() == end_date # run this until the end date is reached
#if datetime.now().minute == 15 # run this every 15 min
os.chdir(path)
df = build_dataset(path) #read the updated dataset
df_time = pd.to_datetime(df['times'])
ind = df[df['buildingcode']==code].index

apt_data = df.iloc[ind]
apt_data = apt_data.drop_duplicates(subset='times').reset_index(drop=True)

apt_data['day'] = df_time.dt.dayofweek
apt_data['hour'] = df_time.dt.hour
apt_data['minute'] = df_time.dt.minute
apt_data['dayofmonth'] = df_time.dt.day
colnames = ['ws_asos','temp_asos','ws_aws','temp_aws','pm10','CA','elect']
apt_data = apt_data.reindex(columns=colnames)

time_range = (apt_data['times']>=str(start_date)) & (apt_data['times']<=str(end_date))
prev_range = (apt_data['times']>=str(last_start)) & (apt_data['times']<=str(last_end))
df_pred['times'] = apt_data.loc[time_range,'times'].reset_index(drop=True)
df_pred['previous'] = apt_data.loc[prev_range,'elect'].reset_index(drop=True)

if cur_timestamp != start_date:
    prev_timestamp = cur_timestamp - timedelta(minutes=15)
    act_data = apt_data[apt_data['times']==str(prev_timestamp)]['elect']
    rep_ind = df_pred['times']==str(prev_timestamp)
    df_pred.loc[rep_ind,'current'] = act_data.values[0]

apt_data = apt_data.drop(columns=['times'])

dataset_updated = apt_data.values

scaler = MinMaxScaler(feature_range=(0,1))
dataset_updated = pd.DataFrame(scaler.fit_transform(dataset_updated))
initial_data = dataset_updated[-n_step:]
initial_data = split_data(initial_data, n_step)

x_predict = np.array(initial_data)
#x_predict = np.rot90(x_predict)
#x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], n_features)

pm10_data = make_dataset(dataset_updated.iloc[-seq_len:,6], seq_len)
pm10_data = np.array(pm10_data)
pm10_data = pm10_data.reshape(pm10_data.shape[0], seq_len, 1)

CA_data = make_dataset(dataset_updated.iloc[-seq_len:,7], seq_len)
CA_data = np.array(CA_data)
CA_data = CA_data.reshape(CA_data.shape[0], seq_len, 1)

ws_asos_data = make_dataset(dataset_updated.iloc[-seq_len:,4], seq_len)
ws_asos_data = np.array(ws_asos_data)
ws_asos_data = ws_asos_data.reshape(ws_asos_data.shape[0], seq_len, 1)
temp_asos_data = make_dataset(dataset_updated.iloc[-seq_len:,5], seq_len)
temp_asos_data = np.array(temp_asos_data)
temp_asos_data = temp_asos_data.reshape(temp_asos_data.shape[0], seq_len, 1)
ws_aws_data = make_dataset(dataset_updated.iloc[-seq_len:,4], seq_len)
ws_aws_data = np.array(ws_aws_data)
ws_aws_data = ws_aws_data.reshape(ws_aws_data.shape[0], seq_len, 1)
temp_aws_data = make_dataset(dataset_updated.iloc[-seq_len:,5], seq_len)
temp_aws_data = np.array(temp_aws_data)
temp_aws_data = temp_aws_data.reshape(temp_aws_data.shape[0], seq_len, 1)


temp_stamps = []
temp_stamps_ = cur_timestamp
for i in range(num_days*24*4):
    temp_stamps.append(temp_stamps_)
    temp_stamps_ += timedelta(minutes=15)
    

period_predict = pd.DataFrame(columns=['times','temp_predict'])
period_predict['times'] = pd.Series(temp_stamps)

next_data = pd.DataFrame()
y_predict = trained_model.predict(x_predict)
#print('y_predict', y_predict, len(y_predict[0]))
CA_predict = CA_model.predict(CA_data)
#print('CA_predict',CA_predict,len(CA_predict))
pm10_predict = pm10_model.predict(pm10_data)
#print('pm10_predict',pm10_predict,len(pm10_predict))
ws_asos_predict = ws_asos_model.predict(ws_asos_data)
#print('ws_asos',ws_asos_predict)
temp_asos_predict = temp_asos_model.predict(temp_asos_data)
#print('temp_asos',temp_asos_predict)
ws_aws_predict = ws_aws_model.predict(ws_aws_data)
temp_aws_predict = temp_aws_model.predict(temp_aws_data)

next_data['day'] = period_predict['times'].dt.dayofweek
next_data['hour'] = period_predict['times'].dt.hour
next_data['minute'] = period_predict['times'].dt.minute
next_data['dayofmonth'] = period_predict['times'].dt.day
next_data['ws_asos'] = pd.Series(ws_asos_predict[0])
next_data['temp_asos'] = pd.Series(temp_asos_predict[0])
next_data['ws_aws'] = pd.Series(ws_asos_predict[0])
next_data['temp_aws'] = pd.Series(temp_aws_predict[0])
next_data['pm10'] = pd.Series(pm10_predict[0])
next_data['CA'] = pd.Series(CA_predict[0])
next_data['elect'] = pd.Series(y_predict[0])
#print('next_data_elect', len(next_data['elect']), next_data['elect'])
#consum_pred = scaler.inverse_transform(y_predict)
consum_pred = scaler.inverse_transform(next_data)
#print('consum_pred', consum_pred, consum_pred[8])
#temp_ind = period_predict[period_predict['times']==timestamp_now].index
period_predict['temp_predict'] = pd.DataFrame(consum_pred)[[8]]

### build dataset for next round
# next_data = np.array(next_data)
# x_predict = np.concatenate((x_predict,next_data[:,None]),axis=1)
# x_predict = x_predict[0][-n_step:]
# x_predict = x_predict.reshape(1,n_step,9)

# x_predict = add_data(x_predict,next_data,n_step,seq_len)
# CA_data = add_data(CA_data,next_data,n_step,seq_len,7)
# pm10_data = add_data(pm10_data,next_data,n_step,seq_len,6)
# if source == 'asos':
#     ws_asos_data = add_data(ws_asos_data,next_data,n_step,seq_len,4)
#     temp_asos_data = add_data(temp_asos_data,next_data,n_step,seq_len,5)
# else:
#     ws_aws_data = add_data(ws_aws_data,next_data,n_step,seq_len,4)
#     temp_aws_data = add_data(temp_aws_data,next_data,n_step,seq_len,5)
   

from_ind = (period_predict['times']>=str(cur_timestamp)) & (period_predict['times']<=str(end_date))
to_ind = (df_pred['times']>=str(cur_timestamp)) & (df_pred['times']<=str(end_date))
df_pred.loc[to_ind,'current']  = period_predict.loc[from_ind,'temp_predict']

if cur_timestamp.hour == 0 and cur_timestamp.minute == 0: ##at every midnight, train the model with the updated data
    trained_model = train_model(df,code,n_step,n_features)
    ### train the following models every night as well then re-load the models
    os.chdir('C:/Users/jbae42/Desktop/솔라미/전력데이터/elect_reduction/pickled_models')
    ws_asos_model = keras.models.load_model("ws_asos_model")
    temp_asos_model = keras.models.load_model("temp_asos_model")
    ws_aws_model = keras.models.load_model("ws_aws_model")
    temp_aws_model = keras.models.load_model("temp_aws_model")
    pm10_model = keras.models.load_model("pm10_model")
    CA_model = keras.models.load_model("CA_model")

elect_diff = sum(df_pred['current']) - usage_rate * sum(df_pred['previous'])
print('elect_diff', elect_diff)
print('this weeks prediction',sum(df_pred['current']))
print('last weeks usage',usage_rate * sum(df_pred['previous']))

if elect_diff > 0:
    time_remained = end_date - cur_timestamp
    timestamps = time_remained.days * 24 * 4 + time_remained.seconds / 900
    print('에너지 절감 필요 - 다음 15분 간 목표 사용량: {}'.format(elect_diff/timestamps))
else:
    next_ind = period_predict['times']==str(cur_timestamp+timedelta(minutes=15))
    next_elect = period_predict.loc[next_ind,'temp_predict']
    print('목표 절감 가능 - 다음 15분 간 목표 사용량: {}'.format(next_elect.values[0]))
    
cur_timestamp += timedelta(minutes=15)

