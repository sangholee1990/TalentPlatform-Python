# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:17:46 2022

@author: jnlee0417
"""

# https://github.com/thieu1995/permetrics
from permetrics.regression import RegressionMetric
from datetime import timedelta
from datetime import datetime
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import os
import datetime
import random
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import glob


# ============================================
# 보조
# ============================================
def split(batch, n_features, n_labels):
    inputs = batch[:, :-1, :n_features]
    labels = batch[:, -1, n_features:]

    return inputs, labels


def make_dataset(data, features, labels, sequence_length, batch_size=64, test=False):
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data[features + labels].astype(np.float32),
        targets=None,
        sequence_length=sequence_length + 1,
        sequence_stride=1,
        shuffle=False if test else True,
        batch_size=batch_size,
        seed=SEED
    )
    ds = ds.map(lambda x: split(x, len(features), len(labels)))

    return ds


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def result_estimate(ob, si):
    x = ob.to_numpy()
    y = si.to_numpy()
    evaluator = RegressionMetric(x, y, decimal=4)
    NSE = evaluator.nash_sutcliffe_efficiency()
    RMSE = evaluator.root_mean_squared_error()
    R2 = evaluator.pearson_correlation_coefficient()
    PBIAS = round((np.sum(y) - np.sum(x)) / np.sum(x) * 100, 5)

    return NSE, RMSE, R2, PBIAS


# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0385'

# 옵션 설정
sysOpt = {
    # 시작/종료 시간
    'srtDate': '2019-01-01'
    , 'endDate': '2023-01-01'
}

globalVar['inpPath'] = '/DATA/INPUT'
globalVar['outPath'] = '/DATA/OUTPUT'
globalVar['figPath'] = '/DATA/FIG'

SEED = 47
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)

station = "yd"

inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'datasets/yd.csv')
fileList = sorted(glob.glob(inpFile))

# F_name = str(os.getcwd())+'\\datasets\\'+station+".csv"
F_name = fileList[0]

df = pd.read_csv(inpFile)
df['date'] = pd.to_datetime(df['date'])
df[['rf', 'inflow', 'etp']].plot(subplots=True)
plt.show()

year = 24 * 60 * 60 * 365.2425
timestamp = df['date'].apply(datetime.datetime.timestamp)
df['Year sin'] = np.sin(timestamp * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp * (2 * np.pi / year))
df['date'].dt.year.unique()

##Training (2002-2015)
train_df = df.loc[df['date'].dt.year.isin(range(2002, 2015 + 1)), ['rf', 'inflow', 'etp', 'Year sin', 'Year cos']]
##Validation (2016-2021)
val_df = df.loc[df['date'].dt.year.isin(range(2016, 2021 + 1)), ['rf', 'inflow', 'etp', 'Year sin', 'Year cos']]
# test ==전체
test_df = df.loc[df['date'].dt.year.isin(range(2002, 2021 + 1)), ['rf', 'inflow', 'etp', 'Year sin', 'Year cos']]
# jnl--assert len(df) == len(train_df) + len(val_df) + len(test_df)
num_features = train_df.shape[1]

train_mean = df[['rf', 'inflow', 'etp']].mean()
train_std = df[['rf', 'inflow', 'etp']].std()
train_df[['rf', 'inflow', 'etp']] = (train_df[['rf', 'inflow', 'etp']] - train_mean) / train_std
val_df[['rf', 'inflow', 'etp']] = (val_df[['rf', 'inflow', 'etp']] - train_mean) / train_std
test_df[['rf', 'inflow', 'etp']] = (test_df[['rf', 'inflow', 'etp']] - train_mean) / train_std

df_std = (df[['rf', 'inflow', 'etp']] - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(['rf', 'inflow', 'etp'], rotation=90)
plt.show()

features = ['Year sin', 'Year cos', 'rf', 'etp']
labels = ['inflow']

##LSTM#################
# for k in range(500):
# unit_num=random.randint(0,300)
# sequence_length =random.randint(17,22)

#####여기서 값 변화는건 87/75 만 변화,,,,
##sy=>
unit_num = 87
sequence_length = 21

##yd=>
# unit_num=75
# sequence_length =21
##87 and 75 units of the SY and YD Dams 


dataset_train = make_dataset(train_df, features, labels, sequence_length, batch_size=64)
dataset_val = make_dataset(val_df, features, labels, sequence_length, batch_size=64)
dataset_test = make_dataset(test_df, features, labels, sequence_length, batch_size=1, test=True)

for batch in dataset_train.take(1):
    inputs, targets = batch
    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)

tf.random.set_seed(SEED)
inputs = Input(shape=(inputs.shape[1], inputs.shape[2]))
x = inputs
x = LSTM(unit_num)(x)
x = Dense(1)(x)
y = x
outputs = y
opt = Adam(learning_rate=0.001)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=opt, loss='mse')
model.summary()

# path_checkpoint = "model/model_checkpoint.h5"
es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=1)

# AI 모델링 (훈련 데이터 : dataset_train, 검증 데이터 : validation_data)
history = model.fit(dataset_train, epochs=500, validation_data=dataset_val, callbacks=[es_callback, tensorboard_callback], )

visualize_loss(history, "Training and Validation Loss")

##요거 잘못된거가? 예측에 dataset_test만 들어간거 같은데....
# 테스트 데이터 (dataset_test)를 이용한 전체 기간 (2002~2021) 예측
# 테스트 데이터의 경우 하단에 sy, et에 따라 분리 (227행, 223행)하여 검증스코어 계산
predict = model.predict(dataset_test).flatten()
obs = np.concatenate([y for x, y in dataset_test]).flatten()
result_df = pd.DataFrame({'obs': obs, 'predict': predict})
result_df = result_df * train_std['inflow'] + train_mean['inflow']
result_df.rename(columns={'obs': 'obs_cms', 'predict': 'sim_cms'}, inplace=True)
result_df
###=>>>>>one하고 출력값 동일하게 맞추기
###>>>>cms->mm변환하기


##date handling==데이터 시프트되는거 반영
date1 = datetime.datetime(2002, 1, 1)
date_delta = timedelta(days=sequence_length)
date2 = date1 + date_delta
print(date2)
date = pd.date_range(start=date2, end='2021-12-31', freq='D')
result_df['date'] = date
print(result_df)
result_df = result_df[['date', 'obs_cms', 'sim_cms']]
sht_result_df = result_df.shift(periods=sequence_length)
# sht_result_df.to_excel("D:/OneDrive/ONE_model/results/LSTM_output_" + station + ".xlsx")
# df.to_excel("D:/OneDrive/ONE_model/lstm-raw_output_"+station+".xlsx")---보관

##Train/Val
st = len(train_df)
et = len(val_df)

##Train
NSE, RMSE, R2, PBIAS = result_estimate(result_df[:et - 1]['obs_cms'], result_df[:et - 1]['sim_cms'])
# print('Train_period:', 'NSE:',round(NSE,2), 'RMSE:',round(RMSE,2), 'R2:',round(R2,2), 'PBIAS:',round(PBIAS,2))
train_text = str('train_period'), round(NSE, 2), round(RMSE, 2), round(R2, 2), round(PBIAS, 2)
print(train_text)

##Val
NSE, RMSE, R2, PBIAS = result_estimate(result_df[et:]['obs_cms'], result_df[et:]['sim_cms'])
# print('valid_period:', 'NSE:',round(NSE,2), 'RMSE:',round(RMSE,2), 'R2:',round(R2,2), 'PBIAS:',round(PBIAS,2))
val_text = str('valid_period'), round(NSE, 2), round(RMSE, 2), round(R2, 2), round(PBIAS, 2)
print(val_text)

##그래프##
stt = '2002-01-01'
ett = '2021-12-31'
fig, (ax1) = plt.subplots(1, 1, figsize=(20, 4))
plt.title("lenth: " + str(sequence_length) + "  unit : " + str(unit_num))
plt.xlabel("Date(day)")
plt.ylabel(r'Runoff ($m^3$/s)')
ax1.semilogy(result_df['date'], result_df['obs_cms'], label='Obs value ($m^3$/s)', linewidth=0.7, color='grey')
ax1.semilogy(result_df['date'], result_df['sim_cms'], label='Sim value ($m^3$/s)', linewidth=0.7, color='k')
plt.text(pd.to_datetime('2005-01-01', format='%Y-%m-%d'), pow(10, 4.5), '                       NSE, RMSE, R2, PBIAS', fontsize=14)
plt.text(pd.to_datetime('2005-01-01', format='%Y-%m-%d'), pow(10, 4.0), train_text, fontsize=14)
plt.text(pd.to_datetime('2005-01-01', format='%Y-%m-%d'), pow(10, 3.5), val_text, fontsize=14)
plt.xlim([pd.to_datetime(stt, format='%Y-%m-%d'), pd.to_datetime(ett, format='%Y-%m-%d')])
plt.legend(loc='upper left')
plt.ylim(pow(10, -1), pow(10, 5))
plt.tight_layout()
# plt.savefig('D:/OneDrive/ONE_model/fig/' + str(NSE) + '_' + str(R2) + '_.jpg')
# plt.grid()
plt.show()

##Training (2002-2015)
##Validation (2016-2021)
