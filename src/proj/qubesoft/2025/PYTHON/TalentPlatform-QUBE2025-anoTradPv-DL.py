# ================================================
# 요구사항
# ================================================
# Python을 이용한 기상청 데이터 모델링 및 DB 적재

# 프로그램 종료
# ps -ef | grep python | grep TalentPlatform-QUBE2025-anoObs.py | awk '{print $2}' | xargs kill -9
# pkill -f TalentPlatform-QUBE2025-anoObs.py

# 프로그램 시작
# conda activate py38

# cd /SYSTEMS/PROG/PYTHON
# /SYSTEMS/LIB/anaconda3/envs/py39/bin/python TalentPlatform-QUBE2025-anoObs.py
# nohup /SYSTEMS/LIB/anaconda3/envs/py39/bin/python TalentPlatform-QUBE2025-anoObs.py &

import glob
# import seaborn as sns
import logging
import logging.handlers
import logging.handlers
import os
import platform
import sys
import argparse
import traceback
import warnings
# import datetime as dt
# from datetime import datetime
# import pvlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# import pyproj
# import xarray as xr
# from scipy.stats import linregress
import pandas as pd
# import cartopy.crs as ccrs
# import math
# from scipy import spatial
# from pandas.tseries.offsets import Day, Hour, Minute, Second
# from scipy.interpolate import Rbf
# from numpy import zeros, newaxis

# import pygrib
# import haversine as hs
import pytz
import datetime
# import h2o
# from pycaret.regression import *
# from sqlalchemy import create_engine
# import re
import configparser
# import sqlalchemy
# from sqlalchemy.ext.declarative import declarative_base
# import random
from urllib.parse import quote_plus
from urllib.parse import unquote_plus
import urllib.parse
# import sqlalchemy
# from sqlalchemy import create_engine, text
# import requests
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.automap import automap_base
# from sqlalchemy import text
import sqlalchemy
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy import create_engine, text
# import requests
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import text
# from pvlib import location
# from pvlib import irradiance
# from multiprocessing import Pool
# import multiprocessing as mp
# import uuid
import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import pickle
from flaml import AutoML
# from sklearn.model_selection import train_test_split
# from pycaret.regression import *
# import pvlib
import h2o
from h2o.automl import H2OAutoML
import uuid
from sklearn.model_selection import train_test_split
# from pycaret.regression import RegressionExperiment

# from pycaret.anomaly import AnomalyExperiment
# import plotly.express as px
# import plotly.graph_objects as go

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# from deepod.models.time_series import TimesNet
from sklearn.preprocessing import StandardScaler
# from deepod.models.time_series import TimesNet
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.models import LightGBMModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import pvlib
from pvlib import location
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =================================================
# 사용자 매뉴얼
# =================================================
# [소스 코드의 실행 순서]
# 1. 초기 설정 : 폰트 설정
# 2. 유틸리티 함수 : 초기화 함수 (로그 설정, 초기 변수, 초기 전달인자 설정) 또는 자주 사용하는 함수
# 3. 주 프로그램 :부 프로그램을 호출
# 4. 부 프로그램 : 자료 처리를 위한 클래스로서 내부 함수 (초기 변수, 비즈니스 로직, 수행 프로그램 설정)
# 4.1. 환경 변수 설정 (로그 설정) : 로그 기록을 위한 설정 정보 읽기
# 4.2. 환경 변수 설정 (초기 변수) : 입력 경로 (inpPath) 및 출력 경로 (outPath) 등을 설정
# 4.3. 초기 변수 (Argument, Option) 설정 : 파이썬 실행 시 전달인자 설정 (pyhton3 *.py argv1 argv2 argv3 ...)
# 4.4. 비즈니스 로직 수행 : 단위 시스템 (unit 파일명)으로 관리 또는 비즈니스 로직 구현

# =================================================
# 1. 초기 설정
# =================================================
warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')
dtKst = datetime.timedelta(hours=9)


# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
    )

    os.makedirs(os.path.dirname(saveLogFile), exist_ok=True)

    # logger instance 생성
    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    # format 생성
    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    # handler 생성
    streamHandler = logging.StreamHandler()
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1, backupCount=30, encoding='utf-8')

    # logger instance에 format 설정
    streamHandler.setFormatter(format)
    fileHandler.setFormatter(format)

    # logger instance에 handler 설정
    log.addHandler(streamHandler)
    log.addHandler(fileHandler)

    # logger instance로 log 기록
    log.setLevel(level=logging.INFO)

    return log


#  초기 변수 설정
def initGlobalVar(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    # 환경 변수 (local, 그 외)에 따라 전역 변수 (입력 자료, 출력 자료 등)를 동적으로 설정
    # 즉 local의 경우 현재 작업 경로 (contextPath)를 기준으로 설정
    # 그 외의 경우 contextPath/resources/input/prjName와 같은 동적으로 구성
    globalVar = {
        'prjName': prjName
        , 'sysOs': platform.system()
        , 'contextPath': contextPath
        , 'resPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'output', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    return globalVar


#  초기 전달인자 설정
def initArgument(globalVar, inParams):
    # 원도우 또는 맥 환경
    if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
        inParInfo = inParams

    # 리눅스 환경
    if globalVar['sysOs'] in 'Linux':
        parser = argparse.ArgumentParser()

        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)

        inParInfo = vars(parser.parse_args())

    log.info("[CHECK] inParInfo : {}".format(inParInfo))

    for key, val in inParInfo.items():
        if val is None: continue
        # 전역 변수에 할당
        globalVar[key] = val

    # 전역 변수
    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)

        globalVar[key] = val.replace('\\', '/')

        log.info("[CHECK] {} : {}".format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar

def initCfgInfo(config, key):

    result = None

    try:
        # log.info(f'[CHECK] key : {key}')
        dbUser = config.get(key, 'user')
        dbPwd = urllib.parse.quote(config.get(key, 'pwd'))
        dbHost = config.get(key, 'host')
        dbPort = config.get(key, 'port')
        dbName = config.get(key, 'dbName')

        engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}", echo=False, pool_timeout=60*5, pool_recycle=3600)
        sessionMake = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        # session = sessionMake()

        base = automap_base()
        base.prepare(autoload_with=engine)
        tableList = base.classes.keys()

        result = {
            'engine': engine
            , 'sessionMake': sessionMake
            # , 'session': session
            , 'tableList': tableList
            , 'tableCls': base.classes
        }

        return result

    except Exception as e:
        log.error(f'Exception : {e}')
        return result

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'C:/SYSTEMS/PROG/PYTHON/TalentPlatform-Python'
    else:
        # contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON'

    prjName = 'anoTradPv'
    serviceName = 'QUBE2025'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info("[START] __init__ : {}".format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info("[END] __init__ : {}".format("init"))

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info('[START] {}'.format("exec"))

        try:
            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/HDD/DATA/INPUT'
                globalVar['outPath'] = '/HDD/DATA/OUTPUT'
                globalVar['figPath'] = '/HDD/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                # 'srtDate': globalVar['srtDate'],
                # 'endDate': globalVar['endDate'],
                # 'srtDate': '2021-01-01',
                # 'endDate': '2025-11-01',

                # 비동기 다중 프로세스 개수
                # 'cpuCoreNum': globalVar['cpuCoreNum'],
                # 'cpuCoreNum': '5',

                # 설정 파일
                'cfgFile': '/SYSTEMS/PROG/PYTHON/TalentPlatform-Python/resources/config/system.cfg',
                # 'cfgFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
                # 'cfgFile': '/vol01/SYSTEMS/INDIAI/PROG/PYTHON/resources/config/system.cfg',
                # 'cfgFile': '/SYSTEMS/PROG/PYTHON/resources/config/system.cfg',
                'cfgDbKey': 'postgresql-qubesoft.iptime.org-qubesoft-dms02',
                'cfgDb': None,
                'posDataL1': None,

                'timesNet': {
                    'saveModelList': "/DATA/AI/*/*/QUBE2025-{srv}-final-timesNet-anoObs-*.pkl",
                    'saveModel': "/DATA/AI/%Y%m/%d/QUBE2025-{srv}-final-timesNet-anoObs-%Y%m%d.pkl",
                    # 'isOverWrite': True,
                    'isOverWrite': False,
                    'srv': None,
                    'preDt': datetime.datetime.now(),
                },
            }

            # *******************************************************
            # 설정 정보
            # *******************************************************
            config = configparser.ConfigParser()
            config.read(sysOpt['cfgFile'], encoding='utf-8')

            # sysOpt['cfgDb'] = initCfgInfo(config, sysOpt['cfgDbKey'])
            cfgDb = initCfgInfo(config, sysOpt['cfgDbKey'])

            # 관측소 정보
            with cfgDb['sessionMake']() as session:
                query = text("""
                             SELECT *, 'SRV' || LPAD(id::text, 5, '0') as srv
                             FROM tb_stn_info
                             WHERE oper_yn = 'Y'
                             ORDER BY id ASC;
                             """)

                posDataL1 = pd.DataFrame(session.execute(query))

            for i, posInfo in posDataL1.iterrows():
                with cfgDb['sessionMake']() as session:
                    try:
                        srv = posInfo['srv']
                        # srv = 'SRV00009'
                        # query = text("""
                        #     SELECT "srv", "date_time", "date_time_kst", "trad", "srad", "otemp", "ptemp"
                        #     FROM "tb_obs_data"
                        #     WHERE "srv" = :srv
                        #     ORDER BY "srv", "date_time_kst" DESC;
                        #  """)
                        query = text("""
                                     SELECT pv.srv,
                                            pv.date_time,
                                            pv.date_time_kst,
                                            pv.pv,
                                            AVG(obs.trad)  AS trad,
                                            AVG(obs.srad)  AS srad,
                                            AVG(obs.otemp) AS otemp,
                                            AVG(obs.ptemp) AS ptemp
                                     FROM tb_pv_data pv
                                              LEFT JOIN tb_obs_data obs
                                                        ON pv.srv = obs.srv
                                                            AND obs.date_time_kst >= pv.date_time_kst - INTERVAL '5 minutes'
                                         AND obs.date_time_kst <= pv.date_time_kst + INTERVAL '5 minutes'
                                     WHERE pv.srv = :srv
                                     GROUP BY
                                         pv.srv,
                                         pv.date_time_kst,
                                         pv.date_time,
                                         pv.pv
                                     ORDER BY
                                         pv.date_time_kst ASC;
                                     """)

                        data = pd.DataFrame(session.execute(query, {'srv':srv}))
                        if len(data) < 1: continue

                        # dataL1 = data[(data['date_time_kst'].dt.hour >= 6) & (data['date_time_kst'].dt.hour <= 20)].reset_index(drop=True)
                        # dataL1 = data

                        # 2. [Darts 단계] TimeSeries 객체 생성 및 결측치 보간
                        df = data.dropna().reset_index(drop=True)
                        if len(df) < 1: continue

                        lat = posInfo['lat']
                        lon = posInfo['lon']

                        solPosInfo = pvlib.solarposition.get_solarposition(df['date_time'], lat, lon, method='nrel_numpy')
                        df['ext_rad'] = pvlib.irradiance.get_extra_radiation(solPosInfo.index.dayofyear)
                        df['sza'] = solPosInfo['zenith'].values
                        df['aza'] = solPosInfo['azimuth'].values
                        df['et'] = solPosInfo['equation_of_time'].values
                        site = location.Location(latitude=lat, longitude=lon, tz='Asia/Seoul')
                        clearInsInfo = site.get_clearsky(pd.to_datetime(df['date_time'].values))
                        df['ghi_clr'] = clearInsInfo['ghi'].values
                        df['dni_clr'] = clearInsInfo['dni'].values
                        df['dhi_clr'] = clearInsInfo['dhi'].values
                        turbidity = pvlib.clearsky.lookup_linke_turbidity(pd.to_datetime(df['date_time'].values), lat, lon, interp_turbidity=True)
                        df['turb'] = turbidity.values

                        # dataframe을 Darts 전용 시계열 객체로 변환합니다.
                        ts_pv = TimeSeries.from_dataframe(df, time_col='date_time_kst', value_cols='pv', fill_missing_dates=True, freq='1h')

                        # 데이터 스케일링을 위한 라이브러리 추가
                        from darts.dataprocessing.transformers import Scaler
                        from darts.models import BlockRNNModel, TCNModel

                        # Darts 내장 함수를 사용하여 NaN으로 뚫어놓은 센서 고장 구간을 앞뒤 데이터를 통해 선형 보간
                        ts_pv_filled = fill_missing_values(ts_pv)

                        # 딥러닝 모델은 데이터 스케일에 민감하므로 0~1 사이로 정규화
                        # (Data Leakage 방지를 위해 먼저 분할 후 scaler 학습)
                        # train_pv_raw, test_pv_raw = ts_pv_filled.split_before(0.8)
                        train_pv_raw, test_pv_raw = ts_pv_filled.split_before(pd.Timestamp('2026-08-23'))
                        if len(train_pv_raw) < 1: continue
                        if len(test_pv_raw) < 1: continue

                        scaler_pv = Scaler()
                        train_pv = scaler_pv.fit_transform(train_pv_raw)
                        test_pv = scaler_pv.transform(test_pv_raw)

                        # 3가지 일사량 변수 조합 정의
                        cov_configs = {
                            'ai_pv_srad': ['srad', 'otemp', 'ext_rad', 'sza', 'aza', 'et', 'ghi_clr', 'dni_clr', 'dhi_clr', 'turb'],
                            'ai_pv_trad': ['trad', 'ptemp', 'ext_rad', 'sza', 'aza', 'et', 'ghi_clr', 'dni_clr', 'dhi_clr', 'turb'],
                            'ai_pv_srad_trad': ['srad', 'trad', 'otemp', 'ptemp', 'ext_rad', 'sza', 'aza', 'et', 'ghi_clr', 'dni_clr', 'dhi_clr', 'turb']
                        }

                        print("=" * 70)
                        print(f"[{srv}] 딥러닝(LSTM/CNN) 성능 평가 시작")
                        print("=" * 70)

                        df_result = None
                        plot_data = {'model_1h': {}, 'model_6h': {}, 'model_cnnlstm': {}}

                        # 모델이 저장될 디렉토리 정의 및 생성
                        model_dir = os.path.join(globalVar['outPath'], 'models')
                        os.makedirs(model_dir, exist_ok=True)

                        for case_name, cov_cols in cov_configs.items():
                            # 현재 조합에 대한 공변량(Covariates) 시계열 생성 및 보간
                            ts_cov = TimeSeries.from_dataframe(df, time_col='date_time_kst', value_cols=cov_cols, fill_missing_dates=True, freq='1h')
                            ts_cov_filled = fill_missing_values(ts_cov)

                            # train_cov_raw, test_cov_raw = ts_cov_filled.split_before(0.8)
                            train_cov_raw, test_cov_raw = ts_cov_filled.split_before(pd.Timestamp('2026-08-23'))


                            scaler_cov = Scaler()
                            train_cov = scaler_cov.fit_transform(train_cov_raw)
                            # 시계열 예측 시 전체 과거 데이터가 필요하므로 합쳐진 공변량도 스케일링해둡니다.
                            ts_cov_scaled = scaler_cov.transform(ts_cov_filled)

                            # [옵션 1] 딥러닝 LSTM 적용 (과거 24시간 참조 -> 1시간 예측)
                            model_path_1h = os.path.join(model_dir, f"QUBE2025_{srv}_{case_name}_LSTM_24h_1h.pt")
                            # if os.path.exists(model_path_1h):
                            #     model_1h = BlockRNNModel.load(model_path_1h)
                            #     print(f"[{srv}] {case_name} LSTM 로드됨: {model_path_1h}")
                            # else:
                            model_1h = BlockRNNModel(
                                model="LSTM",
                                input_chunk_length=24,
                                output_chunk_length=1,
                                n_epochs=1,
                                # n_epochs=15,    # 딥러닝 에폭 설정
                                random_state=42
                            )
                            # 과거 공변량(past_covariates)으로 날씨 피처 사용
                            model_1h.fit(series=train_pv, past_covariates=train_cov)
                            model_1h.save(model_path_1h)
                            print(f"[{srv}] {case_name} LSTM 저장됨: {model_path_1h}")

                            pred_pv_1h_scaled = model_1h.predict(n=len(test_pv), past_covariates=ts_cov_scaled)
                            # 평가를 위해 예측값을 원래 스케일로 역변환(Inverse Transform)
                            pred_pv_1h = scaler_pv.inverse_transform(pred_pv_1h_scaled)

                            df_1h = test_pv_raw.to_dataframe().rename(columns={'pv': 'actual_pv'})
                            df_1h['expected_pv'] = pred_pv_1h.to_dataframe()['pv']
                            corr_1h = df_1h['actual_pv'].corr(df_1h['expected_pv'])
                            rmse_1h = np.sqrt(mean_squared_error(df_1h['actual_pv'], df_1h['expected_pv']))
                            plot_data['model_1h'][case_name] = df_1h
                            a = plot_data['model_1h']

                            # [옵션 2] 딥러닝 CNN(TCN) 적용 (과거 24시간 참조 -> 1시간 예측)
                            model_path_6h = os.path.join(model_dir, f"QUBE2025_{srv}_{case_name}_CNN_24h_1h.pt")
                            # if os.path.exists(model_path_6h):
                            #     model_6h = TCNModel.load(model_path_6h)
                            #     print(f"[{srv}] {case_name} CNN 로드됨: {model_path_6h}")
                            # else:
                            model_6h = TCNModel(
                                input_chunk_length=24,
                                output_chunk_length=1,
                                n_epochs=1,
                                # n_epochs=15,
                                random_state=42
                            )
                            model_6h.fit(series=train_pv, past_covariates=train_cov)
                            model_6h.save(model_path_6h)
                            print(f"[{srv}] {case_name} CNN 저장됨: {model_path_6h}")

                            pred_pv_6h_scaled = model_6h.predict(n=len(test_pv), past_covariates=ts_cov_scaled)
                            # 평가를 위해 예측값을 원래 스케일로 역변환(Inverse Transform)
                            pred_pv_6h = scaler_pv.inverse_transform(pred_pv_6h_scaled)

                            df_6h = test_pv_raw.to_dataframe().rename(columns={'pv': 'actual_pv'})
                            df_6h['expected_pv'] = pred_pv_6h.to_dataframe()['pv']
                            corr_6h = df_6h['actual_pv'].corr(df_6h['expected_pv'])
                            rmse_6h = np.sqrt(mean_squared_error(df_6h['actual_pv'], df_6h['expected_pv']))
                            plot_data['model_6h'][case_name] = df_6h

                            # [옵션 3] 딥러닝 CNN-LSTM 적용 (과거 24시간 참조 -> 1시간 예측)
                            model_path_cnnlstm = os.path.join(model_dir, f"QUBE2025_{srv}_{case_name}_CNNLSTM_24h_1h.pt")

                            input_dim = 1 + len(cov_cols)

                            class CNN_LSTM(nn.Module):
                                def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
                                    super(CNN_LSTM, self).__init__()
                                    self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
                                    self.relu = nn.ReLU()
                                    self.pool = nn.MaxPool1d(kernel_size=2)
                                    self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
                                    self.fc1 = nn.Linear(hidden_dim, 32)
                                    self.fc2 = nn.Linear(32, output_dim)

                                def forward(self, x):
                                    x = x.transpose(1, 2)
                                    x = self.conv1(x)
                                    x = self.relu(x)
                                    x = self.pool(x)
                                    x = x.transpose(1, 2)
                                    out, _ = self.lstm(x)
                                    out = out[:, -1, :]
                                    out = self.relu(self.fc1(out))
                                    out = self.fc2(out)
                                    return out

                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            cnn_lstm_model = CNN_LSTM(input_dim=input_dim).to(device)

                            if os.path.exists(model_path_cnnlstm):
                                cnn_lstm_model.load_state_dict(torch.load(model_path_cnnlstm, map_location=device))
                                print(f"[{srv}] {case_name} CNN-LSTM 로드됨: {model_path_cnnlstm}")
                            else:
                                seq_length = 24
                                out_length = 1
                                pv_arr = train_pv.values()
                                cov_arr = train_cov.values()
                                X_train, y_train = [], []
                                for idx in range(len(pv_arr) - seq_length - out_length + 1):
                                    X_train.append(np.concatenate([pv_arr[idx:idx+seq_length], cov_arr[idx:idx+seq_length]], axis=1))
                                    y_train.append(pv_arr[idx+seq_length:idx+seq_length+out_length, 0])

                                X_train_t = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
                                y_train_t = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)

                                dataset = TensorDataset(X_train_t, y_train_t)
                                dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

                                criterion = nn.MSELoss()
                                optimizer = torch.optim.Adam(cnn_lstm_model.parameters(), lr=0.001)

                                cnn_lstm_model.train()
                                for epoch in range(25):
                                    for batch_x, batch_y in dataloader:
                                        optimizer.zero_grad()
                                        out = cnn_lstm_model(batch_x)
                                        loss = criterion(out, batch_y)
                                        loss.backward()
                                        optimizer.step()

                                torch.save(cnn_lstm_model.state_dict(), model_path_cnnlstm)
                                print(f"[{srv}] {case_name} CNN-LSTM 저장됨: {model_path_cnnlstm}")

                            cnn_lstm_model.eval()
                            preds = []
                            history_pv = train_pv.values()[-24:]
                            history_cov_idx = len(train_pv) - 24
                            cov_full_vals = ts_cov_scaled.values()

                            for idx in range(len(test_pv)):
                                current_cov = cov_full_vals[history_cov_idx + idx : history_cov_idx + idx + 24]
                                x_input = np.concatenate([history_pv, current_cov], axis=1)
                                x_input_t = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)

                                with torch.no_grad():
                                    pred = cnn_lstm_model(x_input_t).item()

                                preds.append([pred])
                                history_pv = np.append(history_pv[1:], [[pred]], axis=0)

                            preds_ts = TimeSeries.from_times_and_values(test_pv.time_index, np.array(preds))
                            preds_scaled = scaler_pv.inverse_transform(preds_ts)

                            df_cnnlstm = test_pv_raw.to_dataframe().rename(columns={'pv': 'actual_pv'})
                            df_cnnlstm['expected_pv'] = preds_scaled.values()[:, 0]
                            corr_cnnlstm = df_cnnlstm['actual_pv'].corr(df_cnnlstm['expected_pv'])
                            rmse_cnnlstm = np.sqrt(mean_squared_error(df_cnnlstm['actual_pv'], df_cnnlstm['expected_pv']))
                            plot_data['model_cnnlstm'][case_name] = df_cnnlstm

                            # 결과 출력
                            print(f"[{case_name}]")
                            print(f"  - LSTM (1h)      -> Corr: {corr_1h:.4f}, RMSE: {rmse_1h:.4f}")
                            print(f"  - CNN (1h)       -> Corr: {corr_6h:.4f}, RMSE: {rmse_6h:.4f}")
                            print(f"  - CNN-LSTM (1h)  -> Corr: {corr_cnnlstm:.4f}, RMSE: {rmse_cnnlstm:.4f}\n")

                            # 기존 DB 적재 로직과 호환되도록 가장 성능이 좋은 ai_pv_trad의 결과를 df_result로 저장
                            if case_name == 'ai_pv_trad':
                                df_result = df_cnnlstm  # 기준을 CNN-LSTM 모델로 설정
                                df_result['ai_pv_trad'] = df_result['expected_pv']
                                df_result['error'] = df_result['actual_pv'] - df_result['expected_pv']

                        # --- [추가] 통합 산점도 시각화 및 저장 ---
                        fig_dir = os.path.join(globalVar['figPath'], 'validation')
                        os.makedirs(fig_dir, exist_ok=True)

                        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
                        fig.suptitle(f'[{srv}] 실제 발전량 vs 딥러닝(LSTM/CNN/CNN-LSTM) 예측 발전량 시뮬레이션 검증', fontsize=20, fontweight='bold')

                        model_labels = {'model_1h': 'LSTM', 'model_6h': 'CNN', 'model_cnnlstm': 'CNN-LSTM'}
                        for i, model_type in enumerate(['model_1h', 'model_6h', 'model_cnnlstm']):
                            for j, case_name in enumerate(cov_configs.keys()):
                                df_plot = plot_data[model_type][case_name]
                                ax = axes[i, j]

                                # 실제값 vs 예측값 산점도
                                ax.scatter(df_plot['actual_pv'], df_plot['expected_pv'], alpha=0.6, edgecolors='w', linewidth=0.5, label='Predicted')

                                # 이상적인 기준선 (y=x)
                                min_val = min(df_plot['actual_pv'].min(), df_plot['expected_pv'].min())
                                max_val = max(df_plot['actual_pv'].max(), df_plot['expected_pv'].max())

                                # 만약 값이 비정상일 경우를 대비
                                if pd.isna(min_val) or pd.isna(max_val):
                                    min_val, max_val = 0, 100

                                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')

                                # 상관계수 및 RMSE 추출
                                corr = df_plot['actual_pv'].corr(df_plot['expected_pv'])
                                rmse = np.sqrt(mean_squared_error(df_plot['actual_pv'], df_plot['expected_pv']))

                                model_label = model_labels[model_type]
                                ax.set_title(f"{case_name} ({model_label})\nCorr: {corr:.3f}, RMSE: {rmse:.3f}", fontsize=14)
                                ax.set_xlabel('Actual PV', fontsize=12)
                                ax.set_ylabel('Expected PV', fontsize=12)
                                ax.grid(True, linestyle=':', alpha=0.7)
                                ax.legend()

                        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitle이 겹치지 않게 여백 조정

                        save_fig_path = os.path.join(fig_dir, f"QUBE2025_{srv}_DL_scatter_validation.png")
                        plt.savefig(save_fig_path, dpi=300)
                        plt.close()
                        print(f"[{srv}] 통합 산점도 저장 완료: {save_fig_path}")

                        # --- [추가] 통합 시계열 그래프 시각화 및 저장 ---
                        fig_ts, axes_ts = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
                        fig_ts.suptitle(f'[{srv}] 실제 발전량 vs 딥러닝(LSTM/CNN/CNN-LSTM) 예측 발전량 시계열 비교', fontsize=20, fontweight='bold')

                        model_labels = {'model_1h': 'LSTM', 'model_6h': 'CNN', 'model_cnnlstm': 'CNN-LSTM'}
                        for i, model_type in enumerate(['model_1h', 'model_6h', 'model_cnnlstm']):
                            for j, case_name in enumerate(cov_configs.keys()):
                                df_plot = plot_data[model_type][case_name]
                                ax_ts = axes_ts[i, j]

                                # 시계열 선 그래프 (실측치 vs 예측치)
                                ax_ts.plot(df_plot.index, df_plot['actual_pv'], label='Actual PV', color='#1f77b4', linewidth=1.5, alpha=0.8)
                                ax_ts.plot(df_plot.index, df_plot['expected_pv'], label='Expected PV', color='#ff7f0e', linewidth=1.5, alpha=0.8)

                                # 상관계수 및 RMSE 추출 (타이틀용)
                                corr = df_plot['actual_pv'].corr(df_plot['expected_pv'])
                                rmse = np.sqrt(mean_squared_error(df_plot['actual_pv'], df_plot['expected_pv']))

                                model_label = model_labels[model_type]
                                ax_ts.set_title(f"{case_name} ({model_label})\nCorr: {corr:.3f}, RMSE: {rmse:.3f}", fontsize=14)
                                ax_ts.set_xlabel('Time', fontsize=12)
                                ax_ts.set_ylabel('PV', fontsize=12)
                                ax_ts.grid(True, linestyle=':', alpha=0.7)
                                ax_ts.legend(loc='upper right')
                                ax_ts.tick_params(axis='x', rotation=30) # X축 날짜 라벨 겹침 방지

                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                        save_ts_path = os.path.join(fig_dir, f"QUBE2025_{srv}_DL_timeseries_validation.png")
                        plt.savefig(save_ts_path, dpi=300)
                        plt.close()
                        print(f"[{srv}] 통합 시계열 그래프 저장 완료: {save_ts_path}")
                    except Exception as e:
                        log.error("Exception : {}".format(e))
        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e

        finally:
            log.info('[END] {}'.format("exec"))

# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print('[START] {}'.format("main"))

    try:

        # 파이썬 실행 시 전달인자를 초기 환경변수 설정
        inParams = {}

        print("[CHECK] inParams : {}".format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
