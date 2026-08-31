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
                    # srv = posInfo['srv']
                    srv = 'SRV00009'
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
                    dataL1 = data

                    from darts import TimeSeries
                    from darts.utils.missing_values import fill_missing_values
                    from darts.models import LightGBMModel
                    from darts.utils.timeseries_generation import datetime_attribute_timeseries
                    import pvlib
                    from pvlib import location
                    from sklearn.metrics import mean_squared_error


                    # df = data
                    df = data.dropna().reset_index(drop=True)
                    # 2. [Darts 단계] TimeSeries 객체 생성 및 결측치 보간

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
                    # ts_cov = TimeSeries.from_dataframe(df, time_col='date_time_kst',value_cols=['trad', 'otemp', 'ptemp'], fill_missing_dates=True, freq='1h')
                    # ts_cov = TimeSeries.from_dataframe(df, time_col='date_time_kst',value_cols=['srad', 'otemp', 'ptemp'], fill_missing_dates=True, freq='1h')
                    # ts_cov = TimeSeries.from_dataframe(df, time_col='date_time_kst',value_cols=['trad', 'srad', 'otemp', 'ptemp'], fill_missing_dates=True, freq='1h')

                    ts_cov = TimeSeries.from_dataframe(df, time_col='date_time_kst',value_cols=['srad', 'otemp', 'ext_rad', 'sza', 'aza', 'et', 'ghi_clr', 'dni_clr', 'dhi_clr', 'turb'], fill_missing_dates=True, freq='1h')
                    # ts_cov = TimeSeries.from_dataframe(df, time_col='date_time_kst',value_cols=['trad', 'ptemp', 'ext_rad', 'sza', 'aza', 'et', 'ghi_clr', 'dni_clr', 'dhi_clr', 'turb'], fill_missing_dates=True, freq='1h')
                    # ts_cov = TimeSeries.from_dataframe(df, time_col='date_time_kst',value_cols=['srad', 'trad', 'otemp', 'ptemp', 'ext_rad', 'sza', 'aza', 'et', 'ghi_clr', 'dni_clr', 'dhi_clr', 'turb'], fill_missing_dates=True, freq='1h')

                    # Darts 내장 함수를 사용하여 NaN으로 뚫어놓은 센서 고장 구간을 앞뒤 데이터를 통해 선형 보간합니다.
                    ts_pv_filled = fill_missing_values(ts_pv)
                    ts_cov_filled = fill_missing_values(ts_cov)

                    # (학습용과 테스트용 데이터 분리 - 예: 마지막 7일을 테스트로 사용)
                    train_pv, test_pv = ts_pv_filled.split_before(pd.Timestamp('2026-07-25'))
                    train_cov, test_cov = ts_cov_filled.split_before(pd.Timestamp('2026-07-25'))

                    # 3. [Darts 단계] 기대 발전량(Expected PV) 예측 모델 학습
                    # 태양광 발전은 '현재 시점'의 일사량과 온도에 즉각적으로 반응하므로
                    # lags_future_covariates=[0] 을 사용하여 "t 시점의 날씨로 t 시점의 발전을 예측"하도록 설정합니다.
                    model = LightGBMModel(
                        lags=1,
                        lags_future_covariates=[0],
                        output_chunk_length=1,
                        random_state=42
                    )
                    # model = LightGBMModel(
                    #     lags=48,
                    #     lags_future_covariates=[0],
                    #     output_chunk_length=48,
                    #     random_state=42
                    # )

                    # model = LightGBMModel(
                    #     lags=48,  # 직전(t-1) 시점의 발전량 참고
                    #     lags_future_covariates=[0],  # 현재(t) 시점의 정제된 기상 데이터(trad, ptemp) 참고
                    #     output_chunk_length=48,
                    #     random_state=42
                    # )

                    # 정상적인 데이터 구간을 통해 발전소의 '정상 패턴' 학습
                    model.fit(series=train_pv, future_covariates=train_cov)

                    # 4. [Darts 단계] 테스트 데이터 구간의 기대 발전량 예측
                    pred_pv = model.predict(n=len(test_pv), future_covariates=test_cov)

                    # 5. [분석 단계] 오차 방향을 고려한 이상감지 룰 적용
                    # 예측 결과와 실제 데이터를 다시 Pandas DataFrame으로 합쳐서 비즈니스 룰을 적용합니다.
                    df_result = test_pv.to_dataframe().rename(columns={'pv': 'actual_pv'})
                    df_result['ai_pv_srad'] = pred_pv.to_dataframe()['pv']
                    # df_result['ai_pv_trad'] = pred_pv.to_dataframe()['pv']
                    # df_result['ai_pv_srad_trad'] = pred_pv.to_dataframe()['pv']



                    # df_result['trad'] = test_cov.to_dataframe()['trad']  # 야간 필터링을 위해 일사량 가져오기

                    # 오차 = 실제 발전량 - 기대 발전량
                    # df_result['error'] = df_result['actual_pv'] - df_result['expected_pv']
                    # corr = df_result['actual_pv'].corr(df_result['expected_pv'])
                    # rmse = np.sqrt(mean_squared_error(df_result['actual_pv'], df_result['expected_pv']))
                    # print(corr, rmse)

                    df_result.to_csv('/DATA/INPUT/QUBE2026/df_result.csv', index=False)


                    # 0.91090328617368 94.92862966539832
                    # 0.9642343185339545 61.06584839511654
                    # 0.9616116807136832 64.70938528789236

                    # dataL3 = df_result.reset_index()
                    # dataL3['srv'] = posInfo['srv']
                    # dataL3['date_time'] = dataL3['date_time_kst'] - dtKst
                    #
                    # with cfgDb['sessionMake']() as session:
                    #     try:
                    #         tbTmp = f"tbTm_{uuid.uuid4().hex}"
                    #         with session.begin():
                    #             dbEngine = session.get_bind()
                    #
                    #             dataL3.to_sql(
                    #                 name=tbTmp,
                    #                 con=dbEngine,
                    #                 if_exists="replace",
                    #                 index=False,
                    #                 chunksize=1000
                    #             )
                    #
                    #             query = text(f"""
                    #                 INSERT INTO "tb_obs_data" (
                    #                       "srv", "date_time", "date_time_kst", "ai_ano_score", "ai_ano"
                    #                 )
                    #                 SELECT
                    #                       "srv", "date_time", "date_time_kst", "ai_ano_score", "ai_ano"
                    #                 FROM "{tbTmp}"
                    #                 ON CONFLICT ("srv", "date_time")
                    #                 DO UPDATE SET
                    #                     "ai_ano_score" = excluded."ai_ano_score",
                    #                     "ai_ano" = excluded."ai_ano"
                    #                   """)
                    #             result = session.execute(query)
                    #             log.info(f"result : {result.rowcount}")
                    #             session.execute(text(f'DROP TABLE IF EXISTS "{tbTmp}"'))
                    #     except Exception as e:
                    #         log.error(f"Exception : {e}")
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
