
import glob
# import seaborn as sns
import logging
import logging.handlers
import os
import platform
import sys
import argparse
import traceback
import warnings
from builtins import enumerate
from sklearn.metrics import mean_squared_error
from flaml import AutoML

# import datetime as dt
# from datetime import datetime
# import pvlib
import matplotlib.dates as mdates
import matplotlib.cm as cm
# from sklearn.metrics import mean_absolute_error


import matplotlib as mpl
import matplotlib.pyplot as  plt
import numpy as np
from scipy.stats import linregress
import pandas as pd
import math
from scipy import spatial
from pandas.tseries.offsets import Day, Hour, Minute, Second
from scipy.interpolate import Rbf
from numpy import zeros, newaxis

# os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share"
# from pygem import IDW
# import eccodes
import pygrib
# import pykrige.kriging_tools as kt
import pytz
import requests
import datetime
import h2o
# from pycaret.regression import *
from sqlalchemy import create_engine
import re
import configparser
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pymysql
import re
import configparser
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from scipy.stats import linregress
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split

import xarray as xr
from pvlib import location
from pvlib import irradiance

import h2o
from h2o.automl import H2OAutoML

from matplotlib import font_manager, rc

try:
    from pycaret.regression import *
except Exception as e:
    print("Exception : {}".format(e))

try:
    from pycaret.regression import *
except Exception as e:
    print("Exception : {}".format(e))


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost
# from sklearn.utils import safe_indexing

import os
import pickle
import ray


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
# font_manager._rebuild()

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

    saveLogFile = "{}/{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.node()
        , prjName
        , datetime.datetime.now().strftime("%Y%m%d")
    )

    if not os.path.exists(os.path.dirname(saveLogFile)):
        os.makedirs(os.path.dirname(saveLogFile))

    # logger instance 생성
    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    # format 생성
    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    # handler 생성
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(saveLogFile)

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
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
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

        # 글꼴 설정
        plt.rc('font', family='Malgun Gothic')

    # 리눅스 환경
    if globalVar['sysOs'] in 'Linux':
        parser = argparse.ArgumentParser()

        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)

        inParInfo = vars(parser.parse_args())

        # 글꼴 설정
        fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        plt.rcParams['font.family'] = fontName

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

        log.info("[CHECK] {}".format(val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar


# 딥러닝 예측
def makeDlModel(subOpt=None, xCol=None, yCol=None, inpData=None):

    log.info('[START] {}'.format('makeDlModel'))
    log.info('[CHECK] subOpt : {}'.format(subOpt))

    result = None

    try:

        saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['modelPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'h2o', 'for', '*')
        saveModelList = sorted(glob.glob(saveModel), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        data = inpData[xyCol]
        dataL1 = data.dropna()

        # h2o.shutdown(prompt=False)

        if (not subOpt['isInit']):
            h2o.init()
            h2o.no_progress()
            subOpt['isInit'] = True

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            # trainData, validData = train_test_split(dataL1, test_size=0.3)

            # dlModel = H2OAutoML(max_models=30, max_runtime_secs=99999, balance_classes=True, seed=123)
            dlModel = H2OAutoML(max_models=20, max_runtime_secs=99999, balance_classes=True, seed=123)

            #dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(trainData), validation_frame=h2o.H2OFrame(validData))
            #dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(dataL1), validation_frame=h2o.H2OFrame(dataL1))
            dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(dataL1), validation_frame=h2o.H2OFrame(dataL1))
            
            fnlModel = dlModel.get_best_model()
            #fnlModel = dlModel.leader

            # 학습 모델 저장
            saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['modelPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'h2o', 'for', datetime.datetime.now().strftime('%Y%m%d'))
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)

            # h2o.save_model(model=fnlModel, path=os.path.dirname(saveModel), filename=os.path.basename(saveModel), force=True)
            fnlModel.save_mojo(path=os.path.dirname(saveModel), filename=os.path.basename(saveModel), force=True)
        else:
            saveModel = saveModelList[0]
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            fnlModel = h2o.import_mojo(saveModel)

        result = {
            'msg': 'succ'
            , 'dlModel': fnlModel
            , 'saveModel': saveModel
            , 'isExist': os.path.exists(saveModel)
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeDlModel'))


# 머신러닝 예측
def makeMlModel(subOpt=None, xCol=None, yCol=None, inpData=None):

    log.info('[START] {}'.format('makeMlModel'))
    log.info('[CHECK] subOpt : {}'.format(subOpt))

    result = None

    try:
        saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model.pkl'.format(globalVar['outPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'pycaret', 'for', '*')
        saveModelList = sorted(glob.glob(saveModel), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        data = inpData[xyCol]
        dataL1 = data.dropna()

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            # trainData, validData = train_test_split(data, test_size=0.3)
            # trainData = inpData

            # from pycaret.regression import *
            # from pycaret.regression import setup
            # from pycaret.regression import compare_models

            # from sklearn.impute import SimpleImputer

            dataL1['is_weekday'] = dataL1['is_weekday'].astype('category')
            dataL1['is_spring'] = dataL1['is_spring'].astype('category')
            dataL1['is_summer'] = dataL1['is_summer'].astype('category')
            dataL1['is_fall'] = dataL1['is_fall'].astype('category')
            dataL1['is_winter'] = dataL1['is_winter'].astype('category')

            # dataL2 = dataL1[['CA_TOT', 'SG_PWRER_USE_AM']]

            pyModel = setup(
                data=dataL1
                , session_id=123
                , silent=True
                , target=yCol
                # 2022.11.02
                # , normalize=True
            )

            # pyModel = setup(
            #     data=dataL1
            #     , session_id=123
            #     , silent=True
            #     , target=yCol
            #     , remove_outliers= True
            #     , remove_multicollinearity = True
            #     , ignore_low_variance = True
            #     , normalize=True
            #     , transformation= True
            #     , transform_target = True
            #     , combine_rare_levels = True
            # )

            # 각 모형에 따른 자동 머신러닝
            # modelList = compare_models(sort='RMSE', n_select=3)
            # modelList = compare_models(sort='RMSE', n_select=2)
            modelList = compare_models(sort='RMSE')

            # 앙상블 모형
            # blendModel = blend_models(estimator_list=modelList, fold=10)

            # 앙상블 파라미터 튜닝
            # tuneModel = tune_model(blendModel, fold=10, choose_better=True)

            # 학습 모형
            fnlModel = finalize_model(modelList)

            # 학습 모형 저장
            saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'pycaret', 'for', datetime.datetime.now().strftime('%Y%m%d'))
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)
            save_model(fnlModel, saveModel)

        else:
            saveModel = saveModelList[0]
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            fnlModel = load_model(os.path.splitext(saveModel)[0])

        result = {
            'msg': 'succ'
            , 'mlModel': fnlModel
            , 'saveModel': saveModel
            , 'isExist': os.path.exists(saveModel)
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeMlModel'))


### 주말이면 1, 평일이면 0 반환
def check_weekend(date):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day = date.weekday()
    if days[day] in ['Sat', 'Sun']:
        return 1
    else:
        return 0


### 봄 여부
def check_spring(date):
    month = date.month
    if month in [3, 4, 5]:
        return 1
    else:
        return 0


### 여름 여부
def check_summer(date):
    month = date.month
    if month in [6, 7, 8]:
        return 1
    else:
        return 0


### 가을 여부
def check_fall(date):
    month = date.month
    if month in [9, 10, 11]:
        return 1
    else:
        return 0


### 겨울 여부
def check_winter(date):
    month = date.month
    if month in [12, 1, 2]:
        return 1
    else:
        return 0


# 머신러닝 예측
def makeFlamlModel(subOpt=None, xCol=None, yCol=None, trainData=None, testData=None):
    log.info('[START] {}'.format('makeFlamlModel'))
    log.info('[CHECK] subOpt : {}'.format(subOpt))

    result = None

    try:
        saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'flaml', 'for', '*')
        saveModelList = sorted(glob.glob(saveModel), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)

        trainDataL1 = trainData[xyCol]
        trainDataL2 = trainDataL1.dropna()

        testDataL1 = testData[xyCol]
        testDataL2 = testDataL1.dropna()

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            # trainData, validData = train_test_split(dataL1, test_size=0.3)

            # 전체 학습 데이터
            # trainData = dataL1

            # flModel = MultiOutputRegressor(
            #     AutoML(
            #         task="regression"
            #         , metric='rmse'
            #         , time_budget=60
            #     )
            # )

            ray.init(num_cpus=12, ignore_reinit_error=True)

            flModel = AutoML(
                task="regression"
                , metric='rmse'
                # , time_budget=600
                # , early_stop = True
                , ensemble = False
                # , ensemble = True
                , seed = 123
                , time_budget=60
            )

            # 각 모형에 따른 자동 머신러닝
            flModel.fit(X_train=trainData[xCol], y_train=trainData[yCol], n_jobs=12)
            # flModel.fit(X_train=trainData[xCol], y_train=trainData[yCol], n_jobs=4, n_concurrent_trials=4)
            # flModel.fit(X_train=trainData[xCol], y_train=trainData[yCol], n_jobs=4)
            # flModel.fit(X_train=trainDataL2[xCol], y_train=trainDataL2[yCol], X_test=testDataL2[xCol], y_test=testDataL2[yCol])
            # flModel.fit(X_train=trainDataL2[xCol], y_train=trainDataL2[yCol])

            # 학습 모형 저장
            saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'flaml', 'for', datetime.datetime.now().strftime('%Y%m%d'))
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)

            with open(saveModel, 'wb') as f:
                pickle.dump(flModel, f, pickle.HIGHEST_PROTOCOL)

        else:
            saveModel = saveModelList[0]
            log.info('[CHECK] saveModel : {}'.format(saveModel))

            with open(saveModel, 'rb') as f:
                flModel = pickle.load(f)

        result = {
            'msg': 'succ'
            , 'mlModel': flModel
            , 'saveModel': saveModel
            , 'isExist': os.path.exists(saveModel)
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeFlamlModel'))

def short_term_predict(raw_data, CNSMR_NO):
    print("Predict {} Start!!".format(CNSMR_NO))
    data = raw_data.loc[raw_data['CNSMR_NO'] == CNSMR_NO, ['SRV', 'MESURE_DATE_TM', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SZA', 'AZA', 'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR', 'SG_PWRER_USE_AM']].reset_index(drop=True)
    data['MESURE_DATE_TM'] = pd.to_datetime(data['MESURE_DATE_TM'])
    data['is_weekday'] = data['MESURE_DATE_TM'].map(check_weekend)
    data['is_spring'] = data['MESURE_DATE_TM'].map(check_spring)
    data['is_summer'] = data['MESURE_DATE_TM'].map(check_summer)
    data['is_fall'] = data['MESURE_DATE_TM'].map(check_fall)
    data['is_winter'] = data['MESURE_DATE_TM'].map(check_winter)
    data = data.loc[:, ['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SZA', 'AZA', 'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR', 'SG_PWRER_USE_AM', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']]
    # 독립변수 목록 정의
    independent_variable = ['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SZA', 'AZA', 'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']
    minmax_scaler = MinMaxScaler()
    minmax_scaler = minmax_scaler.fit_transform(data.loc[:, independent_variable])
    data.loc[:, independent_variable] = minmax_scaler

    ### 단기 예측
    t = 48
    X = np.array(data.loc[:, independent_variable])
    y = data['SG_PWRER_USE_AM']

    train_X = X[:-t, :]
    test_X = X[-t:, :]
    train_y = np.array(y[:-t])
    test_y = np.array(y[-t:])

    filename = './model/short_term/' + CNSMR_NO + '.model'
    model = pickle.load(open(filename, 'rb'))

    test_predict = model.predict(test_X)
    # print("RMSE: {}".format(np.sqrt(mean_squared_error(test_y, test_predict))))
    # print("MAPE: {}".format(mean_absolute_percentage_error(test_y, test_predict)))
    return train_X, test_X, train_y, test_y, model

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 한국 월별 수출액과 코스피 종간 간의 회귀분석

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'PRJ2022'

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

            if (platform.system() == 'Windows'):

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': '2021-10-01'
                    # , 'endDate': '2021-10-10'
                    'srtDate': '2019-01-01'
                    , 'endDate': '2022-05-22'
                    # , 'endDate': '2021-11-01'

                    , 'stnId': '1'

                    #  딥러닝
                    , 'dlModel': {
                        # 초기화
                        'isInit': False

                        # 모형 업데이트 여부
                        , 'isOverWrite': True
                        #, 'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }

                    #  머신러닝
                    , 'mlModel': {
                        # 모델 업데이트 여부
                        # 'isOverWrite': True
                        'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': globalVar['srtDate']
                    # , 'endDate': globalVar['endDate']
                    # , 'stnId': globalVar['stnId']
                    'srtDate': '2019-01-01'
                    , 'endDate': '2022-05-22'
                    , 'stnId': '1'

                    #  딥러닝
                    , 'dlModel': {
                        # 초기화
                        'isInit': False

                        # 모형 업데이트 여부
                        , 'isOverWrite': True
                        #, 'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }

                    #  머신러닝
                    , 'mlModel': {
                        # 모델 업데이트 여부
                        'isOverWrite': True
                        #'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }
                }

            globalVar['inpPath'] = '/DATA/INPUT'
            globalVar['outPath'] = '/DATA/OUTPUTT'
            globalVar['figPath'] = '/DATA/FIG'

            # ******************************************************************************
            # 크몽 전문가 검증 테스트
            # ******************************************************************************
            inpFile = '{}/{}/{}.csv'.format(globalVar['inpPath'], serviceName, 'FINAL_PW_DATA')
            fileList = sorted(glob.glob(inpFile))
            # pw_data = pd.read_csv('FINAL_PW_DATA.csv')
            pw_data = pd.read_csv(fileList[0])

            # pw_data.columns

            inpFile = '{}/{}/{}.csv'.format(globalVar['inpPath'], serviceName, 'FINAL_FORE_DATA')
            fileList = sorted(glob.glob(inpFile))
            # fore_data = pd.read_csv('FINAL_FORE_DATA.csv')
            fore_data = pd.read_csv(fileList[0])

            # fore_data.columns

            # 원본 데이터 : 15,319,449
            # orgData = fore_data.merge(pw_data, on=['SRV', 'MESURE_DATE_TM'])
            # orgData = fore_data.merge(pw_data, on=['SRV', 'MESURE_DATE_TM'])
            raw_data = fore_data.merge(pw_data, on=['SRV', 'MESURE_DATE_TM'])

            # orgData.columns

            # # 1차 가공 데이터 : 466,195
            # orgDataL1 = orgData
            # orgDataL1 = orgData.loc[orgData['SRV'] == 'SRV5011012200']
            # orgDataL1['MESURE_DATE_TM'] = pd.to_datetime(orgDataL1['MESURE_DATE_TM'])
            # orgDataL1['is_weekday'] = orgDataL1['MESURE_DATE_TM'].map(check_weekend)
            # orgDataL1['is_spring'] = orgDataL1['MESURE_DATE_TM'].map(check_spring)
            # orgDataL1['is_summer'] = orgDataL1['MESURE_DATE_TM'].map(check_summer)
            # orgDataL1['is_fall'] = orgDataL1['MESURE_DATE_TM'].map(check_fall)
            # orgDataL1['is_winter'] = orgDataL1['MESURE_DATE_TM'].map(check_winter)
            # orgDataL1.head()
            #
            #
            # # 2차 가공 데이터 : 28,111
            # data = orgData.groupby(['SRV', 'MESURE_DATE_TM', 'CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR'])['SG_PWRER_USE_AM'].mean().reset_index()
            #
            # data['MESURE_DATE_TM'] = pd.to_datetime(data['MESURE_DATE_TM'])
            # data['is_weekday'] = data['MESURE_DATE_TM'].map(check_weekend)
            # data.head()
            #
            # data['is_spring'] = data['MESURE_DATE_TM'].map(check_spring)
            # data['is_summer'] = data['MESURE_DATE_TM'].map(check_summer)
            # data['is_fall'] = data['MESURE_DATE_TM'].map(check_fall)
            # data['is_winter'] = data['MESURE_DATE_TM'].map(check_winter)
            # data.head()
            #
            # dataL1 = data.loc[data['SRV'] == 'SRV5011012200']

            # SRV1141012000 = data.loc[data['SRV'] == 'SRV1141012000', ['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'SG_PWRER_USE_AM', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']].reset_index(drop = True)
            # SRV1153010100 = data.loc[data['SRV'] == 'SRV1153010100', ['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'SG_PWRER_USE_AM', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']].reset_index(drop = True)
            # SRV3011010500 = data.loc[data['SRV'] == 'SRV3011010500', ['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'SG_PWRER_USE_AM', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']].reset_index(drop = True)
            # SRV5011012200 = data.loc[data['SRV'] == 'SRV5011012200', ['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'SG_PWRER_USE_AM', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']].reset_index(drop = True)
            #
            # # 독립변수 목록 정의
            # independent_variable = ['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']
            #
            # minmax_scaler = MinMaxScaler()
            # minmax_scaler = minmax_scaler.fit_transform(SRV5011012200.loc[:, independent_variable])
            # SRV5011012200.loc[:, independent_variable] = minmax_scaler
            # SRV5011012200.head()

            # ==============================================================================
            # 단기 예측
            # ==============================================================================
            CNSMR_NO_list = list(set(raw_data['CNSMR_NO']))
            RMSE_list = []
            # MAPE_list = []

            for i, CNSMR_NO in enumerate(CNSMR_NO_list):

                # CNSMR_NO = 'SE0300104001404'
                # CNSMR_NO = 'SE0300111002104'

                saveFile = '{}/{}/model/short_term/{}.model'.format(globalVar['inpPath'], serviceName, CNSMR_NO)
                if (not os.path.exists(saveFile)): continue

                log.info("[CHECK] CNSMR_NO : {} / {}".format(CNSMR_NO, round(i / len(CNSMR_NO_list) * 100.0, 2)))
                # short_term_predict(raw_data, CNSMR_NO)

                data = raw_data.loc[raw_data['CNSMR_NO'] == CNSMR_NO, ['SRV', 'MESURE_DATE_TM', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SZA', 'AZA', 'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR', 'SG_PWRER_USE_AM']].reset_index(drop=True)
                data['MESURE_DATE_TM'] = pd.to_datetime(data['MESURE_DATE_TM'])
                data['is_weekday'] = data['MESURE_DATE_TM'].map(check_weekend)
                data['is_spring'] = data['MESURE_DATE_TM'].map(check_spring)
                data['is_summer'] = data['MESURE_DATE_TM'].map(check_summer)
                data['is_fall'] = data['MESURE_DATE_TM'].map(check_fall)
                data['is_winter'] = data['MESURE_DATE_TM'].map(check_winter)
                dataL1 = data.loc[:, ['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SZA', 'AZA', 'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR', 'SG_PWRER_USE_AM', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']]

                # 독립변수 목록 정의
                independent_variable = ['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SZA', 'AZA', 'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']
                minmax_scaler = MinMaxScaler()
                minmax_scaler = minmax_scaler.fit_transform(dataL1.loc[:, independent_variable])
                dataL1.loc[:, independent_variable] = minmax_scaler

                ### 단기 예측
                t = 48
                X = np.array(dataL1.loc[:, independent_variable])
                y = dataL1['SG_PWRER_USE_AM']

                # 학습 및 테스트 데이터 분류
                train_X = X[:-t, :]
                train_y = np.array(y[:-t])
                test_X = X[-t:, :]
                test_y = np.array(y[-t:])

                prdData = data[-t:]

                model = pickle.load(open(saveFile, 'rb'))
                prdData['prd'] = model.predict(test_X)

                data['is_weekday'] = data['is_weekday'].astype('category')
                data['is_spring'] = data['is_spring'].astype('category')
                data['is_summer'] = data['is_summer'].astype('category')
                data['is_fall'] = data['is_fall'].astype('category')
                data['is_winter'] = data['is_winter'].astype('category')

                trainData = data[:-t]
                testData = data[-t:]

                # 자동 머신러닝
                xCol = ['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SZA', 'AZA', 'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']
                # xCol = ['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SZA', 'AZA', 'ET', 'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']
                # xCol = ['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SZA', 'AZA', 'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR']
                yCol = 'SG_PWRER_USE_AM'


                sysOpt['mlModel'].update(
                    {
                        'srvId': CNSMR_NO
                        , 'modelKey': 'AI-FOR-20221101'
                        , 'isOverWrite': False
                        # , 'isOverWrite': True
                    }
                )

                # resPycaret = makeMlModel(sysOpt['mlModel'], xCol, yCol, trainData)
                # log.info('[CHECK] resPycaret : {}'.format(resPycaret))
                #
                # pycaretModel = resPycaret['mlModel']
                # prdData['pycaret'] = pycaretModel.predict(prdData)

                # 분석 시간을 기준으로 0~5시간 내의 예보 시간 사용
                # rtnData = pd.DataFrame()
                # for j, anaTimeInfo in enumerate(umDataL4['anaTime'].values):
                #     for k, timeInfo in enumerate(umDataL4['time'].values):
                #         hourDiff = (timeInfo - anaTimeInfo) / pd.Timedelta(hours=1)
                #         if (0 > hourDiff) or (hourDiff > 5): continue
                #
                #         dict = {
                #             'anaTime': [anaTimeInfo]
                #             , 'time': [timeInfo]
                #         }
                #
                #         rtnData = pd.concat([rtnData, pd.DataFrame.from_dict(dict)], ignore_index=True)

                # 머신러닝 불러오기
                # resFlaml = makeFlamlModel(sysOpt['mlModel'], xCol, yCol, trainData)
                resFlaml = makeFlamlModel(sysOpt['mlModel'], xCol, yCol, trainData, testData)
                log.info('[CHECK] resFlaml : {}'.format(resFlaml))

                # 머신러닝 예측
                flamlModel = resFlaml['mlModel']
                prdData['flaml'] = flamlModel.predict(prdData)

                import optuna.integration.lightgbm as lgb

                dtrain = lgb.Dataset(trainData[xCol], trainData[yCol])
                deval = lgb.Dataset(testData[xCol], testData[yCol], reference=dtrain)

                best_params, history = {}, []

                params = {
                    'objective' : 'regression'
                    , 'metric' : 'rmse'
                    , 'verbosity' : -1
                    , 'n_jobs' : -1
                }

                # model = lgb.train(params=params, train_set=dtrain, num_boost_round=5000, early_stopping_rounds=5, valid_sets=[dtrain, deval], verbose_eval=False)
                # model = lgb.train(params=params, train_set=dtrain, num_boost_round=5000, early_stopping_rounds=50, valid_sets=[dtrain, deval], verbose_eval=False)
                # model = lgb.train(params=params, train_set=dtrain, num_boost_round=10000, early_stopping_rounds=50, valid_sets=[dtrain, deval], verbose_eval=False)
                model = lgb.train(params=params, train_set=dtrain, num_boost_round=10000, early_stopping_rounds=100, valid_sets=[dtrain, deval], verbose_eval=False)

                saveModel = '{}/{}/{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], 'AI-FOR-20221101', serviceName, CNSMR_NO, 'final', 'lightgbm', 'for')
                os.makedirs(os.path.dirname(saveModel), exist_ok=True)
                log.info('[CHECK] saveFile : {}'.format(saveModel))

                if (not os.path.exists(saveModel)):
                    pickle.dump(model, open(saveModel, 'wb'))

                model = pickle.load(open(saveModel, 'rb'))

                # {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'n_jobs': -1, 'feature_pre_filter': False, 'lambda_l1': 0.0, 'lambda_l2': 0.0, 'num_leaves': 178, 'feature_fraction': 0.4, 'bagging_fraction': 0.5303062852270854, 'bagging_freq': 7, 'min_child_samples': 20, 'num_iterations': 5000, 'early_stopping_round': 50, 'categorical_column': [15, 16, 17, 18, 19]}
                # best_params = model.params
                # print(best_params)

                prdData['ML3'] = model.predict(data=testData[xCol])
                # 19.93
                # (mean_squared_error(prdData['SG_PWRER_USE_AM'], prdData['ML3'], squared=False) / np.nanmean(prdData['SG_PWRER_USE_AM'])) * 100.0

                # import optuna
                # def objective(trial):
                #     dtrain = lgb.Dataset(trainData[xCol], trainData[yCol])
                #     deval = lgb.Dataset(testData[xCol], testData[yCol], reference=dtrain)
                #
                #     param = {
                #         'objective': 'regression',  # 회귀
                #         'verbose': -1,
                #         'metric': 'rmse',
                #         'max_depth': trial.suggest_int('max_depth', 3, 15),
                #         'learning_rate': trial.suggest_loguniform("learning_rate", 1e-8, 1e-2),
                #         'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                #         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                #         'subsample': trial.suggest_loguniform('subsample', 0.4, 1),
                #     }
                #
                #     model = lgb.LGBMRegressor(**param)
                #     lgb_model = model.fit(trainData[xCol], trainData[yCol], eval_set=[(testData[xCol], testData[yCol])], verbose=0, early_stopping_rounds=25)
                #     rmse = RMSE(h_valid_y, lgb_model.predict(h_valid_X))

                    # return rmse

                # study_lgb = optuna.create_study(direction='minimize', sampler=sampler)
                # study_lgb.optimize(objective, n_trials=100)

                # import xgboost as xgb
                # import optuna
                #
                # # xgboostを実行するには特殊なmatrixにする必要がある
                # dtrain = xgb.DMatrix(trainData[xCol], label=trainData[yCol], enable_categorical = True)
                # dtest = xgb.DMatrix(testData[xCol], label=testData[yCol], enable_categorical = True)

                # def objective(trial):
                #
                #     params = {
                #         # "silent": 1,
                #         "max_depth": trial.suggest_int("max_depth", 3, 9, step=2),
                #         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                #         "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
                #         "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                #         "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
                #         "tree_method": "exact",
                #         # "objective": "reg:linear",
                #         "objective": "reg:squarederror",
                #         "eval_metric": "rmse",
                #         "predictor": "cpu_predictor"
                #     }
                #
                #     cv_results = xgb.cv(
                #         params,
                #         dtrain,
                #         num_boost_round=1000,
                #         seed=123,
                #         nfold=5,
                #         metrics={"rmse"},
                #         early_stopping_rounds=5
                #     )
                #
                #     return cv_results["test-rmse-mean"].min()
                #
                # study = optuna.create_study()
                # # study.optimize(objective, n_trials=5000)
                # study.optimize(objective, n_trials=100, timeout=600)
                #
                # print('Number of finished trials:', len(study.trials))
                # print('Best trial:', study.best_trial.params)
                #
                # for key, val in study.best_trial.params.items():
                #     params[key] = val
                #
                # model = xgb.train(params=params,
                #                   dtrain=dtrain,
                #                   num_boost_round=1000,
                #                   early_stopping_rounds=5,
                #                   evals=[(dtest, "test")])
                #
                # xgb.plot_importance(model, height=0.8)
                # plt.show()
                #
                # prdData['ML4'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
                # import optuna
                # import xgboost as xgb
                #
                # def objective(trial, trainData=trainData, testData=testData, xCol=xCol, yCol=yCol):
                #
                #     # train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.15, random_state=42)
                #
                #     param = {
                #         'tree_method': 'gpu_hist',  # this parameter means using the GPU when training our model to speedup the training process
                #         'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                #         'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                #         'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                #         'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
                #         'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
                #         'n_estimators': 10000,
                #         'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17]),
                #         'random_state': trial.suggest_categorical('random_state', [2020]),
                #         'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
                #     }
                #     # model = xgb.XGBRegressor(**param)
                #
                #     train_set = xgb.DMatrix(trainData[xCol], label=trainData[yCol], enable_categorical=True)
                #     test_set = xgb.DMatrix(testData[xCol], label=testData[yCol], enable_categorical=True)
                #
                #     # model.fit(train_set[xCol], trainData[xCol], eval_set=[(testData[xCol], testData[yCol])], early_stopping_rounds=100, verbose=False)
                #     # model.fit(train_set, test_set, eval_set=[(train_set, test_set)], early_stopping_rounds=100, verbose=False)
                #     model = xgb.train(
                #         param,
                #         train_set,
                #         evals=[(test_set, "eval")],
                #         verbose_eval=False)
                #         # callbacks=[TuneReportCheckpointCallback(filename="model.xgb")])
                #
                #     # preds = model.predict(testData[xCol])
                #     preds = model.predict(test_set)
                #
                #     # rmse = mean_squared_error(testData[yCol], preds, squared=False)
                #     rmse = mean_squared_error(test_set, preds, squared=False)
                #
                #     return rmse
                #
                # # import optuna
                # import optuna
                #
                # study = optuna.create_study(direction='minimize')
                # study.optimize(objective, n_trials=30)
                # print('Number of finished trials:', len(study.trials))
                # print('Best trial:', study.best_trial.params)


                #
                # trainData['is_weekday'] = trainData['is_weekday'].astype('category')
                # trainData['is_spring'] = trainData['is_spring'].astype('category')
                # trainData['is_summer'] = trainData['is_summer'].astype('category')
                # trainData['is_fall'] = trainData['is_fall'].astype('category')
                # trainData['is_winter'] = trainData['is_winter'].astype('category')


                # from lazypredict.Supervised import LazyRegressor
                # import lazypredict
                # reg = LazyRegressor(
                #     verbose=0,
                #     ignore_warnings=True,
                #     custom_metric=None,
                #     predictions=True,
                #     random_state=123,
                # )
                # # X_train, X_test, y_train, y_test
                # models, predictions = reg.fit(X_train = trainData[xCol], y_train = trainData[yCol], X_test = testData[xCol], y_test = testData[yCol])
                # prdData['lazy'] = models.predict(prdData)

                # pycaretModel = resPycaret['mlModel']
                # prdData['pycaret'] = pycaretModel.predict(prdData)

                # 모델 영향도
                # saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], sysOpt['mlModel']['modelKey'], CNSMR_NO, 'feat')
                # #
                # featData = pd.DataFrame([flamlModel.feature_names_in_, flamlModel.feature_importances_], index=['key', 'val']).transpose().sort_values(by=['val'], ascending=True)
                # # featData = pd.DataFrame([pycaretModel.feature_names_in_, pycaretModel.feature_importances_], index=['key', 'val']).transpose().sort_values(by=['val'], ascending=True)
                # plt.barh(featData['key'], featData['val'])
                # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                # plt.savefig(saveImg, dpi=600, bbox_inches='tight')
                # plt.show()
                # plt.close()

                mainTitle = '[48시간 단기] [{0:s}-{1:s}]'.format(prdData['SRV'].iloc[0], CNSMR_NO)
                saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                # saveImg = '{}/{}/short-term_{}-{}-{}.png'.format(globalVar['figPath'], serviceName, prdData['SRV'].iloc[0], CNSMR_NO, 'pycaret')
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                log.info('[CHECK] saveImg : {}'.format(saveImg))

                plt.plot(prdData['MESURE_DATE_TM'], prdData['SG_PWRER_USE_AM'], marker='o', label='실측')
                plt.plot(prdData['MESURE_DATE_TM'], prdData['prd'], label='예측 (RMSE : {:.2f}, {:.2f}%)'.format(
                    mean_squared_error(prdData['SG_PWRER_USE_AM'], prdData['prd'], squared=False)
                    , (mean_squared_error(prdData['SG_PWRER_USE_AM'], prdData['prd'], squared=False) / np.nanmean(prdData['SG_PWRER_USE_AM'])) * 100.0)
                    )
                plt.plot(prdData['MESURE_DATE_TM'], prdData['flaml'], label='예측2 (RMSE : {:.2f}, {:.2f}%)'.format(
                    mean_squared_error(prdData['SG_PWRER_USE_AM'], prdData['flaml'], squared=False)
                    , (mean_squared_error(prdData['SG_PWRER_USE_AM'], prdData['flaml'], squared=False) / np.nanmean(prdData['SG_PWRER_USE_AM'])) * 100.0)
                    )
                plt.plot(prdData['MESURE_DATE_TM'], prdData['ML3'], label='예측3 (RMSE : {:.2f}, {:.2f}%)'.format(
                    mean_squared_error(prdData['SG_PWRER_USE_AM'], prdData['ML3'], squared=False)
                    , (mean_squared_error(prdData['SG_PWRER_USE_AM'], prdData['ML3'], squared=False) / np.nanmean(prdData['SG_PWRER_USE_AM'])) * 100.0)
                    )
                # plt.plot(prdData['MESURE_DATE_TM'], prdData['ML4'], label='예측4 (RMSE : {:.2f}, {:.2f}%)'.format(
                #     mean_squared_error(prdData['SG_PWRER_USE_AM'], prdData['ML4'], squared=False)
                #     , (mean_squared_error(prdData['SG_PWRER_USE_AM'], prdData['ML4'], squared=False) / np.nanmean(prdData['SG_PWRER_USE_AM'])) * 100.0)
                #     )
                # plt.plot(prdData['MESURE_DATE_TM'], prdData['lazy'], label='예측 (lazy) : {:.3f}'.format(round(np.sqrt(mean_squared_error(prdData['SG_PWRER_USE_AM'], prdData['flaml'])), 2)))
                # plt.plot(prdData['MESURE_DATE_TM'], prdData['pycaret'], label='예측 (pycaret) : {:.3f}'.format(round(np.sqrt(mean_squared_error(prdData['SG_PWRER_USE_AM'], prdData['pycaret'])), 2)))

                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d %H'))
                plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
                plt.gcf().autofmt_xdate()
                # plt.xticks(rotation=45, ha='right')
                plt.xticks(rotation=45)
                plt.title(mainTitle)
                plt.legend()
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                plt.show()
                plt.close()





            sys.exit(0)

            # *************************************************************************************
            #
            # *************************************************************************************
            CNSMR_NO = CNSMR_NO_list[0]
            # for CNSMR_NO in CNSMR_NO_list:
            for i, CNSMR_NO in enumerate(CNSMR_NO_list):
                # print("Modeling {} Start!!".format(CNSMR_NO))

                saveFile = '{}/{}/model/short_term/{}.model'.format(globalVar['inpPath'], serviceName, CNSMR_NO)
                if (os.path.exists(saveFile)): continue

                log.info("[CHECK] CNSMR_NO : {} / {}".format(CNSMR_NO, round(i / len(CNSMR_NO_list) * 100.0, 2)))

                data = raw_data.loc[raw_data['CNSMR_NO'] == CNSMR_NO, ['SRV', 'MESURE_DATE_TM', 'CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'SG_PWRER_USE_AM']].reset_index(drop = True)
                data['MESURE_DATE_TM'] = pd.to_datetime(data['MESURE_DATE_TM'])
                data['is_weekday'] = data['MESURE_DATE_TM'].map(check_weekend)
                data['is_spring'] = data['MESURE_DATE_TM'].map(check_spring)
                data['is_summer'] = data['MESURE_DATE_TM'].map(check_summer)
                data['is_fall'] = data['MESURE_DATE_TM'].map(check_fall)
                data['is_winter'] = data['MESURE_DATE_TM'].map(check_winter)
                dataL1 = data.loc[:, ['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'SG_PWRER_USE_AM', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']]

                # 독립변수 목록 정의
                independent_variable = ['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']
                minmax_scaler = MinMaxScaler()
                minmax_scaler = minmax_scaler.fit_transform(dataL1.loc[:, independent_variable])
                dataL1.loc[:, independent_variable] = minmax_scaler

                ### 단기 예측
                t = 48
                X = np.array(dataL1.loc[:, independent_variable])
                y = dataL1['SG_PWRER_USE_AM']

                train_X = X[:-t,:]
                test_X = X[-t:, :]
                train_y = np.array(y[:-t])
                test_y = np.array(y[-t:])

                # 수동 머신러닝
                n_estimators_candidate = [100, 300,500]
                max_depth_candidate = [3, 5, 7]
                learning_rate_candidate = [0.1, 0.01, 0.001]

                # 결과를 저장할 빈 리스트 생성
                n_estimators_list = []
                max_depth_list = []
                learning_rate_list = []
                train_score_list = []
                val_score_list = []

                # XGBoost의 n_estimators 파라미터에 대해서
                for n_estimators in n_estimators_candidate:
                    # XGBoost의 max_depth 파라미터에 대해서
                    for max_depth in max_depth_candidate:
                        # XGBoost의 learning_rate 파라미터에 대해서
                        for learning_rate in learning_rate_candidate:
                            # 모델 생성 및 학습
                            # model = xgboost.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, objective='reg:squarederror').fit(train_X, train_y)
                            # Train 데이터에 대한 결과 확인
                            train_predict = model.predict(train_X)
                            # train_score_list.append(np.sqrt(mean_squared_error(train_y, train_predict)))
                            # Test 데이터에 대한 결과 확인
                            test_predict = model.predict(test_X)
                            # val_score_list.append(np.sqrt(mean_squared_error(test_y, test_predict)))
                            # Parameter 저장
                            n_estimators_list.append(n_estimators)
                            max_depth_list.append(max_depth)
                            learning_rate_list.append(learning_rate)

                result = pd.DataFrame({"n_estimators": n_estimators_list, "max_depth": max_depth_list, "learning_rate": learning_rate_list, 'Train Score': train_score_list, 'Test Score': val_score_list})
                result = result.loc[result['Test Score'] == min(result['Test Score']), :].reset_index(drop = True)
                # model = xgboost.XGBRegressor(n_estimators=result['n_estimators'][0], max_depth=result['max_depth'][0], learning_rate=result['learning_rate'][0], objective='reg:squarederror').fit(train_X, train_y)
                test_predict = model.predict(test_X)
                # RMSE_list.append(np.sqrt(mean_squared_error(test_y, test_predict)))
                # MAPE_list.append(mean_absolute_percentage_error(test_y, test_predict))

                # os.makedirs('./model', exist_ok = True)
                # os.makedirs('./model/short_term', exist_ok = True)
                # filename = './model/short_term/' + CNSMR_NO + '.model'
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                log.info('[CHECK] saveFile : {}'.format(saveFile))
                pickle.dump(model, open(saveFile, 'wb'))

            # performance = pd.DataFrame({"CNSMR_NO": CNSMR_NO_list, 'RMSE': RMSE_list, 'MAPE': MAPE_list})

            saveResFile = '{}/{}/model/{}.csv'.format(globalVar['inpPath'], serviceName, 'short_term_performance')
            log.info('[CHECK] saveResFile : {}'.format(saveResFile))
            # performance.to_csv('./model/short_term_performance.csv', index = False)
            # performance.to_csv(saveResFile, index = False)


            # # ***********************************************************
            # # 가공 2차 및 AI 학습모형을 이용한 예측 결과
            # # ***********************************************************
            # dataL1['prd'] = model.predict(dataL1[['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']])
            #
            # plt.plot(dataL1['MESURE_DATE_TM'], dataL1['SG_PWRER_USE_AM'], label='실측')
            # plt.plot(dataL1['MESURE_DATE_TM'], dataL1['prd'], label='예측')
            # mainTitle = '[short-term] [{0:s}] RMSE = {1:.2f}'.format(
            #     dataL1['SRV'].iloc[0]
            #     , np.sqrt(mean_squared_error(dataL1['SG_PWRER_USE_AM'], dataL1['prd']))
            # )
            # plt.title(mainTitle)
            #
            # saveImg = '{}/short-term_{}.png'.format(globalVar['figPath'], dataL1['SRV'].iloc[0])
            # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent = True)
            # plt.show()
            # plt.close()
            #
            # # ***********************************************************
            # # 가공 1차 및 AI 학습모형을 이용한 예측 결과
            # # ***********************************************************
            # cnsmrNoList = np.unique(orgDataL1['CNSMR_NO'])
            # for i, cnsmrNo in enumerate(cnsmrNoList):
            #
            #     orgDataL2 = orgDataL1.loc[orgDataL1['CNSMR_NO'] == cnsmrNo]
            #     print(cnsmrNo, len(orgDataL2))
            #
            #     orgDataL2['prd'] = model.predict(orgDataL2[['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']])
            #
            #     # plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['SG_PWRER_USE_AM'])
            #     # plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['prd'])
            #     # plt.show()
            #
            #     plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['SG_PWRER_USE_AM'], label='실측')
            #     plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['prd'], label='예측')
            #     mainTitle = '[short-term] [{0:s}-{1:s}] RMSE = {2:.2f}'.format(
            #         orgDataL2['SRV'].iloc[0]
            #         , cnsmrNo
            #         , np.sqrt(mean_squared_error(orgDataL2['SG_PWRER_USE_AM'], orgDataL2['prd']))
            #     )
            #     plt.title(mainTitle)
            #
            #     saveImg = '{}/short-term_{}-{}.png'.format(globalVar['figPath'], dataL1['SRV'].iloc[0], cnsmrNo)
            #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.show()
            #     plt.close()

            # ==============================================================================
            # 장기 예측
            # ==============================================================================
            # t = 720
            #
            # X = np.array(SRV5011012200.loc[:, independent_variable])
            # y = SRV5011012200['SG_PWRER_USE_AM']
            #
            # train_X = X[:-t, :]
            # test_X = X[-t:, :]
            # train_y = np.array(y[:-t])
            # test_y = np.array(y[-t:])
            #
            # n_estimators_candidate = [100, 200, 300, 400, 500]
            # max_depth_candidate = [3, 5, 7, 9]
            # learning_rate_candidate = [0.1, 0.01, 0.001, 0.0001]
            #
            # # 결과를 저장할 빈 리스트 생성
            # n_estimators_list = []
            # max_depth_list = []
            # learning_rate_list = []
            # train_score_list = []
            # val_score_list = []
            #
            # # XGBoost의 n_estimators 파라미터에 대해서
            # for n_estimators in n_estimators_candidate:
            #     # XGBoost의 max_depth 파라미터에 대해서
            #     for max_depth in max_depth_candidate:
            #         # XGBoost의 learning_rate 파라미터에 대해서
            #         for learning_rate in learning_rate_candidate:
            #             print('[CHECK] n_estimators : {} / max_depth : {} / learning_rate : {}'.format(n_estimators, max_depth, learning_rate))
            #
            #             # 모델 생성 및 학습
            #             model = xgboost.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
            #                                          learning_rate=learning_rate, objective='reg:squarederror').fit(
            #                 train_X, train_y)
            #             # Train 데이터에 대한 결과 확인
            #             train_predict = model.predict(train_X)
            #             train_score_list.append(np.sqrt(mean_squared_error(train_y, train_predict)))
            #             # Test 데이터에 대한 결과 확인
            #             test_predict = model.predict(test_X)
            #             val_score_list.append(np.sqrt(mean_squared_error(test_y, test_predict)))
            #             # Parameter 저장
            #             n_estimators_list.append(n_estimators)
            #             max_depth_list.append(max_depth)
            #             learning_rate_list.append(learning_rate)
            #
            # result = pd.DataFrame({"n_estimators": n_estimators_list, "max_depth": max_depth_list, "learning_rate": learning_rate_list,'Train Score': train_score_list, 'Test Score': val_score_list})
            # result = result.loc[result['Test Score'] == min(result['Test Score']), :].reset_index(drop=True)
            #
            # model = xgboost.XGBRegressor(n_estimators=result['n_estimators'][0], max_depth=result['max_depth'][0], learning_rate=result['learning_rate'][0], objective='reg:squarederror').fit(train_X, train_y)
            # test_predict = model.predict(test_X)
            #
            # print("RMSE: {}".format(np.sqrt(mean_squared_error(test_y, test_predict))))
            # print("MAPE: {}".format(mean_absolute_percentage_error(test_y, test_predict)))


            # ***********************************************************
            # 가공 2차 및 AI 학습모형을 이용한 예측 결과
            # ***********************************************************
            # dataL1['prd'] = model.predict(dataL1[['CA_TOT',	'HM', 'PA', 'TA', 'TD',	'WD', 'WS', 'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']])
            #
            # plt.plot(dataL1['MESURE_DATE_TM'], dataL1['SG_PWRER_USE_AM'], label='실측')
            # plt.plot(dataL1['MESURE_DATE_TM'], dataL1['prd'], label='예측')
            # mainTitle = '[short-term] [{0:s}] RMSE = {1:.2f}'.format(
            #     dataL1['SRV'].iloc[0]
            #     , np.sqrt(mean_squared_error(dataL1['SG_PWRER_USE_AM'], dataL1['prd']))
            # )
            # plt.title(mainTitle)
            #
            # saveImg = '{}/long-term_{}.png'.format(globalVar['figPath'], dataL1['SRV'].iloc[0])
            # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent = True)
            # plt.show()
            # plt.close()

            # # ***********************************************************
            # # 가공 1차 및 AI 학습모형을 이용한 예측 결과
            # # ***********************************************************
            # cnsmrNoList = np.unique(orgDataL1['CNSMR_NO'])
            # for i, cnsmrNo in enumerate(cnsmrNoList):
            #
            #     orgDataL2 = orgDataL1.loc[orgDataL1['CNSMR_NO'] == cnsmrNo]
            #     print(cnsmrNo, len(orgDataL2))
            #
            #     orgDataL2['prd'] = model.predict(orgDataL2[['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']])
            #
            #     # plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['SG_PWRER_USE_AM'])
            #     # plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['prd'])
            #     # plt.show()
            #
            #     plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['SG_PWRER_USE_AM'], label='실측')
            #     plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['prd'], label='예측')
            #     mainTitle = '[short-term] [{0:s}-{1:s}] RMSE = {2:.2f}'.format(
            #         orgDataL2['SRV'].iloc[0]
            #         , cnsmrNo
            #         , np.sqrt(mean_squared_error(orgDataL2['SG_PWRER_USE_AM'], orgDataL2['prd']))
            #     )
            #     plt.title(mainTitle)
            #
            #     saveImg = '{}/long-term_{}-{}.png'.format(globalVar['figPath'], dataL1['SRV'].iloc[0], cnsmrNo)
            #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.show()
            #     plt.close()


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
