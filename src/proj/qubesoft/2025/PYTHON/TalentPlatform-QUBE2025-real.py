# ================================================
# 요구사항
# ================================================
# Python을 이용한 재처리

# 프로그램 종료
# ps -ef | grep python | grep TalentPlatform-QUBE2025-real.py | awk '{print $2}' | xargs kill -9
# pkill -f TalentPlatform-QUBE2025-real.py

# 프로그램 시작
# conda activate py39

# cd /SYSTEMS/PROG/PYTHON
# /SYSTEMS/LIB/anaconda3/envs/py39/bin/python /SYSTEMS/PROG/PYTHON/TalentPlatform-QUBE2025-real.py
# nohup /SYSTEMS/LIB/anaconda3/envs/py39/bin/python /SYSTEMS/PROG/PYTHON/TalentPlatform-QUBE2025-real.py --srtDate "$(date -d "2 days ago" +\%Y-\%m-\%d)" --endDate "$(date -d "2 days" +\%Y-\%m-\%d)" &

# 20,50 * * * * cd /SYSTEMS/PROG/PYTHON && /SYSTEMS/LIB/anaconda3/envs/py39/bin/python /SYSTEMS/PROG/PYTHON/TalentPlatform-QUBE2025-real.py --srtDate "$(date -d "2 days ago" +\%Y-\%m-\%d)" --endDate "$(date -d "2 days" +\%Y-\%m-\%d)"

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
from pycaret.regression import RegressionExperiment
import requests

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
import pvlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
from scipy.stats import linregress
import pandas as pd
# import cartopy.crs as ccrs
import math
from scipy import spatial
from pandas.tseries.offsets import Day, Hour, Minute, Second
from scipy.interpolate import Rbf
from numpy import zeros, newaxis

import pygrib
import haversine as hs
import pytz
import datetime
# import h2o
# from pycaret.regression import *
from sqlalchemy import create_engine
import re
import configparser
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
import random
from urllib.parse import quote_plus
from urllib.parse import unquote_plus
import urllib.parse
import sqlalchemy
from sqlalchemy import create_engine, text
import requests
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import text
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
import requests
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import text
from pvlib import location
from pvlib import irradiance
from multiprocessing import Pool
import multiprocessing as mp
import uuid
from sqlalchemy.pool import NullPool
from joblib import parallel_backend

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

        log.info("[CHECK] {} / val : {}".format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar

def cartesian(latitude, longitude, elevation=0):
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)

    return (X, Y, Z)

def initCfgInfo(config, key):

    result = None

    try:
        # log.info(f'[CHECK] key : {key}')
        dbUser = config.get(key, 'user')
        dbPwd = urllib.parse.quote(config.get(key, 'pwd'))
        dbHost = config.get(key, 'host')
        dbPort = config.get(key, 'port')
        dbName = config.get(key, 'dbName')

        # engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}", echo=False, pool_timeout=60*5, pool_recycle=3600)
        engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}", echo=False, poolclass=NullPool)
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

def makeLgbModel(subOpt=None, xCol=None, yCol=None, trainData=None, testData=None):

    # log.info(f'[START] makeLgbModel')
    # log.info(f'[CHECK] subOpt : {subOpt}')

    result = None

    try:

        saveModelList = sorted(glob.glob(subOpt['saveModelList'].format(srv = subOpt['srv'])), reverse=True)

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):
            xyCol = xCol.copy()
            xyCol.append(yCol)

            trainDataL1 = trainData[xyCol].dropna().copy()
            testDataL1 = testData[xyCol].dropna().copy()

            lgbParams = {
                # 연속 예측
                'objective': 'regression',
                'metric': 'rmse',

                # 이진 분류
                # 'objective': 'binary',
                # 'metric': 'auc',

                'n_jobs': -1,
                'verbosity': -1,
                # 'seed': int(subOpt['preDt'].timestamp()),
                'seed': int(datetime.datetime.now().timestamp()),
            }

            lgbTrainData = lgb.Dataset(trainDataL1[xCol], trainDataL1[yCol])
            lgbTestData = lgb.Dataset(testDataL1[xCol], testDataL1[yCol], reference=lgbTrainData)

            # 학습
            fnlModel = lgb.train(
                params=lgbParams,
                num_boost_round=10000,
                train_set=lgbTrainData,
                valid_sets=[lgbTrainData, lgbTestData],
                valid_names=["train", "valid"],
                callbacks=[
                    # early_stopping(stopping_rounds=10000),
                    early_stopping(stopping_rounds=1000),
                    log_evaluation(period=200),
                ],
            )

            # 학습 모형 저장
            saveModel = subOpt['preDt'].strftime(subOpt['saveModel']).format(srv = subOpt['srv'])
            log.info(f'[CHECK] saveModel : {saveModel}')
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)
            with open(saveModel, 'wb') as file:
                pickle.dump(fnlModel, file, pickle.HIGHEST_PROTOCOL)

            # 변수 중요도 저장
            # try:
            #     mainTitle = '{}'.format('lgb-importnce')
            #     saveImg = subOpt['preDt'].strftime(subOpt['saveImg'])
            #     lgb.plot_importance(fnlModel)
            #     plt.title(mainTitle)
            #     plt.tight_layout()
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight')
            #     # plt.show()
            #     plt.close()
            # except Exception as e:
            #     pass

        else:
            saveModel = saveModelList[0]
            log.info(f'[CHECK] saveModel : {saveModel}')
            with open(saveModel, 'rb') as file:
                fnlModel = pickle.load(file)

        result = {
            'msg': 'succ'
            , 'mlModel': fnlModel
            , 'saveModel': saveModel
            , 'isExist': os.path.exists(saveModel)
        }

        return result

    except Exception as e:
        log.error(f"Exception : {e}")
        return result

def makePycaretModel(subOpt=None, xCol=None, yCol=None, trainData=None, testData=None):

    # log.info(f'[START] makePycaretModel')
    # log.info(f'[CHECK] subOpt : {subOpt}')

    result = None

    try:
        saveModelList = sorted(glob.glob(subOpt['saveModelList'].format(srv = subOpt['srv'])), reverse=True)

        # 학습 모델이 없을 경우
        exp = subOpt['exp']
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):
            xyCol = xCol.copy()
            xyCol.append(yCol)

            trainDataL1 = trainData[xyCol].dropna().copy()
            testDataL1 = testData[xyCol].dropna().copy()

#            trainDataL2 = pd.DataFrame(trainDataL1.values, columns=trainDataL1.columns)
#            testDataL2 = pd.DataFrame(testDataL1.values, columns=testDataL1.columns)

#            trainDataL1 = trainData[xyCol].dropna().reset_index(drop=True)
#            testDataL1 = testData[xyCol].dropna().reset_index(drop=True)

#            trainDataL2 = pd.DataFrame(trainDataL1.values, columns=trainDataL1.columns).infer_objects()
#            testDataL2 = pd.DataFrame(testDataL1.values, columns=testDataL1.columns).infer_objects()


            exp.setup(
                data=trainDataL1,
                test_data=testDataL1,
                session_id=int(datetime.datetime.now().timestamp()),
                target=yCol
            )

            with parallel_backend('threading'):
                modelList = exp.compare_models(sort='RMSE', n_select=3, budget_time=60)
                blendModel = exp.blend_models(estimator_list=modelList, fold=10)
                tuneModel = exp.tune_model(blendModel, fold=10, choose_better=True)
                fnlModel = exp.finalize_model(tuneModel)
#            # 각 모형에 따른 자동 머신러닝
#            modelList = exp.compare_models(sort='RMSE', n_select=3, budget_time=60)
#
#            # 앙상블 모형
#            blendModel = exp.blend_models(estimator_list=modelList, fold=10)
#
#            # 앙상블 파라미터 튜닝
#            tuneModel = exp.tune_model(blendModel, fold=10, choose_better=True)
#
#            # 학습 모형
#            fnlModel = exp.finalize_model(tuneModel)
#            #fnlModel = tuneModel

            # 학습 모형 저장
            saveModel = subOpt['preDt'].strftime(subOpt['saveModel']).format(srv = subOpt['srv'])
            log.info(f'[CHECK] saveModel : {saveModel}')
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)
            exp.save_model(fnlModel, saveModel)
        else:
            saveModel = saveModelList[0]
            log.info(f'[CHECK] saveModel : {saveModel}')
            fnlModel = exp.load_model(os.path.splitext(saveModel)[0])

        result = {
            'msg': 'succ'
            , 'mlModel': fnlModel
            , 'saveModel': saveModel
            , 'isExist': os.path.exists(saveModel)
        }

        return result

    except Exception as e:
        log.error(f"Exception : {e}")
        return result

def makeFlamlModel(subOpt=None, xCol=None, yCol=None, trainData=None, testData=None):

    # log.info(f'[START] makeFlamlModel')
    # log.info(f'[CHECK] subOpt : {subOpt}')

    result = None

    try:
        saveModelList = sorted(glob.glob(subOpt['saveModelList'].format(srv = subOpt['srv'])), reverse=True)

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):
            xyCol = xCol.copy()
            xyCol.append(yCol)
            trainDataL1 = trainData[xyCol].dropna().copy()
            testDataL1 = testData[xyCol].dropna().copy()

            # 7:3에 대한 학습/테스트 분류
            # trainData, validData = train_test_split(dataL1, test_size=0.3)

            # 전체 학습 데이터
            # trainData = dataL1

            fnlModel = AutoML(
                # 연속 예측
                task="regression"
                , metric='rmse'

                # 이진 분류
                # task="classification"
                # , metric='accuracy'

                # , ensemble = False
                , ensemble = True
                # , seed = int(subOpt['preDt'].timestamp())
                , seed = int(datetime.datetime.now().timestamp())
                , time_budget=60
                # , time_budget=600
            )

            # 각 모형에 따른 자동 머신러닝
            fnlModel.fit(X_train=trainDataL1[xCol], y_train=trainDataL1[yCol], X_val=testDataL1[xCol], y_val=testDataL1[yCol])
            # fnlModel.fit(X_train=trainDataL1[xCol], y_train=trainDataL1[yCol], n_jobs=12, n_concurrent_trials=4)

            # 학습 모형 저장
            saveModel = subOpt['preDt'].strftime(subOpt['saveModel']).format(srv = subOpt['srv'])
            log.info(f"[CHECK] saveModel : {saveModel}")
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)

            with open(saveModel, 'wb') as file:
                pickle.dump(fnlModel, file, pickle.HIGHEST_PROTOCOL)
        else:
            saveModel = saveModelList[0]
            log.info(f"[CHECK] saveModel : {saveModel}")

            with open(saveModel, 'rb') as f:
                fnlModel = pickle.load(f)

        result = {
            'msg': 'succ'
            , 'mlModel': fnlModel
            , 'saveModel': saveModel
            , 'isExist': os.path.exists(saveModel)
        }

        return result

    except Exception as e:
        log.error(f"Exception : {e}")
        return result

def makeH2oModel(subOpt=None, xCol=None, yCol=None, trainData=None, testData=None):

    # log.info(f'[START] makeH2oModel')
    # log.info(f'[CHECK] subOpt : {subOpt}')

    result = None

    try:
        saveModelList = sorted(glob.glob(subOpt['saveModelList'].format(srv = subOpt['srv'])), reverse=True)

        if (not subOpt['isInit']):
            h2o.init()
            h2o.no_progress()
            subOpt['isInit'] = True

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):
            xyCol = xCol.copy()
            xyCol.append(yCol)
            trainDataL1 = trainData[xyCol].dropna().copy()
            testDataL1 = testData[xyCol].dropna().copy()

            #dlModel = H2OAutoML(max_models=20, max_runtime_secs=60 * 1, balance_classes=True, seed=int(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
            # dlModel = H2OAutoML(max_models=20, max_runtime_secs=60 * 1, balance_classes=True, seed=int(subOpt['preDt'].timestamp()))
            dlModel = H2OAutoML(max_models=20, max_runtime_secs=60 * 1, balance_classes=True, seed=int(datetime.datetime.now().timestamp()))

            # 각 모형에 따른 자동 머신러닝
            dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(trainDataL1), validation_frame=h2o.H2OFrame(testDataL1))

            fnlModel = dlModel.get_best_model()

            # 학습 모형 저장
            saveModel = subOpt['preDt'].strftime(subOpt['saveModel']).format(srv = subOpt['srv'])
            log.info(f"[CHECK] saveModel : {saveModel}")
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)

            # h2o.save_model(model=fnlModel, path=os.path.dirname(saveModel), filename=os.path.basename(saveModel), force=True)
            fnlModel.save_mojo(path=os.path.dirname(saveModel), filename=os.path.basename(saveModel), force=True)
        else:
            saveModel = saveModelList[0]
            log.info(f"[CHECK] saveModel : {saveModel}")
            fnlModel = h2o.import_mojo(saveModel)

        result = {
            'msg': 'succ'
            , 'mlModel': fnlModel
            , 'saveModel': saveModel
            , 'isExist': os.path.exists(saveModel)
        }

        return result

    except Exception as e:
        log.error(f"Exception : {e}")
        return result


def propUmkr(sysOpt, dtDateInfo):
    try:
        # procInfo = mp.current_process()
        umDataList = []
        efList = sysOpt['UMKR'][f"ef{dtDateInfo.strftime('%H')}"]
        for ef in efList:
            inpFile = dtDateInfo.strftime(sysOpt['UMKR']['inpUmFile']).format(ef=ef)
            fileList = sorted(glob.glob(inpFile))

            for fileInfo in fileList:
                try:
                    grb = pygrib.open(fileInfo)
                    grbInfo = grb.select(name='Temperature')[1]

                    # validIdx = int(re.findall('H\d{3}', fileInfo)[0].replace('H', ''))
                    validIdx = int(ef)
                    dtValidDate = grbInfo.validDate
                    dtAnalDate = grbInfo.analDate

                    row2D = sysOpt['row2D']
                    col2D = sysOpt['col2D']
                    uVec = grb.select(name='10 metre U wind component')[0].values[row2D, col2D]
                    vVec = grb.select(name='10 metre V wind component')[0].values[row2D, col2D]
                    WD = (270 - np.rad2deg(np.arctan2(vVec, uVec))) % 360
                    WS = np.sqrt(np.square(uVec) + np.square(vVec))
                    PA = grb.select(name='Surface pressure')[0].values[row2D, col2D]
                    TA = grb.select(name='Temperature')[0].values[row2D, col2D]
                    TD = grb.select(name='Dew point temperature')[0].values[row2D, col2D]
                    HM = grb.select(name='Relative humidity')[0].values[row2D, col2D]
                    lowCA = grb.select(name='Low cloud cover')[0].values[row2D, col2D]
                    medCA = grb.select(name='Medium cloud cover')[0].values[row2D, col2D]
                    higCA = grb.select(name='High cloud cover')[0].values[row2D, col2D]
                    CA_TOT = np.mean([lowCA, medCA, higCA], axis=0)
                    SWR = grb.select(name='unknown')[0].values[row2D, col2D]

                    lat1D = sysOpt['lat1D']
                    lon1D = sysOpt['lon1D']
                    umData = xr.Dataset(
                        {
                            'uVec': (('anaTime', 'time', 'lat', 'lon'), (uVec).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'vVec': (('anaTime', 'time', 'lat', 'lon'), (vVec).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'WD': (('anaTime', 'time', 'lat', 'lon'), (WD).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'WS': (('anaTime', 'time', 'lat', 'lon'), (WS).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'PA': (('anaTime', 'time', 'lat', 'lon'), (PA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'TA': (('anaTime', 'time', 'lat', 'lon'), (TA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'TD': (('anaTime', 'time', 'lat', 'lon'), (TD).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'HM': (('anaTime', 'time', 'lat', 'lon'), (HM).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'lowCA': (('anaTime', 'time', 'lat', 'lon'), (lowCA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'medCA': (('anaTime', 'time', 'lat', 'lon'), (medCA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'higCA': (('anaTime', 'time', 'lat', 'lon'), (higCA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'CA_TOT': (('anaTime', 'time', 'lat', 'lon'),(CA_TOT).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'SWR': (('anaTime', 'time', 'lat', 'lon'), (SWR).reshape(1, 1, len(lat1D), len(lon1D)))
                        }
                        , coords={
                            'anaTime': pd.date_range(dtAnalDate, periods=1)
                            , 'time': pd.date_range(dtValidDate, periods=1)
                            , 'lat': lat1D
                            , 'lon': lon1D
                        }
                    )

                    # umDataL1 = xr.merge([umDataL1, umData])
                    umDataList.append(umData)
                except Exception as e:
                    log.error(f"Exception : {e}")

        if len(umDataList) < 1: return
        umDataL1 = xr.concat(umDataList, dim='time')

        posDataL1 = sysOpt['posDataL1']
        for kk, posInfo in posDataL1.iterrows():
            srv = posInfo['srv']
            posLat = posInfo['lat']
            posLon = posInfo['lon']

            # DB 적재
            with cfgDb['sessionMake']() as session:
                with session.begin():
                    tbTmp = f"tmp_{uuid.uuid4().hex}"
                    conn = session.connection()

                    try:
                        umDataL2 = umDataL1.sel(lat=posLat, lon=posLon)
                        umDataL3 = umDataL2.to_dataframe().dropna().reset_index(drop=False)
                        umDataL3['ana_date'] = umDataL3['anaTime']
                        umDataL3['date_time'] = umDataL3['time']
                        umDataL3['date_time_kst'] = umDataL3['time'] + dtKst
                        umDataL3['srv'] = srv

                        umDataL3['ta'] = umDataL3['TA'] - 273.15
                        umDataL3['td'] = umDataL3['TD'] - 273.15
                        umDataL3['pa'] = umDataL3['PA'] / 100.0
                        umDataL3['ca_tot'] = np.where(umDataL3['CA_TOT'] < 0, 0, umDataL3['CA_TOT'])
                        umDataL3['ca_tot'] = np.where(umDataL3['ca_tot'] > 1, 1, umDataL3['ca_tot'])
                        umDataL3['wd'] = umDataL3['WD']
                        umDataL3['ws'] = umDataL3['WS']
                        umDataL3['hm'] = umDataL3['HM']
                        umDataL3['swr'] = umDataL3['SWR']

                        solPosInfo = pvlib.solarposition.get_solarposition(umDataL3['date_time'], posLat, posLon,
                                                                           pressure=umDataL3['pa'] * 100.0,
                                                                           temperature=umDataL3['ta'], method='nrel_numpy')
                        umDataL3['ext_rad'] = pvlib.irradiance.get_extra_radiation(solPosInfo.index.dayofyear)
                        umDataL3['sza'] = solPosInfo['zenith'].values
                        umDataL3['aza'] = solPosInfo['azimuth'].values
                        umDataL3['et'] = solPosInfo['equation_of_time'].values

                        site = location.Location(latitude=posLat, longitude=posLon, tz='Asia/Seoul')
                        clearInsInfo = site.get_clearsky(pd.to_datetime(umDataL3['date_time'].values))
                        umDataL3['ghi_clr'] = clearInsInfo['ghi'].values
                        umDataL3['dni_clr'] = clearInsInfo['dni'].values
                        umDataL3['dhi_clr'] = clearInsInfo['dhi'].values

                        turbidity = pvlib.clearsky.lookup_linke_turbidity(pd.to_datetime(umDataL3['date_time'].values), posLat, posLon, interp_turbidity=True)
                        umDataL3['turb'] = turbidity.values

                        dbColList = [
                            "srv", "ana_date", "date_time", "date_time_kst",
                            "ca_tot", "hm", "pa", "ta", "td", "wd", "ws",
                            "sza", "aza", "et", "turb",
                            "ghi_clr", "dni_clr", "dhi_clr", "swr", "ext_rad"
                        ]
                        umDataL4 = umDataL3[dbColList].copy()

                        # DB 적재
                        umDataL4.to_sql(
                            name=tbTmp,
                            con=conn,
                            if_exists="replace",
                            index=False,
                            chunksize=1000
                        )

                        query = text(f"""
                            INSERT INTO tb_for_data (
                                  srv, ana_date, date_time, date_time_kst,
                                  ca_tot, hm, pa, ta, td, wd, ws,
                                  sza, aza, et, turb,
                                  ghi_clr, dni_clr, dhi_clr, swr, ext_rad,
                                  reg_date
                            )
                            SELECT
                                  srv, ana_date, date_time, date_time_kst,
                                  ca_tot, hm, pa, ta, td, wd, ws,
                                  sza, aza, et, turb,
                                  ghi_clr, dni_clr, dhi_clr, swr, ext_rad,
                                  now()
                            FROM {tbTmp}
                            ON CONFLICT (srv, ana_date, date_time)
                            DO UPDATE SET
                                  date_time_kst = excluded.date_time_kst,
                                  ca_tot = excluded.ca_tot, 
                                  hm = excluded.hm, 
                                  pa = excluded.pa, 
                                  ta = excluded.ta, 
                                  td = excluded.td, 
                                  wd = excluded.wd, 
                                  ws = excluded.ws,
                                  sza = excluded.sza, 
                                  aza = excluded.aza, 
                                  et = excluded.et, 
                                  turb = excluded.turb,
                                  ghi_clr = excluded.ghi_clr, 
                                  dni_clr = excluded.dni_clr, 
                                  dhi_clr = excluded.dhi_clr, 
                                  swr = excluded.swr,
                                  ext_rad = excluded.ext_rad,
                                  mod_date = now();
                              """)
                        result = session.execute(query)
                        log.info(f"dtDateInfo : {dtDateInfo} / srv : {srv} / result : {result.rowcount}")
                    except Exception as e:
                        log.error(f"Exception : {e}")
                        raise e
                    finally:
                        session.execute(text(f"DROP TABLE IF EXISTS {tbTmp}"))
    except Exception as e:
        log.error(f'Exception : {e}')

def subPvProc(sysOpt, cfgDb):
    try:
        with cfgDb['sessionMake']() as session:
            query = text("""
                         SELECT srv, date_time
                         FROM tb_pv_data
                         WHERE pv > 0 
                           AND date_time BETWEEN :srtDate AND :endDate;
                         """)
            cfgData = pd.DataFrame(session.execute(query, {'srtDate': sysOpt['srtDate'], 'endDate': sysOpt['endDate']}))
            cfgDataL1 = cfgData[cfgData['srv'].isin(sysOpt['posDataL1']['srv'])].reset_index(drop=True)
            cfgDataList = set(zip(cfgDataL1['srv'], cfgDataL1['date_time'].dt.strftime('%Y-%m-%d')))

        dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
        dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
        dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

        resDataL1 = pd.DataFrame()
        for dtDateInfo in reversed(dtDateList):
            for i, posInfo in sysOpt['posDataL1'].iterrows():
                id = posInfo['id']
                srv = posInfo['srv']

                if (srv, dtDateInfo.strftime('%Y-%m-%d')) in cfgDataList: continue

                reqUrl = dtDateInfo.strftime(sysOpt['cfgApi']['url']).format(id=id, token=sysOpt['cfgApi']['token'])
                res = requests.get(reqUrl)
                if not res.status_code == 200: continue
                resJson = res.json()

                if not (resJson['success'] == True): continue
                resInfo = resJson['pvs']
                if len(resInfo) < 1: continue
                resData = pd.DataFrame(resInfo).rename(
                    {
                        'pv': 'pv'
                    }
                    , axis='columns'
                )

                resData['srv'] = srv
                resData['date_time_kst'] = pd.to_datetime(resData['date'], format='%Y-%m-%d %H')
                resData['date_time'] = resData['date_time_kst'] - dtKst

                resDataL1 = pd.concat([resDataL1, resData], ignore_index=True)

        if len(resDataL1) < 1: return
        with cfgDb['sessionMake']() as session:
            with session.begin():
                tbTmp = f"tmp_{uuid.uuid4().hex}"
                conn = session.connection()

                try:
                    # DB 적재
                    resDataL1.to_sql(
                        name=tbTmp,
                        con=conn,
                        if_exists="replace",
                        index=False,
                        chunksize=1000
                    )

                    query = text(f"""
                        INSERT INTO tb_pv_data (
                            srv, date_time, date_time_kst, pv, reg_date
                        )
                        SELECT 
                            srv, date_time, date_time_kst, pv, now()
                        FROM {tbTmp}
                        ON CONFLICT (srv, date_time)
                        DO UPDATE SET
                            date_time_kst = excluded.date_time_kst,
                            pv = excluded.pv, 
                            mod_date = now();
                        """)
                    result = session.execute(query)
                    log.info(f"id : {id} / dtDateInfo : {dtDateInfo} / id : {id} / result : {result.rowcount}")
                except Exception as e:
                    log.error(f"Exception : {e}")
                    raise e
                finally:
                    session.execute(text(f"DROP TABLE IF EXISTS {tbTmp}"))
    except Exception as e:
        log.error(f'Exception : {e}')
        raise e

def initWorker(cfbDbInfo):
    global cfgDb
    cfgDb = cfbDbInfo

def subPropProc(sysOpt, cfgDb):
    try:
        cfgUmFile = sysOpt['UMKR']['cfgUmFile']
        log.info(f"cfgUmFile : {cfgUmFile}")

        cfgInfo = pygrib.open(cfgUmFile).select(name='Temperature')[1]
        lat2D, lon2D = cfgInfo.latlons()

        # 최근접 좌표
        posList = []

        # kdTree를 위한 초기 데이터
        for i in range(0, lon2D.shape[0]):
            for j in range(0, lon2D.shape[1]):
                coord = [lat2D[i, j], lon2D[i, j]]
                posList.append(cartesian(*coord))

        tree = spatial.KDTree(posList)

        row1D = []
        col1D = []
        for ii, posInfo in sysOpt['posDataL1'].iterrows():
            # coord = cartesian(posInfo['LAT'], posInfo['LON'])
            coord = cartesian(posInfo['lat'], posInfo['lon'])
            closest = tree.query([coord], k=1)
            cloIdx = closest[1][0]
            row = int(cloIdx / lon2D.shape[1])
            col = cloIdx % lon2D.shape[1]
            row1D.append(row)
            col1D.append(col)
        sysOpt['row2D'], sysOpt['col2D'] = np.meshgrid(row1D, col1D)

        # **************************************************************************************************************
        # 비동기 다중 프로세스 수행
        # **************************************************************************************************************
        # 비동기 다중 프로세스 개수
        # pool = Pool(int(sysOpt['cpuCoreNum']))
        pool = Pool(
            processes=int(sysOpt['cpuCoreNum']),
            initializer=initWorker,
            initargs=(cfgDb,)
        )

        with cfgDb['sessionMake']() as session:
            query = text("""
                         SELECT srv, ana_date, date_time
                         FROM tb_for_data
                         WHERE swr > 0
                           AND ana_date BETWEEN :srtDate AND :endDate;
                         """)
            cfgData = pd.DataFrame(session.execute(query, {'srtDate': sysOpt['srtDate'], 'endDate': sysOpt['endDate']}))
            cfgDataL1 = cfgData[cfgData['srv'].isin(sysOpt['posDataL1']['srv'])].reset_index(drop=True)

            cfgDataL2 = cfgDataL1.groupby(['ana_date', 'date_time']).size().reset_index(name='cnt')
            cfgDataL3 = cfgDataL2[cfgDataL2['cnt'] == len(sysOpt['posDataL1']['srv'])]
            cfgDataList = set(zip(cfgDataL3['ana_date'].dt.strftime('%Y-%m-%d %H:%M')))

        dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
        dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
        dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['UMKR']['invDate'])
        for dtDateInfo in reversed(dtDateList):
            if (dtDateInfo.strftime('%Y-%m-%d %H:%M'), ) in cfgDataList: continue

            # propUmkr(sysOpt, dtDateInfo)
            pool.apply_async(propUmkr, args=(sysOpt, dtDateInfo))
        pool.close()
        pool.join()
    except Exception as e:
        log.error(f'Exception : {e}')
        raise e

def subModelProc(sysOpt, cfgDb):
    try:
        for i, posInfo in sysOpt['posDataL1'].iterrows():
            id = posInfo['id']
            srv = posInfo['srv']

            with cfgDb['sessionMake']() as session:
                with session.begin():
                    conn = session.connection()
                    tbTmp = f"tmp_{uuid.uuid4().hex}"

                    try:
                        query = text("""
                                     SELECT pv."pv",
                                            lf.*
                                     FROM "tb_pv_data" AS pv
                                              LEFT JOIN
                                          "tb_for_data" AS lf ON pv."srv" = lf."srv" AND pv."date_time" = lf."date_time"
                                     WHERE pv."srv" = :srv
                                       AND pv.pv IS NOT NULL
                                       AND (EXTRACT(EPOCH FROM (lf."date_time" - lf."ana_date")) / 3600.0) <= 5
                                     ORDER BY "srv", "date_time_kst" DESC;
                                     """)

                        # trainData = pd.DataFrame(session.execute(query, {'srv':srv}))
                        # trainData = data[data['DATE_TIME_KST'] < pd.to_datetime('2025-01-01')].reset_index(drop=True)
                        # testData = data[data['DATE_TIME_KST'] >= pd.to_datetime('2025-01-01')].reset_index(drop=True)
                        data = pd.DataFrame(session.execute(query, {'srv': srv}))
                        if data['pv'].sum() == 0:
                            log.info(f"srv : {srv} / pv : {data['pv'].sum()}")
                            continue

                        trainData, testData = train_test_split(data, test_size=0.2, random_state=int(datetime.datetime.now().timestamp()))

                        query = text("""
                                     SELECT lf.*
                                     FROM "tb_for_data" AS lf
                                     WHERE lf."srv" = :srv
                                       AND (
                                         "ml" IS NULL
                                             OR "dl" IS NULL
                                             OR "ai" IS NULL
                                             OR "ai2" IS NULL
                                             OR "ai3" IS NULL
                                         )
                                     ORDER BY "srv", "date_time_kst" DESC;
                                     """)
                        prdData = pd.DataFrame(session.execute(query, {'srv': srv}))
                        if len(prdData) < 1: continue

                        for key in ['orgPycaret', 'orgH2o', 'lgb', 'flaml', 'pycaret']:
                            sysOpt['MODEL'][key]['srv'] = srv

                        exp = RegressionExperiment()
                        sysOpt['MODEL']['orgPycaret']['exp'] = exp
                        sysOpt['MODEL']['pycaret']['exp'] = exp

                        # ****************************************************************************
                        # 독립/종속 변수 설정
                        # ****************************************************************************
                        xColOrg = ['ca_tot', 'hm', 'pa', 'ta', 'td', 'wd', 'ws', 'sza', 'aza', 'et', 'swr']
                        yColOrg = 'pv'
                        xCol = ['ca_tot', 'hm', 'pa', 'ta', 'td', 'wd', 'ws', 'sza', 'aza', 'et', 'turb', 'ghi_clr', 'dni_clr', 'dhi_clr', 'swr', 'ext_rad']
                        yCol = 'pv'

                        # ****************************************************************************
                        # 과거 학습 모델링 (orgPycaret)
                        # ****************************************************************************
                        resOrgPycaret = makePycaretModel(sysOpt['MODEL']['orgPycaret'], xColOrg, yColOrg, trainData, testData)
                        # log.info(f'resOrgPycaret : {resOrgPycaret}')

                        if resOrgPycaret:
                            prdVal = exp.predict_model(resOrgPycaret['mlModel'], data=prdData[xCol])['prediction_label']
                            prdData['ml'] = np.where(prdVal > 0, prdVal, 0)

                        # ****************************************************************************
                        # 과거 학습 모델링 (orgH2o)
                        # ****************************************************************************
                        resOrgH2o = makeH2oModel(sysOpt['MODEL']['orgH2o'], xColOrg, yColOrg, trainData, testData)
                        # log.info(f'resOrgH2o : {resOrgH2o}')

                        if resOrgH2o:
                            prdVal = resOrgH2o['mlModel'].predict(h2o.H2OFrame(prdData)).as_data_frame()
                            prdData['dl'] = np.where(prdVal > 0, prdVal, 0)

                        # ****************************************************************************
                        # 수동 학습 모델링 (lgb)
                        # ****************************************************************************
                        resLgb = makeLgbModel(sysOpt['MODEL']['lgb'], xCol, yCol, trainData, testData)
                        # log.info(f'resLgb : {resLgb}')

                        if resLgb:
                            prdVal = resLgb['mlModel'].predict(data=prdData[xCol])
                            prdData['ai'] = np.where(prdVal > 0, prdVal, 0)

                        # ****************************************************************************
                        # 자동 학습 모델링 (flaml)
                        # ****************************************************************************
                        resFlaml = makeFlamlModel(sysOpt['MODEL']['flaml'], xCol, yCol, trainData, testData)
                        # log.info(f'resFlaml : {resFlaml}')

                        if resFlaml:
                            prdVal = resFlaml['mlModel'].predict(prdData)
                            prdData['ai2'] = np.where(prdVal > 0, prdVal, 0)

                        # ****************************************************************************
                        # 자동 학습 모델링 (pycaret)
                        # ****************************************************************************
                        resPycaret = makePycaretModel(sysOpt['MODEL']['pycaret'], xColOrg, yColOrg, trainData, testData)
                        # log.info(f'resPycaret : {resPycaret}')
                        
                        if resPycaret:
                            prdVal = exp.predict_model(resPycaret['mlModel'], data=prdData[xCol])['prediction_label']
                            prdData['ai3'] = np.where(prdVal > 0, prdVal, 0)

                        # DB 적재
                        prdData.to_sql(
                            name=tbTmp,
                            con=conn,
                            if_exists="replace",
                            index=False,
                            chunksize=1000
                        )

                        query = text(f"""
                              INSERT INTO tb_for_data (
                                    srv, ana_date, date_time, dl, ml, ai, ai2
                              )
                              SELECT
                                    srv, ana_date, date_time, dl, ml, ai, ai2
                              FROM {tbTmp}
                              ON CONFLICT (srv, ana_date, date_time)
                              DO UPDATE SET
                                  dl = excluded.dl,
                                  ml = excluded.ml,
                                  ai = excluded.ai,
                                  ai2 = excluded.ai2,
                                  ai3 = excluded.ai3,
                           """)
                        result = session.execute(query)
                        log.info(f"id : {id} / result : {result.rowcount}")

                        if result.rowcount > 0:
                            query = text("""
                                         UPDATE tb_stn_info
                                         SET init_yn  = 'N',
                                             mod_date = now()
                                         WHERE id = :id
                                         """)
                            session.execute(query, {'id': id})

                    except Exception as e:
                        log.error(f"Exception : {e}")
                        raise e
                    finally:
                        session.execute(text(f"DROP TABLE IF EXISTS {tbTmp}"))
    except Exception as e:
        log.error(f'Exception : {e}')
        raise e

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar, cfgDb

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'
        # contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON'

    prjName = 'test'
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
                'srtDate': globalVar['srtDate'],
                'endDate': globalVar['endDate'],
                # 'srtDate': '2020-01-01',
                # 'endDate': None,
                'invDate': '1d',

                # 비동기 다중 프로세스 개수
                # 'cpuCoreNum': globalVar['cpuCoreNum'],
                'cpuCoreNum': '5',
                # 'cpuCoreNum': '10',
                # 'cpuCoreNum': '1',

                # 설정 파일
                # 'cfgFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
                # 'cfgFile': '/vol01/SYSTEMS/INDIAI/PROG/PYTHON/resources/config/system.cfg',
                'cfgFile': '/SYSTEMS/PROG/PYTHON/resources/config/system.cfg',
                'cfgDbKey': 'postgresql-qubesoft.iptime.org-qubesoft-dms02',
                'posDataL1': None,
                'cfgApiKey': 'pv',
                'cfgApi': None,
                'row2D': None,
                'col2D': None,
                'lat1D': None,
                'lon1D': None,

                # 예보 모델
                'UMKR': {
                    # 'cfgUmFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/modelInfo/UMKR_l015_unis_H000_202110010000.grb2',
                    # 'inpUmFile': '/HDD/DATA/MODEL/%Y%m/%d/UMKR_l015_unis_H{ef}_%Y%m%d%H%M.grb2',
                    # 'cfgUmFile': '/DATA/COLCT/UMKR/201901/01/UMKR_l015_unis_H00_201901010000.grb2',
                    # 'inpUmFile': '/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H{ef}_%Y%m%d%H%M.grb2',
                    'cfgUmFile': '/DATA/MODEL/202001/01/UMKR_l015_unis_H00_202001010000.grb2',
                    'inpUmFile': '/DATA/MODEL/%Y%m/%d/UMKR_l015_unis_H{ef}_%Y%m%d%H%M.grb2',
                    'ef00': ['00', '01', '02', '03', '04', '05', '15', '16', '17', '18', '19', '20', '21', '22', '23','24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38'],
                    # 'ef00': ['00', '01', '02', '03', '04', '05'],
                    'ef06': ['00', '01', '02', '03', '04', '05'],
                    'ef12': ['00', '01', '02', '03', '04', '05'],
                    'ef18': ['00', '01', '02', '03', '04', '05', '21', '22', '23', '24', '25', '26', '27', '28', '29','30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44'],
                    # 'ef18': ['00', '01', '02', '03', '04', '05'],
                    'invDate': '6h',
                },
                # 자동화/수동화 모델링
                'MODEL': {
                    'orgPycaret': {
                        'saveModelList': "/DATA/AI/*/*/LSH0255-{srv}-final-pycaret-for-*.model.pkl",
                        'saveModel': "/DATA/AI/%Y%m/%d/LSH0255-{srv}-final-pycaret-for-%Y%m%d.model",
                        # 'isOverWrite': True,
                        'isOverWrite': False,
                        'srv': None,
                        'preDt': datetime.datetime.now(),
                        'exp': None
                    },
                    'orgH2o': {
                        'saveModelList': "/DATA/AI/*/*/LSH0255-{srv}-final-h2o-for-*.model",
                        'saveModel': "/DATA/AI/%Y%m/%d/LSH0255-{srv}-final-h2o-for-%Y%m%d.model",
                        'isInit': False,
                        # 'isOverWrite': True,
                        'isOverWrite': False,
                        'srv': None,
                        'preDt': datetime.datetime.now(),
                    },
                    'lgb': {
                        'saveModelList': "/DATA/AI/*/*/QUBE2025-{srv}-final-lgb-for-*.model",
                        'saveModel': "/DATA/AI/%Y%m/%d/QUBE2025-{srv}-final-lgb-for-%Y%m%d.model",
                        'saveImg': "/DATA/AI/%Y%m/%d/QUBE2025-{srv}-final-lgb-for-%Y%m%d.png",
                        'isOverWrite': True,
                        # 'isOverWrite': False,
                        'srv': None,
                        'preDt': datetime.datetime.now(),
                    },
                    'flaml': {
                        'saveModelList': "/DATA/AI/*/*/QUBE2025-{srv}-final-flaml-for-*.model",
                        'saveModel': "/DATA/AI/%Y%m/%d/QUBE2025-{srv}-final-flaml-for-%Y%m%d.model",
                        'saveImg': "/DATA/AI/%Y%m/%d/QUBE2025-{srv}-final-flaml-for-%Y%m%d.png",
                        # 'isOverWrite': True,
                        'isOverWrite': False,
                        'srv': None,
                        'preDt': datetime.datetime.now(),
                    },
                    'pycaret': {
                        'saveModelList': "/DATA/AI/*/*/QUBE2025-{srv}-final-pycaret-for-*.model.pkl",
                        'saveModel': "/DATA/AI/%Y%m/%d/QUBE2025-{srv}-final-pycaret-for-%Y%m%d.model",
                        'saveImg': "/DATA/AI/%Y%m/%d/QUBE2025-{srv}-final-pycaret-for-%Y%m%d.png",
                        # 'isOverWrite': True,
                        'isOverWrite': False,
                        'srv': None,
                        'preDt': datetime.datetime.now(),
                        'exp': None
                    },
                },
            }

            # *******************************************************
            # 설정 정보
            # *******************************************************
            config = configparser.ConfigParser(interpolation=None)
            config.read(sysOpt['cfgFile'], encoding='utf-8')

            # sysOpt['cfgDb'] = initCfgInfo(config, sysOpt['cfgDbKey'])
            cfgDb = initCfgInfo(config, sysOpt['cfgDbKey'])
            cfgApi = {
                'url': config.get(sysOpt['cfgApiKey'], 'url'),
                'token': config.get(sysOpt['cfgApiKey'], 'token'),
            }
            sysOpt['cfgApi'] = cfgApi
            # sysOpt['endDate'] = datetime.datetime.now().strftime('%Y-%m-%d')

            # 관측소 정보
            with cfgDb['sessionMake']() as session:
                query = text("""
                             SELECT *, 'SRV' || LPAD(id::text, 5, '0') as srv
                             FROM tb_stn_info
                             WHERE oper_yn = 'Y'
                             ORDER BY id ASC;
                             """)

                posDataL1 = pd.DataFrame(session.execute(query))
                log.info(f'posDataL1 : {posDataL1}')

            sysOpt['posDataL1'] = posDataL1
            sysOpt['lat1D'] = np.array(posDataL1['lat'])
            sysOpt['lon1D'] = np.array(posDataL1['lon'])

            subPvProc(sysOpt, cfgDb)
            subPropProc(sysOpt, cfgDb)
            subModelProc(sysOpt, cfgDb)

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
