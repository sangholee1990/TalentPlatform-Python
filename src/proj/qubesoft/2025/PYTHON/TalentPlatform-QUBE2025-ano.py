# ================================================
# 요구사항
# ================================================
# Python을 이용한 기상청 데이터 모델링 및 DB 적재

# 프로그램 종료
# ps -ef | grep python | grep TalentPlatform-QUBE2025-model-save-ano.py | awk '{print $2}' | xargs kill -9
# pkill -f TalentPlatform-QUBE2025-model-save-ano.py

# 프로그램 시작
# conda activate py38

# cd /SYSTEMS/PROG/PYTHON
# /SYSTEMS/LIB/anaconda3/envs/py39/bin/python TalentPlatform-QUBE2025-model-save-ano.py
# nohup /SYSTEMS/LIB/anaconda3/envs/py39/bin/python TalentPlatform-QUBE2025-model-save-ano.py &

# 20,50 * * * * cd /SYSTEMS/PROG/PYTHON && /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/TalentPlatform-QUBE2025-model-save-ano.py

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

from pycaret.anomaly import AnomalyExperiment
import plotly.express as px
import plotly.graph_objects as go

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

def makeLgbModel(subOpt=None, xCol=None, yCol=None, trainData=None, testData=None):

    # log.info(f'[START] makeLgbModel')
    # log.info(f'[CHECK] subOpt : {subOpt}')

    result = None

    try:

        saveModelList = sorted(glob.glob(subOpt['saveModelList'].format(srvId = subOpt['srvId'])), reverse=True)

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
            saveModel = subOpt['preDt'].strftime(subOpt['saveModel']).format(srvId = subOpt['srvId'])
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
        saveModelList = sorted(glob.glob(subOpt['saveModelList'].format(srvId = subOpt['srvId'])), reverse=True)

        # 학습 모델이 없을 경우
        exp = subOpt['exp']
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):
            xyCol = xCol.copy()
            xyCol.append(yCol)
            trainDataL1 = trainData[xyCol].dropna().copy()
            testDataL1 = testData[xyCol].dropna().copy()

            exp.setup(
                data=trainDataL1,
                test_data=testDataL1,
                session_id=int(datetime.datetime.now().timestamp()),
                target=yCol,
            )

            # 각 모형에 따른 자동 머신러닝
            modelList = exp.compare_models(sort='RMSE', n_select=3, budget_time=60)

            # 앙상블 모형
            blendModel = exp.blend_models(estimator_list=modelList, fold=10)

            # 앙상블 파라미터 튜닝
            tuneModel = exp.tune_model(blendModel, fold=10, choose_better=True)

            # 학습 모형
            # fnlModel = exp.finalize_model(tuneModel)
            fnlModel = tuneModel

            # 학습 모형 저장
            saveModel = subOpt['preDt'].strftime(subOpt['saveModel']).format(srvId = subOpt['srvId'])
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
        saveModelList = sorted(glob.glob(subOpt['saveModelList'].format(srvId = subOpt['srvId'])), reverse=True)

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
            saveModel = subOpt['preDt'].strftime(subOpt['saveModel']).format(srvId = subOpt['srvId'])
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
        saveModelList = sorted(glob.glob(subOpt['saveModelList'].format(srvId = subOpt['srvId'])), reverse=True)

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
            saveModel = subOpt['preDt'].strftime(subOpt['saveModel']).format(srvId = subOpt['srvId'])
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
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        # contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON'

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
                # 'srtDate': globalVar['srtDate'],
                # 'endDate': globalVar['endDate'],
                # 'srtDate': '2021-01-01',
                # 'endDate': '2025-11-01',

                # 비동기 다중 프로세스 개수
                # 'cpuCoreNum': globalVar['cpuCoreNum'],
                # 'cpuCoreNum': '5',

                # 설정 파일
                'cfgFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
                # 'cfgFile': '/vol01/SYSTEMS/INDIAI/PROG/PYTHON/resources/config/system.cfg',
                # 'cfgFile': '/SYSTEMS/PROG/PYTHON/resources/config/system.cfg',
                'cfgDbKey': 'postgresql-qubesoft.iptime.org-qubesoft-dms02',
                'cfgDb': None,
                'posDataL1': None,

                # 자동화/수동화 모델링
                'MODEL': {
                    'pycaret': {
                        'saveModelList': "/DATA/AI/*/*/QUBE2025-{srvId}-final-pycaret-ano-*.model.pkl",
                        'saveModel': "/DATA/AI/%Y%m/%d/QUBE2025-{srvId}-final-pycaret-ano-%Y%m%d.model",
                        # 'isOverWrite': True,
                        'isOverWrite': False,
                        'srvId': None,
                        'preDt': datetime.datetime.now(),
                        'exp': None
                    },
                    'h2o': {
                        'saveModelList': "/DATA/AI/*/*/QUBE2025-{srvId}-final-h2o-ano-*.model",
                        'saveModel': "/DATA/AI/%Y%m/%d/QUBE2025-{srvId}-final-h2o-ano-%Y%m%d.model",
                        'isInit': False,
                        # 'isOverWrite': True,
                        'isOverWrite': False,
                        'srvId': None,
                        'preDt': datetime.datetime.now(),
                    },
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
                    srv = posInfo['srv']
                    query = text("""
                        SELECT "date_time_kst", "pv"
                        FROM "tb_pv_data"
                        WHERE "srv" = :srv
                        ORDER BY "srv", "date_time_kst" DESC;
                     """)

                    # trainData = pd.DataFrame(session.execute(query, {'srvId':srvId}))
                    # trainData = data[data['DATE_TIME_KST'] < pd.to_datetime('2025-01-01')].reset_index(drop=True)
                    # testData = data[data['DATE_TIME_KST'] >= pd.to_datetime('2025-01-01')].reset_index(drop=True)
                    data = pd.DataFrame(session.execute(query, {'srv':srv}))
                    dataL1 = data[(data['date_time_kst'].dt.hour >= 6) & (data['date_time_kst'].dt.hour <= 20)].reset_index(drop=True)

                    # dataL1 = data_copy[(data['date_time_kst'].dt.hour >= 7) & (
                    #             data['date_time_kst'].dt.hour <= 19)].reset_index(drop=True)


                    dataL1['date_time_kst'] = pd.to_datetime(dataL1['date_time_kst'])
                    dataL1['year'] = dataL1['date_time_kst'].dt.year
                    dataL1['month'] = dataL1['date_time_kst'].dt.month
                    dataL1['day'] = dataL1['date_time_kst'].dt.day
                    dataL1['hour'] = dataL1['date_time_kst'].dt.hour
                    dataL1['min'] = dataL1['date_time_kst'].dt.minute
                    dataL1['dayofyear'] = dataL1['date_time_kst'].dt.dayofyear
                    dataL1['dayofweek'] = dataL1['date_time_kst'].dt.dayofweek

                    exp = AnomalyExperiment()
                    exp.setup(
                        data=dataL1,
                        # ignore_features=['date_time_kst'],
                        normalize=True,
                        session_id=int(datetime.datetime.now().timestamp()),
                    )

                    # 기본 국밥 모델인 iforest를 먼저 써보고, 아쉬우면 lof로 바꿔보세요.
                    model = exp.create_model("iforest")

                    # 5. 이상치 시각화 (반드시 df가 아닌 'model' 변수를 넣어야 합니다!)
                    # 3D 뷰어 등으로 데이터가 어떻게 뭉쳐있는지 확인
                    # exp.plot_model(model, plot='tsne')

                    # 6. 최종 결과 라벨링 (원래 데이터에 Anomaly, Anomaly_Score 컬럼 추가)
                    df_anomaly = exp.assign_model(model)

                    # 7. 진짜 이상치(1)로 판별된 녀석들만 추출해서 확인
                    outliers = df_anomaly[df_anomaly['Anomaly'] == 1].reset_index(drop=True)
                    print(outliers.head())

                    # # time series anomaly detection methods
                    # from deepod.models.time_series import TimesNet
                    # clf = TimesNet()
                    # clf.fit(X_train)
                    # scores = clf.decision_function(X_test)
                    #
                    # # evaluation of time series anomaly detection
                    # from deepod.metrics import ts_metrics
                    # from deepod.metrics import point_adjustment  # execute point adjustment for time series ad
                    # eval_metrics = ts_metrics(labels, scores)
                    # adj_eval_metrics = ts_metrics(labels, point_adjustment(labels, scores))

                    import pandas as pd
                    import numpy as np
                    from sklearn.preprocessing import StandardScaler
                    from deepod.models.time_series import TimesNet

                    # 1. 데이터 준비
                    # 딥러닝은 PyCaret과 다르게 밤 시간대(0)도 데이터의 중요한 '흐름'으로 인식하므로
                    # 낮 시간만 필터링하지 않고 전체 데이터를 시간순으로 정렬해서 넣는 것이 좋습니다.
                    df = data.copy()
                    df['date_time_kst'] = pd.to_datetime(df['date_time_kst'])
                    df = df.sort_values(by='date_time_kst').reset_index(drop=True)

                    # 2. 데이터 분할 (Train / Test)
                    # 시계열 딥러닝은 무작위 분할이 아닌 '과거'로 학습하고 '최근'을 테스트합니다.
                    # 예: 과거 80% 데이터로 정상 패턴 학습 -> 최근 20% 데이터에서 이상치 탐지
                    # train_size = int(len(df) * 0.8)
                    # train_df = df.iloc[:train_size].copy()
                    # test_df = df.iloc[train_size:].copy()

                    # 3. 스케일링 (딥러닝 필수 과정)
                    # 발전량 수치(수백~수천)를 딥러닝이 소화하기 좋게 압축합니다.
                    # scaler = StandardScaler()
                    # X_train = scaler.fit_transform(train_df[['pv']].values)
                    # X_test = scaler.transform(test_df[['pv']].values)

                    scaler = StandardScaler()
                    X_all = scaler.fit_transform(df[['pv']].values)

                    # 4. TimesNet 모델 세팅 및 학습
                    # seq_len=24 : 24시간 단위로 잘라서 하루의 주기성을 학습하라는 의미 (매우 중요!)
                    print("모델 학습을 시작합니다. (시간이 다소 소요될 수 있습니다)")
                    # clf = TimesNet(seq_len=24, epochs=10, device='cpu')
                    # clf.fit(X_train)
                    clf = TimesNet(seq_len=24, epochs=10, device='cpu')
                    clf.fit(X_all)

                    # 5. Test 데이터(최근 20%)에 대한 이상치 점수 도출
                    # scores = clf.decision_function(X_test)
                    # test_df['Anomaly_Score'] = scores
                    scores = clf.decision_function(X_all)
                    df['Anomaly_Score'] = scores

                    df_sorted = df.sort_values(by='Anomaly_Score', ascending=False).reset_index(drop=True)


                    # 6. 임계치(Threshold) 설정으로 최종 이상치 판별
                    # 정답(Label)이 없으므로, 비정상적으로 튀는 상위 1% 데이터를 이상치로 정의합니다.
                    # threshold = np.percentile(scores, 99)
                    # test_df['Anomaly'] = (test_df['Anomaly_Score'] > threshold).astype(int)
                    threshold = np.percentile(scores, 97.5)
                    df['Anomaly'] = (df['Anomaly_Score'] > threshold).astype(int)


                    # 7. 탐지된 이상치 확인
                    # outliers = test_df[test_df['Anomaly'] == 1].sort_values(by='Anomaly_Score', ascending=False)
                    # print("탐지된 이상치 상위 5개:\n", outliers.head())

                    outliers = df[df['Anomaly'] == 1].sort_values(by='date_time_kst', ascending=False)
                    print("탐지된 이상치 상위 5개:\n", outliers)


                    # # time series anomaly detection methods
                    # from deepod.models.time_series import TimesNet
                    # clf = TimesNet()
                    # clf.fit(X_train)
                    # scores = clf.decision_function(X_test)
                    #
                    # # evaluation of time series anomaly detection
                    # from deepod.metrics import ts_metrics
                    # from deepod.metrics import point_adjustment # execute point adjustment for time series ad
                    # eval_metrics = ts_metrics(labels, scores)
                    # adj_eval_metrics = ts_metrics(labels, point_adjustment(labels, scores))





                    # # df = data
                    # df = dataL1
                    # df['value_lag1'] = df['PV'].shift(1)
                    # df['value_lag2'] = df['PV'].shift(2)
                    #
                    # # 이동 평균 (Rolling Features): 최근 데이터의 흐름
                    # df['rolling_mean_3'] = df['PV'].rolling(window=3).mean()
                    #
                    # # 결측치 제거 (shift로 인해 앞부분에 NaN 발생)
                    # df.dropna(inplace=True)



                    # dataL1 = data[data['PV'] > 0].reset_index(drop=True)

                    exp = AnomalyExperiment()
                    exp.setup(
                        data=dataL1,
                        # data=data,
                        # data=df,
                        # ignore_features=['DATE_TIME_KST'],
                        # test_data=testDataL1,
                        # normalize=True,
                        session_id=int(datetime.datetime.now().timestamp()),
                    )

                    # exp.models()
                    # exp.setup(
                    #     data=trainDataL1,
                    #     test_data=testDataL1,
                    #     session_id=int(datetime.datetime.now().timestamp()),
                    #     target=yCol,
                    # )
                    #
                    # from pycaret.anomaly import *
                    # models()

                    # pycaret.anomaly

                    # model = exp.models()
                    # # exp.evaluate_model()
                    #
                    # votes = pd.DataFrame()
                    #
                    # for model_id, df_result in all_results.items():
                    #     # 컬럼명을 'iforest', 'knn' 등으로 변경하여 병합
                    #     votes[model_id] = df_result['Anomaly']

                    # exp.models()

                    model = exp.create_model("iforest")
                    exp.models()

                    # model.info
                    df_anomaly = exp.assign_model(model)
                    print(df_anomaly.head())

                    df_anomaly2 = df_anomaly[df_anomaly['Anomaly'] == 1].reset_index(drop=True)

                    exp.plot_model(model, plot='tsne')




                    iforest_results = exp.assign_model(model)




                    fig = px.line(iforest_results, x=iforest_results.index, y="pv",
                                  title='NYC TAXI TRIPS - UNSUPERVISED ANOMALY DETECTION', template='plotly_dark')

                    outlier_dates = iforest_results[iforest_results['Anomaly'] == 1].index

                    y_values = [iforest_results.loc[i]['pv'] for i in outlier_dates]

                    fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode='markers',
                                             name='Anomaly',
                                             marker=dict(color='red', size=10)))

                    fig.show()




                    import h2o
                    from h2o.estimators import H2OIsolationForestEstimator

                    # 1. H2O 초기화
                    h2o.init()

                    # 2. 데이터 로드 (예시)
                    # H2OFrame 형태로 데이터를 불러옵니다.
                    # data = h2o.import_file("your_data.csv")

                    # 3. 모델 정의 및 학습
                    iso_forest = H2OIsolationForestEstimator(
                        ntrees=100,  # 생성할 나무의 개수 (기본값: 50)
                        max_depth=8,  # 나무의 최대 깊이 (너무 깊으면 과적합 가능성)
                        sample_rate=0.8,  # 각 나무를 학습할 때 사용할 데이터 비율
                        seed=42
                    )

                    # 학습 (비지도 학습이므로 y(타겟) 지정 불필요)
                    iso_forest.train(training_frame=h2o.H2OFrame(data))

                    # 4. 예측 (이상치 점수 및 평균 경로 길이 반환)
                    predictions = iso_forest.predict(h2o.H2OFrame(data))

                    # 결과 확인 (predict 컬럼은 정규화된 이상치 점수, mean_length는 평균 경로 길이)
                    print(predictions.head())



                    # exp.evaluate_model(model)
                    prd = exp.predict_model(model, data = data.copy())






                    model = exp.create_model("pca")
                    # model.info
                    df_anomaly = exp.assign_model(model)
                    print(df_anomaly.head())

                    df_anomaly2 = df_anomaly[df_anomaly['Anomaly'] == 1].reset_index(drop=True)







                    exp.evaluate_model(model)
                    prd = exp.predict_model(model, data = dataL1.copy())

                    # model.evaluate(df_anomaly)

                    # new_data = data.copy()
                    # prd = model.predict(model, new_data)

                    plt.figure(figsize=(12, 6))
                    plt.plot(df_anomaly.index, df_anomaly['PV'], label='시계열 데이터')
                    anomalies = df_anomaly[df_anomaly['Anomaly'] == 1]
                    plt.scatter(anomalies.index, anomalies['PV'], color='red', label='Anomaly', marker='o')
                    plt.legend()
                    plt.show()



                    trainData, testData = train_test_split(data, test_size=0.2, random_state=int(datetime.datetime.now().timestamp()))

                    # query = text("""
                    #     SELECT lf.*
                    #     FROM "TB_FOR_DATA" AS lf
                    #     WHERE lf."SRV" = :srvId
                    #       AND (
                    #         "ML" IS NULL
                    #             OR "DL" IS NULL
                    #             OR "AI" IS NULL
                    #             OR "AI2" IS NULL
                    #             OR "AI3" IS NULL
                    #         )
                    #     ORDER BY "SRV", "DATE_TIME_KST" DESC;
                    #      """)
                    # prdData = pd.DataFrame(session.execute(query, {'srvId':srvId}))
                    # if len(prdData) < 1: continue
                    #
                    # for key in ['pycaret', 'h2o']:
                    #     sysOpt['MODEL'][key]['srvId'] = srvId
                    #
                    # exp = RegressionExperiment()
                    # sysOpt['MODEL']['orgPycaret']['exp'] = exp
                    # sysOpt['MODEL']['pycaret']['exp'] = exp
                    #
                    # # ****************************************************************************
                    # # 독립/종속 변수 설정
                    # # ****************************************************************************
                    # xColOrg = ['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SZA', 'AZA', 'ET', 'SWR']
                    # yColOrg = 'PV'
                    # xCol = ['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SZA', 'AZA', 'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR', 'EXT_RAD']
                    # yCol = 'PV'
                    #
                    # # ****************************************************************************
                    # # 과거 학습 모델링 (orgPycaret)
                    # # ****************************************************************************
                    # resOrgPycaret = makePycaretModel(sysOpt['MODEL']['orgPycaret'], xColOrg, yColOrg, trainData, testData)
                    # # log.info(f'resOrgPycaret : {resOrgPycaret}')
                    #
                    # if resOrgPycaret:
                    #     exp = RegressionExperiment()
                    #     prdVal = exp.predict_model(resOrgPycaret['mlModel'], data=prdData[xCol])['prediction_label']
                    #     prdData['ML'] = np.where(prdVal > 0, prdVal, 0)
                    #
                    # # ****************************************************************************
                    # # 과거 학습 모델링 (orgH2o)
                    # # ****************************************************************************
                    # resOrgH2o = makeH2oModel(sysOpt['MODEL']['orgH2o'], xColOrg, yColOrg, trainData, testData)
                    # # log.info(f'resOrgH2o : {resOrgH2o}')
                    #
                    # if resOrgH2o:
                    #     prdVal = resOrgH2o['mlModel'].predict(h2o.H2OFrame(prdData)).as_data_frame()
                    #     prdData['DL'] = np.where(prdVal > 0, prdVal, 0)
                    #
                    # # ****************************************************************************
                    # # 수동 학습 모델링 (lgb)
                    # # ****************************************************************************
                    # resLgb = makeLgbModel(sysOpt['MODEL']['lgb'], xCol, yCol, trainData, testData)
                    # # log.info(f'resLgb : {resLgb}')
                    #
                    # if resLgb:
                    #     prdVal = resLgb['mlModel'].predict(data=prdData[xCol])
                    #     prdData['AI'] = np.where(prdVal > 0, prdVal, 0)
                    #
                    # # ****************************************************************************
                    # # 자동 학습 모델링 (flaml)
                    # # ****************************************************************************
                    # resFlaml = makeFlamlModel(sysOpt['MODEL']['flaml'], xCol, yCol, trainData, testData)
                    # # log.info(f'resFlaml : {resFlaml}')
                    #
                    # if resFlaml:
                    #     prdVal = resFlaml['mlModel'].predict(prdData)
                    #     prdData['AI2'] = np.where(prdVal > 0, prdVal, 0)
                    #
                    # # ****************************************************************************
                    # # 자동 학습 모델링 (pycaret)
                    # # ****************************************************************************
                    # # resPycaret = makePycaretModel(sysOpt['MODEL']['pycaret'], xCol, yCol, trainData, testData)
                    # # # log.info(f'resPycaret : {resPycaret}')
                    # #
                    # # if resPycaret:
                    # #     prdVal = exp.predict_model(resPycaret['mlModel'], data=prdData[xCol])['prediction_label']
                    # #     prdData['AI3'] = np.where(prdVal > 0, prdVal, 0)
                    #
                    # # *******************************************************
                    # # DB 적재
                    # # *******************************************************
                    # # sys.exit(1)
                    # with cfgDb['sessionMake']() as session:
                    #     try:
                    #         tbTmp = f"tbTm_{uuid.uuid4().hex}"
                    #         with session.begin():
                    #             dbEngine = session.get_bind()
                    #
                    #             prdData.to_sql(
                    #                 name=tbTmp,
                    #                 con=dbEngine,
                    #                 if_exists="replace",
                    #                 index=False
                    #             )
                    #
                    #             query = text(f"""
                    #                 INSERT INTO "TB_FOR_DATA" (
                    #                       "SRV", "ANA_DATE", "DATE_TIME", "DL", "ML", "AI", "AI2", "AI3"
                    #                 )
                    #                 SELECT
                    #                       "SRV", "ANA_DATE", "DATE_TIME", "DL", "ML", "AI", "AI2", "AI3"
                    #                 FROM "{tbTmp}"
                    #                 ON CONFLICT ("SRV", "ANA_DATE", "DATE_TIME")
                    #                 DO UPDATE SET
                    #                     "DL" = excluded."DL",
                    #                     "ML" = excluded."ML",
                    #                     "AI" = excluded."AI",
                    #                     "AI2" = excluded."AI2"
                    #                   """)
                    #             session.execute(query)
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
