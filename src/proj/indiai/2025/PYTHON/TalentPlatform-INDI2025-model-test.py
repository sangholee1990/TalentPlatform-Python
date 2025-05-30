# ================================================
# 요구사항
# ================================================
# Python을 이용한 태양광 자동화/수동화 모델링
# pip install optuna-integration[lightgbm]
# pip install flaml
# pip install pycaret[full]

# ps -ef | grep "TalentPlatform-INDI2025-model-test.py" | awk '{print $2}' | xargs kill -9

# cd /vol01/SYSTEMS/INDIAI/PROG/PYTHON
# /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-model-test.py
# nohup /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-model-test.py &

import argparse
import glob
import logging
import logging.handlers
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime
from datetime import timedelta
from urllib.parse import quote_plus

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import xarray as xr
from pandas.tseries.offsets import Hour
import yaml
from multiprocessing import Pool
import multiprocessing as mp
from retrying import retry
import shutil

import os
import re
from datetime import datetime
import subprocess
from isodate import parse_duration
from pandas.tseries.offsets import DateOffset
import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import pickle
from flaml import AutoML
from sklearn.model_selection import train_test_split
from pycaret.regression import *
import pvlib

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
dtKst = timedelta(hours=9)


# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}_{}_{}.log".format(
        os.path.join(contextPath, 'log') if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
        , datetime.now().strftime("%Y%m%d")
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
        # , 'resPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        # , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        # , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        # , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        # , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        # , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        # , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        # , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        # , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        # , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        # , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    return globalVar


#  초기 전달인자 설정
def initArgument(globalVar):
    parser = argparse.ArgumentParser()

    for i, argv in enumerate(sys.argv[1:]):
        if not argv.__contains__('--'): continue
        parser.add_argument(argv)

    inParInfo = vars(parser.parse_args())
    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        if env not in 'local' and key.__contains__('Path'):
            os.makedirs(val, exist_ok=True)
        globalVar[key] = val

    return globalVar


def parseDateOffset(invDate):

    unit = invDate[-1]
    value = int(invDate[:-1])

    if unit == 'y':
        return DateOffset(years=value)
    elif unit == 'm':
        return DateOffset(months=value)
    elif unit == 'd':
        return DateOffset(days=value)
    elif unit == 'h':
        return DateOffset(hours=value)
    elif unit == 't':
        return DateOffset(minutes=value)
    elif unit == 's':
        return DateOffset(seconds=value)
    else:
        raise ValueError(f"날짜 파싱 오류 : {unit}")

def convStrToDate(row):

    if '24:00:00' in row:
        convTime = row.replace('24:00:00', '00:00:00')
        return pd.to_datetime(convTime) + parseDateOffset('1d')
    return pd.to_datetime(row)

def makeLgbModel(subOpt=None, xCol=None, yCol=None, trainData=None, testData=None):

    log.info(f'[START] makeLgbModel')
    log.info(f'[CHECK] subOpt : {subOpt}')

    result = None

    try:

        saveModelList = sorted(glob.glob(subOpt['saveModelList']), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        trainDataL1 = trainData[xyCol].dropna()
        testDataL1 = testData[xyCol].dropna()

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            lgbParams = {
                # 연속 예측
                'objective': 'regression', # 회귀/분류 선택
                'metric': 'rmse', # 평가 지표

                # 이진 분류
                # 'objective': 'binary',
                # 'metric': 'auc',

                'boosting_type': 'gbdt',
                "learning_rate": 0.01,  # 낮은 학습률로 성능 안정화
                "num_leaves": 31,  # 트리 복잡도 제어
                "max_depth": -1,  # 깊이 제한 (-1: 제한 없음)
                "feature_fraction": 0.8,  # 랜덤하게 80%의 특성을 사용
                "bagging_fraction": 0.8,  # 샘플링 비율
                "bagging_freq": 5,  # 매 5회 학습마다 샘플링 수행
                "lambda_l1": 0.1,  # L1 정규화
                "lambda_l2": 0.1,  # L2 정규화
                "min_data_in_leaf": 20,  # 최소 잎사귀 데이터 수
                "verbose": -1 , # 로그 출력 수준
                'verbosity': -1,
                'n_jobs': -1,
                'seed': 123,
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
                    early_stopping(stopping_rounds=10000),
                    log_evaluation(period=200),
                ]
                # early_stopping_rounds=100,
                # verbose_eval=200
            )

            # 학습 모형 저장
            saveModel = subOpt['preDt'].strftime(subOpt['saveModel'])
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
        log.error('Exception : {}'.format(e))
        return result

    finally:
        log.info(f'[END] makeLgbModel')


def makePycaretModel(subOpt=None, xCol=None, yCol=None, trainData=None, testData=None):

    log.info(f'[START] makePycaretModel')
    log.info(f'[CHECK] subOpt : {subOpt}')

    result = None

    try:
        saveModelList = sorted(glob.glob(subOpt['saveModelList']), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        data = trainData[xyCol].dropna()
        trainDataL1 = trainData[xyCol].dropna()
        testDataL1 = testData[xyCol].dropna()

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            # trainData, validData = train_test_split(data, test_size=0.3)

            setup(
                data=trainDataL1,
                session_id=123,
                target=yCol,
            )

            # 각 모형에 따른 자동 머신러닝
            modelList = compare_models(sort='RMSE', n_select=3)

            # 앙상블 모형
            blendModel = blend_models(estimator_list=modelList, fold=10)

            # 앙상블 파라미터 튜닝
            tuneModel = tune_model(blendModel, fold=10, choose_better=True)

            # 학습 모형
            fnlModel = finalize_model(tuneModel)

            # 학습 모형 저장
            saveModel = subOpt['preDt'].strftime(subOpt['saveModel'])
            log.info(f'[CHECK] saveModel : {saveModel}')
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)
            save_model(fnlModel, saveModel)

        else:
            saveModel = saveModelList[0]
            log.info(f'[CHECK] saveModel : {saveModel}')
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
        log.info(f'[END] makePycaretModel')

def makeFlamlModel(subOpt=None, xCol=None, yCol=None, trainData=None, testData=None):

    log.info(f'[START] makeFlamlModel')
    log.info(f'[CHECK] subOpt : {subOpt}')

    result = None

    try:
        saveModelList = sorted(glob.glob(subOpt['saveModelList']), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        data = trainData[xyCol].dropna()
        trainDataL1 = trainData[xyCol].dropna()
        testDataL1 = testData[xyCol].dropna()

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

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

                , ensemble = False
                # , ensemble = True
                , seed = 123
                , time_budget=60
                # , time_budget=600
            )

            # 각 모형에 따른 자동 머신러닝
            fnlModel.fit(X_train=trainDataL1[xCol], y_train=trainDataL1[yCol])
            # fnlModel.fit(X_train=trainData[xCol], y_train=trainData[yCol], n_jobs=12, n_concurrent_trials=4)

            # 학습 모형 저장
            saveModel = subOpt['preDt'].strftime(subOpt['saveModel'])
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
        log.error('Exception : {}'.format(e))
        return result

    finally:
        log.info(f'[END] makeFlamlModel')

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'INDI2025'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info(f"[START] __init__ : init")

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info(f"[END] __init__ : init")

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info(f"[START] exec")

        try:
            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            # 옵션 설정
            sysOpt = {
                # 설정 파일
                'CFG': {
                    'siteInfo': {
                        'filePath': '/DATA/PROP/SAMPLE',
                        'fileName': 'site_info.csv',
                    },
                    'energy': {
                        'filePath': '/DATA/PROP/SAMPLE',
                        'fileName': 'energy.csv',
                    },
                    'ulsanObs': {
                        'filePath': '/DATA/PROP/SAMPLE',
                        'fileName': 'ulsan_obs_data.csv',
                    },
                    'ulsanFcst': {
                        'filePath': '/DATA/PROP/SAMPLE',
                        'fileName': 'ulsan_fcst_data.csv',
                    },
                },

                # 전처리 파일
                'UMKR': {
                    'fileList': '/DATA/PROP/UMKR/%Y%m/UMKR_FOR_%Y%m%d.nc',
                },

                # 자동화/수동화 모델링
                'MODEL': {
                    'lgb': {
                        'saveModelList': f"/DATA/MODEL/*/*/*_solar_test_lgb_for.model",
                        'saveModel': f"/DATA/MODEL/%Y%m/%d/%Y%m%d_solar_test_lgb_for.model",
                        'saveImg': f"/DATA/MODEL/%Y%m/%d/%Y%m%d_solar_test_lgb_for.png",
                        'isOverWrite': True,
                        # 'isOverWrite': False,
                        'preDt': datetime.now(),
                    },
                    'flaml': {
                        'saveModelList': f"/DATA/MODEL/*/*/*_solar_test_flaml_for.model",
                        'saveModel': f"/DATA/MODEL/%Y%m/%d/%Y%m%d_solar_test_flaml_for.model",
                        'saveImg': f"/DATA/MODEL/%Y%m/%d/%Y%m%d_solar_test_flaml_for.png",
                        'isOverWrite': True,
                        # 'isOverWrite': False,
                        'preDt': datetime.now(),
                    },
                    'pycaret': {
                        'saveModelList': f"/DATA/MODEL/*/*/*_solar_test_pycaret_for.model.pkl",
                        'saveModel': f"/DATA/MODEL/%Y%m/%d/%Y%m%d_solar_test_pycaret_for.model",
                        'saveImg': f"/DATA/MODEL/%Y%m/%d/%Y%m%d_solar_test_pycaret_for.png",
                        'isOverWrite': True,
                        # 'isOverWrite': False,
                        'preDt': datetime.now(),
                    },
                },
                'FNL': {
                    'saveFile': '/DATA/FNL/%Y%m/%d/%Y%m%d_solar_test_prd_for.csv',
                    'preDt': datetime.now(),
                },
            }

            # **************************************************************************************************************
            # 설정 파일 읽기
            # **************************************************************************************************************
            cfgData = {}
            for key, item in sysOpt['CFG'].items():
                log.info(f"[CHECK] key : {key}")

                filePattern = '{}/{}'.format(sysOpt['CFG'][key]['filePath'], sysOpt['CFG'][key]['fileName'])
                fileList = sorted(glob.glob(filePattern))
                if fileList is None or len(fileList) < 1:
                    raise Exception(f"[ERROR] filePattern : {filePattern} / 파일을 확인해주세요.")
                data = pd.read_csv(fileList[0])

                cfgData[key] = data

            # 실측 데이터
            energyData = cfgData['energy'][['time', 'ulsan']]
            energyData['eneDt'] = energyData['time'].apply(convStrToDate)

            # 실황 학습모델
            # cfgData['ulsanObs']

            # 예보 학습모델
            ulsanFcstData = cfgData['ulsanFcst']
            ulsanFcstData['anaDt'] = pd.to_datetime(ulsanFcstData['Forecast time'])
            ulsanFcstData['forDt'] = ulsanFcstData.apply(lambda row: row['anaDt'] + parseDateOffset(f"{int(row['forecast'])}h"), axis=1)

            # ****************************************************************************
            # 데이터 병합
            # ****************************************************************************
            data = pd.merge(ulsanFcstData, energyData, how='left', left_on='forDt', right_on='eneDt')

            # ****************************************************************************
            # 유의미한 변수 추가
            # ****************************************************************************
            posInfo = cfgData['siteInfo'][cfgData['siteInfo']['Id'] == '울산태양광']
            posLat = posInfo['Latitude'].item()
            posLon = posInfo['Longitude'].item()

            data['forDtUtc'] = pd.to_datetime(data['forDt']).dt.tz_localize('Asia/Seoul').dt.tz_convert('UTC')

            solPosInfo = pvlib.solarposition.get_solarposition(time=pd.to_datetime(data['forDtUtc'].values), latitude=posLat, longitude=posLon, temperature=data['Temperature'].values, method='nrel_numpy')
            data['sza'] = solPosInfo['apparent_zenith'].values
            data['aza'] = solPosInfo['azimuth'].values
            data['et'] = solPosInfo['equation_of_time'].values
            data['extRad'] = pvlib.irradiance.get_extra_radiation(solPosInfo.index.dayofyear)

            site = pvlib.location.Location(posLat, posLon, tz='Asia/Seoul')
            clearInsInfo = site.get_clearsky(pd.to_datetime(data['forDtUtc'].values))
            data['ghiClr'] = clearInsInfo['ghi'].values
            data['dniClr'] = clearInsInfo['dni'].values
            data['dhiClr'] = clearInsInfo['dhi'].values

            # 혼탁도
            turbidity = pvlib.clearsky.lookup_linke_turbidity(pd.to_datetime(data['forDtUtc'].values), posLat, posLon, interp_turbidity=True)
            data['turb'] = turbidity.values

            # ****************************************************************************
            # 인덱스 재 정렬
            # ****************************************************************************
            # trainData = data[data['anaDt'] <= pd.to_datetime('2019-03-01')].reset_index(drop=True)
            # testData = data[data['anaDt'] > pd.to_datetime('2019-03-01')].reset_index(drop=True)
            trainData = data[data['forDt'] < pd.to_datetime('2020-01-01')].reset_index(drop=True)
            testData = data[data['forDt'] >= pd.to_datetime('2020-01-01')].reset_index(drop=True)
            prdData = testData

            # ulsan > 0인 경우
            trainDataL1 = trainData[trainData['ulsan'] > 0].reset_index(drop=True)

            # ****************************************************************************
            # 독립/종속 변수 설정
            # ****************************************************************************
            # xCol = ['Temperature', 'Humidity', 'WindSpeed', 'WindDirection', 'Cloud']
            # xCol = ['Temperature', 'Humidity', 'WindSpeed', 'WindDirection', 'Cloud', 'sza', 'aza', 'et', 'ghiClr', 'dniClr', 'dhiClr', 'turb']
            xCol = ['Temperature', 'Humidity', 'WindSpeed', 'WindDirection', 'Cloud', 'sza', 'aza', 'et', 'ghiClr', 'dniClr', 'dhiClr', 'turb', 'extRad']
            yCol = 'ulsan'

            # ****************************************************************************
            # 수동 학습 모델링 (lgb)
            # ****************************************************************************
            # resLgb = makeLgbModel(sysOpt['MODEL']['lgb'], xCol, yCol, trainData, testData)
            resLgb = makeLgbModel(sysOpt['MODEL']['lgb'], xCol, yCol, trainDataL1, testData)
            # log.info(f'[CHECK] resLgb : {resLgb}')

            # prdData['prd-lgb'] = resLgb['mlModel'].predict(data=prdData[xCol])
            prdData['prd-lgb'] = np.where(prdData['ulsan'] > 0, resLgb['mlModel'].predict(data=prdData[xCol]), 0)

            # ****************************************************************************
            # 자동 학습 모델링 (flaml)
            # ****************************************************************************
            # resFlaml = makeFlamlModel(sysOpt['MODEL']['flaml'], xCol, yCol, trainData, testData)
            resFlaml = makeFlamlModel(sysOpt['MODEL']['flaml'], xCol, yCol, trainDataL1, testData)
            # log.info(f'[CHECK] resFlaml : {resFlaml}')

            # prdData['prd-flaml'] = resFlaml['mlModel'].predict(prdData)
            prdData['prd-flaml'] = np.where(prdData['ulsan'] > 0, resFlaml['mlModel'].predict(prdData), 0)

            # ****************************************************************************
            # 자동 학습 모델링 (pycaret)
            # ****************************************************************************
            # resPycaret = makePycaretModel(sysOpt['MODEL']['pycaret'], xCol, yCol, trainData, testData)
            resPycaret = makePycaretModel(sysOpt['MODEL']['pycaret'], xCol, yCol, trainDataL1, testData)
            # log.info(f'[CHECK] resPycaret : {resPycaret}')

            # prdData['prd-pycaret'] = predict_model(resPycaret['mlModel'], data=prdData[xCol])['prediction_label']

            prdData['prd-pycaret'] = np.where(prdData['ulsan'] > 0, predict_model(resPycaret['mlModel'], data=prdData[xCol])['prediction_label'], 0)

            # ****************************************************************************
            # 자료 저장
            # ****************************************************************************
            subOpt = sysOpt['FNL']
            saveFile = subOpt['preDt'].strftime(subOpt['saveFile'])
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            prdData.to_csv(saveFile, index=False)
            log.info(f'[CHECK] saveFile : {saveFile}')

        except Exception as e:
            log.error(f"Exception : {e}")
            raise e

        finally:
            log.info(f"[END] exec")


# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print(f'[START] main')

    try:
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] main')
