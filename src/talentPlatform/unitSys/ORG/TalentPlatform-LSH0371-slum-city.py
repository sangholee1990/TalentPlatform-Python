# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from _ast import expr
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
from matplotlib import font_manager, rc
import geopandas as gpd
import seaborn as sns
from sklearn.metrics import *

from flaml import AutoML
from lightgbm import plot_importance
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
# import ray

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

import optuna.integration.lightgbm as lgb
import pandas as pd
import xgboost as xgb
from teddynote import models

import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
import json
from sklearn.cluster import KMeans
from scipy.stats import linregress
import matplotlib.cm as cm


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

# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):

    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
        , datetime.now().strftime("%Y%m%d")
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

        log.info("[CHECK] {} : {}".format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar


# 머신러닝 예측
def makeFlamlModel(subOpt=None, xCol=None, yCol=None, trainData=None, testData=None):

    log.info('[START] {}'.format('makeFlamlModel'))
    log.info('[CHECK] subOpt : {}'.format(subOpt))

    result = None

    try:
        # saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'flaml', 'for', '*')
        # saveModel = '{}/{}/{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, subOpt['modelKey'], 'final', 'flaml', 'for', '*')
        saveModel = '{}/{}/{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, subOpt['modelKey'], 'final', 'flaml', 'for')
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

            # ray.init(num_cpus=12, ignore_reinit_error=True)
            # ray.init(num_cpus=12)

            flModel = AutoML(
                # 회귀
                # task="regression"
                # , metric='rmse'

                # 이진 분류
                task="classification"
                , metric='accuracy'

                , ensemble = False
                # , ensemble = True
                , seed = 123
                # , time_budget=60
                , time_budget=600
            )

            # 각 모형에 따른 자동 머신러닝
            flModel.fit(X_train=trainData[xCol], y_train=trainData[yCol])
            # flModel.fit(X_train=trainData[xCol], y_train=trainData[yCol], n_jobs=12, n_concurrent_trials=4)

            # 학습 모형 저장
            # saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'flaml', 'for', datetime.datetime.now().strftime('%Y%m%d'))
            # saveModel = '{}/{}/{}-{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, subOpt['modelKey'], 'final', 'flaml', 'for', datetime.now().strftime('%Y%m%d'))
            saveModel = '{}/{}/{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, subOpt['modelKey'], 'final', 'flaml', 'for')
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


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 아프리카 빈민가 딥러닝 예측

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0371'

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
                    'srtDate': '1990-01-01'
                    , 'endDate': '2022-01-01'

                    #  머신러닝
                    , 'mlModel': {
                        # 모델 업데이트 여부
                        'isOverWrite': True
                        # 'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2019-01-01'
                    , 'endDate': '2023-01-01'

                    #  머신러닝
                    , 'mlModel': {
                        # 모델 업데이트 여부
                        'isOverWrite': True
                        # 'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # ****************************************************************************
            # 파일 읽기
            # ****************************************************************************
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'KNY_MorphSocioCluster_SlumNonslum.csv')
            fileList = sorted(glob.glob(inpFile))
            data = pd.read_csv(fileList[0])

            for i, info in data.iterrows():
                geoInfo = json.loads(info['.geo'])['coordinates']
                data.loc[i, 'lat'] = geoInfo[0]
                data.loc[i, 'lon'] = geoInfo[1]

            # 위/경도를 이용하여 kmean 수행 (4개 city 분류)
            kmeanModel = KMeans(n_clusters=4, random_state=123).fit(data[['lon', 'lat']])
            data['city'] = kmeanModel.labels_

            mainTitle = '{}'.format('kmeans-city')
            saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            sns.scatterplot(data=data, x='lon', y='lat', hue='city')
            plt.title(mainTitle)
            plt.tight_layout()
            plt.savefig(saveImg, dpi=600, bbox_inches='tight')
            plt.show()
            plt.close()

            cityList = np.unique(data['city'])
            # cityInfo = cityList[1]
            for i, cityInfo in enumerate(cityList):

                trainData = data.loc[
                    (data['city'] != cityInfo)
                ].reset_index(drop=True)

                testData = data.loc[
                    (data['city'] == cityInfo)
                ].reset_index(drop=True)

                if (len(trainData) < 5): continue
                log.info('[CHECK] cityInfo : {}'.format(cityInfo))

                modelKey = '{}-city{:05d}'.format('slum', cityInfo)

                # 7:3에 대한 훈련/테스트 데이터 분류
                # trainData, testData = train_test_split(dataL1, test_size=0.3, random_state=123)

                # 인덱스 재 정렬
                trainData = trainData.reset_index(drop=True)
                testData = testData.reset_index(drop=True)

                # ****************************************************************************
                # 독립/종속 변수 설정
                # ****************************************************************************
                xCol = list(data.columns.difference(['system:index', '.geo', 'Altitude', 'class', 'city', 'lat', 'lon', 'DHSwi', 'rwi', 'remapped']))
                yCol = 'class'

                prdData = testData

                # ****************************************************************************
                # 수동 학습 모델링 (xgboost)
                # ****************************************************************************
                xgbTrainData = xgb.DMatrix(trainData[xCol], trainData[yCol])
                xgbTestData = xgb.DMatrix(testData[xCol], testData[yCol])

                xgbParams = {
                    # 회귀
                    # 'objective': 'reg:squarederror'
                    # , 'eval_metric' : 'rmse'

                    # 이진 분류
                    'objective': 'binary:logistic'
                    , 'random_state': 123
                }

                # 학습
                xgbModel = xgb.train(
                    params=xgbParams
                    , dtrain=xgbTrainData
                    , evals=[(xgbTrainData, "train"), (xgbTestData, "valid")]
                    , num_boost_round=10000
                    , verbose_eval=False
                    # , early_stopping_rounds=1000
                    , early_stopping_rounds=10000
                )

                # 학습모형 저장
                saveModel = '{}/{}/{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, modelKey, 'final', 'xgb', 'for')
                os.makedirs(os.path.dirname(saveModel), exist_ok=True)
                pickle.dump(xgbModel, open(saveModel, 'wb'))
                log.info('[CHECK] saveFile : {}'.format(saveModel))

                # 변수 중요도 저장
                try:
                    mainTitle = '{}'.format('xgb-importance')
                    saveImg = '{}/{}/{}-{}.png'.format(globalVar['figPath'], serviceName, modelKey, mainTitle)
                    xgb.plot_importance(xgbModel)
                    plt.title(mainTitle)
                    plt.tight_layout()
                    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
                    plt.show()
                    plt.close()
                except Exception as e:
                    log.error('Exception : {}'.format(e))

                # 학습모형 불러오기
                xgbModel = pickle.load(open(saveModel, 'rb'))

                # 예측
                prdData['ML'] = xgbModel.predict(xgb.DMatrix(testData[xCol]))
                prdData['ML2'] = np.where(prdData['ML'] > 0.5, 1.0, 0.0)

                # ****************************************************************************
                # 수동 학습 모델링 (lightgbm)
                # ****************************************************************************
                lgbTrainData = lgb.Dataset(trainData[xCol], trainData[yCol])
                lgbTestData = lgb.Dataset(testData[xCol], testData[yCol], reference=lgbTrainData)

                lgbParams = {
                    # 회귀
                    # 'objective': 'regression'
                    # , 'metric': 'rmse'

                    # 이진 분류
                    'objective': 'binary'
                    , 'verbosity': -1
                    , 'n_jobs': -1
                    , 'seed': 123
                }

                # 학습
                lgbModel = lgb.train(
                    params=lgbParams
                    , train_set=lgbTrainData
                    , num_boost_round=10000
                    # , early_stopping_rounds=1000
                    , early_stopping_rounds=10000
                    , valid_sets=[lgbTrainData, lgbTestData]
                    , verbose_eval=False
                )

                # 학습모형 저장
                saveModel = '{}/{}/{}-{}-{}-{}.model'.format(globalVar['outPath'], serviceName, modelKey, 'final', 'lgb', 'for')
                os.makedirs(os.path.dirname(saveModel), exist_ok=True)
                pickle.dump(lgbModel, open(saveModel, 'wb'))
                log.info('[CHECK] saveFile : {}'.format(saveModel))

                # 변수 중요도 저장
                try:
                    mainTitle = '{}'.format('lgb-importance')
                    saveImg = '{}/{}/{}-{}.png'.format(globalVar['figPath'], serviceName, modelKey, mainTitle)
                    lgb.plot_importance(lgbModel)
                    plt.title(mainTitle)
                    plt.tight_layout()
                    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
                    plt.show()
                    plt.close()
                except Exception as e:
                    log.error('Exception : {}'.format(e))

                # 학습모형 불러오기
                lgbModel = pickle.load(open(saveModel, 'rb'))

                # 예측
                prdData['ML3'] = lgbModel.predict(data=testData[xCol])
                prdData['ML4'] = np.where(prdData['ML3'] > 0.5, 1.0, 0.0)

                # ****************************************************************************
                # 자동 학습 모델링 (flaml)
                # ****************************************************************************
                # 머신러닝 불러오기
                sysOpt['mlModel'].update(
                    {
                        'srvId': serviceName
                        , 'modelKey': modelKey
                        , 'isOverWrite': True
                    }
                )

                # resFlaml = makeFlamlModel(sysOpt['mlModel'], xCol, yCol, trainData)
                resFlaml = makeFlamlModel(sysOpt['mlModel'], xCol, yCol, trainData, testData)
                log.info('[CHECK] resFlaml : {}'.format(resFlaml))

                # 머신러닝 예측
                flamlModel = resFlaml['mlModel']
                prdData['ML5'] = flamlModel.predict(prdData)
                prdData['ML6'] = np.where(prdData['ML5'] > 0.5, 1.0, 0.0)

                # 변수 중요도 저장
                try:
                    mainTitle = '{}'.format('AutoML-importance')
                    saveImg = '{}/{}/{}-{}.png'.format(globalVar['figPath'], serviceName, modelKey, mainTitle)
                    featData = pd.DataFrame([flamlModel.feature_names_in_, flamlModel.feature_importances_], index=['key', 'val']).transpose().sort_values(by=['val'], ascending=True)
                    plt.barh(featData['key'], featData['val'])
                    plt.title(mainTitle)
                    plt.tight_layout()
                    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
                    plt.show()
                    plt.close()
                except Exception as e:
                    log.error('Exception : {}'.format(e))


                # ****************************************************************************
                # 검증 시각화
                # ****************************************************************************
                # mainTitle = '{}'.format('인덱스에 따른 아프리카 빈민가 딥러닝 예측 결과')
                mainTitle = '{}'.format('African slum deep learning prediction result according to index')
                saveImg = '{}/{}/{}-{}.png'.format(globalVar['figPath'], serviceName, modelKey, mainTitle)
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                log.info('[CHECK] saveImg : {}'.format(saveImg))

                plt.plot(prdData.index, prdData['class'], marker='o', label='obs')
                plt.plot(prdData.index, prdData['ML2'], label='xgb (Accuracy : {:.2f}, Precision : {:.2f}, Recall : {:.2f})'.format(
                    accuracy_score(prdData['class'], prdData['ML2']), precision_score(prdData['class'], prdData['ML2']), recall_score(prdData['class'], prdData['ML2'])
                ))
                plt.plot(prdData.index, prdData['ML4'], label='lgb (Accuracy : {:.2f}, Precision : {:.2f}, Recall : {:.2f})'.format(
                    accuracy_score(prdData['class'], prdData['ML4']), precision_score(prdData['class'], prdData['ML4']), recall_score(prdData['class'], prdData['ML4'])
                ))
                plt.plot(prdData.index, prdData['ML6'], label='AutoML (Accuracy : {:.2f}, Precision : {:.2f}, Recall : {:.2f})'.format(
                    accuracy_score(prdData['class'], prdData['ML6']), precision_score(prdData['class'], prdData['ML6']), recall_score(prdData['class'], prdData['ML6'])
                ))

                plt.title(mainTitle)
                plt.legend()
                plt.tight_layout()
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                plt.show()

                # ****************************************************************************
                # 산점도 시각화
                # ****************************************************************************
                colList = ['ML2', 'ML4', 'ML6']
                # colInfo = colList[1]
                for j, colInfo in enumerate(sorted(set(colList))):

                    selData = prdData[[yCol, colInfo]].dropna()
                    if (len(selData) < 1): continue

                    colName = {'ML2' : 'xgb', 'ML4' : 'lgb', 'ML6' : 'AutoML'}.get(colInfo, 'NA')

                    mainTitle =  'African deep learning prediction result ({} vs class)'.format(colName)
                    saveImg = '{}/{}/{}-{}.png'.format(globalVar['figPath'], serviceName, modelKey, mainTitle)
                    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                    # rtnInfo = makeUserHist2dPlot(selData[colInfo], selData['class'], colName, 'obs', mainTitle, saveImg, 0.0, 1.0, 0.05, 0.15, 30, True)

                    RocCurveDisplay.from_predictions(
                        selData[colInfo].ravel()
                        , selData[yCol].ravel()
                        , color="darkorange"
                    )

                    plt.plot([0, 1], [0, 1], "k--", label='')
                    plt.axis('square')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(mainTitle)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                    plt.show()
                    plt.close()

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
        inParams = { }
        print("[CHECK] inParams : {}".format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))