# ================================================
# 요구사항
# ================================================

# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
# /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-model.py
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-model.py &
# tail -f nohup.out

# pkill -f TalentPlatform-LSH0627-DaemonFramework-model.py
# 0 0 * * * cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys && /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-model.py

import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import pandas as pd
import re
import json
from datetime import datetime, timedelta
# from konlpy.tag import Okt
from collections import Counter
import pytz
import os
import sys
import urllib.request
import os
import sys
import requests
import json
from konlpy.tag import Okt
from newspaper import Article
from darts import TimeSeries
from darts.models import Prophet
from darts.models import DLinearModel

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
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1,
                                                            backupCount=30, encoding='utf-8')

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
        , 'sysCfg': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.json')
        ,
        'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
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

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'LSH0612'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info('[START] {}'.format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("init"))

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info('[START] {}'.format("exec"))

        try:

            if (platform.system() == 'Windows'):
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 입력 데이터
                'inpFile': '/HDD/DATA/OUTPUT/LSH0627/naverShop_자전거.csv',
                'model': {
                    # 시계열 딥러닝
                    'dl': {
                        'input_chunk_length': 2,
                        'input_chunk_length': 7,
                        'input_chunk_length': 50,
                    },
                    # 예측
                    'prdCnt': 7,
                },
                # 예측 데이터
                'saveFile': '/HDD/DATA/OUTPUT/LSH0627/naverShop_prd.csv',
                'preDt': datetime.now(),
            }

            # =================================================================
            # 모델링
            # =================================================================
            mlModel = Prophet()
            dlModel = DLinearModel(
                input_chunk_length=sysOpt['model']['dl']['output_chunk_length'],
                output_chunk_length=sysOpt['model']['dl']['output_chunk_length'],
                n_epochs=sysOpt['model']['dl']['n_epochs'],
            )

            orgData = pd.read_csv(sysOpt['inpFile'])
            data = orgData
            data['dtDate'] = pd.to_datetime(data['date'], format='%Y%m%d')

            # 자전거 상품 별로 최저가 2개 이상인 경우
            modTitleList = sorted(orgData.groupby("title").filter(lambda x: x['lprice'].nunique() > 1)['title'].unique())

            mlPrdDataL1 = pd.DataFrame()
            dlPrdDataL1 = pd.DataFrame()
            for i, modTitleInfo in enumerate(modTitleList):
                per = round(i / len(data) * 100, 1)
                log.info(f'i : {i} / {per}%')

                selData = data[(data['title'] == modTitleInfo)]
                selDataL1 = TimeSeries.from_dataframe(selData, time_col='dtDate', value_cols='lprice')

                try:
                    mlModel.fit(selDataL1)
                    mlPrd = mlModel.predict(n=sysOpt['model']['prdCnt'])
                    mlPrdData = pd.DataFrame({
                        'title': modTitleInfo,
                        'dtDate': mlPrd.time_index,
                        'mlPrd': mlPrd.values().flatten()
                    })

                    if len(mlPrdData) > 0:
                        mlPrdDataL1 = pd.concat([mlPrdDataL1, mlPrdData], ignore_index=True)
                except Exception as e:
                    log.error(f'Exception : {e}')

                try:
                    dlModel.fit(selDataL1)
                    dlPrd = dlModel.predict(n=sysOpt['model']['prdCnt'])
                    dlPrdData = pd.DataFrame({
                        'title': modTitleInfo,
                        'dtDate': dlPrd.time_index,
                        'dlPrd': dlPrd.values().flatten()
                    })

                    if len(dlPrdData) > 0:
                        dlPrdDataL1 = pd.concat([dlPrdDataL1, dlPrdData], ignore_index=True)
                except Exception as e:
                    log.error(f'Exception : {e}')

            if len(mlPrdDataL1) > 0:
                dataL1 = pd.merge(data, mlPrdDataL1, on=['title', 'dtDate'], how='outer')
            if len(dlPrdDataL1) > 0:
                dataL2 = pd.merge(dataL1, dlPrdDataL1, on=['title', 'dtDate'], how='outer')

            if len(dataL2) > 0:
                saveFile = sysOpt['preDt'].strftime(sysOpt['saveFile'])
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL2.to_csv(saveFile, index=False)
                log.info(f"[CHECK] saveFile : {saveFile}")
                
            # # dd = sorted(changed_price_df['title'].unique())
            # dd2 = sorted(changed_price_df['title'].unique())
            # print(len(dd2))
            #
            # orgDataL1 = orgData[(orgData['title'] == dd2[0])]
            #
            # #
            # orgDataL1['dtDate'] = pd.to_datetime(orgDataL1['date'], format='%Y%m%d')
            # orgDataL1.reset_index(drop=True, inplace=True)
            # orgDataL1.index = orgDataL1['dtDate']
            #
            # orgDataL1['ds'] = orgDataL1['dtDate']
            # orgDataL1['y'] = orgDataL1['lprice']

            # len(orgDataL1)
            # from autots import AutoTS
            #
            # model = AutoTS(
            #     forecast_length=5,
            #     frequency='D',
            #     # model_list=['Prophet', 'ARIMA', 'LightGBM'],
            #     ensemble='simple',
            #     model_list='superfast',
            #     num_validations=0
            # )
            #
            # # data = {
            # #     'Date': pd.to_datetime(pd.date_range(start='2025-01-01', periods=100)),
            # #     'Value': 100 + pd.Series(range(100)) * 0.5 + np.sin(np.arange(100) / 7) * 10 + np.random.rand(100) * 5
            # # }
            #
            # model.fit(orgDataL1, date_col='dtDate', value_col='lprice')
            #
            # prediction = model.predict()
            # print(prediction.forecast)

            #                   lprice
            # 2025-10-12  2.426200e+05
            # 2025-10-13  6.596439e+08
            # 2025-10-14  4.563059e+11
            # 2025-10-15  3.153079e+14
            # 2025-10-16  2.178777e+17

            # # pip install --upgrade darts prophet pytorch-lightning
            # from darts import TimeSeries
            # from darts.models import Prophet
            #
            # model = Prophet()
            # series = TimeSeries.from_dataframe(orgDataL1, time_col='dtDate', value_cols='lprice')
            # model.fit(series)
            # prediction = model.predict(n=7)
            # orgDataL1[['dtDate', 'lprice']]
            #
            # df_manual = pd.DataFrame({
            #     'dtDate': prediction.time_index,
            #     'lprice': prediction.values().flatten()
            # })
            #
            #
            # from darts.models import DLinearModel
            #
            #
            # model = DLinearModel(
            #     input_chunk_length=2,
            #     output_chunk_length=7,
            #     n_epochs=50,
            # )
            # model.fit(series)
            # pred = model.predict(7)
            # print(pred.values())
            #
            #
            # # DLinearModel2
            # [[241903.00792747]
            #  [241902.78877793]
            #  [241902.89348256]
            #  [241956.50949091]
            #  [241956.80001005]
            #  [243267.55961479]
            #  [241956.635577  ]]

            # DLinearModel2
            # [[241253.39928345]
            #  [241253.12697573]
            #  [241227.35230956]
            #  [241227.71175859]
            #  [241226.89687106]
            #  [241906.12707926]
            #  [241907.45918962]]

            # Prophet
            #       dtDate         lprice
            # 0 2025-10-13  241253.183421
            # 1 2025-10-14  241253.463072
            # 2 2025-10-15  241226.392595
            # 3 2025-10-16  241226.678872
            # 4 2025-10-17  241252.788388
            # 5 2025-10-18  241907.439282
            # 6 2025-10-19  241906.827484


            # NLinearModel
            # [[241447.74009144]
            #  [241583.97552001]
            #  [241558.05203277]
            #  [241694.34951872]
            #  [243091.41273275]
            #  [241558.33758252]]
            #
            # from darts.models import NLinearModel
            # model_nlinear = NLinearModel(
            #     input_chunk_length=5,
            #     output_chunk_length=4,
            #     n_epochs=50
            # )
            # model_nlinear.fit(series)
            # pred2 = model_nlinear.predict(6)
            # print(pred2.values())


            #
            # from darts.models import AutoARIMA
            # from darts.utils.timeseries_generation import datetime_attribute_timeseries
            #
            #
            # model_nbeats = AutoARIMA()
            #
            # # 2. 학습 데이터로 모델 학습
            # # 딥러닝 모델은 GPU 사용 시 속도가 빠릅니다.
            # print("NBEATSModel 학습 시작...")
            # model_nbeats.fit(series)
            #
            # # 3. 예측 (forecast)
            # prediction_nbeats = model_nbeats.predict(n=forecast_horizon)
            # print("NBEATSModel 예측 완료.")




            # pip install neuralprophet
            # from neuralprophet import NeuralProphet
            #
            # # NeuralProphet 모델 생성 (기본 설정)
            # # n_forecasts: 예측할 미래 시점의 수
            # # n_lags: 과거 몇 개의 시점을 참고할지 (AR 기능)
            # model2 = NeuralProphet()
            #
            #
            # # 모델 학습
            # # freq='D'는 데이터가 일별(Daily) 데이터임을 의미합니다.
            # df_for_fit = orgDataL1[['ds', 'y']]
            # metrics = model2.fit(df_for_fit, freq="D")
            #
            # tsDlFor = model2.predict(n=2)
            # # print(tsDlFor.values)
            #
            # # tsDlFor.pd_series()
            #
            # df_manual = pd.DataFrame({
            #     'dtDate': tsDlFor.time_index,
            #     'lprice': tsDlFor.values().flatten()
            # })

            #       dtDate         lprice
            # 0 2025-10-12  243025.665088
            # 1 2025-10-13  243680.807311
            # 2 2025-10-14  244335.949534
            # 3 2025-10-15  244991.091757
            # 4 2025-10-16  245646.233980

            from flaml import AutoML

            # X_train = np.arange("2014-01", "2022-01", dtype="datetime64[M]")
            # y_train = np.random.random(size=84)
            # automl = AutoML()
            # automl.fit(
            #     X_train=X_train[:84],  # a single column of timestamp
            #     y_train=y_train,  # value for each timestamp
            #     period=12,  # time horizon to forecast, e.g., 12 months
            #     task="ts_forecast",
            #     time_budget=15,  # time budget in seconds
            #     log_file_name="ts_forecast.log",
            #     eval_method="holdout",
            # )
            # print(automl.predict(X_train[84:]))

            # AutoML 객체 초기화
            automl = AutoML()

            # 시계열 예측을 위한 설정
            settings = {
                "time_budget": 10,
                # "time_budget": 60,
                "metric": 'mape',
                "task": 'ts_forecast',
                "time_col": 'dtDate',
                "label": 'lprice',
                "eval_method": 'holdout',
                "seed": 7654321,
                # "estimator_list": ['prophet', 'lgbm']
            }

            # train_df
            # orgDataL2 = orgDataL1[['dtDate', 'lprice']].rename({'dtDate' : 'index'},axis=1)
            # # orgDataL2.reset_index(drop=True, inplace=True)
            # # orgDataL2.index = orgDataL1['dtDate']
            # orgDataL2 = orgDataL2.sort_values(by='index').reset_index(drop=True)
            # orgDataL2['index'] = pd.to_datetime(orgDataL2['index'])

            # orgDataL2['lprice'] = orgDataL2['lprice'].fillna(method='ffill')

            orgDataL1_ts = orgDataL1[['dtDate', 'lprice']]
            # orgDataL1_ts = orgDataL1_ts.set_index('dtDate')

            # 모델 학습 시작
            # period: 예측할 미래 기간. 30일 뒤까지 예측하도록 설정
            # automl.fit(dataframe=orgDataL2, **settings, period=0)
            # automl.fit( dataframe=orgDataL2, **settings, period=1)
            # automl.fit( X_train=orgDataL2, y_train=orgDataL2, **settings, period=1)
            # automl.fit( data, **settings, period1)
            xCol = ['dtDate']
            yCol = 'lprice'
            automl.fit(X_train=orgDataL1[xCol], y_train=orgDataL1[yCol], **settings, period=2)
            automl.fit(X_train=orgDataL1_ts[xCol], y_train=orgDataL1_ts[yCol], **settings, period=2)
            # automl.fit(dataframe=orgDataL1, **settings, period=2)
            # automl.fit(dataframe=orgDataL1_ts, **settings, period=2)

            prdData = pd.DataFrame(prediction.forecast)
            prdData['dtDate'] = prdData.index
            y_pred = automl.predict(prdData)

        # orgDataL1[['dtDate', 'lprice']]
        # Out[147]:
        # 2025-10-13    241240.0
        # 2025-10-14    241240.0
        # 2025-10-15    241240.0
        # 2025-10-16    241240.0
        # 2025-10-17    241240.0

        #               lprice
        # 2025-10-13  237100.0
        # 2025-10-14  230200.0
        # 2025-10-15  220540.0
        # 2025-10-16  208120.0
        # 2025-10-17  192940.0

        #       dtDate  lprice
        # 0 2025-10-12  241240
        # 1 2025-10-11  242620
        # 2 2025-10-10  241240
        # 3 2025-10-09  241240
        # 4 2025-10-08  241240
        # 5 2025-10-07  241240
        # 6 2025-10-06  241240
        # 7 2025-10-05  241240
        # 8 2025-10-04  241240

        # prdData['flaml'] = automl.predict(prdData)

        #
        #
        # def makeFlamlModel(subOpt=None, xCol=None, yCol=None, trainData=None, testData=None):
        #
        #     log.info(f'[START] makeFlamlModel')
        #     log.info(f'[CHECK] subOpt : {subOpt}')
        #
        #     result = None
        #
        #     try:
        #         saveModelList = sorted(glob.glob(subOpt['saveModelList']), reverse=True)
        #         xyCol = xCol.copy()
        #         xyCol.append(yCol)
        #         data = trainData[xyCol].dropna()
        #         trainDataL1 = trainData[xyCol].dropna()
        #         testDataL1 = testData[xyCol].dropna()
        #
        #         # 학습 모델이 없을 경우
        #         if (subOpt['isOverWrite']) or (len(saveModelList) < 1):
        #
        #             # 7:3에 대한 학습/테스트 분류
        #             # trainData, validData = train_test_split(dataL1, test_size=0.3)
        #
        #             # 전체 학습 데이터
        #             # trainData = dataL1
        #
        #             fnlModel = AutoML(
        #                 # 연속 예측
        #                 task="regression"
        #                 , metric='rmse'
        #
        #                 # 이진 분류
        #                 # task="classification"
        #                 # , metric='accuracy'
        #
        #                 , ensemble=False
        #                 # , ensemble = True
        #                 , seed=123
        #                 , time_budget=60
        #                 # , time_budget=600
        #             )
        #
        #             # 각 모형에 따른 자동 머신러닝
        #             fnlModel.fit(X_train=trainDataL1[xCol], y_train=trainDataL1[yCol])
        #             # fnlModel.fit(X_train=trainData[xCol], y_train=trainData[yCol], n_jobs=12, n_concurrent_trials=4)
        #
        #             # 학습 모형 저장
        #             saveModel = subOpt['preDt'].strftime(subOpt['saveModel'])
        #             log.info(f"[CHECK] saveModel : {saveModel}")
        #             os.makedirs(os.path.dirname(saveModel), exist_ok=True)
        #
        #             with open(saveModel, 'wb') as file:
        #                 pickle.dump(fnlModel, file, pickle.HIGHEST_PROTOCOL)
        #
        #         else:
        #             saveModel = saveModelList[0]
        #             log.info(f"[CHECK] saveModel : {saveModel}")
        #
        #             with open(saveModel, 'rb') as f:
        #                 fnlModel = pickle.load(f)
        #
        #         result = {
        #             'msg': 'succ'
        #             , 'mlModel': fnlModel
        #             , 'saveModel': saveModel
        #             , 'isExist': os.path.exists(saveModel)
        #         }
        #
        #         return result
        #
        #     except Exception as e:
        #         log.error('Exception : {}'.format(e))
        #         return result
        #
        #     finally:
        #         log.info(f'[END] makeFlamlModel')
        # python3.9
        #     pip install "flaml[ts]"
        # pip install auto-ts
        # pip install autots

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e

        finally:
            log.info('[END] {}'.format("exec"))


# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print('[START] {}'.format("main"))

    try:
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
