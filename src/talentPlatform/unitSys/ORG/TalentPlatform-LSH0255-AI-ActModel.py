# -*- coding: utf-8 -*-
import glob
import seaborn as sns
import logging
import logging.handlers
import os
import platform
import sys
import argparse
import traceback
import warnings
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# import pyproj
# import xarray as xr
# from mizani.transforms import trans
from scipy.stats import linregress
import pandas as pd
import math
from scipy import spatial
from pandas.tseries.offsets import Day, Hour, Minute, Second
from scipy.interpolate import Rbf

# os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share"
# from pygem import IDW
# import eccodes
# import pygrib
# import pykrige.kriging_tools as kt
# import haversine as hs
import pytz
import pvlib
import pandas as pd

# from auto_ts import auto_timeseries
# from plotnine import ggplot
# from pycaret.regression import setup
# from pycaret.regression import compare_models

import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
import time
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

try:
    from pycaret.regression import *
except Exception as e:
    print("Exception : {}".format(e))

try:
    from pycaret.regression import *
    from pycaret.regression import setup
    from pycaret.regression import compare_models
    from pycaret.utils import check_metric
except Exception as e:
    print("Exception : {}".format(e))

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

    saveLogFile = "{}/{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
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
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        ,
        'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
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


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 한국 월별 수출액과 코스피 종간 간의 회귀분석

    # conda activate
    # python3 TalentPlatform-LSH0255-ASOS.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "2020-09"

    # python3 TalentPlatform-LSH0255-ASOS.py --inpPattzKsth "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "2020-09" &
    # nohup bash RunShell-Python.sh "2020-10" &

    #  /SYSTEMS/OUTPUT/OBS/202109/01/AWS_OBS_202109010000.txt
    # python3 ${contextPath}/TalentPlatform-LSH0255-GK2A.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "$1" --endDate "$2"
    # python3 TalentPlatform-LSH0255-GK2A.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "20210901" --endDate "20210902"
    # bash RunShell-Python.sh "20210901" "20210902"

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PV'

    prjName = 'test'
    serviceName = 'LSH0255'

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

        # import pandas as pd
        # import numpy as np

        h2o.init()

        try:
            if (platform.system() == 'Windows'):

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2021-10-01'
                    , 'endDate': '2021-11-01'
                    , 'isOverWrite': True
                    # , 'isOverWrite': False
                }

                globalVar['inpPath'] = 'E:/DATA'
                globalVar['outPath'] = 'E:/DATA'

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': globalVar['srtDate']
                    , 'endDate': globalVar['endDate']
                    # , 'isOverWrite': True
                    , 'isOverWrite': False
                }

            inpPosFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/GA_STN_INFO.xlsx')
            posData = pd.read_excel(inpPosFile, engine='openpyxl')
            posDataL1 = posData[['id', 'lat', 'lon']]

            modelDirKeyList = ['AI_2Y']
            # modelDirKeyList = ['AI_2Y', 'AI_7D', 'AI_15D', 'AI_1M', 'AI_3M', 'AI_6M']
            for k, modelDirKey in enumerate(modelDirKeyList):
                log.info("[CHECK] modelDirKey : {}".format(modelDirKey))

                for i, posInfo in posDataL1.iterrows():
                    posId = int(posInfo['id'])
                    posLat = posInfo['lat']
                    posLon = posInfo['lon']

                    log.info("[CHECK] posId (posLon, posLat) : {} ({}. {})".format(posId, posLon, posLat))

                    # break
                    inpFile = '{}/{}/{}-SRV{:05d}-{}-{}-{}.xlsx'.format(globalVar['outPath'], 'ACT', serviceName, posId, 'final', 'proc', 'act')
                    fileList = sorted(glob.glob(inpFile))

                    # 파일 없을 경우 예외 처리
                    if fileList is None or len(fileList) < 1:
                        log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                        continue

                    fileInfo = fileList[0]
                    inpData = pd.read_excel(fileInfo, engine='openpyxl')

                    inpData['CA_TOT'][inpData['CA_TOT'] < 0] = np.nan
                    inpData['WS'][inpData['WS'] < 0] = np.nan
                    inpData['WD'][inpData['WD'] < 0] = np.nan
                    inpData['SWR'][inpData['SWR'] < 0] = np.nan
                    inpData['pv'][inpData['pv'] < 0] = np.nan

                    inpDataL1 = inpData.dropna().reset_index(drop=True)

                    # idxInfo = inpDataL1.loc[inpDataL1['dtDateKst'] >= pd.to_datetime('2021-01-01', format='%Y-%m-%d')].index.to_numpy()[0]
                    idxInfo = inpDataL1.loc[inpDataL1['dtDateKst'] >= pd.to_datetime('2021-10-30', format='%Y-%m-%d')].index.to_numpy()
                    if (len(idxInfo) < 1): continue
                    idx = idxInfo[0]

                    # 7일, 15일, 1달, 3달, 6달, 2년
                    if (modelDirKey == 'AI_2Y'):
                        # 2021년 기준으로 데이터 분할
                        # trainData, testData = inpDataL1[0:idx], inpDataL1[idx:len(inpDataL1)]

                        # 전체 데이터
                        # trainData = inpDataL1

                        # 2021년 기준으로 데이터 분할
                        trainData, testData = inpDataL1[0:idx], inpDataL1[idx:len(inpDataL1)]

                    elif (modelDirKey == 'AI_7D'):
                        srtIdx = inpDataL1.loc[inpDataL1['dtDateKst'] >= pd.to_datetime('2021-01-01', format='%Y-%m-%d') - timedelta(days=7)].index.to_numpy()[0]
                        trainData, testData = inpDataL1[srtIdx:idx], inpDataL1[idx:len(inpDataL1)]
                    elif (modelDirKey == 'AI_15D'):
                        srtIdx = inpDataL1.loc[inpDataL1['dtDateKst'] >= pd.to_datetime('2021-01-01', format='%Y-%m-%d') - timedelta(days=15)].index.to_numpy()[0]
                        trainData, testData = inpDataL1[srtIdx:idx], inpDataL1[idx:len(inpDataL1)]
                    elif (modelDirKey == 'AI_1M'):
                        srtIdx = inpDataL1.loc[inpDataL1['dtDateKst'] >= pd.to_datetime('2021-01-01', format='%Y-%m-%d') - relativedelta(months=1)].index.to_numpy()[0]
                        trainData, testData = inpDataL1[srtIdx:idx], inpDataL1[idx:len(inpDataL1)]
                    elif (modelDirKey == 'AI_3M'):
                        srtIdx = inpDataL1.loc[inpDataL1['dtDateKst'] >= pd.to_datetime('2021-01-01', format='%Y-%m-%d') - relativedelta(months=3)].index.to_numpy()[0]
                        trainData, testData = inpDataL1[srtIdx:idx], inpDataL1[idx:len(inpDataL1)]
                    elif (modelDirKey == 'AI_6M'):
                        srtIdx = inpDataL1.loc[inpDataL1['dtDateKst'] >= pd.to_datetime('2021-01-01', format='%Y-%m-%d') - relativedelta(months=6)].index.to_numpy()[0]
                        trainData, testData = inpDataL1[srtIdx:idx], inpDataL1[idx:len(inpDataL1)]

                    # 2021년 기준으로 변경
                    trainDataL1 = trainData[['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'pv', 'sza', 'aza', 'et']]

                    # trainDataL1.describe()



                    # **********************************************************************************************************
                    # 머신러닝
                    # **********************************************************************************************************
                    # 시게열
                    # https://towardsdatascience.com/time-series-forecasting-with-pycaret-regression-module-237b703a0c63
                    #
                    # saveCsvFile = '{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, "trainDataL4")
                    # # trainDataL4.to_csv(saveCsvFile, index=False)
                    # log.info('[CHECK] saveCsvFile : {}'.format(saveCsvFile))
                    #
                    # trainDataL4 = pd.read_csv(saveCsvFile)
                    # trainDataL4.describe()

                    saveMlModel = '{}/{}/{}-SRV{:05d}-{}-{}-{}-{}.model.pkl'.format(globalVar['outPath'], modelDirKey, serviceName, posId, 'final', 'pycaret', 'act', '*')
                    saveMlModelList = sorted(glob.glob(saveMlModel), reverse=True)

                    # 학습 모델이 없을 경우
                    if (sysOpt['isOverWrite']) or (len(saveMlModelList) < 1):
                        pyModel = setup(
                            data=trainDataL1
                            , session_id=123
                            , silent=True
                            , target='pv'
                        )

                        # 각 모형에 따른 자동 머신러닝
                        modelList = compare_models(sort='RMSE', n_select=10)

                        # 앙상블 모형
                        blendModel = blend_models(estimator_list=modelList, fold=10)

                        # 앙상블 튜닝
                        tuneModel = tune_model(blendModel, fold=2, choose_better=True)
                        log.info("[CHECK] tuneModel : {}".format(tuneModel))

                        # 학습 모델
                        fnlModel = finalize_model(tuneModel)
                        log.info("[CHECK] fnlModel : {}".format(fnlModel))

                        # 학습 모델 저장
                        saveModel = '{}/{}/{}-SRV{:05d}-{}-{}-{}-{}.model'.format(globalVar['outPath'], modelDirKey, serviceName, posId, 'final', 'pycaret', 'act', datetime.now().strftime("%Y%m%d"))
                        log.info("[CHECK] saveModel : {}".format(saveModel))
                        os.makedirs(os.path.dirname(saveModel), exist_ok=True)
                        save_model(fnlModel, saveModel)

                    # **********************************************************************************************************
                    # 딥러닝
                    # **********************************************************************************************************
                    saveDlModel = '{}/{}/{}-SRV{:05d}-{}-{}-{}-{}.model'.format(globalVar['outPath'], modelDirKey, serviceName, posId, 'final', 'h2o', 'act', '*')
                    saveDlModelList = sorted(glob.glob(saveDlModel), reverse=True)

                    # 학습 모델이 없을 경우
                    if (sysOpt['isOverWrite']) or (len(saveDlModelList) < 1):
                        # 개수 제한
                        # 10초 제한
                        # dnModel = H2OAutoML(max_models=20, max_runtime_secs=10000, balance_classes=True, seed=123)
                        dnModel = H2OAutoML(max_models=40, max_runtime_secs=20000, balance_classes=True, seed=123)

                        # trainSet, validSet = np.split(trainDataL1, [int(0.70 * len(trainDataL1))])

                        dnModel.train(x=['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'sza', 'aza', 'et'], y='pv', training_frame=h2o.H2OFrame(trainDataL1))
                        # aml.train(x=['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'sza', 'aza', 'et'], y='pv', training_frame=h2o.H2OFrame(trainSetL1), validation_frame=h2o.H2OFrame(validSetL1))
                        # aml.train(x=['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'sza', 'aza', 'et'], y='pv', training_frame=h2o.H2OFrame(trainDataL2), validation_frame=h2o.H2OFrame(testData))
                        # aml.train(x=['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'sza', 'aza', 'et'], y='pv', training_frame=h2o.H2OFrame(trainSet), validation_frame=h2o.H2OFrame(validSet))
                        # aml.train(x=['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'sza', 'aza', 'et'], y='pv', training_frame=h2o.H2OFrame(trainDataL2))

                        # 학습 모델 저장
                        saveModel = '{}/{}/{}-SRV{:05d}-{}-{}-{}-{}.model'.format(globalVar['outPath'], modelDirKey, serviceName, posId, 'final', 'h2o', 'act', datetime.now().strftime("%Y%m%d"))
                        log.info("[CHECK] saveModel : {}".format(saveModel))
                        os.makedirs(os.path.dirname(saveModel), exist_ok=True)
                        h2o.save_model(model=dnModel.get_best_model(), path=os.path.dirname(saveModel), filename=os.path.basename(saveModel), force=True)

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
