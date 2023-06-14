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
from builtins import enumerate

# import datetime as dt
# from datetime import datetime
# import pvlib
import matplotlib.dates as mdates
import matplotlib.cm as cm

import matplotlib as mpl
import matplotlib.pyplot as plt
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
from sklearn.model_selection import train_test_split

import xarray as xr
from pvlib import location
from pvlib import irradiance

import h2o
from h2o.automl import H2OAutoML

from pycaret.regression import *
from matplotlib import font_manager, rc

# try:
#     from pycaret.regression import *
# except Exception as e:
#     print("Exception : {}".format(e))
#
try:
    from pycaret.regression import *
except Exception as e:
    print("Exception : {}".format(e))


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers
# from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.utils import safe_indexing
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


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

#plt.rc('font', family='Malgun Gothic')
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
        # fileList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        # fontName = font_manager.FontProperties(fname=fileList[0]).get_name()
        # plt.rc('font', family=fontName)

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
        saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model.pkl'.format(globalVar['modelPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'pycaret', 'for', '*')
        saveModelList = sorted(glob.glob(saveModel), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        data = inpData[xyCol]
        dataL1 = data.dropna()

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            trainData, validData = train_test_split(data, test_size=0.3)
            # trainData = inpData

            pyModel = setup(
                data=dataL1
                , session_id=123
                , silent=True
                , target=yCol
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
            modelList = compare_models(sort='RMSE', n_select=3)

            # 앙상블 모형
            blendModel = blend_models(estimator_list=modelList, fold=10)

            # 앙상블 파라미터 튜닝
            tuneModel = tune_model(blendModel, fold=10, choose_better=True)

            # 학습 모형
            fnlModel = finalize_model(tuneModel)

            # 학습 모형 저장
            saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['modelPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'pycaret', 'for', datetime.datetime.now().strftime('%Y%m%d'))
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

    env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PV_20220523'

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


            # ******************************************************************************
            # 크몽 전문가 검증 테스트
            # ******************************************************************************


            globalVar['inpPath'] = '/DATA/INPUT'
            globalVar['outPath'] = '/DATA/OUTPUTT'
            globalVar['figPath'] = '/DATA/FIG'

            inpFile = '{}/{}/{}.csv'.format(globalVar['inpPath'], serviceName, 'FINAL_PW_DATA')
            fileList = sorted(glob.glob(inpFile))
            # pw_data = pd.read_csv('FINAL_PW_DATA.csv')
            pw = pd.read_csv(fileList[0])


            inpFile = '{}/{}/{}.csv'.format(globalVar['inpPath'], serviceName, 'FINAL_FORE_DATA')
            fileList = sorted(glob.glob(inpFile))
            # fore_data = pd.read_csv('FINAL_FORE_DATA.csv')
            fore = pd.read_csv(fileList[0])

            # fore 데이터를 기준으로 pw를 병합하기 때문에 pw 데이터의 수용가번호 (CNSMR_NO) 삭제
            # 따라서 pw 데이터를 기준으로 fore 데이터를 좌측 조인해야 함
            full = pd.merge(fore, pw[['SRV', 'MESURE_DATE_TM', 'SG_PWRER_USE_AM']])
            full = full.drop_duplicates().sort_values(['SRV', 'MESURE_DATE_TM'])

            full.columns

            sg_pwrer_use_am_avg=pd.pivot_table(data=full, index=['SRV','MESURE_DATE_TM'],
                           values=['SG_PWRER_USE_AM'], aggfunc='mean').reset_index().sort_values(['SRV','MESURE_DATE_TM'])
            srv_date=full.drop_duplicates(['SRV','MESURE_DATE_TM']).drop('SG_PWRER_USE_AM',axis=1)

            data=pd.merge(sg_pwrer_use_am_avg,srv_date, on=['SRV','MESURE_DATE_TM']).sort_values(['SRV','MESURE_DATE_TM'])



            # 기상변수 파생변수 생성
            # n일전 기상데이터를 변수로 추가
            lag_1=data[['CA_TOT', 'HM', 'PA', 'TA','TD', 'WD', 'WS', 'SZA', 'AZA',
                  'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR']].shift(1)

            lag_2=data[['CA_TOT', 'HM', 'PA', 'TA','TD', 'WD', 'WS', 'SZA', 'AZA',
                  'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR']].shift(2)

            lag_3=data[['CA_TOT', 'HM', 'PA', 'TA','TD', 'WD', 'WS', 'SZA', 'AZA',
                  'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR']].shift(3)

            lag_4=data[['CA_TOT', 'HM', 'PA', 'TA','TD', 'WD', 'WS', 'SZA', 'AZA',
                  'ET', 'TURB', 'GHI_CLR', 'DNI_CLR', 'DHI_CLR', 'SWR']].shift(4)

            lag_df_list=[lag_1,lag_2,lag_3,lag_4]

            for i,df in enumerate(lag_df_list):
                new_col=[]
                for j in df.columns:
                    new_col.append('lag_'+str(i+1)+'_'+j)
                df.columns=new_col

            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split

            # from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

            all_leg_df = pd.concat([data, lag_1, lag_2, lag_3, lag_4], axis=1)
            all_leg_df = all_leg_df.dropna()

            X_train.columns

            X_train, X_test, Y_train, Y_test = train_test_split(
                all_leg_df.drop(['SRV', 'MESURE_DATE_TM', 'SG_PWRER_USE_AM'], axis=1), all_leg_df['SG_PWRER_USE_AM'],
                test_size=0.2, random_state=42)

            rf = RandomForestRegressor()
            rf.fit(X_train, Y_train)
            rf_pred = rf.predict(X_test)
            print('RMSE: ', np.sqrt(mean_squared_error(Y_test, rf_pred)))

            sns.lineplot(rf_pred[:500], label='pred')
            sns.lineplot(Y_test.values[:500], label='real')



            # # 원본 데이터 : 15,319,449
            # # orgData = fore_data.merge(pw_data, on=['SRV', 'MESURE_DATE_TM'])
            # orgData = fore_data.merge(pw_data, on=['SRV', 'MESURE_DATE_TM'])
            #
            # orgData.columns
            #
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
            #
            # # SRV1141012000 = data.loc[data['SRV'] == 'SRV1141012000', ['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'SG_PWRER_USE_AM', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']].reset_index(drop = True)
            # # SRV1153010100 = data.loc[data['SRV'] == 'SRV1153010100', ['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'SG_PWRER_USE_AM', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']].reset_index(drop = True)
            # # SRV3011010500 = data.loc[data['SRV'] == 'SRV3011010500', ['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'SG_PWRER_USE_AM', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']].reset_index(drop = True)
            # SRV5011012200 = data.loc[data['SRV'] == 'SRV5011012200', ['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'SG_PWRER_USE_AM', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']].reset_index(drop = True)
            #
            # # 독립변수 목록 정의
            # independent_variable = ['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']
            #
            # minmax_scaler = MinMaxScaler()
            # minmax_scaler = minmax_scaler.fit_transform(SRV5011012200.loc[:, independent_variable])
            # SRV5011012200.loc[:, independent_variable] = minmax_scaler
            # SRV5011012200.head()
            #
            #
            #
            # # ==============================================================================
            # # 단기 예측
            # # ==============================================================================
            # t = 48
            # X = np.array(SRV5011012200.loc[:, independent_variable])
            # y = SRV5011012200['SG_PWRER_USE_AM']
            # #
            # # train_X = X[:-t, :]
            # # test_X = X[-t:, :]
            # # train_y = np.array(y[:-t])
            # # test_y = np.array(y[-t:])
            # #
            # # # n_estimators_candidate = [100, 300, 500]
            # # # max_depth_candidate = [3, 5, 7]
            # # # min_samples_split_candidate = [5, 7, 9]
            # #
            # # # 결과를 저장할 빈 리스트 생성
            # # n_estimators_candidate = [100, 200, 300, 400, 500]
            # # max_depth_candidate = [3, 5, 7, 9]
            # # learning_rate_candidate = [0.1, 0.01, 0.001, 0.0001]
            # #
            # # # 결과를 저장할 빈 리스트 생성
            # # n_estimators_list = []
            # # max_depth_list = []
            # # learning_rate_list = []
            # # train_score_list = []
            # # val_score_list = []
            # #
            # # # XGBoost의 n_estimators 파라미터에 대해서
            # # for n_estimators in n_estimators_candidate:
            # #     # XGBoost의 max_depth 파라미터에 대해서
            # #     for max_depth in max_depth_candidate:
            # #         # XGBoost의 learning_rate 파라미터에 대해서
            # #         for learning_rate in learning_rate_candidate:
            # #             print('[CHECK] n_estimators : {} / max_depth : {} / learning_rate : {}'.format(n_estimators, max_depth, learning_rate))
            # #
            # #             # 모델 생성 및 학습
            # #             model = xgboost.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, objective='reg:squarederror').fit(train_X, train_y)
            # #             # Train 데이터에 대한 결과 확인
            # #             train_predict = model.predict(train_X)
            # #             train_score_list.append(np.sqrt(mean_squared_error(train_y, train_predict)))
            # #             # Test 데이터에 대한 결과 확인
            # #             test_predict = model.predict(test_X)
            # #             val_score_list.append(np.sqrt(mean_squared_error(test_y, test_predict)))
            # #             # Parameter 저장
            # #             n_estimators_list.append(n_estimators)
            # #             max_depth_list.append(max_depth)
            # #             learning_rate_list.append(learning_rate)
            # #
            # # result = result.loc[result['Test Score'] == min(result['Test Score']), :].reset_index(drop=True)
            # # result
            # #
            # # model = xgboost.XGBRegressor(n_estimators=result['n_estimators'][0], max_depth=result['max_depth'][0], min_samples_split=result['min_samples_split'][0], objective='reg:squarederror').fit(train_X, train_y)
            # # test_predict = model.predict(test_X)
            # #
            # # print("RMSE: {}".format(np.sqrt(mean_squared_error(test_y, test_predict))))
            # # # print("MAPE: {}".format(mean_absolute_percentage_error(test_y, test_predict)))
            # #
            # # # ***********************************************************
            # # # 가공 2차 및 AI 학습모형을 이용한 예측 결과
            # # # ***********************************************************
            # # dataL1['prd'] = model.predict(dataL1[['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']])
            # #
            # # plt.plot(dataL1['MESURE_DATE_TM'], dataL1['SG_PWRER_USE_AM'], label='실측')
            # # plt.plot(dataL1['MESURE_DATE_TM'], dataL1['prd'], label='예측')
            # # mainTitle = '[short-term] [{0:s}] RMSE = {1:.2f}'.format(
            # #     dataL1['SRV'].iloc[0]
            # #     , np.sqrt(mean_squared_error(dataL1['SG_PWRER_USE_AM'], dataL1['prd']))
            # # )
            # # plt.title(mainTitle)
            # #
            # # saveImg = '{}/short-term_{}.png'.format(globalVar['figPath'], dataL1['SRV'].iloc[0])
            # # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            # # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent = True)
            # # plt.show()
            # # plt.close()
            # #
            # # # ***********************************************************
            # # # 가공 1차 및 AI 학습모형을 이용한 예측 결과
            # # # ***********************************************************
            # # cnsmrNoList = np.unique(orgDataL1['CNSMR_NO'])
            # # for i, cnsmrNo in enumerate(cnsmrNoList):
            # #
            # #     orgDataL2 = orgDataL1.loc[orgDataL1['CNSMR_NO'] == cnsmrNo]
            # #     print(cnsmrNo, len(orgDataL2))
            # #
            # #     orgDataL2['prd'] = model.predict(orgDataL2[['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']])
            # #
            # #     # plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['SG_PWRER_USE_AM'])
            # #     # plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['prd'])
            # #     # plt.show()
            # #
            # #     plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['SG_PWRER_USE_AM'], label='실측')
            # #     plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['prd'], label='예측')
            # #     mainTitle = '[short-term] [{0:s}-{1:s}] RMSE = {2:.2f}'.format(
            # #         orgDataL2['SRV'].iloc[0]
            # #         , cnsmrNo
            # #         , np.sqrt(mean_squared_error(orgDataL2['SG_PWRER_USE_AM'], orgDataL2['prd']))
            # #     )
            # #     plt.title(mainTitle)
            # #
            # #     saveImg = '{}/short-term_{}-{}.png'.format(globalVar['figPath'], dataL1['SRV'].iloc[0], cnsmrNo)
            # #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            # #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            # #     plt.show()
            # #     plt.close()
            #
            #
            #
            # # ==============================================================================
            # # 장기 예측
            # # ==============================================================================
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
            # # print("MAPE: {}".format(mean_absolute_percentage_error(test_y, test_predict)))
            #
            #
            # # ***********************************************************
            # # 가공 2차 및 AI 학습모형을 이용한 예측 결과
            # # ***********************************************************
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
            #
            # # # ***********************************************************
            # # # 가공 1차 및 AI 학습모형을 이용한 예측 결과
            # # # ***********************************************************
            # # cnsmrNoList = np.unique(orgDataL1['CNSMR_NO'])
            # # for i, cnsmrNo in enumerate(cnsmrNoList):
            # #
            # #     orgDataL2 = orgDataL1.loc[orgDataL1['CNSMR_NO'] == cnsmrNo]
            # #     print(cnsmrNo, len(orgDataL2))
            # #
            # #     orgDataL2['prd'] = model.predict(orgDataL2[['CA_TOT',	'HM',	'PA',	'TA', 'TD',	'WD',	'WS',	'SZA',	'AZA',	'ET',	'TURB',	'GHI_CLR',	'DNI_CLR',	'DHI_CLR',	'SWR', 'is_weekday', 'is_spring', 'is_summer', 'is_fall', 'is_winter']])
            # #
            # #     # plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['SG_PWRER_USE_AM'])
            # #     # plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['prd'])
            # #     # plt.show()
            # #
            # #     plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['SG_PWRER_USE_AM'], label='실측')
            # #     plt.plot(orgDataL2['MESURE_DATE_TM'], orgDataL2['prd'], label='예측')
            # #     mainTitle = '[short-term] [{0:s}-{1:s}] RMSE = {2:.2f}'.format(
            # #         orgDataL2['SRV'].iloc[0]
            # #         , cnsmrNo
            # #         , np.sqrt(mean_squared_error(orgDataL2['SG_PWRER_USE_AM'], orgDataL2['prd']))
            # #     )
            # #     plt.title(mainTitle)
            # #
            # #     saveImg = '{}/long-term_{}-{}.png'.format(globalVar['figPath'], dataL1['SRV'].iloc[0], cnsmrNo)
            # #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            # #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            # #     plt.show()
            # #     plt.close()


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
