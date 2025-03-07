# ================================================
# 요구사항
# ================================================
# Python을 이용한 태양광 자동화/수동화 모델링
# pip install optuna-integration[lightgbm]
# pip install flaml
# pip install pycaret[full]

# ps -ef | grep "TalentPlatform-INDI2025-model-solar.py" | awk '{print $2}' | xargs kill -9

# cd /vol01/SYSTEMS/INDIAI/PROG/PYTHON
# /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-model-real.py
# nohup /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-model-real.py &

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
from sklearn.metrics import mean_squared_error

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

def calcRmse(grpData, actCol, prdColList):
    result = {}
    for prdCol in prdColList:
        grpDataL1 = grpData[[actCol, prdCol]].dropna()
        if len(grpDataL1) < 1:
            result[f'{prdCol}_RMSE'] = np.nan
            continue

        actData = grpDataL1[actCol]
        prdData = grpDataL1[prdCol]
        result[f'{prdCol}_RMSE'] = np.sqrt(mean_squared_error(actData, prdData))

    return result

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
            }

            # **************************************************************************************************************
            # 설정 파일 읽기
            # **************************************************************************************************************
            filePattern = '{}/{}'.format('/DATA/FNL/202502/21', '20250221_solar_test_prd_for.csv')
            fileList = sorted(glob.glob(filePattern))
            if fileList is None or len(fileList) < 1:
                raise Exception(f"[ERROR] filePattern : {filePattern} / 파일을 확인해주세요.")
            testData = pd.read_csv(fileList[0])
            # testData.columns

            filePattern = '{}/{}'.format('/DATA/FNL/202502/21', '20250221_solar_real_prd_for.csv')
            fileList = sorted(glob.glob(filePattern))
            if fileList is None or len(fileList) < 1:
                raise Exception(f"[ERROR] filePattern : {filePattern} / 파일을 확인해주세요.")
            realData = pd.read_csv(fileList[0])
            # realData.columns

            # ****************************************************************************
            # 데이터 병합
            # ****************************************************************************
            mergeColList = ['forDt', 'forDtUtc', 'ulsan']
            data = pd.merge(testData, realData, how='left', left_on=mergeColList, right_on=mergeColList)
            dataL1 = data[data['ulsan'] > 0].reset_index(drop=False)
            dataL2 = dataL1[['forDt', 'ulsan', 'prd-lgb_x', 'prd-flaml_x', 'prd-pycaret_x', 'prd-lgb_y', 'prd-flaml_y', 'prd-pycaret_y']]

            # 편향 계산
            diffColList = ['prd-lgb_x', 'prd-flaml_x', 'prd-pycaret_x', 'prd-lgb_y', 'prd-flaml_y', 'prd-pycaret_y']
            for diffCol in diffColList:
                dataL2[f'{diffCol}_diff'] = dataL2['ulsan'] - dataL2[f'{diffCol}']

            # 일별 검증
            dataL2['dtDate'] = pd.to_datetime(dataL2['forDt']).dt.date
            grpData = dataL2.groupby('dtDate')

            prdColList = ['prd-lgb_x', 'prd-flaml_x', 'prd-pycaret_x', 'prd-lgb_y', 'prd-flaml_y', 'prd-pycaret_y']
            rmseByDate = grpData.apply(calcRmse, actCol='ulsan', prdColList=prdColList).apply(pd.Series)
            log.info(f'[CHECK] rmseByDate : {rmseByDate}')

            rmseByDateL1 = rmseByDate.mean().to_frame(name='mean')
            log.info(f'[CHECK] rmseByDateL1 : {rmseByDateL1}')

            dataL3 = dataL2.mean().to_frame(name='mean')
            log.info(f'[CHECK] dataL3 : {dataL3}')

            # ****************************************************************************
            # 자료 저장
            # ****************************************************************************
            saveFilePattern = '{}/{}'.format('/DATA/FNL/202502/21', '20250221_solar_merge_prd_for.csv')
            saveFile = saveFilePattern
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            dataL2.to_csv(saveFile, index=False)
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
