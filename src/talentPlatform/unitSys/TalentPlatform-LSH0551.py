# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import xarray as xr
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from multiprocessing import Pool
import multiprocessing as mp

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
        , 'resPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'output', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'mapInfo')
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

def subCalc(metaInfo, metaData, data, colNameList):

    result = None

    try:
        procInfo = mp.current_process()
        fnlData = pd.DataFrame()

        # dateInfo = dateList[0]
        # dateList = ["2019/01/01"]
        dateList = sorted(set(data['Date']))
        for dateInfo in dateList:
            log.info(f'[CHECK] dateInfo : {dateInfo} / pid : {procInfo.pid}')

            dataL1 = data.loc[
                (data['Date'] == dateInfo)
            ]
            if len(dataL1) < 1: continue

            selMetaData = metaData.loc[
                (metaData['city'] == metaInfo['city'])
            ]
            if len(selMetaData) < 1: continue

            # isYn == Y인 경우
            metaDataL4 = pd.DataFrame()
            if metaInfo['isYn'] == 'Y':

                metaDataL3 = dataL1.loc[
                    (dataL1['Level'] == 'County')
                    & (dataL1['State Postal Code'] == metaInfo['code'])
                    & (dataL1['County Name'].isin([f'{county} County' for county in selMetaData['county']]))
                    ]

                if len(metaDataL3) < 1: continue

                statData = metaDataL3[colNameList].sum(skipna=True)
                sumVal = statData[['Population Staying at Home', 'Population Not Staying at Home']].sum(skipna=True)
                allCnt = metaInfo['allCnt']
                weg = allCnt / sumVal
                metaDataL4 = statData.to_frame().transpose() * weg

            else:
                # isYn == N인 경우
                if pd.isna(selMetaData['county']).any():
                    metaDataL3 = dataL1.loc[
                        (dataL1['Level'] == 'County')
                        & (dataL1['State Postal Code'] == metaInfo['code'])
                        ]
                elif pd.isna(selMetaData['code']).any():
                    metaDataL3 = dataL1.loc[
                        (dataL1['Level'] == 'County')
                        & (dataL1['County Name'].isin([f'{county} County' for county in selMetaData['county']]))
                        ]
                else:
                    metaDataL3 = dataL1.loc[
                        (dataL1['Level'] == 'County')
                        & (dataL1['State Postal Code'] == metaInfo['code'])
                        & (dataL1['County Name'].isin([f'{county} County' for county in selMetaData['county']]))
                        ]

                if len(metaDataL3) < 1: continue
                metaDataL4 = metaDataL3[colNameList].sum(skipna=True).to_frame().transpose()

            metaDataL5 = pd.concat([metaInfo.to_frame().transpose().reset_index(drop=True), metaDataL4], axis=True)
            metaDataL5['date'] = dateInfo

            fnlData = pd.concat([fnlData, metaDataL5], ignore_index=True)

        saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, metaInfo['city'], metaInfo['code'])
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        fnlData.to_csv(saveFile, index=False)
        log.info(f'[CHECK] saveFile : {saveFile}')

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        log.info(f'[END] pid : {procInfo.pid}')


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 미국주,도시별 일평균 배출량을 이용하여 도시별 배출량 변환

    # cd /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/unitSys
    # nohup /SYSTEMS/anaconda3/envs/py38-test/bin/python3.8 TalentPlatform-LSH0551.py &
    # tail -f nohup.out

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
    serviceName = 'LSH0551'

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
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            sysOpt = {
                # 비동기 다중 프로세스 개수
                'cpuCoreNum': 40
                # 'cpuCoreNum': 1
            }

            # ********************************************************************
            # 메타 정보
            # ********************************************************************
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'Match.xlsx')
            fileList = sorted(glob.glob(inpFile))
            metaData = pd.read_excel(fileList[0], engine='openpyxl')

            # ********************************************************************
            # 파일 읽기
            # ********************************************************************
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'Trips_by_Distance_20240405.csv')
            fileList = sorted(glob.glob(inpFile))
            data = pd.read_csv(fileList[0])

            # dataL1.columns
            colNameList = ['Population Staying at Home', 'Population Not Staying at Home', 'Number of Trips', 'Number of Trips <1', 'Number of Trips 1-3', 'Number of Trips 3-5', 'Number of Trips 5-10', 'Number of Trips 10-25', 'Number of Trips 25-50', 'Number of Trips 50-100', 'Number of Trips 100-250', 'Number of Trips 250-500', 'Number of Trips >=500']

            # **************************************************************************************************************
            # 비동기 다중 프로세스 수행
            # **************************************************************************************************************
            # 비동기 다중 프로세스 개수
            pool = Pool(sysOpt['cpuCoreNum'])

            metaDataL1 = metaData[['city', 'code', 'isYn', 'allCnt']].drop_duplicates().reset_index(drop=True)
            metaDataL1 = metaDataL1.loc[metaDataL1['city'] == 'Minneapolis']
            for i, metaInfo in metaDataL1.iterrows():
                log.info(f'[CHECK] metaInfo : {metaInfo}')
                pool.apply_async(subCalc, args=(metaInfo, metaData, data, colNameList))
            pool.close()
            pool.join()

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
