# ================================================
# 요구사항
# ================================================
# Python을 이용한

# ps -ef | grep "TalentPlatform-INDI2025-colct-kmaApiHub.py" | awk '{print $2}' | xargs kill -9

# cd /vol01/SYSTEMS/INDIAI/PROG/PYTHON
# /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-colct-kmaApiHub.py --modelList 'UMKR' --cpuCoreNum '5' --srtDate '2019-01-01' --endDate '2020-01-01'

# nohup /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-colct-kmaApiHub.py --modelList 'UMKR' --cpuCoreNum '5' --srtDate '2019-01-01' --endDate '2020-01-01' &

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
# import cdsapi
import shutil

import requests
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime
import subprocess
from isodate import parse_duration
from pandas.tseries.offsets import DateOffset

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

    return globalVar


def propProc(modelType, modelInfo, dtDateInfo):
    try:
        propFunList = {
            'UMKR': propNwp,
        }

        propFun = propFunList.get(modelType)
        propFun(modelInfo, dtDateInfo)

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e

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

@retry(stop_max_attempt_number=10)
def propNwp(dataInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        searchFileInfo = dtDateInfo.strftime(dataInfo['searchFileList'])

        # 파일 검사
        fileList = sorted(glob.glob(searchFileInfo))
        if len(fileList) > 0: return

        # reqUrl = dtDateInfo.strftime(f"{modelInfo['request']['url']}").format(
        #     tmfc=dtDateInfo.strftime('%Y%m%d%H%M'),
        #     tmfc2=(dtDateInfo + parseDateOffset(modelInfo['request']['invDate']) - parseDateOffset('1s')).strftime('%Y%m%d%H%M'),
        #     authKey=modelInfo['request']['authKey']
        # )

        # res = requests.get(reqUrl)
        # if not (res.status_code == 200): return

        # os.makedirs(os.path.dirname(tmpFileInfo), exist_ok=True)
        # os.makedirs(os.path.dirname(updFileInfo), exist_ok=True)
        #
        # if os.path.exists(tmpFileInfo):
        #     os.remove(tmpFileInfo)
        #
        # cmd = f"curl -s -C - '{reqUrl}' --retry 10 -o {tmpFileInfo}"
        # log.info(f'[CHECK] cmd : {cmd}')
        #
        # try:
        #     subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
        # except subprocess.CalledProcessError as e:
        #     raise ValueError(f'[ERROR] 실행 프로그램 실패 : {str(e)}')
        #
        # if os.path.exists(tmpFileInfo):
        #     if os.path.getsize(tmpFileInfo) > 1000:
        #         shutil.move(tmpFileInfo, updFileInfo)
        #         log.info(f'[CHECK] CMD : mv -f {tmpFileInfo} {updFileInfo}')
        #     else:
        #         os.remove(tmpFileInfo)
        #         log.info(f'[CHECK] CMD : rm -f {tmpFileInfo}')

        log.info(f'[END] colctKmaApiHub : {dtDateInfo} / pid : {procInfo.pid}')

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e


def matchStnRadar(sysOpt, modelInfo, code, dtDateList):

    try:
        # ==========================================================================================================
        # 융합 ASOS/AWS 지상 관측소을 기준으로 최근접 레이더 가공파일 화소 찾기 (posRow, posCol, posLat, posLon, posDistKm)
        # ==========================================================================================================
        # 매 5분 순간마다 가공파일 검색/병합
        procFilePattern = '{}/{}'.format(modelInfo['procPath'], modelInfo['procName'])
        # dtHourList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invHour'])

        searchList = []
        for dtDateInfo in dtDateList:
            procFile = dtDateInfo.strftime(procFilePattern).format(code)
            fileList = sorted(glob.glob(procFile))
            if fileList is None or len(fileList) < 1: continue
            searchList.append(fileList[0])
            break

        if searchList is None or len(searchList) < 1:
            log.error(f"[ERROR] procFilePattern : {procFilePattern} / 가공파일을 확인해주세요.")
            return

        # 레이더 가공 파일 일부
        fileInfo = searchList[0]
        cfgData = xr.open_dataset(fileInfo)
        cfgDataL1 = cfgData.to_dataframe().reset_index(drop=False)

        # ASOS/AWS 융합 지상관측소
        inpFilePattern = '{}/{}'.format(sysOpt['stnInfo']['filePath'], sysOpt['stnInfo']['fileName'])
        fileList = sorted(glob.glob(inpFilePattern))
        if fileList is None or len(fileList) < 1:
            log.error(f"[ERROR] inpFilePattern : {inpFilePattern} / 융합 지상관측소를 확인해주세요.")
            return

        fileInfo = fileList[0]
        allStnData = pd.read_csv(fileInfo)
        allStnDataL1 = allStnData[['STN', 'STN_KO', 'LON', 'LAT']]

        # 2024.10.24 수정
        # allStnDataL2 = allStnDataL1[allStnDataL1['STN'].isin(sysOpt['stnInfo']['list'])]
        allStnDataL2 = allStnDataL1

        # 융합 ASOS/AWS 지상 관측소을 기준으로 최근접 레이더 가공파일 화소 찾기 (posRow, posCol, posLat, posLon, posDistKm)
        #      STN STN_KO        LON       LAT  ...  posCol      posLat     posLon  posDistKm
        # 0     90     속초  128.56473  38.25085  ...   456.0  128.565921  38.251865   0.024091
        # 10   104    북강릉  128.85535  37.80456  ...   482.0  128.850344  37.805816   0.072428
        # 11   105     강릉  128.89099  37.75147  ...   486.0  128.894198  37.750990   0.045061
        # 12   106     동해  129.12433  37.50709  ...   507.0  129.124659  37.503421   0.064192
        # 289  520    설악동  128.51818  38.16705  ...   452.0  128.518319  38.171507   0.077807
        # 292  523    주문진  128.82139  37.89848  ...   479.0  128.818774  37.896447   0.050570
        # 424  661     현내  128.40191  38.54251  ...   441.0  128.401035  38.542505   0.011947
        # 432  670     양양  128.62954  38.08874  ...   462.0  128.630338  38.088726   0.010963
        # 433  671     청호  128.59360  38.19091  ...   459.0  128.598611  38.188309   0.082373
        baTree = BallTree(np.deg2rad(cfgDataL1[['lat', 'lon']].values), metric='haversine')
        for i, posInfo in allStnDataL2.iterrows():
            if (pd.isna(posInfo['LAT']) or pd.isna(posInfo['LON'])): continue

            closest = baTree.query(np.deg2rad(np.c_[posInfo['LAT'], posInfo['LON']]), k=1)
            cloDist = closest[0][0][0] * 1000.0
            cloIdx = closest[1][0][0]
            cfgInfo = cfgDataL1.loc[cloIdx]

            allStnDataL2.loc[i, 'posRow'] = cfgInfo['row']
            allStnDataL2.loc[i, 'posCol'] = cfgInfo['col']
            allStnDataL2.loc[i, 'posLat'] = cfgInfo['lon']
            allStnDataL2.loc[i, 'posLon'] = cfgInfo['lat']
            allStnDataL2.loc[i, 'posDistKm'] = cloDist

        # log.info(f"[CHECK] allStnDataL2 : {allStnDataL2}")

        saveFilePattern = '{}/{}'.format(sysOpt['stnInfo']['matchPath'], sysOpt['stnInfo']['matchName'])
        saveFile = dtDateInfo.strftime(saveFilePattern).format(code)
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        allStnDataL2.to_csv(saveFile, index=False)
        log.info(f"[CHECK] saveFile : {saveFile}")

    except Exception as e:
        log.error(f'Exception : {str(e)}')

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
    def __init__(self, inParams):

        log.info(f"[START] __init__ : init")

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

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
                # 예보시간 시작일, 종료일, 시간 간격 (연 1y, 월 1m, 일 1d, 시간 1h, 분 1t, 초 1s)
                'srtDate': '2019-01-01',
                'endDate': '2019-01-04',
                # 'srtDate': globalVar['srtDate'],
                # 'endDate': globalVar['endDate'],

                # 수행 목록
                # 'modelList': ['ACT', 'FOR'],
                'modelList': ['FOR'],
                # 'modelList': [globalVar['modelList']],

                # 비동기 다중 프로세스 개수
                'cpuCoreNum': '5',
                # 'cpuCoreNum': globalVar['cpuCoreNum'],
                # 설정 파일
                'CFG': {
                    'siteInfo': {
                        'filePath': '/DATA/PROP/SAMPLE',
                        'fileName': 'site_info.csv',
                    },
                    # 'energy': {
                    #     'filePath': '/DATA/PROP/SAMPLE',
                    #     'fileName': 'energy.csv',
                    # },
                    # 'ulsanObs': {
                    #     'filePath': '/DATA/PROP/SAMPLE',
                    #     'fileName': 'ulsan_obs_data.csv',
                    # },
                    # 'ulsanFcst': {
                    #     'filePath': '/DATA/PROP/SAMPLE',
                    #     'fileName': 'ulsan_fcst_data.csv',
                    # },
                },

                'FOR': {
                    'UMKR': {
                        'searchFileList': f"/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d%H%M.grb2",
                        'invDate': '6h',
                        # , 'saveFile': '/DATA/PROP/%Y%m/%d/UMKR_l015_unis_H{ef}_%Y%m%d%H%M.grb2'
                    },
                },
                'ACT': {
                    'ASOS': {
                        'searchFileList': f"/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d%H%M.grb2",
                        'invDate': '6h',
                    },
                    'AWS': {
                        'searchFileList': f"/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d%H%M.grb2",
                        'invDate': '6h',
                    },
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
                    log.error(f"[ERROR] filePattern : {filePattern} / 파일을 확인해주세요.")
                    raise Exception(f"[ERROR] filePattern : {filePattern} / 파일을 확인해주세요.")
                data = pd.read_csv(fileList[0])

                cfgData[key] = data

            # **************************************************************************************************************
            # 비동기 다중 프로세스 수행
            # **************************************************************************************************************
            # 비동기 다중 프로세스 개수
            pool = Pool(int(sysOpt['cpuCoreNum']))

            for modelType in sysOpt['modelList']:
                modelInfo = sysOpt.get(modelType)
                if modelInfo is None: continue

                for dataType in modelInfo.keys():
                    dataInfo = modelInfo.get(dataType)

                    # 시작일/종료일 설정
                    dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
                    dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
                    dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=dataInfo['invDate'])

                    for dtDateInfo in dtDateList:
                        # log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
                        pool.apply_async(propProc, args=(dataType, dataInfo, dtDateInfo))

            pool.close()
            pool.join()

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
        print('[END] main')
