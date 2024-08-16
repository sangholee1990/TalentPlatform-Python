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
from viresclient import AeolusRequest

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


@retry(stop_max_attempt_number=10)
def colctRemssSat(modelInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        reqUrl = dtDateInfo.strftime(f"{modelInfo['request']['url']}/{modelInfo['request']['filePath']}")
        res = requests.get(reqUrl)
        if not (res.status_code == 200): return

        soup = BeautifulSoup(res.text, 'html.parser')
        for link in soup.find_all('a'):
            fileInfo = link.get('href')
            fileName = link.text

            match = re.match(modelInfo['request']['fileNamePattern'], fileName)
            if match is None: continue

            dtPreDate = pd.to_datetime(f'{match.group(1)}-{match.group(2)}-{match.group(3)}', format='%Y-%m-%d')
            isDate = (dtDateInfo == dtPreDate)
            if not isDate: continue

            tmpFileInfo = dtDateInfo.strftime(modelInfo['tmp']).format(fileName)
            updFileInfo = dtDateInfo.strftime(modelInfo['target']).format(fileName)

            # 파일 검사
            fileList = sorted(glob.glob(updFileInfo))
            if len(fileList) > 0: continue

            os.makedirs(os.path.dirname(tmpFileInfo), exist_ok=True)
            os.makedirs(os.path.dirname(updFileInfo), exist_ok=True)

            if os.path.exists(tmpFileInfo):
                os.remove(tmpFileInfo)

            cmd = f"curl -s -C - {modelInfo['request']['url']}/{fileInfo} --retry 10 -o {tmpFileInfo}"
            log.info(f'[CHECK] cmd : {cmd}')

            try:
                subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
            except subprocess.CalledProcessError as e:
                raise ValueError(f'[ERROR] 실행 프로그램 실패 : {str(e)}')

            if os.path.exists(tmpFileInfo):
                if os.path.getsize(tmpFileInfo) > 0:
                    shutil.move(tmpFileInfo, updFileInfo)
                    log.info(f'[CHECK] CMD : mv -f {tmpFileInfo} {updFileInfo}')
                else:
                    os.remove(tmpFileInfo)
                    log.info(f'[CHECK] CMD : rm -f {tmpFileInfo}')

            log.info(f'[END] colctRemssSat : {dtDateInfo} / pid : {procInfo.pid}')

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e
    finally:
        if os.path.exists(tmpFileInfo):
            os.remove(tmpFileInfo)

@retry(stop_max_attempt_number=10)
def colctAeolusSat(modelInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        # 변수 설정
        varList = {
            'ray': ["rayleigh_" + varInfo for varInfo in modelInfo['request']['varList']]
            , 'mie': ["mie_" + varInfo for varInfo in modelInfo['request']['varList']]
        }

        # 날짜 설정
        dtSrtDt = dtDateInfo
        dtEndDt = dtDateInfo + pd.Timedelta(hours=23, minutes=59, seconds=59)
        dtDtList = pd.date_range(start=dtSrtDt, end=dtEndDt, freq='1h')

        for i, dtDtInfo in enumerate(dtDtList):
            if (i + 1) == len(dtDtList): continue
            sSrtDt = dtDtList[i].tz_localize('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
            sEndDt = dtDtList[i + 1].tz_localize('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
            log.info(f'[CHECK] sSrtDt : {sSrtDt} ~ sEndDt : {sEndDt}')

            for key, val in varList.items():
                request = AeolusRequest(url=modelInfo['request']['url'], token=modelInfo['request']['token'])
                request.set_collection(modelInfo['request']['varLevel'])

                if re.search('ray', key, re.IGNORECASE):
                    request.set_fields(rayleigh_wind_fields=val)
                    request.set_range_filter(parameter="rayleigh_wind_result_COG_latitude", minimum=0, maximum=90)
                    request.set_range_filter(parameter="rayleigh_wind_result_COG_longitude", minimum=180, maximum=360)
                elif re.search('mie', key, re.IGNORECASE):
                    request.set_fields(mie_wind_fields=val)
                    request.set_range_filter(parameter="mie_wind_result_COG_latitude", minimum=0, maximum=90)
                    request.set_range_filter(parameter="mie_wind_result_COG_longitude", minimum=180, maximum=360)
                else:
                    continue

                tmpFileInfo = dtDtInfo.strftime(modelInfo['tmp']).format(key)
                updFileInfo = dtDtInfo.strftime(modelInfo['target']).format(key)

                # 파일 검사
                fileList = sorted(glob.glob(updFileInfo))
                if len(fileList) > 0: continue

                data = request.get_between(start_time=sSrtDt, end_time=sEndDt, filetype="nc", asynchronous=True)
                dataL1 = data.as_xarray()

                if len(dataL1) < 1: continue

                os.makedirs(os.path.dirname(tmpFileInfo), exist_ok=True)
                os.makedirs(os.path.dirname(updFileInfo), exist_ok=True)

                if os.path.exists(tmpFileInfo):
                    os.remove(tmpFileInfo)

                data.to_file(tmpFileInfo, overwrite=True)
                # dataL1.to_netcdf(tmpFileInfo)

                if os.path.exists(tmpFileInfo):
                    if os.path.getsize(tmpFileInfo) > 0:
                        shutil.move(tmpFileInfo, updFileInfo)
                        log.info(f'[CHECK] CMD : mv -f {tmpFileInfo} {updFileInfo}')
                    else:
                        os.remove(tmpFileInfo)
                        log.info(f'[CHECK] CMD : rm -f {tmpFileInfo}')

                log.info(f'[END] colctAeolusSat : {dtDateInfo} / pid : {procInfo.pid}')

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e
    finally:
        if os.path.exists(tmpFileInfo):
            os.remove(tmpFileInfo)

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 위성측정기반 바람 산출물 수집

    # cd /home/hanul/SYSTEMS/KIER/PROG/PYTHON/colct
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-remssAMSR2.py --modelList SAT-AMSR2 --cpuCoreNum 5 --srtDate 2024-08-01 --endDate 2024-08-15 &

    # ps -ef | grep "TalentPlatform-INDI2024-colct-remssSMAP.py" | awk '{print $2}' | xargs kill -9
    # ps -ef | grep "RunShell-get-gfsncep2.sh" | awk '{print $2}' | xargs kill -9
    # ps -ef | egrep "RunShell|Repro" | awk '{print $2}' | xargs kill -9

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
        # contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'
        contextPath = os.getcwd() if env in 'local' else '/vol01/SYSTEMS/KIER/PROG/PYTHON'

    prjName = 'test'
    serviceName = 'INDI2024'

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
                # 시작일, 종료일, 시간 간격 (연 1y, 월 1h, 일 1d, 시간 1h)
                'srtDate': '2023-01-01'
                , 'endDate': '2023-01-03'
                # 'srtDate': globalVar['srtDate']
                # , 'endDate': globalVar['endDate']
                , 'invDate': '1d'

                # 수행 목록
                # , 'modelList': ['SSMIS', 'AMSR2', 'GMI', 'SMAP', 'ASCAT-B', 'ASCAT-C', 'AEOLUS']
                , 'modelList': ['AEOLUS']
                # 'modelList': [globalVar['modelList']]

                # 비동기 다중 프로세스 개수
                , 'cpuCoreNum': '7'
                # , 'cpuCoreNum': globalVar['cpuCoreNum']

                , 'SSMIS': {
                    'request': {
                        'url': 'https://data.remss.com'
                        , 'filePath': '/ssmi/f18/bmaps_v08/y%Y/m%m'
                        , 'fileNamePattern': 'f18_(\d{4})(\d{2})(\d{2})v(\d+)\.gz'
                    }
                    , 'tmp': '/HDD/DATA/data1/SAT/SSMIS/%Y/%m/.{}'
                    , 'target': '/HDD/DATA/data1/SAT/SSMIS/%Y/%m/{}'
                }
                , 'AMSR2': {
                    'request': {
                        'url': 'https://data.remss.com'
                        , 'filePath': '/amsr2/ocean/L3/v08.2/daily/%Y'
                        , 'fileNamePattern': 'RSS_AMSR2_ocean_L3_daily_(\d{4})-(\d{2})-(\d{2})_v(\d+\.\d+)\.nc'
                    }
                    , 'tmp': '/HDD/DATA/data1/SAT/AMSR2/%Y/%m/.{}'
                    , 'target': '/HDD/DATA/data1/SAT/AMSR2/%Y/%m/{}'
                }
                , 'GMI': {
                    'request': {
                        'url': 'https://data.remss.com'
                        , 'filePath': '/gmi/bmaps_v08.2/y%Y/m%m'
                        , 'fileNamePattern': 'f35_(\d{4})(\d{2})(\d{2})v(\d+\.\d+)\.gz'
                    }
                    , 'tmp': '/HDD/DATA/data1/SAT/GMI/%Y/%m/.{}'
                    , 'target': '/HDD/DATA/data1/SAT/GMI/%Y/%m/{}'
                }
                , 'SMAP': {
                    'request': {
                        'url': 'https://data.remss.com'
                        , 'filePath': '/smap/wind/L3/v01.0/daily/NRT/%Y'
                        , 'fileNamePattern': 'RSS_smap_wind_daily_(\d{4})_(\d{2})_(\d{2})_NRT_v(\d+\.\d+)\.nc'
                    }
                    , 'tmp': '/HDD/DATA/data1/SAT/SMAP/%Y/%m/.{}'
                    , 'target': '/HDD/DATA/data1/SAT/SMAP/%Y/%m/{}'
                }
                , 'ASCAT-B': {
                    'request': {
                        'url': 'https://data.remss.com'
                        , 'filePath': '/ascat/metopb/bmaps_v02.1/y%Y/m%m'
                        , 'fileNamePattern': 'ascatb_(\d{4})(\d{2})(\d{2})_v(\d+\.\d+)\.gz'
                    }
                    , 'tmp': '/HDD/DATA/data1/SAT/ASCAT/%Y/%m/.{}'
                    , 'target': '/HDD/DATA/data1/SAT/ASCAT/%Y/%m/{}'
                }
                , 'ASCAT-C': {
                    'request': {
                        'url': 'https://data.remss.com'
                        , 'filePath': '/ascat/metopc/bmaps_v02.1/y%Y/m%m'
                        , 'fileNamePattern': 'ascatc_(\d{4})(\d{2})(\d{2})_v(\d+\.\d+)\.gz'
                    }
                    , 'tmp': '/HDD/DATA/data1/SAT/ASCAT/%Y/%m/.{}'
                    , 'target': '/HDD/DATA/data1/SAT/ASCAT/%Y/%m/{}'
                }
                , 'AEOLUS': {
                    'request': {
                        'url': 'https://aeolus.services/ows'
                        , 'token': ''
                        , 'varLevel': 'ALD_U_N_2B'
                        , 'varList': [
                            "wind_result_start_time"
                            , "wind_result_stop_time"
                            , "wind_result_COG_time"
                            , "wind_result_bottom_altitude"
                            , "wind_result_top_altitude"
                            , "wind_result_range_bin_number"
                            , "wind_result_start_latitude"
                            , "wind_result_start_longitude"
                            , "wind_result_stop_latitude"
                            , "wind_result_stop_longitude"
                            , "wind_result_COG_latitude"
                            , "wind_result_COG_longitude"
                            , "wind_result_HLOS_error"
                            , "wind_result_wind_velocity"
                            , "wind_result_observation_type"
                            , "wind_result_validity_flag"
                            , "wind_result_alt_of_DEM_intersection"
                        ]
                    }
                    , 'tmp': '/HDD/DATA/data1/SAT/AEOLUS/%Y/%m/%d/.aeolus_wind-{}_%Y%m%d%H%M.nc'
                    , 'target': '/HDD/DATA/data1/SAT/AEOLUS/%Y/%m/%d/aeolus_wind-{}_%Y%m%d%H%M.nc'
                }
            }

            # **************************************************************************************************************
            # 비동기 다중 프로세스 수행
            # **************************************************************************************************************
            # 비동기 다중 프로세스 개수
            pool = Pool(int(sysOpt['cpuCoreNum']))

            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])
            for dtDateInfo in dtDateList:
                log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                for modelType in sysOpt['modelList']:
                    # log.info(f'[CHECK] modelType : {modelType}')

                    modelInfo = sysOpt.get(modelType)
                    if modelInfo is None: continue

                    if re.search('AEOLUS', modelType, re.IGNORECASE):
                        pool.apply_async(colctAeolusSat, args=(modelInfo, dtDateInfo))
                    else:
                        pool.apply_async(colctRemssSat, args=(modelInfo, dtDateInfo))

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
