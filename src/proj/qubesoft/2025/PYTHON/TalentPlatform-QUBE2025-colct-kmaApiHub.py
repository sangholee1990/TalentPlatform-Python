# ================================================
# 요구사항
# ================================================
# Python을 이용한 기상청 API 허브 다운로드

# ps -ef | grep "TalentPlatform-QUBE2025-colct-kmaApiHub.py" | awk '{print $2}' | xargs kill -9
# pkill -f "TalentPlatform-QUBE2025-colct-kmaApiHub.py"

# cd /SYSTEMS/PROG/PYTHON
# /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/TalentPlatform-QUBE2025-colct-kmaApiHub.py --modelList 'UMKR' --cpuCoreNum '5' --srtDate '2024-12-01' --endDate '2024-12-05'
# /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/TalentPlatform-QUBE2025-colct-kmaApiHub.py --modelList 'UMKR' --cpuCoreNum '5' --srtDate '2024-01-01' --endDate '2025-01-01'

# nohup /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/TalentPlatform-QUBE2025-colct-kmaApiHub.py --modelList 'UMKR' --cpuCoreNum '5' --srtDate '2025-01-01' --endDate "$(date -u +\%Y-\%m-\%d)" &

# 2025.11.02
# */10 * * * * cd /SYSTEMS/PROG/PYTHON && /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/TalentPlatform-QUBE2025-colct-kmaApiHub.py --modelList 'UMKR' --cpuCoreNum '5' --srtDate "$(date -d "2 days ago" +\%Y-\%m-\%d)" --endDate "$(date -d "2 days" +\%Y-\%m-\%d)"

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
import configparser
from urllib.parse import urlparse, parse_qs
from lxml import etree

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


def colctProc(modelType, modelInfo, dtDateInfo):
    try:
        colctFunList = {
            'UMKR': colctNwp,
            'KIMG': colctNwp,
            'ASOS': colctObs,
            'AWS': colctObs,
        }

        colctFun = colctFunList.get(modelType)
        colctFun(modelInfo, dtDateInfo)

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
def colctObs(modelInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        tmpFileInfo = dtDateInfo.strftime(modelInfo['tmp'])
        updFileInfo = dtDateInfo.strftime(modelInfo['target'])

        # 파일 검사
        fileList = sorted(glob.glob(updFileInfo))
        if len(fileList) > 0: return

        # reqUrl = dtDateInfo.strftime(f"{modelInfo['request']['url']}").format(tmfc=dtDateInfo.strftime('%Y%m%d%H%M'), tmfc2=(dtDateInfo + parseDateOffset(modelInfo['request']['invDate']) - parseDateOffset('1s')).strftime('%Y%m%d%H%M'), authKey=modelInfo['request']['authKey'])
        reqUrl = dtDateInfo.strftime(f"{modelInfo['request']['url']}").format(tmfc=dtDateInfo.strftime('%Y%m%d%H%M'), tmfc2=(dtDateInfo + parseDateOffset(modelInfo['request']['invDate']) - parseDateOffset('1s')).strftime('%Y%m%d%H%M'), authKey=extAuthKey())

        res = requests.get(reqUrl)
        if not (res.status_code == 200): return

        os.makedirs(os.path.dirname(tmpFileInfo), exist_ok=True)
        os.makedirs(os.path.dirname(updFileInfo), exist_ok=True)

        if os.path.exists(tmpFileInfo):
            os.remove(tmpFileInfo)

        # cmd = f"curl -s -C - '{reqUrl}' --retry 10 -o {tmpFileInfo}"
        cmd = modelInfo['cmd'].format(reqUrl=reqUrl, tmpFileInfo=tmpFileInfo)
        log.info(f'[CHECK] cmd : {cmd}')

        try:
            subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
            # subprocess.run(cmd, shell=False, check=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f'[ERROR] 실행 프로그램 실패 : {str(e)}')

        if os.path.exists(tmpFileInfo):
            if os.path.getsize(tmpFileInfo) > 1000:
                shutil.move(tmpFileInfo, updFileInfo)
                log.info(f'[CHECK] CMD : mv -f {tmpFileInfo} {updFileInfo}')
            else:
                os.remove(tmpFileInfo)
                log.info(f'[CHECK] CMD : rm -f {tmpFileInfo}')

        log.info(f'[END] colctKmaApiHub : {dtDateInfo} / pid : {procInfo.pid}')

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e
    finally:
        if os.path.exists(tmpFileInfo):
            os.remove(tmpFileInfo)


@retry(stop_max_attempt_number=10)
def colctNwp(modelInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        efList = modelInfo['request'][f"ef{dtDateInfo.strftime('%H')}"]
        for ef in efList:
            log.info(f'[CHECK] dtDateInfo : {dtDateInfo} / ef : {ef}')

            tmpFileInfo = dtDateInfo.strftime(modelInfo['tmp']).format(ef=ef)
            updFileInfo = dtDateInfo.strftime(modelInfo['target']).format(ef=ef)

            # 파일 검사
            fileList = sorted(glob.glob(updFileInfo))
            # if len(fileList) > 0: return
            if len(fileList) > 0: continue

            # reqUrl = dtDateInfo.strftime(f"{modelInfo['request']['url']}").format(tmfc=dtDateInfo.strftime('%Y%m%d%H'), ef=ef, authKey=modelInfo['request']['authKey'])
            # reqUrl = dtDateInfo.strftime(f"{modelInfo['request']['url']}").format(tmfc=dtDateInfo.strftime('%Y%m%d%H'), ef=ef, authKey='hQDU-t1aQHaA1PrdWvB2eA')
            reqUrl = dtDateInfo.strftime(f"{modelInfo['request']['url']}").format(tmfc=dtDateInfo.strftime('%Y%m%d%H'), ef=ef, authKey=extAuthKey())
            # res = requests.get(reqUrl)
            # if not (res.status_code == 200): return

            os.makedirs(os.path.dirname(tmpFileInfo), exist_ok=True)
            os.makedirs(os.path.dirname(updFileInfo), exist_ok=True)

            if os.path.exists(tmpFileInfo):
                os.remove(tmpFileInfo)

            # cmd = f"curl -s -C - '{reqUrl}' --retry 10 -o {tmpFileInfo}"
            cmd = modelInfo['cmd'].format(reqUrl=reqUrl, tmpFileInfo=tmpFileInfo)
            log.info(f'[CHECK] cmd : {cmd}')

            try:
                subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
                # subprocess.run(cmd, shell=False, check=True)
            except subprocess.CalledProcessError as e:
                raise ValueError(f'[ERROR] 실행 프로그램 실패 : {str(e)}')

            if os.path.exists(tmpFileInfo):
                if os.path.getsize(tmpFileInfo) > 1000:
                    shutil.move(tmpFileInfo, updFileInfo)
                    log.info(f'[CHECK] CMD : mv -f {tmpFileInfo} {updFileInfo}')
                else:
                    os.remove(tmpFileInfo)
                    log.info(f'[CHECK] CMD : rm -f {tmpFileInfo}')

        log.info(f'[END] colctKmaApiHub : {dtDateInfo} / pid : {procInfo.pid}')

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e
    finally:
        if os.path.exists(tmpFileInfo):
            os.remove(tmpFileInfo)

def extAuthKey():

    authKey = None

    try:
        url = "https://apihub.kma.go.kr/apiList.do"
        res = requests.get(url)
        if res.status_code != 200: return authKey

        soup = BeautifulSoup(res.text, 'html.parser')
        lxml = etree.HTML(str(soup))

        tagList = lxml.xpath('/html/body/div/div[1]/div/p[6]/a')
        for tagInfo in tagList:
            urlHref = tagInfo.get('href')
            urlPar = urlparse(urlHref)
            urlQuery = parse_qs(urlPar.query)
            authKey = urlQuery['authKey'][0]

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e

    return authKey

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
                # 예보시간 시작일, 종료일, 시간 간격 (연 1y, 월 1m, 일 1d, 시간 1h, 분 1t, 초 1s)
                'srtDate': '2024-12-01',
                'endDate': '2024-12-04',
                # 'srtDate': globalVar['srtDat
                # e'],
                # 'endDate': globalVar['endDate'],

                # 수행 목록
                # 'modelList': ['AWS', 'ASOS', 'UMKR', 'KIMG'],
                # 'modelList': ['UMKR', 'KIMG'],
                'modelList': ['UMKR'],
                # 'modelList': globalVar['modelList'].split(','),

                # 비동기 다중 프로세스 개수
                'cpuCoreNum': '1',
                # 'cpuCoreNum': globalVar['cpuCoreNum'],
                'ASOS': {
                    'request': {
                        'url': 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php?tm1={tmfc}&tm2={tmfc2}&stn=0&help=0&authKey={authKey}'
                        , 'authKey': None
                        , 'invDate': '1d'
                    }
                    , 'cmd': 'curl -s -C - "{reqUrl}" --retry 10 -o "{tmpFileInfo}"'
                    , 'tmp': '/DATA/OBS/%Y%m/%d/.ASOS_OBS_%Y%m%d%H%M.txt'
                    , 'target': '/DATA/OBS/%Y%m/%d/ASOS_OBS_%Y%m%d%H%M.txt'
                },
                'AWS': {
                    'request': {
                        'url': 'https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?tm1={tmfc}&tm2={tmfc2}&stn=0&disp=0&help=0&authKey={authKey}'
                        , 'authKey': None
                        , 'invDate': '3h'
                    }
                    , 'cmd': 'curl -s -C - "{reqUrl}" --retry 10 -o "{tmpFileInfo}"'
                    , 'tmp': '/DATA/OBS/%Y%m/%d/.AWS_OBS_%Y%m%d%H%M.txt'
                    , 'target': '/DATA/OBS/%Y%m/%d/AWS_OBS_%Y%m%d%H%M.txt'
                },
                'UMKR': {
                    'request': {
                        # 'url': 'https://apihub-org.kma.go.kr/api/typ06/url/nwp_file_down.php?nwp=l015&sub=unis&tmfc={tmfc}&ef={ef}&authKey={authKey}'
                        'url': 'https://apihub.kma.go.kr/api/typ06/url/nwp_file_down.php?nwp=l015&sub=unis&tmfc={tmfc}&ef={ef}&authKey={authKey}'
                        # , 'ef': ['00', '01', '02', '03', '04', '05']
                        # , 'ef00': ['00', '01', '02', '03', '04', '05']
                        , 'ef00': ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47']
                        , 'ef06': ['00', '01', '02', '03', '04', '05']
                        , 'ef12': ['00', '01', '02', '03', '04', '05']
                        , 'ef18': ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47']
                        , 'authKey': None
                        , 'invDate': '6h'
                    }
                    , 'cmd': 'curl -s -C - "{reqUrl}" --retry 10 -o "{tmpFileInfo}"'
                    , 'tmp': '/DATA/MODEL/%Y%m/%d/.UMKR_l015_unis_H{ef}_%Y%m%d%H%M.grb2'
                    , 'target': '/DATA/MODEL/%Y%m/%d/UMKR_l015_unis_H{ef}_%Y%m%d%H%M.grb2'
                },
                'KIMG': {
                    'request': {
                        # 'url': 'https://apihub-org.kma.go.kr/api/typ06/url/nwp_file_down.php?nwp=k128&sub=unis&tmfc={tmfc}&ef={ef}&authKey={authKey}'
                        'url': 'https://apihub-org.kma.go.kr/api/typ06/url/nwp_file_down.php?nwp=k128&sub=unis&tmfc={tmfc}&ef={ef}&authKey={authKey}'
                        # , 'ef': ['00', '03', '06']
                        , 'ef00': ['00', '03', '06']
                        , 'ef06': ['00', '03', '06']
                        , 'ef12': ['00', '03', '06']
                        , 'ef18': ['00', '03', '06']
                        , 'authKey': None
                        , 'invDate': '6h'
                    }
                    , 'cmd': 'curl -s -C - "{reqUrl}" --retry 10 -o "{tmpFileInfo}"'
                    , 'tmp': '/DATA/MODEL/%Y%m/%d/.KIMG_k128_unis_H{ef}_%Y%m%d%H%M.grb2'
                    , 'target': '/DATA/MODEL/%Y%m/%d/KIMG_k128_unis_H{ef}_%Y%m%d%H%M.grb2'
                },

                # 설정 정보
                'cfgFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
            }

            # **********************************************************************************************************
            # 설정 정보
            # **********************************************************************************************************
            config = configparser.ConfigParser()
            config.read(sysOpt['cfgFile'], encoding='utf-8')
            # sysOpt['ASOS']['request']['authKey'] = config.get('apihub-api-key', 'asos')
            # sysOpt['AWS']['request']['authKey'] = config.get('apihub-api-key', 'aws')
            # sysOpt['UMKR']['request']['authKey'] = config.get('apihub-api-key', 'umkr')
            # sysOpt['KIMG']['request']['authKey'] = config.get('apihub-api-key', 'kimg')

            # **************************************************************************************************************
            # 비동기 다중 프로세스 수행
            # **************************************************************************************************************
            # 비동기 다중 프로세스 개수
            pool = Pool(int(sysOpt['cpuCoreNum']))

            for modelType in sysOpt['modelList']:
                modelInfo = sysOpt.get(modelType)
                if modelInfo is None: continue

                # 시작일/종료일 설정
                dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
                dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
                dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=modelInfo['request']['invDate'])

                # for dtDateInfo in dtDateList:
                for dtDateInfo in reversed(dtDateList):
                    # log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
                    pool.apply_async(colctProc, args=(modelType, modelInfo, dtDateInfo))

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
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] main')
