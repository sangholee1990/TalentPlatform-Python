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
import cdsapi
import shutil

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
def subColct(modelInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        for key in ['year', 'month', 'day', 'time']:
            modelInfo['request'][key] = [dtDateInfo.strftime(fmt) for fmt in modelInfo['request'][key]]

        modelInfo['target'] = dtDateInfo.strftime(modelInfo['target'])
        modelInfo['tmp'] = dtDateInfo.strftime(modelInfo['tmp'])

        # 파일 검사
        fileList = sorted(glob.glob(modelInfo['target']))
        if len(fileList) > 0: return

        log.info(f'[START] subColct : {dtDateInfo} / pid : {procInfo.pid}')

        tmpFileInfo = modelInfo['tmp']
        updFileInfo = modelInfo['target']
        os.makedirs(os.path.dirname(tmpFileInfo), exist_ok=True)
        os.makedirs(os.path.dirname(updFileInfo), exist_ok=True)

        if os.path.exists(tmpFileInfo):
            os.remove(tmpFileInfo)

        # c = cdsapi.Client(timeout=9999999, quiet=True, debug=True, url=modelInfo['api']['url'], key=modelInfo['api']['key'])
        # c = cdsapi.Client(timeout=9999999, quiet=False, debug=True, url=modelInfo['api']['url'], key=modelInfo['api']['key'])
        c = cdsapi.Client(timeout=9999999, quiet=False, debug=False, url=modelInfo['api']['url'], key=modelInfo['api']['key'])
        c.retrieve(name=modelInfo['name'], request=modelInfo['request'], target=modelInfo['tmp'])

        if os.path.exists(tmpFileInfo):
            # if not os.path.exists(updFileInfo) or os.path.getsize(tmpFileInfo) > os.path.getsize(updFileInfo):
            # if not os.path.exists(updFileInfo) or os.path.getsize(tmpFileInfo) > 0:
            if os.path.getsize(tmpFileInfo) > 0:
                shutil.move(tmpFileInfo, updFileInfo)
                log.info(f'[CHECK] CMD : mv -f {tmpFileInfo} {updFileInfo}')
            else:
                os.remove(tmpFileInfo)
                log.info(f'[CHECK] CMD : rm -f {tmpFileInfo}')

        log.info(f'[END] subColct : {dtDateInfo} / pid : {procInfo.pid}')

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
    # Python을 이용한 ECMWF 재분석자료 수집

    # , 'key': '307569:99d328b8-16ca-4bbe-a2c6-fb348c5c4219'
    # , 'key': '292516:2df989f2-40aa-454f-9b83-daf3517aa2f9'
    # , 'key': '38372:e61b5517-d919-47b6-93bf-f9a01ee4246f'
    # , 'key': '314000:5f2ea8cc-f1c3-4626-8d3c-4c573c28135d'
    # , 'key': '313996:a9827fcb-bc34-4b1a-816c-8b6ab0915fb2'
    # , 'key': '313999:09d74faf-b856-40fc-8047-46d669fb56eb'

    # python TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList "REANALY-ERA5-25K-UNIS" --cpuCoreNum "1" --srtDate "2024-01-01" --endDate "2024-01-02" --key "313996:a9827fcb-bc34-4b1a-816c-8b6ab0915fb2"
    # python TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList "REANALY-ERA5-25K-PRES" --cpuCoreNum "1" --srtDate "2024-01-01" --endDate "2024-01-02" --key "313996:a9827fcb-bc34-4b1a-816c-8b6ab0915fb2"

    # cd /home/hanul/SYSTEMS/KIER/PROG/PYTHON/colct
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList REANALY-ERA5-25K-UNIS --cpuCoreNum 5 --srtDate 2022-01-01 --endDate 2022-06-01 --key 292516:2df989f2-40aa-454f-9b83-daf3517aa2f9 &
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList REANALY-ERA5-25K-UNIS --cpuCoreNum 5 --srtDate 2022-06-01 --endDate 2023-01-01 --key 38372:e61b5517-d919-47b6-93bf-f9a01ee4246f &
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList REANALY-ERA5-25K-UNIS --cpuCoreNum 5 --srtDate 2023-01-01 --endDate 2023-06-01 --key 307569:99d328b8-16ca-4bbe-a2c6-fb348c5c4219 &
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList REANALY-ERA5-25K-UNIS --cpuCoreNum 5 --srtDate 2023-06-01 --endDate 2024-01-01 --key 314000:5f2ea8cc-f1c3-4626-8d3c-4c573c28135d &
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList REANALY-ERA5-25K-UNIS --cpuCoreNum 5 --srtDate 2023-10-01 --endDate 2024-01-01 --key 313999:09d74faf-b856-40fc-8047-46d669fb56eb &
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList REANALY-ERA5-25K-UNIS --cpuCoreNum 5 --srtDate 2024-01-01 --endDate 2024-06-01 --key 313996:a9827fcb-bc34-4b1a-816c-8b6ab0915fb2 &

    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList REANALY-ERA5-25K-PRES --cpuCoreNum 5 --srtDate 2022-01-01 --endDate 2022-06-01 --key 292516:2df989f2-40aa-454f-9b83-daf3517aa2f9 &
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList REANALY-ERA5-25K-PRES --cpuCoreNum 5 --srtDate 2022-06-01 --endDate 2023-01-01 --key 38372:e61b5517-d919-47b6-93bf-f9a01ee4246f &
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList REANALY-ERA5-25K-PRES --cpuCoreNum 5 --srtDate 2023-01-01 --endDate 2023-06-01 --key 307569:99d328b8-16ca-4bbe-a2c6-fb348c5c4219 &
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList REANALY-ERA5-25K-PRES --cpuCoreNum 5 --srtDate 2023-06-01 --endDate 2024-01-01 --key 314000:5f2ea8cc-f1c3-4626-8d3c-4c573c28135d &
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList REANALY-ERA5-25K-PRES --cpuCoreNum 5 --srtDate 2023-10-01 --endDate 2024-01-01 --key 313999:09d74faf-b856-40fc-8047-46d669fb56eb &
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-reanalyEra5.py --modelList REANALY-ERA5-25K-PRES --cpuCoreNum 5 --srtDate 2024-01-01 --endDate 2024-06-01 --key 313996:a9827fcb-bc34-4b1a-816c-8b6ab0915fb2 &


    # ps -ef | grep "TalentPlatform-INDI2024-colct-reanalyEra5.py" | awk '{print $2}' | xargs kill -9
    # ps -ef | grep "TalentPlatform-INDI2024-colct-reanalyEra5.py" | grep "REANALY-ERA5-25K-UNIS" | grep "2023-01-01" | awk '{print $2}' | xargs kill -9
    # ps -ef | grep "TalentPlatform-INDI2024-colct-reanalyEra5.py" | grep "REANALY-ERA5-25K-UNIS" | awk '{print $2}' | xargs kill -9
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
                # 수행 목록
                # 'modelList': ['REANALY-ERA5-25K-UNIS']
                # 'modelList': ['REANALY-ERA5-25K-PRES']
                'modelList': [globalVar['modelList']]

                # 비동기 다중 프로세스 개수
                # , 'cpuCoreNum': '1'
                , 'cpuCoreNum': globalVar['cpuCoreNum']

                , 'REANALY-ERA5-25K-UNIS': {

                    # 시작일, 종료일, 시간 간격 (연 1y, 월 1h, 일 1d, 시간 1h)
                    # 'srtDate': '2024-01-01'
                    # , 'endDate': '2024-01-02'
                    'srtDate': globalVar['srtDate']
                    , 'endDate': globalVar['endDate']
                    # , 'invDate': '1d'
                    , 'invDate': '1h'
                    , 'api': {
                        'url': 'https://cds.climate.copernicus.eu/api/v2'
                        # , 'key': ''
                        , 'key': globalVar['key']
                    }

                    , 'name': 'reanalysis-era5-single-levels'
                    , 'request': {
                        'product_type': 'reanalysis',
                        'format': 'grib',
                        'variable': [
                            '10m_u_component_of_wind','10m_v_component_of_wind','2m_dewpoint_temperature','2m_temperature','land_sea_mask','mean_sea_level_pressure','sea_ice_cover','sea_surface_temperature','skin_temperature','snow_depth','soil_temperature_level_1','soil_temperature_level_2','soil_temperature_level_3','soil_temperature_level_4','surface_pressure','volumetric_soil_water_layer_1','volumetric_soil_water_layer_2','volumetric_soil_water_layer_3','volumetric_soil_water_layer_4'
                        ],
                        'year': [
                            '%Y'
                        ],
                        'month': [
                            '%m'
                        ],
                        'day': [
                            '%d'
                        ],
                        'time': [
                            # '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
                             '%H:%M'
                        ],
                        'area': [
                            90, -180, -90, 180,
                            # 30, 120, 31, 121
                        ],
                    }
                    # , 'tmp': '/DATA/INPUT/INDI2024/DATA/REANALY-ERA5/%Y/%m/%d/.reanaly-era5-unis_%Y%m%d.grib'
                    # , 'target': '/DATA/INPUT/INDI2024/DATA/REANALY-ERA5/%Y/%m/%d/reanaly-era5-unis_%Y%m%d.grib'
                    # , 'tmp': '/data1/REANALY-ERA5/%Y/%m/%d/.reanaly-era5-unis_%Y%m%d.grib'
                    # , 'target': '/data1/REANALY-ERA5/%Y/%m/%d/reanaly-era5-unis_%Y%m%d.grib'
                    # , 'tmp': '/DATA/INPUT/INDI2024/DATA/REANALY-ERA5/%Y/%m/%d/.reanaly-era5-unis_%Y%m%d%H%M.grib'
                    # , 'target': '/DATA/INPUT/INDI2024/DATA/REANALY-ERA5/%Y/%m/%d/reanaly-era5-unis_%Y%m%d%H%M.grib'
                    , 'tmp': '/data1/REANALY-ERA5/%Y/%m/%d/.reanaly-era5-unis_%Y%m%d%H%M.grib'
                    , 'target': '/data1/REANALY-ERA5/%Y/%m/%d/reanaly-era5-unis_%Y%m%d%H%M.grib'
                }

                , 'REANALY-ERA5-25K-PRES': {
                    # 'srtDate': '2024-01-01'
                    # , 'endDate': '2024-01-02'
                    'srtDate': globalVar['srtDate']
                    , 'endDate': globalVar['endDate']
                    , 'invDate': '1h'
                    , 'api': {
                        'url': 'https://cds.climate.copernicus.eu/api/v2'
                        # , 'key': ''
                        , 'key': globalVar['key']
                    }

                    , 'name': 'reanalysis-era5-pressure-levels'
                    , 'request': {
                        'product_type': 'reanalysis',
                        'format': 'grib',
                        'variable': [
                            'geopotential','relative_humidity','specific_humidity','temperature','u_component_of_wind','v_component_of_wind','vertical_velocity'
                        ],
                        'pressure_level': [
                            '1','2','3','5','7','10','20','30','50','70','100','125','150','175','200','225','250','300','350','400','450','500','550','600','650','700','750','775','800','825','850','875','900','925','950','975','1000'
                        ],
                        'year': [
                            '%Y'
                        ],
                        'month': [
                            '%m'
                        ],
                        'day': [
                            '%d'
                        ],
                        'time': [
                            '%H:%M'
                            # '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
                        ],
                        'area': [
                            90, -180, -90, 180
                            # 30, 120, 31, 121
                        ],
                    }
                    # , 'tmp': '/DATA/INPUT/INDI2024/DATA/REANALY-ERA5/%Y/%m/%d/.reanaly-era5-pres_%Y%m%d%H%M.grib'
                    # , 'target': '/DATA/INPUT/INDI2024/DATA/REANALY-ERA5/%Y/%m/%d/reanaly-era5-pres_%Y%m%d%H%M.grib'
                    , 'tmp': '/data1/REANALY-ERA5/%Y/%m/%d/.reanaly-era5-pres_%Y%m%d%H%M.grib'
                    , 'target': '/data1/REANALY-ERA5/%Y/%m/%d/reanaly-era5-pres_%Y%m%d%H%M.grib'
                }
            }

            # **************************************************************************************************************
            # 비동기 다중 프로세스 수행
            # **************************************************************************************************************
            # 비동기 다중 프로세스 개수
            pool = Pool(int(sysOpt['cpuCoreNum']))

            for modelType in sysOpt['modelList']:
                log.info(f'[CHECK] modelType : {modelType}')

                modelInfo = sysOpt.get(modelType)
                if modelInfo is None: continue

                # 시작일/종료일 설정
                dtSrtDate = pd.to_datetime(modelInfo['srtDate'], format='%Y-%m-%d')
                dtEndDate = pd.to_datetime(modelInfo['endDate'], format='%Y-%m-%d')
                dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=modelInfo['invDate'])
                
                for dtDateInfo in dtDateList:
                    log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
                    pool.apply_async(subColct, args=(modelInfo, dtDateInfo))

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
