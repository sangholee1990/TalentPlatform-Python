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
import pandas as pd
import xarray as xr
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import xarray as xr
import seaborn as sns
from pandas.tseries.offsets import Day, Hour, Minute, Second
import re
import tempfile
import shutil
import pymannkendall as mk
from dask.distributed import Client
import dask

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

def calcMannKendall(data, colName):
    try:
        # trend 추세, p 유의수준, Tau 상관계수, z 표준 검정통계량, s 불일치 개수, slope 기울기
        result = mk.original_test(data)
        return getattr(result, colName)
    except Exception:
        return np.nan


def calcMaxContDay(isMask):
    arr = isMask.astype(int)
    sumCum = np.where(arr, arr.cumsum(axis=0), 0)

    diff = np.diff(sumCum, axis=0, prepend=0)
    sumCumData = diff * arr

    result = sumCumData.max(axis=0) - sumCumData.min(axis=0)

    return result

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 시간별 재분석 ERA5 모델 (NetCDF)로부터 월간 온도 및 강수량 분석 그리고 MK 검정 (Mann-Kendall)

    # 1) 월간 데이터에서 격자별
    # TXx: Montly maximum value of daily maximum temperature
    # R10: Number of heavy precipitation days(precipitation >10mm)
    # CDD: The largests No. of consecutive days with, 1mm of precipitation
    # 값을 추출하고 새로운 3장(각각 1장)의 netCDF 파일로 생성

    # (후에 다른 연도 파일도 같은 방식으로 처리하고 합쳐서 trend를 파악하려고 함)
    # 2) 각 격자별 trend를 계산해서 지도로 시각화/ Mann Kendall 검정
    # (2개월 데이터로만 처리해주셔도 됩니다. 첨부사진처럼 시각화하려고 합니다. )

    # 제가 시도해봤던 코드도 보내드립니다. 하나는 대략적인 평균 강수량 및 평균 온도를 시각화한 코드이고,
    # 다른 하나는 새로운 netCDF파일 생성하려고 한 코드입니다. time (1,)에 위도 경도 는 바뀌지 않아야 하는데 다 동일하게 변경이 되네요ㅠㅠ

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0544'

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

            # 옵션 설정
            sysOpt = {
                # 시작일, 종료일, 시간 간격 (연 1y, 월 1h, 일 1d, 시간 1h)
                'srtDate': '2023-11-01'
                , 'endDate': '2024-01-01'
                , 'invDate': '1m'

                # 수행 목록
                , 'modelList': ['REANALY-ECMWF-1D']

                # 일 평균
                , 'REANALY-ECMWF-1D': {
                    # 원본 파일 정보
                    'filePath': '/DATA/INPUT/LSH0544/%Y%m'
                    , 'fileName': 'data.nc'
                    , 'varList': ['t2m', 'tp', 'tp']

                    # 가공 파일 덮어쓰기 여부
                    , 'isOverWrite': True
                    # , 'isOverWrite': False

                    # 가공 변수
                    , 'procList': ['TXX', 'R10', 'CDD']

                    # 가공 파일 정보
                    , 'procPath': '/DATA/OUTPUT/LSH0544'
                    , 'procName': '{}_{}-{}_{}-{}.nc'
                }
            }

            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

            # 멀티코어 설정
            # client = Client(n_workers=os.cpu_count(), threads_per_worker=os.cpu_count())
            # dask.config.set(scheduler='processes')

            # ===================================================================================
            # 가공 파일 생산
            # ===================================================================================
            for modelType in sysOpt['modelList']:
                log.info(f'[CHECK] modelType : {modelType}')

                modelInfo = sysOpt.get(modelType)
                if modelInfo is None: continue

                # 시작일/종료일에 따른 데이터 병합
                # mrgData = xr.Dataset()
                for dtDateInfo in dtDateList:
                    log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                    inpFilePattern = '{}/{}'.format(modelInfo['filePath'], modelInfo['fileName'])
                    inpFile = dtDateInfo.strftime(inpFilePattern)
                    fileList = sorted(glob.glob(inpFile))

                    if fileList is None or len(fileList) < 1:
                        # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                        continue

                    fileInfo = fileList[0]
                    data = xr.open_dataset(fileInfo)
                    log.info(f'[CHECK] fileInfo : {fileInfo}')

                    dataL1 = data

                    # 변수 삭제
                    selList = ['expver']
                    for selInfo in selList:
                        try:
                            dataL1 = dataL1.isel(expver=1).drop_vars([selInfo])
                        except Exception as e:
                            pass

                    # mrgData = xr.merge([mrgData, dataL1])
                    mrgData = dataL1

                    # ******************************************************************************************************
                    # 1) 월간 데이터에서 격자별 값을 추출하고 새로운 3장(각각 1장)의 netCDF 파일로 생성
                    # ******************************************************************************************************
                    for varIdx, varInfo in enumerate(modelInfo['varList']):
                        procInfo = modelInfo['procList'][varIdx]
                        log.info(f'[CHECK] varInfo : {varInfo} / procInfo : {procInfo}')

                        # TXX: Montly maximum value of daily maximum temperature
                        if re.search('TXX', procInfo, re.IGNORECASE):
                            # 0 초과 필터, 그 외 결측값 NA
                            varData = mrgData[varInfo]

                            varDataL1 = varData.where(varData > 0).resample(time='1D').max(skipna=False)
                            varDataL2 = varDataL1.resample(time='1M').max(skipna=False)

                        # R10: Number of heavy precipitation days(precipitation > 10mm)
                        elif re.search('R10', procInfo, re.IGNORECASE):
                            # 단위 변환 (m/hour -> mm/day)
                            varData = mrgData[varInfo] * 24 * 1000

                            varDataL1 = varData.resample(time='1D').sum(skipna=False)
                            varDataL2 = varDataL1.where(varDataL1 > 10.0, drop=False).resample(time='1M').count()

                        # CDD: The largests No. of consecutive days with, 1mm of precipitation
                        elif re.search('CDD', procInfo, re.IGNORECASE):

                            # 단위 변환 (m/hour -> mm/day)
                            varData = mrgData[varInfo] * 24 * 1000

                            varDataL1 = varData.resample(time='1D').sum(skipna=False)

                            # True: 1 mm 미만 강수량 / False: 그 외
                            # varDataL1 = varDataL1 >= 1.0
                            varDataL2 = (varDataL1 < 1.0).resample(time='1M').apply(calcMaxContDay)
                        else:
                            continue

                        # 마스킹 데이터
                        maskData = varData.isel(time=0)
                        maskDataL1 = xr.where(np.isnan(maskData), np.nan, 1)

                        varDataL2 = varDataL2 * maskDataL1
                        varDataL2.name = procInfo

                        # varDataL3.isel(time = 0).plot()
                        # plt.show()

                        timeList = varDataL2['time'].values
                        minDate = pd.to_datetime(timeList).min().strftime("%Y%m%d")
                        maxDate = pd.to_datetime(timeList).max().strftime("%Y%m%d")

                        procFilePattern = '{}/{}'.format(modelInfo['procPath'], modelInfo['procName'])
                        procFile = procFilePattern.format(modelType, procInfo, 'ORG', minDate, maxDate)
                        os.makedirs(os.path.dirname(procFile), exist_ok=True)
                        varDataL2.to_netcdf(procFile)
                        log.info(f'[CHECK] procFile : {procFile}')

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
