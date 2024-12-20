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

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 ECMWF ERA5 기후 데이터를 이용하여 일간-월간-연간 통계 생산

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
    serviceName = 'LSH0515'

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
                'srtDate': '2020-01-01'
                , 'endDate': '2021-02-01'
                , 'invDateList': ['1y', '1m', '1d']

                # 경도 최소/최대/간격
                , 'lonMin': -180
                , 'lonMax': 180
                , 'lonInv': 0.5

                # 위도 최소/최대/간격
                , 'latMin': -90
                , 'latMax': 90
                , 'latInv': 0.5

                # 수행 목록
                , 'modelList': ['ECMWF-SFC']

                # 모델 정보
                , 'ECMWF-SFC': {
                    # 원본 파일 정보
                    'filePath': '/DATA/INPUT/LSH0515/data07/climate_change/prc/ecmwf/era5/%Y/%m/%d/{}'
                    , 'fileName': '{}_%Y%m%d_%H.txt'
                    , 'varList': ['2t']
                    , 'comVar': {'Longitude': 'lon', 'Latitude': 'lat', 'Value': '{}'}

                    # 가공 파일 덮어쓰기 여부
                    , 'isOverWrite': True
                    # , 'isOverWrite': False

                    # 가공 파일 정보
                    , 'procPath': '/DATA/OUTPUT/LSH0515/PROC/%Y%m/%d/%H'
                    , 'procName': '{}_1h_{}_out_%Y%m%d%H%M.nc'

                    # 검색 파일 정보
                    , 'searchPath': {
                        '1y': '/DATA/OUTPUT/LSH0515/PROC/%Y*/*/*'
                        , '1m': '/DATA/OUTPUT/LSH0515/PROC/%Y%m/*/*'
                        , '1d': '/DATA/OUTPUT/LSH0515/PROC/%Y%m/%d/*'
                    }
                    , 'searchName': {
                        '1y': '{}_1h_{}_out_%Y*.nc'
                        , '1m': '{}_1h_{}_out_%Y%m*.nc'
                        , '1d': '{}_1h_{}_out_%Y%m%d*.nc'
                    }

                    # 저장 파일 정보
                    , 'savePath': {
                        '1y': '/DATA/OUTPUT/LSH0515/%Y'
                        , '1m': '/DATA/OUTPUT/LSH0515/%Y%m'
                        , '1d': '/DATA/OUTPUT/LSH0515/%Y%m/%d'
                    }
                    , 'saveName': {
                        '1y': '{}_{}_{}_{}_out_%Y.csv'
                        , '1m': '{}_{}_{}_{}_out_%Y%m.csv'
                        , '1d': '{}_{}_{}_{}_out_%Y%m%d.csv'
                    }
                }
            }

            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1h')

            # 기준 위도/경도 설정
            lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
            latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])

            log.info(f'[CHECK] len(lonList) : {len(lonList)}')
            log.info(f'[CHECK] len(latList) : {len(latList)}')

            # ===================================================================================
            # 가공 파일 생산
            # ===================================================================================
            for dtDateIdx, dtDateInfo in enumerate(dtDateList):
                # log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                # dataL1 = xr.Dataset()
                for modelIdx, modelType in enumerate(sysOpt['modelList']):
                    # log.info(f'[CHECK] modelType : {modelType}')

                    modelInfo = sysOpt.get(modelType)
                    if modelInfo is None: continue

                    for varIdx, varInfo in enumerate(modelInfo['varList']):
                        # log.info(f'[CHECK] varInfo : {varInfo}')

                        procFilePattern = '{}/{}'.format(modelInfo['procPath'], modelInfo['procName'])
                        procFile = dtDateInfo.strftime(procFilePattern).format(modelType.lower(), varInfo)

                        # 파일 덮어쓰기 및 파일 존재 여부
                        if not modelInfo['isOverWrite'] and os.path.exists(procFile): continue

                        inpFilePattern = '{}/{}'.format(modelInfo['filePath'], modelInfo['fileName'])
                        inpFile = dtDateInfo.strftime(inpFilePattern).format(varInfo, varInfo)
                        fileList = sorted(glob.glob(inpFile))

                        if fileList is None or len(fileList) < 1:
                            # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                            continue

                        # 파일 읽기
                        for j, fileInfo in enumerate(fileList):
                            data = pd.read_csv(fileInfo, sep='\s+')
                            log.info(f'[CHECK] fileInfo : {fileInfo}')

                            modelInfo['comVar']['Value'] = varInfo
                            dataL1 = data.rename(columns = modelInfo['comVar'])[modelInfo['comVar'].values()]
                            dataL1['time'] = dtDateInfo

                            if (len(dataL1) < 1): continue

                            # CSV to NetCDF 변환
                            dataL2 = dataL1.set_index(['time', 'lat', 'lon'])
                            dataL3 = dataL2.to_xarray()

                            # 특정 변수 선택 및  위경도 내삽
                            dataL4 = dataL3[varInfo].interp({'lon': lonList, 'lat': latList}, method='linear')

                            # 0 초과 필터, 그 외 결측값 NA
                            dataL5 = dataL4.where((dataL4 > 0))

                            # dataL5.isel(time = 0).plot()
                            # plt.show()

                            # NetCDF 저장
                            os.makedirs(os.path.dirname(procFile), exist_ok=True)
                            dataL5.to_netcdf(procFile)
                            log.info(f'[CHECK] procFile : {procFile}')

            # ===================================================================================
            # 통계 파일 생산
            # ===================================================================================
            for invIdx, invDate in enumerate(sysOpt['invDateList']):
                log.info(f'[CHECK] invDate : {invDate}')

                dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=invDate)
                for dtDateIdx, dtDateInfo in enumerate(dtDateList):
                    # log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                    # dataL1 = xr.Dataset()
                    for modelIdx, modelType in enumerate(sysOpt['modelList']):
                        # log.info(f'[CHECK] modelType : {modelType}')

                        modelInfo = sysOpt.get(modelType)
                        if modelInfo is None: continue

                        for varIdx, varInfo in enumerate(modelInfo['varList']):
                            # log.info(f'[CHECK] varInfo : {varInfo}')

                            searchFilePattern = '{}/{}'.format(modelInfo['searchPath'][invDate], modelInfo['searchName'][invDate])
                            searchFile = dtDateInfo.strftime(searchFilePattern).format(modelType.lower(), varInfo)
                            fileList = sorted(glob.glob(searchFile))

                            if fileList is None or len(fileList) < 1:
                                # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                                continue

                            data = xr.open_mfdataset(fileList)

                            # 통계 종류 (연, 월, 일)
                            if re.search('y', invDate, re.IGNORECASE):
                                statType = 'time.year'
                            elif re.search('m', invDate, re.IGNORECASE):
                                statType = 'time.month'
                            elif re.search('d', invDate, re.IGNORECASE):
                                statType = 'time.day'
                            else:
                                continue

                            # 주요 변수에 따라 통계 (평균 mean, 합계 sum) 계산
                            if re.search('2t|2t', varInfo, re.IGNORECASE):
                                statData = data.groupby(statType).mean(skipna=True)
                            elif re.search('cp|cp', varInfo, re.IGNORECASE):
                                statData = data.groupby(statType).sum(skipna=True)
                            else:
                                continue

                            # cntData = data.groupby('time.day').count().isel(day = 0).rename({'2t': 'cnt'})

                            # 불필요한 변수/차원 삭제
                            delList = ['year', 'month', 'day']
                            for i, delInfo in enumerate(delList):
                                try:
                                    statData = statData.drop_vars(delInfo)
                                    statData = statData[varInfo].isel({delInfo : 0})
                                    break
                                except Exception as e:
                                    pass

                            # dataL1 = xr.merge([statData, cntData])
                            dataL1 = statData
                            dataL2 = dataL1.to_dataframe().reset_index(drop=False)

                            # CSV 생성
                            saveFilePattern = '{}/{}'.format(modelInfo['savePath'][invDate], modelInfo['saveName'][invDate])
                            saveFile = dtDateInfo.strftime(saveFilePattern).format(modelType.lower(), invDate, varInfo, len(fileList))
                            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                            dataL2.to_csv(saveFile, index=False)
                            log.info(f'[CHECK] saveFile : {saveFile}')

                            # NetCDF 생성
                            saveNcFile = saveFile.replace('.csv', '.nc')
                            os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
                            statData.to_netcdf(saveNcFile)
                            log.info(f'[CHECK] saveNcFile : {saveNcFile}')

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
