# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from pandas.tseries.offsets import Day, Hour, Minute, Second
import re
import pygrib

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
    # Python을 이용한 ECMWF 및 GFS 예보모델 자료 처리

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
    serviceName = 'LSH0462'

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

            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            # 옵션 설정
            sysOpt = {
                # 시작일, 종료일, 시간 간격
                'srtDate': '2022-06-01'
                , 'endDate': '2022-06-02'
                # 'srtDate': globalVar['srtDate']
                # , 'endDate': globalVar['endDate']
                , 'invHour': 1

                # 경도 최소/최대/간격
                , 'lonMin': 120
                , 'lonMax': 145
                , 'lonInv': 0.25

                # 위도 최소/최대/간격
                , 'latMin': 20
                , 'latMax': 44.75
                , 'latInv': 0.25

                # 기압 설정
                , 'levList': [500, 700, 850, 1000]

                # 수행 목록
                , 'modelList': ['GFS', 'ECMWF']

                # 모델 정보 : 파일 경로, 파일명, 데이터/DB 컬럼 (지표면 wrfsolar 동적 설정, 상층면 wrfout 정적 설정), 시간 간격
                , 'GFS': {
                    'SFC': {
                        'filePath': '/DATA/INPUT/LSH0462'
                        , 'fileName': 'gfs.0p25.%Y%m%d%H.f*.gr_crop.grib2'
                        , 'comVar': {'lon': 'lon_0', 'lat': 'lat_0'}
                        , 'level': [-1]
                        , 'orgVar': ['TMP_P0_L1_GLL0']
                        , 'newVar': ['T2']
                    }
                    , 'PRE': {
                        'filePath': '/DATA/INPUT/LSH0462'
                        , 'fileName': 'gfs.0p25.%Y%m%d%H.f*.gr_crop.grib2'
                        , 'comVar': {'lon': 'lon_0', 'lat': 'lat_0', 'lev': 'lv_ISBL0'}
                        , 'level': [500, 700, 850, 1000]
                        , 'orgVar': ['TMP_P0_L100_GLL0', 'TMP_P0_L100_GLL0', 'TMP_P0_L100_GLL0', 'TMP_P0_L100_GLL0']
                        , 'newVar': ['T500', 'T700', 'T850', 'T1000']
                    }
                }
                , 'ECMWF': {
                    'SFC': {
                        'filePath': '/DATA/INPUT/LSH0462'
                        , 'fileName': 'reanalysis-era5-single-levels_%Y%m%d_%H_asia.grib'
                        , 'comVar': {'lon': 'g0_lon_1', 'lat': 'g0_lat_0'}
                        , 'level': [-1]
                        , 'orgVar': ['2T_GDS0_SFC']
                        , 'newVar': ['T2']
                    }
                    , 'PRE': {
                        'filePath': '/DATA/INPUT/LSH0462'
                        , 'fileName': 'reanalysis-era5-pressure-levels_%Y%m%d_%H_asia.grib'
                        , 'comVar': {'lon': 'g0_lon_2', 'lat': 'g0_lat_1', 'lev': 'lv_ISBL0'}
                        , 'level': [500, 700, 850, 1000]
                        , 'orgVar': ['T_GDS0_ISBL', 'T_GDS0_ISBL', 'T_GDS0_ISBL', 'T_GDS0_ISBL']
                        , 'newVar': ['T500', 'T700', 'T850', 'T1000']
                    }
                }
            }

            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(sysOpt['invHour']))

            # 기준 위도, 경도, 기압 설정
            lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
            latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])
            levList = np.array(sysOpt['levList'])

            log.info(f'[CHECK] len(lonList) : {len(lonList)}')
            log.info(f'[CHECK] len(latList) : {len(latList)}')
            log.info(f'[CHECK] len(levList) : {len(levList)}')

            for dtDateIdx, dtDateInfo in enumerate(dtDateList):
                log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                dataL1 = xr.Dataset()
                for modelIdx, modelType in enumerate(sysOpt['modelList']):
                    log.info(f'[CHECK] modelType : {modelType}')

                    for i, modelKey in enumerate(sysOpt[modelType]):
                        log.info(f'[CHECK] modelKey : {modelKey}')

                        modelInfo = sysOpt[modelType].get(modelKey)
                        if modelInfo is None: continue

                        inpFile = '{}/{}'.format(modelInfo['filePath'], modelInfo['fileName'])
                        inpFileDate = dtDateInfo.strftime(inpFile)
                        fileList = sorted(glob.glob(inpFileDate))

                        if fileList is None or len(fileList) < 1:
                            # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                            continue

                        # NetCDF 파일 읽기
                        for j, fileInfo in enumerate(fileList):

                            data = xr.open_dataset(fileInfo, engine='pynio')
                            log.info(f'[CHECK] fileInfo : {fileInfo}')

                            # pygrib에서 분석/예보 시간 추출
                            gribData = pygrib.open(fileInfo).select()[0]
                            anaDt = gribData.analDate
                            fotDt = gribData.validDate

                            log.info(f'[CHECK] anaDt : {anaDt} / fotDt : {fotDt}')

                            # 파일명에서 분석/예보 시간 추출
                            # isMatch = re.search(r'f(\d+)', fileInfo)
                            # if not isMatch: continue
                            # int(isMatch.group(1))

                            # anaDt = dtDateInfo
                            # fotDt = anaDt + pd.Timedelta(hours = int(isMatch.group(1)))

                            for level, orgVar, newVar in zip(modelInfo['level'], modelInfo['orgVar'], modelInfo['newVar']):
                                if data.get(orgVar) is None: continue

                                try:
                                    if level == -1:
                                        selData = data[orgVar].interp({modelInfo['comVar']['lon']: lonList, modelInfo['comVar']['lat']: latList}, method='linear')
                                        selDataL1 = selData
                                    else:
                                        selData = data[orgVar].interp({modelInfo['comVar']['lon']: lonList, modelInfo['comVar']['lat']: latList, modelInfo['comVar']['lev']: levList}, method='linear')
                                        selDataL1 = selData.sel({modelInfo['comVar']['lev']: level})

                                    selDataL2 = xr.Dataset(
                                        {
                                            f'{modelType}_{newVar}': (('anaDt', 'fotDt', 'lat', 'lon'), (selDataL1.values).reshape(1, 1, len(latList), len(lonList)))
                                        }
                                        , coords={
                                            'anaDt': pd.date_range(anaDt, periods=1)
                                            , 'fotDt': pd.date_range(fotDt, periods=1)
                                            , 'lat': latList
                                            , 'lon': lonList
                                        }
                                    )

                                    dataL1 = xr.merge([dataL1, selDataL2])
                                except Exception as e:
                                    log.error(f'Exception : {e}')

                if len(dataL1) < 1: continue

                # NetCDF 자료 저장
                saveFile = '{}/{}/{}_{}.nc'.format(globalVar['outPath'], serviceName, 'ecmwf-gfs_model', dtDateInfo.strftime('%Y%m%d%H%M'))
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL1.to_netcdf(saveFile)
                log.info(f'[CHECK] saveFile : {saveFile}')

                # 비교
                dataL1['DIFF_T2'] = dataL1['ECMWF_T2'] - dataL1['GFS_T2']

                # 시각화
                saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, 'ecmwf_t2', dtDateInfo.strftime('%Y%m%d%H%M'))
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                dataL1['ECMWF_T2'].isel(anaDt=0, fotDt=0).plot()
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                plt.show()
                log.info(f'[CHECK] saveImg : {saveImg}')

                saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, 'gfs_t2', dtDateInfo.strftime('%Y%m%d%H%M'))
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                dataL1['GFS_T2'].isel(anaDt=0, fotDt=0).plot()
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                plt.show()
                log.info(f'[CHECK] saveImg : {saveImg}')

                saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, 'diff_t2', dtDateInfo.strftime('%Y%m%d%H%M'))
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                dataL1['DIFF_T2'].isel(anaDt=0, fotDt=0).plot()
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                plt.show()
                log.info(f'[CHECK] saveImg : {saveImg}')

        except Exception as e:
            log.error(f'Exception : {e}')

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
