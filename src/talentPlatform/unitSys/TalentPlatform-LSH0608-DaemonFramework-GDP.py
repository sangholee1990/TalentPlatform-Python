# ================================================
# 요구사항
# ================================================
# Python 이용한 NetCDF 파일 처리 및 3종 증발산량 (Penman, Hargreaves, Thornthwaite) 계산

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
from pyproj import Proj
import rioxarray as rio
import cftime
import subprocess
# from global_land_mask import globe
from pyproj import Proj
import rioxarray as rio
import cftime
import subprocess
import gc


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

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0608'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info("[START] __init__ : {}".format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

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
                pass
                # globalVar['inpPath'] = '/DATA/INPUT'
                # globalVar['outPath'] = '/DATA/OUTPUT'
                # globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': '1990-01-01'
                , 'endDate': '2022-01-01'

                # 경도 최소/최대/간격
                , 'lonMin': -180
                , 'lonMax': 180
                , 'lonInv': 0.1

                # 위도 최소/최대/간격
                , 'latMin': -90
                , 'latMax': 90
                , 'latInv': 0.1
            }

            # 도법 설정
            proj4326 = 'epsg:4326'
            mapProj4326 = Proj(proj4326)

            lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
            latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])

            log.info('[CHECK] len(lonList) : {}'.format(len(lonList)))
            log.info('[CHECK] len(latList) : {}'.format(len(latList)))

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1Y')
            # dtIncDateInfo = dtIncDateList[0]

            inpFilePattern = '{}'.format('GDP/rast_adm2_gdp_perCapita_1990_2022.tif')
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, inpFilePattern)
            fileList = sorted(glob.glob(inpFile))

            if len(fileList) < 1: 
                raise Exception(f"파일 없음 : {inpFile}")
            fileInfo = fileList[0]
            
            # 세부 adm2, 1990~2022 연도
            data = xr.open_rasterio(fileInfo)
            
            descList = data.attrs['descriptions']
            dataL5 = xr.Dataset()
            for idx, desc in enumerate(descList):
                log.info(f"[CHECK] idx : {idx} / desc : {desc}")

                dtDateInfo = pd.to_datetime(desc, format='gdp_pc_%Y')
                sYear = dtDateInfo.strftime('%Y')

                dataL1 = data.isel(band=idx)

                dataL2 = dataL1.rio.reproject(proj4326)
                dataL3 = dataL2.interp(x=lonList, y=latList, method='nearest')

                # 결측값 처리
                dataL3 = xr.where((dataL3 < 0), np.nan, dataL3)

                lon1D = dataL3['x'].values
                lat1D = dataL3['y'].values

                dataL4 = xr.Dataset(
                    {
                        'GDP': (('time', 'lat', 'lon'), (dataL3.values).reshape(1, len(lat1D), len(lon1D)))
                    }
                    , coords={
                        'time': pd.date_range(sYear, periods=1)
                        , 'lat': lat1D
                        , 'lon': lon1D
                    }
                )

                if len(dataL5) < 1:
                    dataL5 = dataL4
                else:
                    dataL5 = xr.concat([dataL5, dataL4], "time")

            timeList = dataL5['time'].values
            minYear = pd.to_datetime(timeList.min()).strftime('%Y')
            maxYear = pd.to_datetime(timeList.max()).strftime('%Y')

            saveFile = '{}/{}/{}_{}-{}.nc'.format(globalVar['outPath'], serviceName, 'GDP', minYear, maxYear)
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            dataL5.to_netcdf(saveFile)
            log.info(f'[CHECK] saveFile : {saveFile}')

            # dataL5 = xr.Dataset()
            # for j, dtIncDateInfo in enumerate(dtIncDateList):
            #     log.info(f"[CHECK] dtIncDateInfo : {dtIncDateInfo}")
            #     sYear = dtIncDateInfo.strftime('%Y')
            # 
            #     saveFile = '{}/{}/{}-{}.nc'.format(globalVar['outPath'], serviceName, 'GDP', sYear)
            #     fileChkList = glob.glob(saveFile)
            #     if (len(fileChkList) > 0): continue
            # 
            #     # inpFilePattern = '{}/CarbonMonitor_*{}*_y{}_m{}.nc'.format(serviceName, keyInfo, dtYear, dtMonth)
            #     # inpFilePattern = '{1:s}/{0:s}/{0:s}{1:s}.tif'.format(sYear, 'GDP')
            #     inpFilePattern = '{1:s}/{0:s}{1:s}.tif'.format(sYear, 'GDP')
            #     inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, inpFilePattern)
            #     fileList = sorted(glob.glob(inpFile))
            # 
            #     if (len(fileList) < 1): continue
            # 
            #     fileInfo = fileList[0]
            # 
            #     # 파일 읽기
            #     # data = xr.open_rasterio(fileInfo)
            #     data = xr.open_rasterio(fileInfo, chunks={"band": 1, "x": 500, "y": 500})
            # 
            #     # proj4326 도법 변환
            #     dataL1 = data.rio.reproject(proj4326)
            #     dataL2 = dataL1.sel(band = 1)
            # 
            #     dataL3 = dataL2.interp(x=lonList, y=latList, method='nearest')
            # 
            #     # 결측값 처리
            #     dataL3 = xr.where((dataL3 < 0), np.nan, dataL3)
            # 
            #     lon1D = dataL3['x'].values
            #     lat1D = dataL3['y'].values
            # 
            #     dataL4 = xr.Dataset(
            #         {
            #             'GDP': (('time', 'lat', 'lon'), (dataL3.values).reshape(1, len(lat1D), len(lon1D)))
            #         }
            #         , coords={
            #             'time': pd.date_range(sYear, periods=1)
            #             , 'lat': lat1D
            #             , 'lon': lon1D
            #         }
            #     )
            # 
            #     # if (len(dataL5) < 1):
            #     #     dataL5 = dataL4
            #     # else:
            #     #     dataL5 = xr.concat([dataL5, dataL4], "time")
            # 
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     dataL4.to_netcdf(saveFile)
            #     log.info(f'[CHECK] saveFile : {saveFile}')
            # 
            #     # 데이터셋 닫기 및 메모리에서 제거
            #     data.close()
            #     del data
            #     dataL1.close()
            #     del dataL1
            #     dataL2.close()
            #     del dataL2
            #     dataL3.close()
            #     del dataL3
            #     dataL4.close()
            #     del dataL4
            # 
            #     # 가비지 수집기 강제 실행
            #     gc.collect()

            # inpFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'GDP-????.nc')
            # fileList = sorted(glob.glob(inpFile))
            # dataL5 = xr.open_mfdataset(fileList)
            #
            # timeList = dataL5['time'].values
            # minYear = pd.to_datetime(timeList.min()).strftime('%Y')
            # maxYear = pd.to_datetime(timeList.max()).strftime('%Y')
            #
            # saveFile = '{}/{}/{}_{}-{}.nc'.format(globalVar['outPath'], serviceName, 'GDP', minYear, maxYear)
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # dataL5.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

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
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))