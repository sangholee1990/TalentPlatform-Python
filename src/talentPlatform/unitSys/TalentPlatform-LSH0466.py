# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import urllib
import warnings
from datetime import datetime
import pytz
import googlemaps
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xmltodict
from sklearn.neighbors import BallTree
import requests
import json
import pymysql
import requests
from urllib.parse import quote_plus
import configparser
import pymysql
import zipfile

from sqlalchemy.dialects.mysql import insert
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Float
from sqlalchemy.dialects.mysql import DOUBLE
from sqlalchemy import Table, Column, Integer, String, MetaData
from sqlalchemy import Column, Numeric
import xarray as xr
import xclim as xc
from sklearn.linear_model import LinearRegression
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

Base = declarative_base()

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
    # Python을 이용한 전 지구 규모의 일 단위 강수량 편의보정 및 성능평가

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
    serviceName = 'LSH0466'

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
                # 시작/종료 시간
                'srtDate': '1990-01-01'
                , 'endDate': '1991-01-01'

                # 경도 최소/최대/간격
                , 'lonMin': 0
                , 'lonMax': 360
                , 'lonInv': 1

                # 위도 최소/최대/간격
                , 'latMin': -90
                , 'latMax': 90
                , 'latInv': 1
            }


            # 변수 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1D')
            dtMonthList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1M')

            lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
            latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])
            log.info(f'[CHECK] len(lonList) : {len(lonList)}')
            log.info(f'[CHECK] len(latList) : {len(latList)}')

            # 날짜 기준으로 반복문
            # for dtDayIdx, dtDayInfo in enumerate(dtDayList):
                # print(f'[CHECK] dtDayInfo : {dtDayInfo}')

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ERA5_1979_2020-004.nc')
            fileList = sorted(glob.glob(inpFile))
            obsData = xr.open_dataset(fileList[0]).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))

            # 경도 변환 (-180~180 to 0~360)
            obsDataL1 = obsData
            obsDataL1.coords['lon'] = (obsDataL1.coords['lon']) % 360
            obsDataL1 = obsDataL1.sortby(obsDataL1.lon)

            obsDataL2 = obsDataL1.interp({'lon': lonList, 'lat': latList}, method='linear')



            # obsData['rain'].isel(time = 0).plot()
            # obsDataL2['rain'].isel(time = 0).plot()
            # plt.show()

            # mm d-1
            # obsData['rain'].attrs

            # if fileList is None or len(fileList) < 1:
            #     continue

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'pr_day_MRI-ESM2-0_historical_r1i1p1f1_gn_19500101-19991231-003.nc')
            fileList = sorted(glob.glob(inpFile))
            modData = xr.open_dataset(fileList[0]).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate'])).interp({'lon': lonList, 'lat': latList}, method='linear')

            # modData['pr'].attrs

            # 일 강수량 단위 환산 : 60 * 60 * 24
            modData['pr'] = modData['pr'] * 86400
            modData['pr'].attrs["units"] = "mm d-1"

            modDataL2 = modData

            # 경도 변환 (-180~180 to 0~360)
            # modDataL1 = modData
            # modDataL1.coords['lon'] = (modDataL1.coords['lon']) % 360
            # modDataL1 = modDataL1.sortby(modDataL1.lon)

            # modData['time'].values
            # modData['pr'] = modData['pr'] * 2592000 / (10 ** 6)

            # timeList = modData['time'].values
            # bndList = modData['bnds'].values
            # modData['pr'].isel(time=0).plot()
            # # modDataL1['pr'].isel(time=0).plot()
            # plt.show()

            # https://xclim.readthedocs.io/en/stable/notebooks/sdba.html
            from xclim.sdba import QuantileDeltaMapping

            # group = xc.sdba.Grouper('time.dayofyear', window=5)
            # qdm = QuantileDeltaMapping.train(obsData['rain'], modData['pr'], group=group)
            # qdm = QuantileDeltaMapping.train(obsData['rain'], modData['pr'])
            # qdm = QuantileDeltaMapping.train(obsDataL2['rain'], modDataL2['pr'])
            qdm = QuantileDeltaMapping.train(obsDataL2['rain'], modDataL2['pr'], nquantiles=15, group="time", kind="+")
            # corData = qdm.adjust(modDataL1['pr'])
            # corData = qdm.adjust(modData['pr'])
            corData = qdm.adjust(modData['pr'])

            corData.isel(time=0).plot()
            plt.show()

            # qdm = QuantileDeltaMapping.train(obsData['rain'], modData['pr'])


            # 0 또는 360도의 경우 동일하기 때문에 변환시 문제
            # 따라서 360도 제거
            # correctedL1 = corrected.where(corrected['lon'] != 360, drop=True)

            # 경도 변환 (0~360 to -180~180)
            # corrected.coords['lon'] = (corrected.coords['lon'] + 180) % 360 - 180
            # corrected = corrected.sortby(corrected.lon)

            # corrected.plot()
            corrected.isel(time = 10).plot(x='lon', y='lat')
            plt.show()

            obs_array = obsData['pr'].sel(lat=some_lat, lon=some_lon).values
            cor_array = corrected.sel(lat=some_lat, lon=some_lon).values
            X = obs_array.reshape(-1, 1)
            y = cor_array

            # X = obsData['pr'].values.reshape(-1, 1)
            # y = corrected.values

            reg = LinearRegression().fit(X, y)

            mse = mean_squared_error(y, reg.predict(X))

            n = len(y)
            k = 2  # parameters: intercept and coefficient for linear regression

            aic = n * np.log(mse) + 2 * k
            bic = n * np.log(mse) + k * np.log(n)




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
