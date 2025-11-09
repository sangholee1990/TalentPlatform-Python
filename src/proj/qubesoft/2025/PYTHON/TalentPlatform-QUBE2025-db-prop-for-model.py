# ================================================
# 요구사항
# ================================================
# Python을 이용한 데이터베이스

# 프로그램 종료
# ps -ef | grep python | grep TalentPlatform-QUBE2025-db-prop-for-real.py | awk '{print $2}' | xargs kill -9
# pkill -f TalentPlatform-QUBE2025-db-prop-for-real.py

# 프로그램 시작
# conda activate py38

# cd /SYSTEMS/PROG/PYTHON
# nohup /SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-QUBE2025-db-prop-for-real.py --srtDate "2022-02-18" --endDate "2025-11-04" &

# * 1 * * * cd /SYSTEMS/PROG/PYTHON && /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/TalentPlatform-QUBE2025-db-prop-for-real.py --srtDate "$(date -d "2 days ago" +\%Y-\%m-\%d)" --endDate "$(date -d "2 days" +\%Y-\%m-\%d)"

import glob
# import seaborn as sns
import logging
import logging.handlers
import logging.handlers
import os
import platform
import sys
import argparse
import traceback
import warnings
# import datetime as dt
# from datetime import datetime
import pvlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
from scipy.stats import linregress
import pandas as pd
# import cartopy.crs as ccrs
import math
from scipy import spatial
from pandas.tseries.offsets import Day, Hour, Minute, Second
from scipy.interpolate import Rbf
from numpy import zeros, newaxis

# import pygrib
# import haversine as hs
import pytz
import datetime
# import h2o
# from pycaret.regression import *
from sqlalchemy import create_engine
import re
import configparser
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
import random
from urllib.parse import quote_plus
from urllib.parse import unquote_plus
import urllib.parse
import sqlalchemy
from sqlalchemy import create_engine, text
import requests
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import text
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
import requests
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import text
from pvlib import location
from pvlib import irradiance
from multiprocessing import Pool
import multiprocessing as mp
import uuid

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
dtKst = datetime.timedelta(hours=9)


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

        log.info("[CHECK] {} / val : {}".format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar


def cartesian(latitude, longitude, elevation=0):
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)

    return (X, Y, Z)


def initCfgInfo(config, key):

    result = None

    try:
        log.info(f'[CHECK] key : {key}')

        dbUser = config.get(key, 'user')
        dbPwd = urllib.parse.quote(config.get(key, 'pwd'))
        dbHost = config.get(key, 'host')
        dbPort = config.get(key, 'port')
        dbName = config.get(key, 'dbName')

        engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}", echo=False)
        sessionMake = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        # session = sessionMake()

        base = automap_base()
        base.prepare(autoload_with=engine)
        tableList = base.classes.keys()

        result = {
            'engine': engine
            , 'sessionMake': sessionMake
            # , 'session': session
            , 'tableList': tableList
            , 'tableCls': base.classes
        }

        return result

    except Exception as e:
        log.error(f'Exception : {e}')
        return result


def propUmkr(sysOpt, cfgDb, dtDateInfo):
    try:
        procInfo = mp.current_process()

        for ef in sysOpt['UMKR']['ef']:
            inpFile = dtDateInfo.strftime(sysOpt['UMKR']['inpUmFile']).format(ef=ef)
            fileList = sorted(glob.glob(inpFile))
            if len(fileList) < 1: continue

            for jj, fileInfo in enumerate(fileList):
                log.info(f"[CHECK] fileInfo : {fileInfo}")

                umData = None
                try:
                    grb = pygrib.open(fileInfo)
                    grbInfo = grb.select(name='Temperature')[1]

                    # validIdx = int(re.findall('H\d{3}', fileInfo)[0].replace('H', ''))
                    validIdx = int(ef)
                    dtValidDate = grbInfo.validDate
                    dtAnalDate = grbInfo.analDate

                    row2D = sysOpt['row2D']
                    col2D = sysOpt['col2D']
                    uVec = grb.select(name='10 metre U wind component')[0].values[row2D, col2D]
                    vVec = grb.select(name='10 metre V wind component')[0].values[row2D, col2D]
                    WD = (270 - np.rad2deg(np.arctan2(vVec, uVec))) % 360
                    WS = np.sqrt(np.square(uVec) + np.square(vVec))
                    PA = grb.select(name='Surface pressure')[0].values[row2D, col2D]
                    TA = grb.select(name='Temperature')[0].values[row2D, col2D]
                    TD = grb.select(name='Dew point temperature')[0].values[row2D, col2D]
                    HM = grb.select(name='Relative humidity')[0].values[row2D, col2D]
                    lowCA = grb.select(name='Low cloud cover')[0].values[row2D, col2D]
                    medCA = grb.select(name='Medium cloud cover')[0].values[row2D, col2D]
                    higCA = grb.select(name='High cloud cover')[0].values[row2D, col2D]
                    CA_TOT = np.mean([lowCA, medCA, higCA], axis=0)
                    SWR = grb.select(name='unknown')[0].values[row2D, col2D]

                    lat1D = sysOpt['lat1D']
                    lon1D = sysOpt['lon1D']
                    umData = xr.Dataset(
                        {
                            'uVec': (('anaTime', 'time', 'lat', 'lon'), (uVec).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'vVec': (('anaTime', 'time', 'lat', 'lon'), (vVec).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'WD': (('anaTime', 'time', 'lat', 'lon'), (WD).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'WS': (('anaTime', 'time', 'lat', 'lon'), (WS).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'PA': (('anaTime', 'time', 'lat', 'lon'), (PA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'TA': (('anaTime', 'time', 'lat', 'lon'), (TA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'TD': (('anaTime', 'time', 'lat', 'lon'), (TD).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'HM': (('anaTime', 'time', 'lat', 'lon'), (HM).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'lowCA': (('anaTime', 'time', 'lat', 'lon'), (lowCA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'medCA': (('anaTime', 'time', 'lat', 'lon'), (medCA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'higCA': (('anaTime', 'time', 'lat', 'lon'), (higCA).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'CA_TOT': (('anaTime', 'time', 'lat', 'lon'), (CA_TOT).reshape(1, 1, len(lat1D), len(lon1D)))
                            , 'SWR': (('anaTime', 'time', 'lat', 'lon'), (SWR).reshape(1, 1, len(lat1D), len(lon1D)))
                        }
                        , coords={
                            'anaTime': pd.date_range(dtAnalDate, periods=1)
                            , 'time': pd.date_range(dtValidDate, periods=1)
                            , 'lat': lat1D
                            , 'lon': lon1D
                        }
                    )

                except Exception as e:
                    log.error(f"Exception : {e}")

                if umData is None: continue

                posDataL1 = sysOpt['posDataL1']
                for kk, posInfo in posDataL1.iterrows():
                    posId = int(posInfo['ID'])
                    posLat = posInfo['LAT']
                    posLon = posInfo['LON']

                    # log.info(f"[CHECK] posId (posLon, posLat) : {posId} ({posLon}. {posLat})")

                    dtAnaTimeInfo = umData['anaTime'].values
                    umDataL2 = umData.sel(lat=posLat, lon=posLon, anaTime=dtAnaTimeInfo)
                    umDataL3 = umDataL2.to_dataframe().dropna().reset_index(drop=True)
                    # umDataL3['dtDate'] = pd.to_datetime(dtAnaTimeInfo) + (umDataL3.index.values * datetime.timedelta(hours=1))
                    umDataL3['DATE_TIME'] = pd.to_datetime(dtAnaTimeInfo) + (validIdx * datetime.timedelta(hours=1))
                    # umDataL3['dtDateKst'] = umDataL3.index.tz_localize(tzUtc).tz_convert(tzKst)
                    umDataL3['DATE_TIME_KST'] = umDataL3['DATE_TIME'] + dtKst
                    umDataL3['ANA_DATE'] = pd.to_datetime(dtAnaTimeInfo)
                    umDataL3['SRV'] = 'SRV{:05d}'.format(posId)
                    umDataL3['TA'] = umDataL3['TA'] - 273.15
                    umDataL3['TD'] = umDataL3['TD'] - 273.15
                    umDataL3['PA'] = umDataL3['PA'] / 100.0
                    umDataL3['CA_TOT'] = np.where(umDataL3['CA_TOT'] < 0, 0, umDataL3['CA_TOT'])
                    umDataL3['CA_TOT'] = np.where(umDataL3['CA_TOT'] > 1, 1, umDataL3['CA_TOT'])

                    solPosInfo = pvlib.solarposition.get_solarposition(umDataL3['DATE_TIME'], posLat, posLon,
                                                                       pressure=umDataL3['PA'] * 100.0,
                                                                       temperature=umDataL3['TA'], method='nrel_numpy')
                    umDataL3['EXT_RAD'] = pvlib.irradiance.get_extra_radiation(solPosInfo.index.dayofyear)
                    umDataL3['SZA'] = solPosInfo['zenith'].values
                    umDataL3['AZA'] = solPosInfo['azimuth'].values
                    umDataL3['ET'] = solPosInfo['equation_of_time'].values

                    site = location.Location(latitude=posLat, longitude=posLon, tz='Asia/Seoul')
                    clearInsInfo = site.get_clearsky(pd.to_datetime(umDataL3['DATE_TIME'].values))
                    umDataL3['GHI_CLR'] = clearInsInfo['ghi'].values
                    umDataL3['DNI_CLR'] = clearInsInfo['dni'].values
                    umDataL3['DHI_CLR'] = clearInsInfo['dhi'].values
                    #
                    # poaInsInfo = irradiance.get_total_irradiance(
                    #     surface_tilt=posInfo['STN_SZA'],
                    #     surface_azimuth=posInfo['STN_AZA'],
                    #     dni=clearInsInfo['dni'],
                    #     ghi=clearInsInfo['ghi'],
                    #     dhi=clearInsInfo['dhi'],
                    #     solar_zenith=solPosInfo['apparent_zenith'].values,
                    #     solar_azimuth=solPosInfo['azimuth'].values
                    # )
                    # umDataL3['GHI_POA'] = poaInsInfo['poa_global'].values
                    # umDataL3['DNI_POA'] = poaInsInfo['poa_direct'].values
                    # umDataL3['DHI_POA'] = poaInsInfo['poa_diffuse'].values

                    # 혼탁도
                    turbidity = pvlib.clearsky.lookup_linke_turbidity(pd.to_datetime(umDataL3['DATE_TIME'].values), posLat, posLon, interp_turbidity=True)
                    umDataL3['TURB'] = turbidity.values

                    # *******************************************************
                    # DB 적재
                    # *******************************************************
                    try:
                        # fileName = os.path.basename(fileInfo)
                        # tbTmp = f"tbTm_{fileName}"
                        tbTmp = f"tbTm_{uuid.uuid4().hex}"
                        with cfgDb['sessionMake']() as session:
                            with session.begin():
                                db_engine = session.get_bind()
                                umDataL3.to_sql(
                                    name=tbTmp,
                                    con=db_engine,
                                    if_exists="replace",
                                    index=False
                                )

                                query = text(f"""
                                    INSERT INTO "TB_FOR_DATA" (
                                          "SRV", "ANA_DATE", "DATE_TIME", "DATE_TIME_KST",
                                          "CA_TOT", "HM", "PA", "TA", "TD", "WD", "WS",
                                          "SZA", "AZA", "ET", "TURB",
                                          "GHI_CLR", "DNI_CLR", "DHI_CLR", "SWR", "EXT_RAD",
                                          "REG_DATE"
                                    )
                                    SELECT
                                          "SRV", "ANA_DATE", "DATE_TIME", "DATE_TIME_KST",
                                          "CA_TOT", "HM", "PA", "TA", "TD", "WD", "WS",
                                          "SZA", "AZA", "ET", "TURB",
                                          "GHI_CLR", "DNI_CLR", "DHI_CLR", "SWR", "EXT_RAD",
                                          now()
                                    FROM "{tbTmp}"
                                    ON CONFLICT ("SRV", "ANA_DATE", "DATE_TIME")
                                    DO UPDATE SET
                                          "DATE_TIME_KST" = excluded."DATE_TIME_KST",
                                          "CA_TOT" = excluded."CA_TOT", 
                                          "HM" = excluded."HM", 
                                          "PA" = excluded."PA", 
                                          "TA" = excluded."TA", 
                                          "TD" = excluded."TD", 
                                          "WD" = excluded."WD", 
                                          "WS" = excluded."WS",
                                          "SZA" = excluded."SZA", 
                                          "AZA" = excluded."AZA", 
                                          "ET" = excluded."ET", 
                                          "TURB" = excluded."TURB",
                                          "GHI_CLR" = excluded."GHI_CLR", 
                                          "DNI_CLR" = excluded."DNI_CLR", 
                                          "DHI_CLR" = excluded."DHI_CLR", 
                                          "SWR" = excluded."SWR",
                                          "EXT_RAD" = excluded."EXT_RAD",
                                          "MOD_DATE" = now();
                                      """)
                                session.execute(query)
                                session.execute(text(f'DROP TABLE IF EXISTS "{tbTmp}"'))
                    except Exception as e:
                        log.error(f"Exception : {e}")
            # log.info(f'[END] propUmkr : {dtDateInfo} / pid : {procInfo.pid}')
    except Exception as e:
        log.error(f'Exception : {e}')
        raise e

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'
        # contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON'

    prjName = 'test'
    serviceName = 'LSH0255'

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
                globalVar['inpPath'] = '/HDD/DATA/INPUT'
                globalVar['outPath'] = '/HDD/DATA/OUTPUT'
                globalVar['figPath'] = '/HDD/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                # 'srtDate': globalVar['srtDate'],
                # 'endDate': globalVar['endDate'],
                'srtDate': '2021-01-01',
                'endDate': '2025-11-01',

                # 비동기 다중 프로세스 개수
                # 'cpuCoreNum': globalVar['cpuCoreNum'],
                'cpuCoreNum': '5',

                # 설정 파일
                'cfgFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
                # 'cfgFile': '/vol01/SYSTEMS/INDIAI/PROG/PYTHON/resources/config/system.cfg',
                # 'cfgFile': '/SYSTEMS/PROG/PYTHON/resources/config/system.cfg',
                'cfgDbKey': 'postgresql-qubesoft.iptime.org-qubesoft-dms02',
                'cfgDb': None,
                'posDataL1': None,
                'row2D': None,
                'col2D': None,
                'lat1D': None,
                'lon1D': None,

                # 예보 모델
                'UMKR': {
                    'cfgUmFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/modelInfo/UMKR_l015_unis_H000_202110010000.grb2',
                    'inpUmFile': '/HDD/DATA/MODEL/%Y%m/%d/UMKR_l015_unis_H{ef}_%Y%m%d%H%M.grb2',
                    # 'cfgUmFile': '/DATA/COLCT/UMKR/201901/01/UMKR_l015_unis_H00_201901010000.grb2',
                    # 'inpUmFile': '/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H{ef}_%Y%m%d%H%M.grb2',
                    # 'cfgUmFile': '/DATA/MODEL/202001/01/UMKR_l015_unis_H00_202001010000.grb2',
                    # 'inpUmFile': '/DATA/MODEL/%Y%m/%d/UMKR_l015_unis_H{ef}_%Y%m%d%H%M.grb2',
                    'ef': ['00', '01', '02', '03', '04', '05'],
                    'invDate': '6h',
                },
            }

            # *******************************************************
            # 설정 정보
            # *******************************************************
            config = configparser.ConfigParser()
            config.read(sysOpt['cfgFile'], encoding='utf-8')

            # sysOpt['cfgDb'] = initCfgInfo(config, sysOpt['cfgDbKey'])
            cfgDb = initCfgInfo(config, sysOpt['cfgDbKey'])

            # 관측소 정보
            query = text("""
                         SELECT *
                         FROM "TB_STN_INFO"
                         WHERE "OPER_YN" = 'Y'
                         ORDER BY "ID" ASC;
                         """)
            posDataL1 = pd.DataFrame(cfgDb['sessionMake']().execute(query))

            for i, posInfo in posDataL1.iterrows():
                posId = posInfo['ID']
                srvId = f"SRV{posId:05d}"

                query = text("""
                            SELECT
                                pv."PV",
                                lf.*
                            FROM
                                "TB_PV_DATA" AS pv
                            LEFT JOIN
                                "TB_FOR_DATA" AS lf ON pv."SRV" = lf."SRV" AND pv."DATE_TIME" = lf."DATE_TIME"
                            WHERE pv."SRV" = :srvId
                            ORDER BY "SRV", "DATE_TIME_KST" DESC;
                             """)
                data = pd.DataFrame(cfgDb['sessionMake']().execute(query, {'srvId':srvId}))

                # data is None
                # len(data) < 1

            # lat1D = np.array(posDataL1['LAT'])
            # lon1D = np.array(posDataL1['LON'])
            #
            # sysOpt['posDataL1'] = posDataL1
            # sysOpt['lat1D'] = lat1D
            # sysOpt['lon1D'] = lon1D

            # *******************************************************
            # UM 자료 읽기
            # # *******************************************************
            # cfgUmFile = sysOpt['UMKR']['cfgUmFile']
            # log.info(f"[CHECK] cfgUmFile : {cfgUmFile}")
            #
            # cfgInfo = pygrib.open(cfgUmFile).select(name='Temperature')[1]
            # lat2D, lon2D = cfgInfo.latlons()
            #
            # # 최근접 좌표
            # posList = []
            #
            # # kdTree를 위한 초기 데이터
            # for i in range(0, lon2D.shape[0]):
            #     for j in range(0, lon2D.shape[1]):
            #         coord = [lat2D[i, j], lon2D[i, j]]
            #         posList.append(cartesian(*coord))
            #
            # tree = spatial.KDTree(posList)
            #
            # # coord = cartesian(posInfo['lat'], posInfo['lon'])
            # row1D = []
            # col1D = []
            # for ii, posInfo in posDataL1.iterrows():
            #     coord = cartesian(posInfo['LAT'], posInfo['LON'])
            #     closest = tree.query([coord], k=1)
            #     cloIdx = closest[1][0]
            #     row = int(cloIdx / lon2D.shape[1])
            #     col = cloIdx % lon2D.shape[1]
            #     row1D.append(row)
            #     col1D.append(col)
            # sysOpt['row2D'], sysOpt['col2D'] = np.meshgrid(row1D, col1D)
            #
            # # **************************************************************************************************************
            # # 비동기 다중 프로세스 수행
            # # **************************************************************************************************************
            # # 비동기 다중 프로세스 개수
            # # pool = Pool(int(sysOpt['cpuCoreNum']))
            #
            # dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            # dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            # dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['UMKR']['invDate'])
            # for dtDateInfo in reversed(dtDateList):
            #     propUmkr(sysOpt, cfgDb, dtDateInfo)
            #     # pool.apply_async(propUmkr, args=(sysOpt, cfgDb, dtDateInfo))
            #
            # # pool.close()
            # # pool.join()

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
