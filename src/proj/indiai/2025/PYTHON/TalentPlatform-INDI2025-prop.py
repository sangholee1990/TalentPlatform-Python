# ================================================
# 요구사항
# ================================================
# Python을 이용한 UMKR 수치모델 전처리

# ps -ef | grep "TalentPlatform-INDI2025-colct-kmaApiHub.py" | awk '{print $2}' | xargs kill -9

# cd /vol01/SYSTEMS/INDIAI/PROG/PYTHON
# /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-prop.py --modelList 'UMKR' --cpuCoreNum '5' --srtDate '2019-01-01' --endDate '2021-01-01'
# nohup /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-prop.py --modelList 'UMKR' --cpuCoreNum '10' --srtDate '2019-01-01' --endDate '2021-01-01' &

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
from sklearn.neighbors import BallTree
import pygrib
from matplotlib import font_manager, rc
from metpy.units import units
from metpy.calc import wind_components, wind_direction, wind_speed

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

# def propProc(modelType, modelInfo, dtDateInfo):
#     try:
#         propFunList = {
#             'UMKR': propNwp,
#         }
#
#         propFun = propFunList.get(modelType)
#         propFun(modelInfo, dtDateInfo)
#
#     except Exception as e:
#         log.error(f'Exception : {str(e)}')
#         raise e

def matchStnFor(subOpt, subData):
    
    try:
        # ==========================================================================================================
        # 지상 관측소을 기준으로 최근접 가공파일 화소 찾기 (posRow, posCol, posLat, posLon, posDistKm)
        # ==========================================================================================================
        fileInfo = subOpt['umkrFileInfo']

        if fileInfo is None or len(fileInfo) < 1:
            log.error(f"[ERROR] fileInfo : {fileInfo} / 가공파일을 확인해주세요.")
            return

        # 가공 파일 일부
        grb = pygrib.open(fileInfo)
        grbInfo = grb.select(name='Temperature')[0]
        lat2D, lon2D = grbInfo.latlons()
        row2D, col2D = np.indices(lat2D.shape)
        cfgDataL3 = pd.DataFrame({'row': row2D.flatten(), 'col': col2D.flatten(), 'lat': lat2D.flatten(), 'lon': lon2D.flatten()})

        #         row  col  TMP_P0_L1_GLC0        lat         lon
        # 0         0    0      281.500610  32.256874  121.834427
        # 1         0    1      281.522095  32.257580  121.850502
        # cfgData = xr.open_dataset(fileInfo, engine='pynio')
        # cfgDataL1 = cfgData[['TMP_P0_L1_GLC0']]
        # cfgDataL2 = cfgDataL1.to_dataframe().reset_index(drop=False)
        # cfgDataL3 = cfgDataL2.rename(
        #     columns={
        #         'ygrid_0': 'row'
        #         , 'xgrid_0': 'col'
        #         , 'gridlat_0': 'lat'
        #         , 'gridlon_0': 'lon'
        #     }
        # )

        # 지상관측소
        allStnData = subData
        allStnDataL1 = subData[['Id', 'Latitude', 'Longitude']]

        # 2024.10.24 수정
        # allStnDataL2 = allStnDataL1[allStnDataL1['STN'].isin(sysOpt['stnInfo']['list'])]
        allStnDataL2 = allStnDataL1

        # 지상 관측소를 기준으로 최근접 가공파일 화소 찾기 (posRow, posCol, posLat, posLon, posDistKm)
        # 0, 당진수상태양광, 37.0507527, 126.5102993, 343.0, 288.0, 37.054656982421875, 126.50814819335938, 0.07443994101426843
        # 1, 당진자재창고태양광, 37.0507527, 126.5102993, 343.0, 288.0, 37.054656982421875, 126.50814819335938, 0.07443994101426843
        # 2, 당진태양광, 37.0507527, 126.5102993, 343.0, 288.0, 37.054656982421875, 126.50814819335938, 0.07443994101426843
        # 3, 울산태양광, 35.47765089999999, 129.380778, 233.0, 459.0, 35.48063278198242, 129.3863983154297, 0.09525458282421134
        baTree = BallTree(np.deg2rad(cfgDataL3[['lat', 'lon']].values), metric='haversine')
        for i, posInfo in allStnDataL2.iterrows():
            if (pd.isna(posInfo['Latitude']) or pd.isna(posInfo['Longitude'])): continue

            closest = baTree.query(np.deg2rad(np.c_[posInfo['Latitude'], posInfo['Longitude']]), k=1)
            cloDist = closest[0][0][0] * 1000.0
            cloIdx = closest[1][0][0]
            cfgInfo = cfgDataL3.loc[cloIdx]

            allStnDataL2.loc[i, 'posRow'] = cfgInfo['row']
            allStnDataL2.loc[i, 'posCol'] = cfgInfo['col']
            allStnDataL2.loc[i, 'posLat'] = cfgInfo['lat']
            allStnDataL2.loc[i, 'posLon'] = cfgInfo['lon']
            allStnDataL2.loc[i, 'posDistKm'] = cloDist

        return allStnDataL2

    except Exception as e:
        log.error(f'Exception : {str(e)}')


@retry(stop_max_attempt_number=10)
def propUmkr(modelInfo, cfgDataL1, dtDateInfo):
    try:
        procInfo = mp.current_process()

        # 저장 파일 검사
        saveFile = dtDateInfo.strftime(modelInfo['saveFile'])
        saveFileList = sorted(glob.glob(saveFile))
        if len(saveFileList) > 0: return

        # 입력 파일 검사
        filePattern = dtDateInfo.strftime(modelInfo['fileList'])
        fileList = sorted(glob.glob(filePattern))
        if len(fileList) < 1: return

        # 다중 인덱스 기반 데이터 추출
        cfgDataL2 = cfgDataL1[['posLat', 'posLon', 'posRow', 'posCol']].drop_duplicates()

        lat1D = cfgDataL2['posLat'].values
        lon1D = cfgDataL2['posLon'].values
        row1D = cfgDataL2['posRow'].astype(int).values
        col1D = cfgDataL2['posCol'].astype(int).values
        row2D, col2D = np.meshgrid(row1D, col1D)

        # fileInfo = fileList[1]
        dsDataL1 = xr.Dataset()
        for fileInfo in fileList:
            # log.info(f'[CHECK] fileInfo : {fileInfo}')

            try:
                # xarray 기반 자료처리
                # umData = xr.open_dataset(fileInfo, engine='pynio')
                # if len(umData) < 1: continue
                # log.info(f'[CHECK] fileInfo : {fileInfo}')

                # attrInfo = umData[list(umData.dtypes)[0]].attrs
                # anaDate = pd.to_datetime(attrInfo['initial_time'], format="%m/%d/%Y (%H:%M)")
                # forDate = anaDate + pd.DateOffset(hours=int(attrInfo['forecast_time'][0]))

                # pygrib 기반 자료처리
                grb = pygrib.open(fileInfo)
                grbInfo = grb.select(name='Temperature')[0]

                anaDate = grbInfo.analDate
                forDate = grbInfo.validDate

                # 변수 목록
                # grbList = grb.select()
                # for grbInfo in grbList:
                #     log.info(f'[CHECK] grbInfo : {grbInfo}')

                # 메타정보 보기
                # grbInfo = grb.select()[0]
                # for key in grbInfo.keys():
                #     log.info(f'[CHECK] key : {key}, / val : {grbInfo[key]}')

                # 단일 위경도 기반 데이터 추출
                # grb.select(name='10 metre U wind component')[0].data(lon1=128, lat1=38)

                uVec = grb.select(name='10 metre U wind component')[0].values[row2D, col2D]
                vVec = grb.select(name='10 metre V wind component')[0].values[row2D, col2D]
                WD = (270 - np.rad2deg(np.arctan2(vVec, uVec))) % 360
                WS = np.sqrt(np.square(uVec) + np.square(vVec))
                # WD = wind_direction(uVec.data * units('m/s'), vVec.data * units('m/s'), convention='from')
                # WS = wind_speed(uVec.data * units('m/s'), vVec.data * units('m/s'))
                PA = grb.select(name='Surface pressure')[0].values[row2D, col2D]
                TA = grb.select(name='Temperature')[0].values[row2D, col2D]
                TD = grb.select(name='Dew point temperature')[0].values[row2D, col2D]
                HM = grb.select(name='Relative humidity')[0].values[row2D, col2D]
                lowCA = grb.select(name='Low cloud cover')[0].values[row2D, col2D]
                medCA = grb.select(name='Medium cloud cover')[0].values[row2D, col2D]
                higCA = grb.select(name='High cloud cover')[0].values[row2D, col2D]
                CA = np.mean([lowCA, medCA, higCA], axis=0)
                TDSWS = grb.select(name='unknown')[3].values[row2D, col2D]

                dsData = xr.Dataset(
                    {
                        'uVec': (('forDate', 'lat', 'lon'), (uVec).reshape(1, len(lat1D), len(lon1D))),
                        'vVec': (('forDate', 'lat', 'lon'), (vVec).reshape(1, len(lat1D), len(lon1D))),
                        'WD': (('forDate', 'lat', 'lon'), (WD).reshape(1, len(lat1D), len(lon1D))),
                        'WS': (('forDate', 'lat', 'lon'), (WS).reshape(1, len(lat1D), len(lon1D))),
                        'PA': (('forDate', 'lat', 'lon'), (PA).reshape(1, len(lat1D), len(lon1D))),
                        'TA': (('forDate', 'lat', 'lon'), (TA).reshape(1, len(lat1D), len(lon1D))),
                        'TD': (('forDate', 'lat', 'lon'), (TD).reshape(1, len(lat1D), len(lon1D))),
                        'HM': (('forDate', 'lat', 'lon'), (HM).reshape(1, len(lat1D), len(lon1D))),
                        'lowCA': (('forDate', 'lat', 'lon'), (lowCA).reshape(1, len(lat1D), len(lon1D))),
                        'medCA': (('forDate', 'lat', 'lon'), (medCA).reshape(1, len(lat1D), len(lon1D))),
                        'higCA': (('forDate', 'lat', 'lon'), (higCA).reshape(1, len(lat1D), len(lon1D))),
                        'CA_TOT': (('forDate', 'lat', 'lon'), (CA).reshape(1, len(lat1D), len(lon1D))),
                        'TDSWS': (('forDate', 'lat', 'lon'), (TDSWS).reshape(1, len(lat1D), len(lon1D))),
                        # 'uVec': (('anaDate', 'forDate', 'lat', 'lon'), (uVec).reshape(1, 1, len(lat1D), len(lon1D))),
                        # 'vVec': (('anaDate', 'forDate', 'lat', 'lon'), (vVec).reshape(1, 1, len(lat1D), len(lon1D))),
                        # 'WD': (('anaDate', 'forDate', 'lat', 'lon'), (WD).reshape(1, 1, len(lat1D), len(lon1D))),
                        # 'WS': (('anaDate', 'forDate', 'lat', 'lon'), (WS).reshape(1, 1, len(lat1D), len(lon1D))),
                        # 'PA': (('anaDate', 'forDate', 'lat', 'lon'), (PA).reshape(1, 1, len(lat1D), len(lon1D))),
                        # 'TA': (('anaDate', 'forDate', 'lat', 'lon'), (TA).reshape(1, 1, len(lat1D), len(lon1D))),
                        # 'TD': (('anaDate', 'forDate', 'lat', 'lon'), (TD).reshape(1, 1, len(lat1D), len(lon1D))),
                        # 'HM': (('anaDate', 'forDate', 'lat', 'lon'), (HM).reshape(1, 1, len(lat1D), len(lon1D))),
                        # 'lowCA': (('anaDate', 'forDate', 'lat', 'lon'), (lowCA).reshape(1, 1, len(lat1D), len(lon1D))),
                        # 'medCA': (('anaDate', 'forDate', 'lat', 'lon'), (medCA).reshape(1, 1, len(lat1D), len(lon1D))),
                        # 'higCA': (('anaDate', 'forDate', 'lat', 'lon'), (higCA).reshape(1, 1, len(lat1D), len(lon1D))),
                        # 'CA_TOT': (('anaDate', 'forDate', 'lat', 'lon'), (CA).reshape(1, 1, len(lat1D), len(lon1D))),
                        # 'TDSWS': (('anaDate', 'forDate', 'lat', 'lon'), (TDSWS).reshape(1, 1, len(lat1D), len(lon1D))),
                    }
                    , coords={
                        # 'anaDate': pd.date_range(anaDate, periods=1),
                        'forDate': pd.date_range(forDate, periods=1),
                        'lat': lat1D,
                        'lon': lon1D,
                    }
                )

                dsDataL1 = xr.merge([dsDataL1, dsData])

            except Exception as e:
                log.error(f'Exception : {e}')

        if len(dsDataL1) > 0:
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            dsDataL1.to_netcdf(saveFile)
            log.info('[CHECK] saveFile : {}'.format(saveFile))

            # dsDataL2 = dsDataL1.interp(lat=35.480634, lon=129.386405)
            # dsDataL3 = dsDataL2.to_dataframe().reset_index()

            log.info(f'[END] propUmkr : {dtDateInfo} / pid : {procInfo.pid}')

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
                # 'srtDate': globalVar['srtDate'],
                # 'endDate': globalVar['endDate'],
                'srtDate': '2019-01-01',
                'endDate': '2019-01-04',
                'invDate': '1d',

                # 수행 목록
                # 'modelList': [globalVar['modelList']],
                'modelList': ['UMKR'],

                # 비동기 다중 프로세스 개수
                # 'cpuCoreNum': globalVar['cpuCoreNum'],
                'cpuCoreNum': '5',

                # 설정 파일
                'CFG': {
                    'siteInfo': '/DATA/PROP/SAMPLE/site_info.csv',
                    'umkrFileInfo': '/DATA/COLCT/UMKR/201901/01/UMKR_l015_unis_H00_201901011200.grb2',
                },

                'UMKR': {
                    'fileList': '/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d*.grb2',
                    'saveFile': '/DATA/PROP/UMKR/%Y%m/UMKR_FOR_%Y%m%d.nc',
                },
                # 'ACT': {
                #     'ASOS': {
                #         'searchFileList': f"/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d%H%M.grb2",
                #         'invDate': '6h',
                #     },
                #     'AWS': {
                #         'searchFileList': f"/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d%H%M.grb2",
                #         'invDate': '6h',
                #     },
                # },
            }

            # **************************************************************************************************************
            # 설정 파일 읽기
            # **************************************************************************************************************
            filePattern = sysOpt['CFG']['siteInfo']
            fileList = sorted(glob.glob(filePattern))
            if fileList is None or len(fileList) < 1:
                log.error(f"[ERROR] filePattern : {filePattern} / 파일을 확인해주세요.")
                raise Exception(f"[ERROR] filePattern : {filePattern} / 파일을 확인해주세요.")
            cfgData = pd.read_csv(fileList[0])
            cfgDataL1 = matchStnFor(sysOpt['CFG'], cfgData)

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
                dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

                for dtDateInfo in dtDateList:
                    # log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
                    pool.apply_async(propUmkr, args=(modelInfo, cfgDataL1, dtDateInfo))

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
