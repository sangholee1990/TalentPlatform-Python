# -*- coding: utf-8 -*-
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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from global_land_mask import globe
import cftime

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


def makeLandMask(data):

    log.info('[START] {}'.format('makeLandMask'))
    result = None

    try:

        lon1D = sorted(set(data['lon'].values))
        lat1D = sorted(set(data['lat'].values))
        time1D = sorted(set(data['time'].values))

        # 경도 변환 (0~360 to -180~180)
        convLon1D = []
        for i, lon in enumerate(lon1D):
            convLon1D.append((lon + 180) % 360 - 180)

        lon2D, lat2D = np.meshgrid(convLon1D, lat1D)
        isLand2D = globe.is_land(lat2D, lon2D)
        isLand3D = np.tile(isLand2D, (len(time1D), 1, 1))

        landData = xr.Dataset(
            {
                'isLand': (('time', 'lat', 'lon'), (isLand3D).reshape(len(time1D), len(lat1D), len(lon1D)))
            }
            , coords={
                'lat': lat1D
                , 'lon': lon1D
                , 'time': time1D
            }
        )

        # landData = xr.Dataset(
        #     {
        #         'isLand': (('lat', 'lon'), (isLand3D).reshape(len(lat1D), len(lon1D)))
        #     }
        #     , coords={
        #         'lat': lat1D
        #         , 'lon': lon1D
        #     }
        # )

        # 육해상에 대한 강수량
        dataL1 = xr.merge([data, landData])

        result = {
            'msg': 'succ'
            , 'lon1D': lon1D
            , 'lat1D': lat1D
            , 'time1D': time1D
            , 'isLand2D': isLand2D
            , 'isLand3D': isLand3D
            , 'resData': dataL1
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeLandMask'))

def makeLandMaskYear(data):

    log.info('[START] {}'.format('makeLandMaskYear'))
    result = None

    try:

        lon1D = sorted(set(data['lon'].values))
        lat1D = sorted(set(data['lat'].values))
        time1D = sorted(set(data['year'].values))

        # 경도 변환 (0~360 to -180~180)
        convLon1D = []
        for i, lon in enumerate(lon1D):
            convLon1D.append((lon + 180) % 360 - 180)

        lon2D, lat2D = np.meshgrid(convLon1D, lat1D)
        isLand2D = globe.is_land(lat2D, lon2D)
        isLand3D = np.tile(isLand2D, (len(time1D), 1, 1))

        landData = xr.Dataset(
            {
                'isLand': (('year', 'lat', 'lon'), (isLand3D).reshape(len(time1D), len(lat1D), len(lon1D)))
            }
            , coords={
                'lat': lat1D
                , 'lon': lon1D
                , 'year': time1D
            }
        )

        # landData = xr.Dataset(
        #     {
        #         'isLand': (('lat', 'lon'), (isLand3D).reshape(len(lat1D), len(lon1D)))
        #     }
        #     , coords={
        #         'lat': lat1D
        #         , 'lon': lon1D
        #     }
        # )

        # 육해상에 대한 강수량
        dataL1 = xr.merge([data, landData])

        result = {
            'msg': 'succ'
            , 'lon1D': lon1D
            , 'lat1D': lat1D
            , 'time1D': time1D
            , 'isLand2D': isLand2D
            , 'isLand3D': isLand3D
            , 'resData': dataL1
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeLandMaskYear'))


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 앙상블 모델 구축, 과거대비 변화율 산정, 연 강수량 및 평균온도 계산

    # 새롭게 문의드릴 내용은 다음과 같습니다.
    # 1) 가중치를 적용하여 각 모델의 어셈블 모델을 구축하려고합니다!(미래, 과거 데이터 다 확보하였고 가중치는 제가 직접계산하여 파일로 적용할 예정입니다.)
    # 2.) 과거기간대비 변화율을 산정하려고합니다. (1980년-2014년) 미래 (2031-2060)/ (2071-2100) 입니다.
    # (미래 - 현재)/현재 * 100
    # 3) 연 총 강수량과 연 평균 온도를 구하려고합니다.

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PV'

    prjName = 'test'
    serviceName = 'LSH0318'

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
                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2020-09-01'
                    , 'endDate': '2020-09-03'

                    # 경도 최소/최대/간격
                    , 'lonMin': 0
                    , 'lonMax': 360
                    , 'lonInv': 0.5

                    # 위도 최소/최대/간격
                    , 'latMin': -90
                    , 'latMax': 90
                    , 'latInv': 0.5
                }
            else:
                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': globalVar['srtDate']
                    # , 'endDate': globalVar['endDate']
                }


            # ************************************************************************************************
            # 입력 자료 (M1, M2, M3 별로 가중치 적용) 읽기
            # [변경] 입력 자료 (위도별 가중치 적용) 읽기
            # ************************************************************************************************
            # weight 자료
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'Pr weight.xlsx')
            fileList = sorted(glob.glob(inpFile))

            # weiData = pd.read_csv(fileList[0], index_col=False)
            weiData = pd.read_excel(fileList[0], sheet_name='순위합법', index_col=False)

            # ************************************************************************************************
            # historical (미래, 먼 미래) 파일 읽기
            # ************************************************************************************************
            # for i, weiInfo in weiData.iterrows():
            #
            #     modelInfo = weiInfo['model']
            #     log.info('[CHECK] modelInfo : {}'.format(modelInfo))
            #
            #     # if modelInfo != 'ACCESS-CM2': continue
            #     # if i > 2: break
            #
            #     saveFile = '{}/{}/{}_hist-tas.nc'.format(globalVar['outPath'], serviceName, modelInfo)
            #     # if (os.path.exists(saveFile)): continue
            #
            #     # Future 파일 읽기
            #     inpFile = '{}/{}/Historical/{}*historical*tas.nc'.format(globalVar['inpPath'], serviceName, modelInfo)
            #     fileList = sorted(glob.glob(inpFile))
            #
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해 주세요.'))
            #         continue
            #
            #     # inpData = xr.open_mfdataset(fileList)
            #     # inpData = xr.open_mfdataset(fileList).sel(time = slice('1950-01', '2014-12'))
            #     # 테스트
            #     inpData = xr.open_mfdataset(fileList).sel(time = slice('2013-01', '2014-12'))
            #
            #     highBotData = inpData.copy().sel(lat = slice(-90, -61))
            #     highBotData['time'] = pd.to_datetime(highBotData.indexes['time'].to_datetimeindex().strftime("%Y-%m"), format='%Y-%m') \
            #         if type(highBotData['time'].values[0]) == cftime._cftime.DatetimeNoLeap else \
            #         pd.to_datetime(pd.to_datetime(highBotData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #
            #     middleBotData = inpData.copy().sel(lat = slice(-60, -31))
            #     middleBotData['time'] = pd.to_datetime(middleBotData.indexes['time'].to_datetimeindex().strftime("%Y-%m"), format='%Y-%m') \
            #         if type(middleBotData['time'].values[0]) == cftime._cftime.DatetimeNoLeap else \
            #         pd.to_datetime(pd.to_datetime(middleBotData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #
            #     lowBotData = inpData.copy().sel(lat = slice(-30, 0))
            #     lowBotData['time'] = pd.to_datetime(lowBotData.indexes['time'].to_datetimeindex().strftime("%Y-%m"), format='%Y-%m') \
            #         if type(lowBotData['time'].values[0]) == cftime._cftime.DatetimeNoLeap else \
            #         pd.to_datetime(pd.to_datetime(lowBotData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #
            #     lowTopData = inpData.copy().sel(lat = slice(0.5, 30))
            #     lowTopData['time'] = pd.to_datetime(lowTopData.indexes['time'].to_datetimeindex().strftime("%Y-%m"), format='%Y-%m') \
            #         if type(lowTopData['time'].values[0]) == cftime._cftime.DatetimeNoLeap else \
            #         pd.to_datetime(pd.to_datetime(lowTopData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #
            #     middleTopData = inpData.copy().sel(lat = slice(31, 60))
            #     middleTopData['time'] = pd.to_datetime(middleTopData.indexes['time'].to_datetimeindex().strftime("%Y-%m"), format='%Y-%m') \
            #         if type(middleTopData['time'].values[0]) == cftime._cftime.DatetimeNoLeap else \
            #         pd.to_datetime(pd.to_datetime(middleTopData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #
            #     highTopData = inpData.copy().sel(lat = slice(61, 90))
            #     highTopData['time'] = pd.to_datetime(highTopData.indexes['time'].to_datetimeindex().strftime("%Y-%m"), format='%Y-%m') \
            #         if type(highTopData['time'].values[0]) == cftime._cftime.DatetimeNoLeap else \
            #         pd.to_datetime(pd.to_datetime(highTopData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #
            #     # 가중치를 통해 자료 병합
            #     tmpData = xr.merge(
            #         [
            #             highBotData['tas'] * weiInfo['highBot']
            #             , middleBotData['tas'] * weiInfo['middleBot']
            #             , lowBotData['tas'] * weiInfo['lowBot']
            #             , lowTopData['tas'] * weiInfo['lowTop']
            #             , middleTopData['tas'] * weiInfo['middleTop']
            #             , highTopData['tas'] * weiInfo['highTop']
            #         ]
            #     )
            #
            #     tmpData = tmpData.rename( { 'tas' : modelInfo } )
            #
            #     # 파일 저장 시 오래 걸림
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     try:
            #         tmpData.to_netcdf(saveFile)
            #     except Exception as e:
            #         log.error('Exception : {}'.format(e))
            #     log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # # 모델 가중치 파일 읽기
            # inpFile = '{}/{}/*_hist-tas.nc'.format(globalVar['outPath'], serviceName)
            # fileList = sorted(glob.glob(inpFile))
            # log.info('[CHECK] fileList : {}'.format(fileList))
            #
            # dsData = xr.open_mfdataset(fileList)
            #
            # dsSumData = dsData.to_array().sum('variable', skipna = True)
            # lon1D = sorted(set(dsData['lon'].values))
            # lat1D = sorted(set(dsData['lat'].values))
            # time1D = sorted(set(dsData['time'].values))
            #
            # dsDataL1 = xr.Dataset(
            #     {
            #         'tas': (('time', 'lat', 'lon'), (dsSumData.values).reshape(len(time1D), len(lat1D), len(lon1D)))
            #     }
            #     , coords = {
            #         'lat' : lat1D
            #         , 'lon' : lon1D
            #         , 'time' : time1D
            #     }
            # )
            #
            # data = dsDataL1
            # saveFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'tas-hist_MME.nc')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # data.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # dataL1 = makeLandMask(data)['resData']
            # # 육지에 대한 강수량
            # dataL2 = dataL1.where(dataL1['isLand'] == True).sel(lat = slice(-60, 90))
            #
            # saveFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'tas-hist_MME_land.nc')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # dataL2.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # # # ************************************************************************************************
            # # # ssp126 (현재) 파일 읽기
            # # # ************************************************************************************************
            # for i, weiInfo in weiData.iterrows():
            #
            #     modelInfo = weiInfo['model']
            #     log.info('[CHECK] modelInfo : {}'.format(modelInfo))
            #
            #     # if i > 2: break
            #     # if modelInfo != 'ACCESS-CM2': continue
            #
            #     saveFile = '{}/{}/{}_tas.nc'.format(globalVar['outPath'], serviceName, modelInfo)
            #     # if (os.path.exists(saveFile)): continue
            #
            #     # Future 파일 읽기
            #     inpFile = '{}/{}/Future/{}*ssp126*tas.nc'.format(globalVar['inpPath'], serviceName, modelInfo)
            #     fileList = sorted(glob.glob(inpFile))
            #
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해 주세요.'))
            #         continue
            #
            #     # inpData = xr.open_dataset(fileList[0])
            #     # inpData = xr.open_mfdataset(fileList)
            #     # 테스트
            #     inpData = xr.open_mfdataset(fileList).sel(time = slice('2015-01', '2016-12'))
            #
            #     highBotData = inpData.copy().sel(lat = slice(-90, -61))
            #     highBotData['time'] = pd.to_datetime(highBotData.indexes['time'].to_datetimeindex().strftime("%Y-%m"), format='%Y-%m') \
            #         if type(highBotData['time'].values[0]) == cftime._cftime.DatetimeNoLeap else \
            #         pd.to_datetime(pd.to_datetime(highBotData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #
            #     middleBotData = inpData.copy().sel(lat = slice(-60, -31))
            #     middleBotData['time'] = pd.to_datetime(middleBotData.indexes['time'].to_datetimeindex().strftime("%Y-%m"), format='%Y-%m') \
            #         if type(middleBotData['time'].values[0]) == cftime._cftime.DatetimeNoLeap else \
            #         pd.to_datetime(pd.to_datetime(middleBotData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #
            #     lowBotData = inpData.copy().sel(lat = slice(-30, 0))
            #     lowBotData['time'] = pd.to_datetime(lowBotData.indexes['time'].to_datetimeindex().strftime("%Y-%m"), format='%Y-%m') \
            #         if type(lowBotData['time'].values[0]) == cftime._cftime.DatetimeNoLeap else \
            #         pd.to_datetime(pd.to_datetime(lowBotData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #
            #     lowTopData = inpData.copy().sel(lat = slice(0.5, 30))
            #     lowTopData['time'] = pd.to_datetime(lowTopData.indexes['time'].to_datetimeindex().strftime("%Y-%m"), format='%Y-%m') \
            #         if type(lowTopData['time'].values[0]) == cftime._cftime.DatetimeNoLeap else \
            #         pd.to_datetime(pd.to_datetime(lowTopData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #
            #     middleTopData = inpData.copy().sel(lat = slice(31, 60))
            #     middleTopData['time'] = pd.to_datetime(middleTopData.indexes['time'].to_datetimeindex().strftime("%Y-%m"), format='%Y-%m') \
            #         if type(middleTopData['time'].values[0]) == cftime._cftime.DatetimeNoLeap else \
            #         pd.to_datetime(pd.to_datetime(middleTopData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #
            #     highTopData = inpData.copy().sel(lat = slice(61, 90))
            #     highTopData['time'] = pd.to_datetime(highTopData.indexes['time'].to_datetimeindex().strftime("%Y-%m"), format='%Y-%m') \
            #         if type(highTopData['time'].values[0]) == cftime._cftime.DatetimeNoLeap else \
            #         pd.to_datetime(pd.to_datetime(highTopData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #
            #
            #     # ************************************************************************************************
            #     # 1) 가중치를 적용하여 각 모델의 어셈블 모델을 구축하려고합니다!(미래, 과거 데이터 다 확보하였고
            #     # 가중치는 제가 직접계산하여 파일로 적용할 예정입니다.)
            #     # ************************************************************************************************
            #     # 가중치를 통해 자료 병합
            #     tmpData = xr.merge(
            #         [
            #             highBotData['tas'] * weiInfo['highBot']
            #             , middleBotData['tas'] * weiInfo['middleBot']
            #             , lowBotData['tas'] * weiInfo['lowBot']
            #             , lowTopData['tas'] * weiInfo['lowTop']
            #             , middleTopData['tas'] * weiInfo['middleTop']
            #             , highTopData['tas'] * weiInfo['highTop']
            #         ]
            #     )
            #
            #     tmpData = tmpData.rename( { 'tas' : modelInfo } )
            #     # dsData = dsData.merge(tmpData)
            #
            #     # 파일 저장 시 오래 걸림
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     try:
            #         tmpData.to_netcdf(saveFile)
            #     except Exception as e:
            #         log.error('Exception : {}'.format(e))
            #     log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # # 모델 가중치 파일 읽기
            # inpFile = '{}/{}/*_tas.nc'.format(globalVar['outPath'], serviceName)
            # fileList = sorted(glob.glob(inpFile))
            # log.info('[CHECK] fileList : {}'.format(fileList))
            #
            # dsData = xr.open_mfdataset(fileList)
            #
            # dsSumData = dsData.to_array().sum('variable', skipna = True)
            # lon1D = sorted(set(dsData['lon'].values))
            # lat1D = sorted(set(dsData['lat'].values))
            # time1D = sorted(set(dsData['time'].values))
            #
            # dsDataL1 =  xr.Dataset(
            #     {
            #         'tas': (('time', 'lat', 'lon'), (dsSumData.values).reshape(len(time1D), len(lat1D), len(lon1D)))
            #     }
            #     , coords = {
            #         'lat' : lat1D
            #         , 'lon' : lon1D
            #         , 'time' : time1D
            #     }
            # )
            #
            # data = dsDataL1
            # saveFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'tas_MME.nc')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # data.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # dataL1 = makeLandMask(data)['resData']
            # # 육지에 대한 강수량
            # dataL2 = dataL1.where(dataL1['isLand'] == True).sel(lat = slice(-60, 90))
            #
            # saveFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'tas_MME_land.nc')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # dataL2.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # # sys.exit(0)
            #
            # # ************************************************************************************************
            # # 2) 과거기간대비 변화율을 산정하려고합니다. (1980년-2014년) 미래 (2031-2065)/ (2066-2100) 입니다.
            # # 변화율 = (미래 - 현재)/현재 * 100
            # # 현재 데이터 없음
            # # ************************************************************************************************
            inpFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'tas_MME_land.nc')
            fileList = sorted(glob.glob(inpFile))
            dataL2 = xr.open_mfdataset(fileList)

            inpFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'tas-hist_MME_land.nc')
            fileList = sorted(glob.glob(inpFile))
            inpHistData = xr.open_mfdataset(fileList)

            dataL3 = xr.merge( [ dataL2, inpHistData ] )

            resData = makeLandMask(dataL3)
            lon1D = resData['lon1D']
            lat1D = resData['lat1D']
            isLand2D = resData['isLand2D']

            # 현재 (1980년-2014년) : 자료 없음
            nowData = dataL3.sel(time = slice('1980-01', '2014-12'))

            # 위경도에 따른 연별 합계
            # nowMean = nowData['tas'].groupby('time.year').sum(skipna = True)

            # 위경도에 따른 평균 수행
            nowMean = nowData['tas'].groupby('time.year').mean(skipna = True).mean(dim = ['year'], skipna = True)
            # nowMean = nowData['tas'].groupby('time.year').mean(skipna = True)

            # 평균 수행
            # nowMean = nowData['tas'].groupby('time.year').sum(skipna = True).mean(skipna = True)

            # 현재 봄 (3, 4, 5)
            nowSpringData = nowData.sel(time = nowData.time.dt.month.isin([3, 4, 5]))
            nowSpringMean = nowSpringData['tas'].groupby('time.year').mean(skipna = True).mean(dim = ['year'], skipna = True)

            # 현재 여름 (6, 7, 8)
            nowSumerData = nowData.sel(time = nowData.time.dt.month.isin([6, 7, 8]))
            nowSumerMean = nowSumerData['tas'].groupby('time.year').mean(skipna = True).mean(dim = ['year'], skipna = True)

            # 현재 가을 (9, 10, 11)
            nowFailData = nowData.sel(time = nowData.time.dt.month.isin([9, 10, 11]))
            nowFailMean = nowFailData['tas'].groupby('time.year').mean(skipna = True).mean(dim = ['year'], skipna = True)

            # 현재 겨울 (12, 1, 2)
            nowWntrData = nowData.sel(time = nowData.time.dt.month.isin([1, 2, 12]))
            nowWntrMean = nowWntrData['tas'].groupby('time.year').mean(skipna = True).mean(dim = ['year'], skipna = True)

            # 미래 (2031-2065)
            # nextData = dataL3.sel(time = slice('2031-01', '2065-12'))
            # 테스트
            nextData = dataL3.sel(time = slice('2014-01', '2014-12'))
            nextMean = nextData['tas'].groupby('time.year').mean(skipna = True).mean(dim = ['year'], skipna = True)

            # 미래 봄 (3, 4, 5)
            nextSpringData = nextData.sel(time = nextData.time.dt.month.isin([3, 4, 5]))
            nextSpringMean = nextSpringData['tas'].groupby('time.year').mean(skipna = True).mean(dim = ['year'], skipna = True)

            # 미래 여름 (6, 7, 8)
            nextSumerData = nextData.sel(time=nextData.time.dt.month.isin([6, 7, 8]))
            nextSumerMean = nextSumerData['tas'].groupby('time.year').mean(skipna=True).mean(dim = ['year'], skipna=True)

            # 미래 가을 (9, 10, 11)
            nextFailData = nextData.sel(time = nextData.time.dt.month.isin([9, 10, 11]))
            nextFailMean = nextFailData['tas'].groupby('time.year').mean(skipna = True).mean(dim = ['year'], skipna = True)

            # 미래 겨울 (12, 1, 2)
            nextWntrData = nextData.sel(time=nextData.time.dt.month.isin([1, 2, 12]))
            nextWntrMean = nextWntrData['tas'].groupby('time.year').mean(skipna=True).mean(dim = ['year'], skipna=True)

            # 먼 미래 (2066-2100)
            # futData = dataL3.sel(time = slice('2066-01', '2100-12'))
            futData = dataL3.sel(time = slice('2015-01', '2015-12'))
            futMean = futData['tas'].groupby('time.year').mean(skipna = True).mean(dim = ['year'], skipna = True)

            # 미래 봄 (3, 4, 5)
            futSpringData = futData.sel(time = futData.time.dt.month.isin([3, 4, 5]))
            futSpringMean = futSpringData['tas'].groupby('time.year').mean(skipna = True).mean(dim = ['year'], skipna = True)

            # 먼 미래 여름 (6, 7, 8)
            futSumerData = futData.sel(time=futData.time.dt.month.isin([6, 7, 8]))
            futSumerMean = futSumerData['tas'].groupby('time.year').mean(skipna=True).mean(dim = ['year'], skipna=True)

            # 미래 가을 (9, 10, 11)
            futFailData = futData.sel(time = futData.time.dt.month.isin([9, 10, 11]))
            futFailMean = futFailData['tas'].groupby('time.year').mean(skipna = True).mean(dim = ['year'], skipna = True)

            # 먼 미래 겨울 (12, 1, 2)
            futWntrData = futData.sel(time=futData.time.dt.month.isin([1, 2, 12]))
            futWntrMean = futWntrData['tas'].groupby('time.year').mean(skipna=True).mean(dim = ['year'], skipna=True)

            # 육해상에 대한 강수량
            VarRatData = xr.Dataset(
                {
                    # 일반 통계
                    'nowMean': (('lat', 'lon'), (nowMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'nowSpringMean': (('lat', 'lon'), (nowSpringMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'nowSumerMean': (('lat', 'lon'), (nowSumerMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'nowFailMean': (('lat', 'lon'), (nowFailMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'nowWntrMean': (('lat', 'lon'), (nowWntrMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'nextMean': (('lat', 'lon'), (nextMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'nextSpringMean': (('lat', 'lon'), (nextSpringMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'nextSumerMean': (('lat', 'lon'), (nextSumerMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'nextFailMean': (('lat', 'lon'), (nextFailMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'nextWntrMean': (('lat', 'lon'), (nextWntrMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'futMean': (('lat', 'lon'), (futMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'futSpringMean': (('lat', 'lon'), (futSpringMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'futSumerMean': (('lat', 'lon'), (futSumerMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'futFailMean': (('lat', 'lon'), (futFailMean.values).reshape(len(lat1D), len(lon1D)))
                    , 'futWntrMean': (('lat', 'lon'), (futWntrMean.values).reshape(len(lat1D), len(lon1D)))

                    # 변화율
                    , 'Annual-P1': (('lat', 'lon'), ( ( (nextMean - nowMean) / nowMean ).values * 100.0 ).reshape(len(lat1D), len(lon1D)))
                    , 'Spring-P1': (('lat', 'lon'), ( ( (nextSpringMean - nowSpringMean) / nowSpringMean ).values * 100.0 ).reshape(len(lat1D), len(lon1D)))
                    , 'Sumer-P1': (('lat', 'lon'), ( ( (nextSumerMean - nowSumerMean) / nowSumerMean ).values * 100.0 ).reshape(len(lat1D), len(lon1D)))
                    , 'Fail-P1': (('lat', 'lon'), ( ( (nextFailMean - nowFailMean) / nowFailMean ).values * 100.0 ).reshape(len(lat1D), len(lon1D)))
                    , 'Wntr-P1': (('lat', 'lon'), ( ( (nextWntrMean - nowWntrMean) / nowWntrMean ).values * 100.0 ).reshape(len(lat1D), len(lon1D)))

                    , 'Annual-P3': (('lat', 'lon'), ( ( (futMean - nowMean) / nowMean ).values * 100.0 ).reshape(len(lat1D), len(lon1D)))
                    , 'Spring-P3': (('lat', 'lon'), ( ( (futSpringMean - nowSpringMean) / nowSpringMean ).values * 100.0 ).reshape(len(lat1D), len(lon1D)))
                    , 'Sumer-P3': (('lat', 'lon'), ( ( (futSumerMean - nowSumerMean) / nowSumerMean ).values * 100.0 ).reshape(len(lat1D), len(lon1D)))
                    , 'Fail-P3': (('lat', 'lon'), ( ( (futFailMean - nowFailMean) / nowFailMean ).values * 100.0 ).reshape(len(lat1D), len(lon1D)))
                    , 'Wntr-P3': (('lat', 'lon'), ( ( (futWntrMean - nowWntrMean) / nowWntrMean ).values * 100.0 ).reshape(len(lat1D), len(lon1D)))

                    # 육해상
                    , 'isLand': (('lat', 'lon'), (isLand2D).reshape(len(lat1D), len(lon1D)))
                }
                , coords={
                    'lat': lat1D
                    , 'lon': lon1D
                }
            )

            saveFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'varRat-tas.nc')
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            VarRatData.to_netcdf(saveFile)
            log.info('[CHECK] saveFile : {}'.format(saveFile))

            # 육지에 대한 강수량
            VarRatDataL1 = VarRatData.where(VarRatData['isLand'] == True).sel(lat=slice(-60, 90))

            saveFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'varRat-tas_land.nc')
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            VarRatDataL1.to_netcdf(saveFile)
            log.info('[CHECK] saveFile : {}'.format(saveFile))

            # ************************************************************************************************
            # 3) 연 총 강수량과 연 평균 온도를 구하려고합니다.
            # ************************************************************************************************
            # +++++++++++++++++++++++++++++++++++++++++++++++
            # 평균 온도
            # +++++++++++++++++++++++++++++++++++++++++++++++
            inpFile = '{}/{}/*_tas.nc'.format(globalVar['outPath'], serviceName)
            fileList = sorted(glob.glob(inpFile))
            log.info('[CHECK] fileList : {}'.format(fileList))
            dsData = xr.open_mfdataset(fileList)

            inpFile = '{}/{}/*_hist-tas.nc'.format(globalVar['outPath'], serviceName)
            fileList = sorted(glob.glob(inpFile))
            log.info('[CHECK] fileList : {}'.format(fileList))
            dsHistData = xr.open_mfdataset(fileList)

            inpFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'tas_MME.nc')
            fileList = sorted(glob.glob(inpFile))
            log.info('[CHECK] fileList : {}'.format(fileList))
            dataL2 = xr.open_mfdataset(fileList)

            inpFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, 'tas-hist_MME.nc')
            fileList = sorted(glob.glob(inpFile))
            log.info('[CHECK] fileList : {}'.format(fileList))
            inpHistData = xr.open_mfdataset(fileList)

            inpFile = '{}/{}/OBS/{}'.format(globalVar['inpPath'], serviceName, 'CRU TEMP OBS.nc')
            fileList = sorted(glob.glob(inpFile))
            log.info('[CHECK] fileList : {}'.format(fileList))
            # obsData = xr.open_mfdataset(fileList).sel(time = slice('1950-01', '2014-12'))
            obsData = xr.open_mfdataset(fileList).sel(time = slice('2013-01', '2015-12'))
            obsDataL1 = obsData.rename( { 'tmp' : 'obs' } )

            # 데이터 융합
            dataL3 = xr.merge( [ dsData, dsHistData, dataL2, inpHistData, obsDataL1 ] )
            # dataL3 = xr.merge( [ dsHistData, dataL2, inpHistData, obsDataL1 ] )

            # sumData = dataL3['tas'].groupby('time.year').sum(skipna = True)
            # sumData = dataL3.groupby('time.year').sum(skipna = True)
            statData = dataL3.groupby('time.year').mean(skipna = True)
            resData = makeLandMaskYear(statData)

            # 연도별 평균 온도 (육해상)
            statDataL1 = resData['resData']

            # saveFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'sum_pr')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # sumDataL1.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

            # 연도별 평균 온도 (육지)
            statDataL2 = statDataL1.where(statDataL1['isLand'] == True).sel(lat=slice(-60, 90))

            # saveFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'sum_pr_land')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # sumDataL2.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

            # 연도별 평균 온도의 평균 (육해상)
            # meanSumData = sumData.mean(dim = ['year'], skipna=True)

            # saveFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'mean-sum_pr')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # meanSumData.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

            # 연도별 평균 온도의 평균 (육해상)
            meanStatDataL2 = statDataL2.mean(dim = ['year'], skipna=True)
            # meanSumDataL2 = sumDataL2.mean(dim = ['year'], skipna=True).sel(year = slice('2031', '2065')).mean(dim = ['year'], skipna = True)

            # 미래 (2031-2065)
            nextData = statDataL2.sel(year = slice('2031', '2065'))
            nextMeanData = nextData.rename( { 'tas' : 'next' } )['next'].mean(dim = ['year'], skipna = True)

            # 먼 미래 (2066-2100)
            futData = statDataL2.sel(year = slice('2066', '2100'))
            futMeanData = futData.rename( { 'tas' : 'fut' } )['fut'].mean(dim = ['year'], skipna = True)

            meanSumDataL3 = xr.merge( [ meanStatDataL2, nextMeanData, futMeanData ] )

            # fut, next에 대한 연도별 평균 온도의 평균
            saveFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'mean-mean_tas_land')
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            meanSumDataL3.to_netcdf(saveFile)
            log.info('[CHECK] saveFile : {}'.format(saveFile))

            # 위도별 평균 온도의 평균
            meanSumLatData = statDataL2.mean(dim = ['lon'], skipna=True)

            saveFile = '{0}/{1}/{1}_{2}.xlsx'.format(globalVar['outPath'], serviceName, 'mean-mean-lat_tas_land')
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            writer = pd.ExcelWriter(saveFile, engine='openpyxl')

            varList = list(meanSumLatData.data_vars)
            for j, varInfo in enumerate(varList):
                log.info('[CHECK] varInfo : {}'.format(varInfo))

                meanSumLatDataL1 = meanSumLatData[varInfo].to_dataframe().reset_index()
                meanSumLatDataL2 = meanSumLatDataL1.pivot(index='lat', columns='year')

                meanSumLatDataL2.to_excel(writer, sheet_name=varInfo, index=True)

            writer.save()
            log.info('[CHECK] saveFile : {}'.format(saveFile))

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
        inParams = { }

        print("[CHECK] inParams : {}".format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
