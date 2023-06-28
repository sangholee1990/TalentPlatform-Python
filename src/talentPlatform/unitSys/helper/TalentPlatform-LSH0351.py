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

from global_land_mask import globe

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
    # Python을 이용한 증발산 2종 결과에 대한 과거대비 변화율 산정, 통계 계산

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/pycharm_project_83'
        # contextPath = os.getcwd() if env in 'local' else '/home/dxinyu/SYSTEMS/PROG/PYTHON'

    prjName = 'test'
    serviceName = 'LSH0351'

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
                    'srtDate': '1990-01-01'
                    , 'endDate': '2022-01-01'

                    # 경도 최소/최대/간격
                    , 'lonMin': -180
                    , 'lonMax': 180
                    , 'lonInv': 0.1
                    # , 'lonInv': 5

                    # 위도 최소/최대/간격
                    , 'latMin': -90
                    , 'latMax': 90
                    , 'latInv': 0.1
                    # , 'latInv': 5
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': globalVar['srtDate']
                    # , 'endDate': globalVar['endDate']
                    'srtDate': '2019-01-01'
                    , 'endDate': '2023-01-01'

                    # 경도 최소/최대/간격
                    , 'lonMin': -180
                    , 'lonMax': 180
                    , 'lonInv': 0.1

                    # 위도 최소/최대/간격
                    , 'latMin': -90
                    , 'latMax': 90
                    , 'latInv': 0.1
                }

                # globalVar['inpPath'] = '/data3/dxinyu/graced/Emissionmaps'
                # globalVar['outPath'] = '/data1/dxinyu/TEST/OUTPUT'
                # globalVar['figPath'] = '/DATA/FIG'

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

                # globalVar['inpPath'] = 'G:/Climate variables/PET/OUTPUT'
                # globalVar['outPath'] = 'G:/Climate variables/PET/OUTPUT'

            # 모형 설정
            # modelList = ['har_ACCESS-CM2', 'har_ACCESS-ESM1-5', 'OBS']
            # modelList = ['har_ACCESS-CM2', 'har_ACCESS-ESM1-5']
            modelList = ['har_ACCESS-CM2', 'har_ACCESS-ESM1-5']
            keyList = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

            # ************************************************************************************************
            # 2) 과거기간대비 변화율을 산정하려고합니다. (1980년-2014년) 미래 (2031-2065)/ (2066-2100) 입니다.
            # 변화율 = (미래 - 현재)/현재 * 100
            # 현재 데이터 없음
            # ************************************************************************************************
            for k, keyInfo in enumerate(keyList):
                log.info('[CHECK] keyInfo : {}'.format(keyInfo))

                for i, modelInfo in enumerate(modelList):
                    # inpFile = '{}/{}/*{}*.nc'.format(globalVar['inpPath'], 'LSH0346', modelInfo)
                    inpKeyFile = '{}/{}/*{} {}*.nc'.format(globalVar['inpPath'], 'LSH0346', modelInfo, keyInfo)
                    fileKeyList = sorted(glob.glob(inpKeyFile))

                    inpRefFile = '{}/{}/*{}_eto.nc'.format(globalVar['inpPath'], 'LSH0346', modelInfo)
                    fileRefList = sorted(glob.glob(inpRefFile))

                    fileList = fileKeyList + fileRefList

                    if fileList is None or len(fileList) < 1:
                        log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

                    # fileInfo = fileList[0]
                    dataL2 = xr.Dataset()
                    for j, fileInfo in enumerate(fileList):

                        log.info('[CHECK] fileInfo : {}'.format(fileInfo))
                        data = xr.open_dataset(fileInfo)

                        try:
                            for k, varInfo in enumerate(data.data_vars.keys()):
                                varInfoOrg = varInfo
                                varInfoNew = varInfo.replace(' ssp126', '')

                            data = data.rename( { varInfoOrg : varInfoNew } )
                        except Exception as e:
                            log.error('Exception : {}'.format(e))

                        dataL2 = xr.merge( [dataL2, data] )

                    for j, varInfo in enumerate(dataL2.data_vars.keys()):

                        log.info('[CHECK] varInfo : {}'.format(varInfo))
                        dataL3 = dataL2[varInfo]

                        resData = makeLandMask(dataL3)
                        lon1D = resData['lon1D']
                        lat1D = resData['lat1D']
                        isLand2D = resData['isLand2D']

                        # 현재 (1980년-2014년) : 자료 없음
                        nowData = dataL3.sel(time = slice('1980-01', '2014-12'))

                        # 위경도에 따른 연별 합계
                        # nowMean = nowData['pr'].groupby('time.year').sum(skipna = True)

                        # 위경도에 따른 평균 수행
                        nowMean = nowData.groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)
                        # nowMean = nowData['pr'].groupby('time.year').mean(skipna = True)

                        # 평균 수행
                        # nowMean = nowData['pr'].groupby('time.year').sum(skipna = True).mean(skipna = True)

                        # 현재 봄 (3, 4, 5)
                        nowSpringData = nowData.sel(time = nowData.time.dt.month.isin([3, 4, 5]))
                        nowSpringMean = nowSpringData.groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)
                        # nowSpringMean = nowSpringData['pr'].groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)

                        # 현재 여름 (6, 7, 8)
                        nowSumerData = nowData.sel(time = nowData.time.dt.month.isin([6, 7, 8]))
                        nowSumerMean = nowSumerData.groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)
                        # nowSumerMean = nowSumerData['pr'].groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)

                        # 현재 가을 (9, 10, 11)
                        nowFailData = nowData.sel(time = nowData.time.dt.month.isin([9, 10, 11]))
                        nowFailMean = nowFailData.groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)
                        # nowFailMean = nowFailData['pr'].groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)

                        # 현재 겨울 (12, 1, 2)
                        nowWntrData = nowData.sel(time = nowData.time.dt.month.isin([1, 2, 12]))
                        nowWntrMean = nowWntrData.groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)
                        # nowWntrMean = nowWntrData['pr'].groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)

                        # 미래 (2031-2065)
                        nextData = dataL3.sel(time = slice('2031-01', '2065-12'))
                        nextMean = nextData.groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)
                        # nextMean = nextData['pr'].groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)

                        # 미래 봄 (3, 4, 5)
                        nextSpringData = nextData.sel(time = nextData.time.dt.month.isin([3, 4, 5]))
                        nextSpringMean = nextSpringData.groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)
                        # nextSpringMean = nextSpringData['pr'].groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)

                        # 미래 여름 (6, 7, 8)
                        nextSumerData = nextData.sel(time=nextData.time.dt.month.isin([6, 7, 8]))
                        nextSumerMean = nextSumerData.groupby('time.year').sum(skipna=True).mean(dim = ['year'], skipna=True)
                        # nextSumerMean = nextSumerData['pr'].groupby('time.year').sum(skipna=True).mean(dim = ['year'], skipna=True)

                        # 미래 가을 (9, 10, 11)
                        nextFailData = nextData.sel(time = nextData.time.dt.month.isin([9, 10, 11]))
                        nextFailMean = nextFailData.groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)
                        # nextFailMean = nextFailData['pr'].groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)

                        # 미래 겨울 (12, 1, 2)
                        nextWntrData = nextData.sel(time=nextData.time.dt.month.isin([1, 2, 12]))
                        nextWntrMean = nextWntrData.groupby('time.year').sum(skipna=True).mean(dim = ['year'], skipna=True)
                        # nextWntrMean = nextWntrData['pr'].groupby('time.year').sum(skipna=True).mean(dim = ['year'], skipna=True)

                        # 먼 미래 (2066-2100)
                        futData = dataL3.sel(time = slice('2066-01', '2100-12'))
                        futMean = futData.groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)
                        # futMean = futData['pr'].groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)

                        # 미래 봄 (3, 4, 5)
                        futSpringData = futData.sel(time = futData.time.dt.month.isin([3, 4, 5]))
                        futSpringMean = futSpringData.groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)
                        # futSpringMean = futSpringData['pr'].groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)

                        # 먼 미래 여름 (6, 7, 8)
                        futSumerData = futData.sel(time=futData.time.dt.month.isin([6, 7, 8]))
                        futSumerMean = futSumerData.groupby('time.year').sum(skipna=True).mean(dim = ['year'], skipna=True)
                        # futSumerMean = futSumerData['pr'].groupby('time.year').sum(skipna=True).mean(dim = ['year'], skipna=True)

                        # 미래 가을 (9, 10, 11)
                        futFailData = futData.sel(time = futData.time.dt.month.isin([9, 10, 11]))
                        futFailMean = futFailData.groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)
                        # futFailMean = futFailData['pr'].groupby('time.year').sum(skipna = True).mean(dim = ['year'], skipna = True)

                        # 먼 미래 겨울 (12, 1, 2)
                        futWntrData = futData.sel(time=futData.time.dt.month.isin([1, 2, 12]))
                        futWntrMean = futWntrData.groupby('time.year').sum(skipna=True).mean(dim = ['year'], skipna=True)
                        # futWntrMean = futWntrData['pr'].groupby('time.year').sum(skipna=True).mean(dim = ['year'], skipna=True)

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

                        saveFile = '{}/{}/{}-{}-{}.nc'.format(globalVar['outPath'], serviceName, varInfo, keyInfo, 'varRat')
                        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                        VarRatData.to_netcdf(saveFile)
                        log.info('[CHECK] saveFile : {}'.format(saveFile))

                        # 육지에 대한 강수량
                        VarRatDataL1 = VarRatData.where(VarRatData['isLand'] == True).sel(lat=slice(-60, 90))

                        saveFile = '{}/{}/{}-{}-{}.nc'.format(globalVar['outPath'], serviceName, varInfo, keyInfo, 'varRat_land')
                        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                        VarRatDataL1.to_netcdf(saveFile)
                        log.info('[CHECK] saveFile : {}'.format(saveFile))

                    # ************************************************************************************************
                    # 3) 연 총 강수량과 연 평균 온도를 구하려고합니다.
                    # ************************************************************************************************
                    # +++++++++++++++++++++++++++++++++++++++++++++++
                    # 평균 온도
                    # +++++++++++++++++++++++++++++++++++++++++++++++
                    for j, varInfo in enumerate(dataL2.data_vars.keys()):

                        log.info('[CHECK] varInfo : {}'.format(varInfo))
                        dataL3 = dataL2[varInfo]

                        # sumData = dataL3['tas'].groupby('time.year').sum(skipna = True)
                        # sumData = dataL3.groupby('time.year').sum(skipna = True)
                        statData = dataL3.groupby('time.year').mean(skipna=True)
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
                        meanStatDataL2 = statDataL2.mean(dim=['year'], skipna=True)
                        # meanSumDataL2 = sumDataL2.mean(dim = ['year'], skipna=True).sel(year = slice('2031', '2065')).mean(dim = ['year'], skipna = True)

                        # 미래 (2031-2065)
                        nextData = statDataL2.sel(year=slice('2031', '2065'))
                        nextMeanData = nextData.rename( { varInfo : 'next' } )['next'].mean(dim=['year'], skipna=True)

                        # 먼 미래 (2066-2100)
                        futData = statDataL2.sel(year=slice('2066', '2100'))
                        futMeanData = futData.rename( { varInfo : 'fut' } )['fut'].mean(dim=['year'], skipna=True)

                        meanSumDataL3 = xr.merge([meanStatDataL2, nextMeanData, futMeanData])

                        # fut, next에 대한 연도별 평균 온도의 평균
                        saveFile = '{}/{}/{}-{}-{}.nc'.format(globalVar['outPath'], serviceName, varInfo, keyInfo, 'mean-mean_tas_land')
                        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                        meanSumDataL3.to_netcdf(saveFile)
                        log.info('[CHECK] saveFile : {}'.format(saveFile))

                        # 위도별 평균 온도의 평균
                        meanSumLatData = statDataL2.mean(dim=['lon'], skipna=True)

                        saveFile = '{}/{}/{}-{}-{}.xlsx'.format(globalVar['outPath'], serviceName, varInfo, keyInfo, 'mean-mean-lat_tas_land')
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