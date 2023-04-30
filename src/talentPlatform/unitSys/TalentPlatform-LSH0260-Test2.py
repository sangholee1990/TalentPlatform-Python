# -*- coding: utf-8 -*-
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
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
import multiprocessing as mp
import warnings
from datetime import timedelta, date
import matplotlib.dates as mdates

# =================================================
# 사용자 매뉴얼
# =================================================
# [소스 코드의 실행 순서]
# 1. 초기 설정 : 폰트 설정
# 2. 유틸리티 함수 : 초기화 함수 (로그 설정, 초기 변수, 초기 전달인자 설정) 또는 자주 사용하는 함수
# 3. 주 프로그램

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

    saveLogFile = "{}/{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
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

# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):

    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
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

    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)
        globalVar[key] = val.replace('\\', '/')

    return globalVar

# 맵 시각화
def makeMapPlot(dsSwe, timeInfo):

    log.info('[START] {}'.format('makeMapPlot'))

    result = None

    try:
        log.info('[CHECK] timeInfo : {}'.format(timeInfo))

        getDsSwe = dsSwe.sel(time=timeInfo)

        latList = getDsSwe.lat.data
        lonList = getDsSwe.lon.data
        sweList = getDsSwe.data

        dtDateTime = pd.to_datetime(timeInfo)

        saveImg = '{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, 'swe', dtDateTime.strftime('%Y%m%d%H%M'))

        plt.pcolormesh(lonList, latList, sweList, cmap='coolwarm')
        plt.colorbar(orientation="vertical")
        plt.title(dtDateTime.strftime('%Y-%m-%d %H:%M'))
        plt.savefig(saveImg, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeMapPlot'))

# 맵 시각화
def makeHgtMapPlot(data, timeInfo, levelInfo, key):

    log.info('[START] {}'.format('makeHgtMapPlot'))

    result = None

    try:
        log.info('[CHECK] timeInfo : {}'.format(timeInfo))
        log.info('[CHECK] levelInfo : {}'.format(levelInfo))

        getData = data.sel(time=timeInfo, level=levelInfo)

        latList = getData.lat.data
        lonList = getData.lon.data
        valList = getData.data

        dtDateTime = pd.to_datetime(timeInfo)
        mainTitle = '{}_{}'.format(dtDateTime.strftime('%Y-%m-%d %H:%M'), levelInfo)

        saveImg = '{}/{}_{}_{}_{}.png'.format(globalVar['figPath'], serviceName, key, dtDateTime.strftime('%Y%m%d%H%M'), levelInfo)

        plt.pcolormesh(lonList, latList, valList, cmap='coolwarm')
        plt.colorbar(orientation="vertical")
        plt.title(mainTitle)
        plt.savefig(saveImg, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeHgtMapPlot'))


# 맵 시각화
def makeHgtTimePlot(sysOpt, data, key):

    log.info('[START] {}'.format('makeHgtTimePlot'))

    result = None

    try:
        timeList = data.time.data

        dtDateTime = pd.to_datetime(timeList).strftime('%Y%m%d%H%M')
        saveImg = '{}/{}_{}_{}-{}.png'.format(globalVar['figPath'], serviceName, key, min(dtDateTime), max(dtDateTime))

        # data.plot()
        # data.contourf(levels=40)
        data.plot.contourf(levels=20, vmin=sysOpt['minVal'], vmax=sysOpt['maxVal'])

        plt.gca().invert_yaxis()
        plt.xlabel('Date [Day-Month]')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        plt.yscale('log')
        plt.ylabel('hPa')
        plt.savefig(saveImg, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeHgtTimePlot'))

# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print('[START] {}'.format("main"))

    try:

        global env, contextPath, prjName, serviceName, log, globalVar

        # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
        env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
        # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
        prjName = 'test'
        serviceName = 'LSH0260'

        # 환경 변수 설정 (로그 설정)
        log = initLog(env, contextPath, prjName)

        # 환경 변수 설정 (초기 변수)
        globalVar = initGlobalVar(env, contextPath, prjName)

        for key, val in globalVar.items():
            log.info("[CHECK] globalVar[{}] {}".format(key, val))

        # 비즈니스 로직
        
        # 옵션 설정
        sysOpt = {
            # 시작/종료 시간
            'srtDate': '1963-04-20'
            , 'endDate': '1963-06-05'

            # , 'srtDate': '1963-01-01'
            # , 'endDate': '1963-12-31'

            # 이동평균 정보
            , 'movAvgInfo': 5

            # 특정 위도, 경도
            , 'posLat': -60
            , 'posLon': 200

            # 값 설정
            , 'minVal': 0
            , 'maxVal': 35000
        }

        # [단위 시스템] swe 데이터 처리 및 시각화
        # result = unitSweDataProcVis()
        # log.info('[CHECK] result : {}'.format(result))

        inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ght/hgt*.nc')

        fileList = glob.glob(inpFile)
        if fileList is None or len(fileList) < 1:
            log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

        ds = xr.open_mfdataset(fileList)

        dtSrtDate = (pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d') - timedelta(
            days=sysOpt['movAvgInfo'] - 1)).strftime("%Y-%m-%d")
        dtEndDate = (pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')).strftime("%Y-%m-%d")

        hgtData = ds['hgt']

        # 결측값 자동 제거
        # hgtData = hgtData.where((sys < hgtData) & (hgtData < 35000))

        # 시간 필터
        hgtDataL1 = hgtData.sel(
            time=slice(dtSrtDate, dtEndDate)
        )

        # 시간, 레벨에 따른 맵 시각화
        # timeList = hgtDataL1.time.data
        # levelList = hgtDataL1.level.data
        #
        # for timeInfo in timeList:
        #     for levelInfo in levelList:
        #         rtnInfo = makeHgtMapPlot(hgtDataL1, timeInfo, levelInfo, 'map-ori')
        #         log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

        # 시간을 기준으로 5일 이동평균
        # hgtDataL2 = hgtDataL1.rolling(time=sysOpt['movAvgInfo']).mean().dropna('time')
        hgtDataL2 = hgtDataL1.rolling(time=1).mean().dropna('time')

        # 시간, 레벨에 따른 맵 시각화
        # timeList = hgtDataL2.time.data
        # levelList = hgtDataL2.level.data

        # for timeInfo in timeList:
        #     for levelInfo in levelList:
        #         rtnInfo = makeHgtMapPlot(hgtDataL2, timeInfo, levelInfo, 'map-mv')
        #         log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

        # 특정 지점 (위도, 경도)에 대한 시계열
        hgtDataL3 = hgtDataL2.sel(lat=sysOpt['posLat'], lon=sysOpt['posLon']).transpose('level', 'time')
        rtnInfo = makeHgtTimePlot(sysOpt, hgtDataL3, 'pos')
        log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

        # 위도/경도에 대한 평균
        hgtDataL4 = hgtDataL2.mean(['lon', 'lat']).transpose('level', 'time')
        rtnInfo = makeHgtTimePlot(sysOpt, hgtDataL4, 'mean')
        log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
