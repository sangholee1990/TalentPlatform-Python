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
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta, date
import matplotlib.dates as mdates
import datetime as dt

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
# 4.4. 비즈니스 로직 수행 : 단위 시스템 (unit 파일명)으로 관리하거나 로직 구현

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


#  초기 전달인자 설정
def initArgument(globalVar, inParams):

    for i, key in enumerate(inParams):
        # 리눅스 환경
        if globalVar['sysOs'] in 'Linux':
            if i >= len(sys.argv[1:]): continue
            if inParams[key] is None: continue
            val = inParams[key] if sys.argv[i + 1] is None else sys.argv[i + 1]

        # 원도우 또는 맥 환경
        if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
            if inParams[key] is None: continue
            val = inParams[key]

        # self 변수에 할당
        # setattr(self, key, val)

        # 전역 변수에 할당
        globalVar[key] = val
        log.info("[CHECK] {} / val : {}".format(key, val))

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

        result = {'msg': 'succ'}

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
def makeHgtTimePlot(sysOpt, data, key, isSetVal=None, levelVal = None):

    log.info('[START] {}'.format('makeHgtTimePlot'))

    if isSetVal is None: isSetVal = False
    if levelVal is None: levelVal = 20

    result = None

    try:
        timeList = data.time.data

        dtDateTime = pd.to_datetime(timeList).strftime('%Y%m%d%H%M')
        saveImg = '{}/{}_{}_{}-{}.png'.format(globalVar['figPath'], serviceName, key, min(dtDateTime), max(dtDateTime))

        # data.plot()
        # data.contourf(levels=40)
        if (isSetVal == True):
            data.plot.contourf(levels=levelVal, vmin=sysOpt['minVal'], vmax=sysOpt['maxVal'])
        else:
            data.plot.contourf(levels=levelVal)

        plt.gca().invert_yaxis()
        plt.xlabel('Date [Day-Month]')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        plt.yscale('log')
        plt.ylabel('hPa')
        plt.yticks([10, 100, 500, 1000], labels=[10, 100, 500, 1000])

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

# 단위시스템
def unitSweDataProcVis():

    log.info('[START] {}'.format('unitSweDataProcVis'))

    result = None

    try:

        inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'swe/1980/01/*swe*.nc')

        fileList = glob.glob(inpFile)
        if fileList is None or len(fileList) < 1:
            log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

        ds = xr.open_mfdataset(fileList)
        dsSwe = ds['swe']

        # 0-500 값만 처리
        dsSwe = dsSwe.where((0 < dsSwe) & (dsSwe < 500))

        timeList = dsSwe.time.data

        # 단일 코어
        for timeInfo in timeList:
            rtnInfo = makeMapPlot(dsSwe, timeInfo)
            log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

        # 다중 코어 프로세스
        # rtnInfo = Parallel(n_jobs=mp.cpu_count())([delayed(makeMapPlot)(dsSwe, timeInfo) for timeInfo in timeList])
        # log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

        result = {'msg': 'succ'}

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('unitSweDataProcVis'))


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 xarray 데이터 처리 및 시각화

    # grib이 아니라 nc로 드릴께요.
    # 제가 짧은 기간(1963년-1973년 )에 대해서만 드릴테니 테스트 해보시고 제가 전기간에 돌리면 될것 같습니다.
    # 의뢰내용에 약간 수정사항이 있는데 이게 일자료라서 5일 평균장으로 그려주시면 될것같아요,
    # 4월20일부터-6월5일까지 기간데 대해서, 지위고도를 5일moving average로 계산해서 예제 그림처럼 나타내면 될것 같습니다,
    # 예를 들어 4월 20-25일 평균을 4월 20일, 4월 21-26일 평균을 4월 21일 이런식으로 해서 마지막 날짜는 6월 5일까지만 그려지게요.
    # 자료 확인해보시고 연락주세요

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0260'

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
            log.info("[CHECK] inParams : {}".format(inParams))

            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

            for key, val in globalVar.items():
                log.info("[CHECK] globalVar[{}] {}".format(key, val))

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

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': '1963-04-20'
                , 'endDate': '1963-06-05'

                # 이동평균 정보
                , 'movAvgInfo': 5

                # 영역 위도
                , 'srtLat': 65
                , 'endLat': 90

                # 특정 위도, 경도
                , 'posLat': 70
                , 'posLon': 200

                # 값 설정
                , 'minVal': 0
                , 'maxVal': 35000
            }

            dtSrtDate = (pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d') - timedelta(days=sysOpt['movAvgInfo'] - 1)).strftime("%Y-%m-%d")
            dtEndDate = (pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')).strftime("%Y-%m-%d")

            dtSrtJul = (pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d') - timedelta(days=sysOpt['movAvgInfo'] - 1)).dayofyear
            dtEndJul = (pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')).dayofyear

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ght/hgt*.nc')

            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

            # 자료 읽기
            ds = xr.open_mfdataset(fileList)
            dsData = ds

            # inpFile 경로에서 선정한 쥴리안 데이를 통해 자료 정제
            # 즉 쥴리안 데이 (106-156)의 경우에서만 수행
            # 만약에 특정 연도를 수행할 경우 반복문에서 조건문 필요
            # dsData = xr.Dataset()
            # timeList = ds['time'].data
            #
            # for timeInfo in timeList:
            #     getJul = pd.to_datetime(timeInfo).dayofyear
            #
            #     if (dtSrtJul > getJul or getJul > dtEndJul): continue
            #     getDsData = ds.sel(time=timeInfo)
            #
            #     if (len(dsData) < 1):
            #         dsData = getDsData
            #     else:
            #         dsData = xr.concat([dsData, getDsData], "time")

            # 위도 65-90으로 설정
            dsDataL1 = dsData.sel(
                lat = slice(sysOpt['endLat'], sysOpt['srtLat'])
            )

            # 데이터 읽기
            hgtData = dsDataL1['hgt']

            # 5일 이동평균
            hgtDataL1 = hgtData.rolling(time=sysOpt['movAvgInfo']).mean().dropna('time')

            # [기후] 기압에 따른 평균
            clmMean = hgtDataL1.mean(['lon', 'lat', 'time'])

            # [기후] 기압에 따른 표준편차
            clmStd = hgtDataL1.std(['lon', 'lat', 'time'])

            # [현재] 기압 및 시간에 따른 평균
            hgtDataL2 = hgtDataL1.sel(
                time=slice(dtSrtDate, dtEndDate)
            ).mean(['lon', 'lat']).transpose('level', 'time')

            # [현재] 기압에 따른 평균
            preMean = hgtDataL2.mean(['time'])

            # [현재] 기압에 따른 표준편차
            preStd = hgtDataL2.std(['time'])

            # [현재] 정규화
            preData = (hgtDataL2 - preMean) / preStd

            # [기후] 정규화
            clmData = (hgtDataL2 - clmMean) / clmStd

            # 아노말리
            anaData = preData - clmData
            rtnInfo = makeHgtTimePlot(sysOpt, anaData, 'pos-normal', False, 40)
            log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

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
        # 입력 자료 : inpPath
        # 그림 자료 : figPath
        # 출력 자료 : outPath
        # 로그 자료 : logPath
        inParams = {
            # 'inpPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
            # , 'figPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
            # , 'outPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
            # , 'logPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
        }

        print("[CHECK] inParams : {}".format(inParams))

        # ================================================
        # 4. 부 프로그램
        # ================================================
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
