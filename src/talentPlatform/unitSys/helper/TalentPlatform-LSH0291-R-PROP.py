# -*- coding: utf-8 -*-

import os
os.environ['PROJ_LIB'] = os.path.join(os.environ["CONDA_PREFIX"], 'share', 'proj')

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

from matplotlib import font_manager, rc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.basemap import Basemap

import numpy as np
import math
from scipy.stats import t
import numpy as np, scipy.stats as st
from matplotlib import colors

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

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
plt.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus'] = False

plt.rc('savefig', dpi = 600, transparent = True)

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


# 맵 시각화
def makeMapPlot(lon2D, lat2D, val2D, mainTitle, saveImg, isCbarLog=None):

    log.info('[START] {}'.format('makeMapPlot'))

    if isCbarLog is None: isCbarLog = False

    result = None

    try:
        plt.figure(figsize=(10, 8))
        map = Basemap(projection="cyl", lon_0=0.0, lat_0=0.0, resolution="c")

        if (isCbarLog == True):
            cs = map.scatter(lon2D, lat2D, c=val2D, s=0.01, marker='s', norm=colors.LogNorm(), cmap=plt.cm.get_cmap('Spectral_r'))
        else:
            # cs = map.scatter(lon2D, lat2D, c=val2D, s=0.05, cmap=plt.cm.get_cmap('Spectral_r'), vmin=0, vmax=30)
            cs = map.scatter(lon2D, lat2D, c=val2D, s=0.01, marker='s', cmap=plt.cm.get_cmap('Spectral_r'))

        map.drawcoastlines()
        map.drawmapboundary()
        map.drawcountries(linewidth=1, linestyle='solid', color='k')
        map.drawmeridians(range(-180, 180, 30), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 0, 1])
        map.drawparallels(range(-90, 90, 30), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 0, 0, 0])

        plt.ylabel(None, fontsize=15, labelpad=35)
        plt.xlabel(None, fontsize=15, labelpad=20)
        cbar = map.colorbar(cs, location='right')
        cbar.set_label(None, fontsize=13)
        plt.title(mainTitle, fontsize=15)
        plt.savefig(saveImg, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result
    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeMapPlot'))


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 한국 월별 수출액과 코스피 종간 간의 회귀분석

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
        contextPath = os.getcwd() if env in 'local' else '/home/dxinyu/TEST'

    prjName = 'test'
    serviceName = 'LSH0291'

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
                    'srtDate': '2019-01-01'
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

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': globalVar['srtDate']
                    # , 'endDate': globalVar['endDate']
                }

            lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
            latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])

            log.info('[CHECK] len(lonList) : {}'.format(len(lonList)))
            log.info('[CHECK] len(latList) : {}'.format(len(latList)))

            keyList = {
                'RMODEL'
            }

            for i, keyInfo in enumerate(keyList):
                log.info("[CHECK] keyInfo : {}".format(keyInfo))

                dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
                dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
                dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1Y')

                # dtIncDateInfo = dtIncDateList[0]
                for j, dtIncDateInfo in enumerate(dtIncDateList):
                    log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))
                    sYear = dtIncDateInfo.strftime('%Y')

                    inpFilePattern = 'R{}_RES.csv'.format(sYear)
                    inpFile = '{}/{}/{}/{}'.format(globalVar['inpPath'], serviceName, keyInfo, inpFilePattern)
                    fileList = sorted(glob.glob(inpFile))

                    if (len(fileList) < 1): continue

                    fileInfo = fileList[0]

                    # 파일 읽기
                    data = pd.read_csv(fileInfo)
                    data.columns = ['lon', 'lat', 'ems', 'flag']

                    # meanVal = np.nanmean(data['ems'].values)
                    # mainTitle = '[{}] {} {} ({:.2f})'.format(sYearMonth, keyInfo, 'ems', meanVal)
                    # saveImg = '{}/{}_{}_{}_{}.png'.format(globalVar['figPath'], serviceName, keyInfo, 'ems', sYearMonth)
                    # rtnInfo = makeMapPlot(data['lon'], data['lat'], data['ems'].values, mainTitle, saveImg, None)
                    # log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

                    # 위도, 경도, 날짜에 따른 합계
                    dataL1 = data.set_index(['lat', 'lon'])
                    dataL2 = dataL1.groupby(by=['lat', 'lon']).sum()
                    dataL3 = dataL2.to_xarray()

                    # dataL4 = dataL3.interp(lat=latList, lon=lonList, method='nearest', kwargs={'fill_value': 'extrapolate'})
                    # dataL4 = dataL3.interp(lat=latList, lon=lonList, kwargs={'fill_value': 'extrapolate'})
                    dataL4 = dataL3.interp(lat=latList, lon=lonList, method='nearest')
                    # dataL4 = dataL3.interp(lat=latList, lon=lonList, method='linear')

                    # dataL4['ems'].plot()
                    # plt.show()

                    lon1D = dataL4['lon'].values
                    lat1D = dataL4['lat'].values
                    # lon2D, lat2D = np.meshgrid(lon1D, lat1D)

                    # meanVal = np.nanmean(dataL4['ems'].values)
                    # mainTitle = '[{}] {} {} ({:.2f})'.format(sYearMonth, keyInfo, 'ems PROP', meanVal)
                    # saveImg = '{}/{}_{}_{}_{}.png'.format(globalVar['figPath'], serviceName, keyInfo, 'ems PROP', sYearMonth)
                    # rtnInfo = makeMapPlot(lon2D, lat2D, dataL4['ems'].values, mainTitle, saveImg, None)
                    # log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

                    # saveFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, serviceName, keyInfo, sYearMonth)
                    # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    # dataL4.to_netcdf(saveFile)
                    # log.info('[CHECK] saveFile : {}'.format(saveFile))

                    saveData = xr.Dataset(
                        {
                            'ems': (('key', 'date', 'lat', 'lon'), (dataL4['ems'].values.reshape(1, 1, len(lat1D), len(lon1D))))
                            , 'flag': (('key', 'date', 'lat', 'lon'), (dataL4['flag'].values.reshape(1, 1, len(lat1D), len(lon1D))))
                        }
                        , coords={
                            'lon': lon1D
                            , 'lat': lat1D
                            , 'key': [keyInfo]
                            , 'date': pd.date_range(dtIncDateInfo, periods=1)
                        }
                    )

                    saveFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, serviceName, keyInfo, sYear)
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    saveData.to_netcdf(saveFile)
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
