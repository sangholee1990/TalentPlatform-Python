# -*- coding: utf-8 -*-

import os
os.environ['PROJ_LIB'] = os.path.join(os.environ["CONDA_PREFIX"], 'share', 'proj')

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
        meanVal = np.nanmean(val2D)
        log.info('[CHECK] val2D : {}'.format(meanVal))

        plt.figure(figsize=(10, 8))
        map = Basemap(projection="cyl", lon_0=0.0, lat_0=0.0, resolution="c")

        if (isCbarLog == True):
            cs = map.scatter(lon2D, lat2D, c=val2D, s=0.05, norm=colors.LogNorm(), cmap=plt.cm.get_cmap('Spectral_r'))
        else:
            # cs = map.scatter(lon2D, lat2D, c=val2D, s=0.05, cmap=plt.cm.get_cmap('Spectral_r'), vmin=0, vmax=30)
            cs = map.scatter(lon2D, lat2D, c=val2D, s=1.0, cmap=plt.cm.get_cmap('Spectral_r'))

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

            # bash RunShell-Python.sh "TalentPlatform-LSH0291-Analy.py" "2018-01-01" "2021-01-01"
            # nohup bash RunShell-Python.sh "TalentPlatform-LSH0291-Analy.py" "2018-01-01" "2021-01-01" &
            # /home/dxinyu/TEST/OUTPUT
            # /home/dxinyu/TEST/OUTPUT

            # python3 "/home/dxinyu/TEST/TalentPlatform-LSH0291-DataMerge.py" --inpPath "/home/dxinyu/TEST/OUTPUT" --outPath "/home/dxinyu/TEST/OUTPUT"

            if (platform.system() == 'Windows'):

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2018-01-01'
                    , 'endDate': '2021-01-01'
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': globalVar['srtDate']
                    # , 'endDate': globalVar['endDate']
                }

            inpPosFile = '{}/{}'.format(globalVar['cfgPath'], 'nationInfo/GEO_NATION_CONT_INFO.csv')
            posData = pd.read_csv(inpPosFile)
            posDataL1 = posData[['lon', 'lat', 'landSea', 'cont']]

            posDataL1['lon'] = np.round(posDataL1['lon'], 1)
            posDataL1['lat'] = np.round(posDataL1['lat'], 1)

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'odiac2020b_1x1d_2019.nc')
            log.info("[CHECK] inpFile : {}".format(inpFile))

            fileList = sorted(glob.glob(inpFile))
            log.info('[CHECK] fileList : {}'.format(fileList))
            # if (len(fileList) < 1): continue

            dsData = xr.open_mfdataset(fileList)
            dsData = xr.where((dsData == 0), np.nan, dsData)

            cnt2D = dsData.count(['month'])
            mean2D = dsData.mean(['month'])
            sd2D = dsData.std(['month'])
            sum2D = dsData.sum(['month'])

            time1D = dsData['month'].values
            lon1D = dsData['lon'].values
            lat1D = dsData['lat'].values
            lon2D, lat2D = np.meshgrid(lon1D, lat1D)

            # *****************************************************************************
            # 확장/상대 불확도 계산
            # *****************************************************************************
            # 자유도
            df = len(time1D)

            # t값
            tVal = t(df)

            # 신뢰구간 95%에 대한 t값
            t025 = tVal.ppf(0.975)

            # 신뢰구간 95%  불확실성 범위
            # leftConf = mean2D - t025 * (sd2D / np.sqrt(df))
            # rightConf = mean2D + t025 * (sd2D / np.sqrt(df))

            # 확장 불확도
            extndUncrt = t025 * (sd2D / np.sqrt(df))

            # 상대 불확도 (%)
            rltvUncrt = (extndUncrt * 100) / mean2D

            # 총 불확도
            totalUncrt = (rltvUncrt * extndUncrt) / np.abs(extndUncrt)

            dtYear = 2019
            keyInfo = 'land'

            meanTotalUncrt = np.nanmean(totalUncrt[keyInfo].values)
            mainTitle = '[{}] {} {} ({:.2f})'.format(dtYear, keyInfo, 'total uncertainty', meanTotalUncrt)
            saveImg = '{}/{}_{}_{}-{}.png'.format(globalVar['figPath'], serviceName, keyInfo, 'totalUncertainty', dtYear)
            rtnInfo = makeMapPlot(lon2D, lat2D, totalUncrt[keyInfo].values, mainTitle, saveImg, None)
            log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

            keyInfo = 'intl_bunker'
            meanTotalUncrt = np.nanmean(totalUncrt[keyInfo].values)
            mainTitle = '[{}] {} {} ({:.2f})'.format(dtYear, keyInfo, 'total uncertainty', meanTotalUncrt)
            saveImg = '{}/{}_{}_{}-{}.png'.format(globalVar['figPath'], serviceName, keyInfo, 'totalUncertainty',  dtYear)
            rtnInfo = makeMapPlot(lon2D, lat2D, totalUncrt[keyInfo].values, mainTitle, saveImg, None)
            log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

            totalUncrt['sum'] = totalUncrt['land'] + totalUncrt['intl_bunker']

            keyInfo = 'sum'
            meanTotalUncrt = np.nanmean(totalUncrt[keyInfo].values)
            mainTitle = '[{}] {} {} ({:.2f})'.format(dtYear, keyInfo, 'total uncertainty', meanTotalUncrt)
            saveImg = '{}/{}_{}_{}-{}.png'.format(globalVar['figPath'], serviceName, keyInfo, 'totalUncertainty',  dtYear)
            rtnInfo = makeMapPlot(lon2D, lat2D, totalUncrt[keyInfo].values, mainTitle, saveImg, None)
            log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))


            dsDataL3 = totalUncrt.to_dataframe().reset_index()
            dsDataL4 = dsDataL3.merge(posDataL1, how='left', left_on=['lat', 'lon'], right_on=['lat', 'lon'])

            dsDataL3.describe()
            posDataL1.describe()

            # from global_land_mask import globe
            # is_on_land = globe.is_land(dsDataL3['lon'], dsDataL3['lat'])


            # try:
            #     totalUncrtTotal = dsDataL4.mean()
            #     totalUncrtLandSea = dsDataL4.groupby(by=['landSea']).mean()
            #     totalUncrtCont = dsDataL4.groupby(by=['cont']).mean()
            #
            #     emissionTotal = dsDataL4.mean()['mean']
            #     emissionLandSea = dsDataL4.groupby(by=['landSea']).mean()['mean']
            #     emissionCont = dsDataL4.groupby(by=['cont']).mean()['mean']
            #
            #     dict = {
            #         'year': [dtYear]
            #         , 'key': [keyInfo]
            #         , 'rltvUncrt total': [rltvUncrtTotal]
            #         , 'rltvUncrt land': [rltvUncrtLandSea['land']]
            #         , 'rltvUncrt sea': [rltvUncrtLandSea['sea']]
            #         , 'rltvUncrt Africa': [rltvUncrtCont['Africa']]
            #         , 'rltvUncrt Antarctica': [rltvUncrtCont['Antarctica']]
            #         , 'rltvUncrt Asia': [rltvUncrtCont['Asia']]
            #         , 'rltvUncrt Australia': [rltvUncrtCont['Australia']]
            #         , 'rltvUncrt Europe': [rltvUncrtCont['Europe']]
            #         , 'rltvUncrt NorthAmerica': [rltvUncrtCont['NorthAmerica']]
            #         , 'rltvUncrt SouthAmerica': [rltvUncrtCont['SouthAmerica']]
            #     }
            #
            #     statData = statData.append(pd.DataFrame.from_dict(dict))




            # mainTitle = '[{}] {} {}'.format(dtYear, keyInfo, 'emission')
            # saveImg = '{}/{}_{}_{}-{}.png'.format(globalVar['figPath'], serviceName, keyInfo, 'emission', dtYear)
            # rtnInfo = makeMapPlot(lon2D, lat2D, mean2D['emission'].values, mainTitle, saveImg, True)
            # log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

            # *******************************************************
            # 육/해상, 대륙별 배출량
            # *******************************************************
            keyList = ['total', 'Power', 'Industry', 'Residential', 'GroundTransportation', 'InternationalAviation', 'InternationalShipping', 'DomesticAviation']

            statData = pd.DataFrame()
            for i, keyInfo in enumerate(keyList):

                # dtYear = 2019
                for dtYear in range(2018, 2022):
                    log.info("[CHECK] keyInfo : {}".format(keyInfo))
                    log.info("[CHECK] dtYear : {}".format(dtYear))

                    inpFilePattern = '{}_{}_{}*.nc'.format(serviceName, keyInfo, dtYear)
                    inpFile = '{}/{}'.format(globalVar['outPath'], inpFilePattern)
                    log.info("[CHECK] inpFile : {}".format(inpFile))

                    fileList = sorted(glob.glob(inpFile))
                    log.info('[CHECK] fileList : {}'.format(fileList))
                    if (len(fileList) < 1): continue

                    dsData = xr.open_mfdataset(fileList)
                    # log.info('[CHECK] dsData : {}'.format(dsData))

                    time1D = dsData['time'].values
                    lon1D = dsData['lon'].values
                    lat1D = dsData['lat'].values
                    lon2D, lat2D = np.meshgrid(lon1D, lat1D)

                    # 결측값 처리
                    dsData = xr.where((dsData == 0), np.nan, dsData)

                    # *****************************************************************************
                    # 위/경도에 따른 통계 계산
                    # *****************************************************************************
                    cnt2D = dsData.count(['time'])
                    mean2D = dsData.mean(['time'])
                    sd2D = dsData.std(['time'])
                    sum2D = dsData.sum(['time'])

                    # cntVal = np.nanmean(cnt2D['emission'])
                    # log.info('[CHECK] cntVal : {}'.format(cntVal))
                    #
                    # sumVal = np.nanmean(sum2D['emission'])
                    # log.info('[CHECK] sumVal : {}'.format(sumVal))
                    #
                    # meanVal = np.nanmean(mean2D['emission'])
                    # log.info('[CHECK] meanVal : {}'.format(meanVal))
                    #
                    # sdVal = np.nanmean(sd2D['emission'])
                    # log.info('[CHECK] sdVal : {}'.format(sdVal))


                    # *****************************************************************************
                    # 확장/상대 불확도 계산
                    # *****************************************************************************
                    # 자유도
                    df = len(time1D)

                    # t값
                    tVal = t(df)

                    # 신뢰구간 95%에 대한 t값
                    t025 = tVal.ppf(0.975)

                    # 신뢰구간 95%  불확실성 범위
                    # leftConf = mean2D - t025 * (sd2D / np.sqrt(df))
                    # rightConf = mean2D + t025 * (sd2D / np.sqrt(df))

                    # 확장 불확도
                    extndUncrt = t025 * (sd2D / np.sqrt(df))

                    # 상대 불확도 (%)
                    rltvUncrt = (extndUncrt * 100) / mean2D

                    # NetCDF 생산
                    dsDataL2 = xr.Dataset(
                        {
                            'mean': (('lat', 'lon'), (mean2D['emission'].values).reshape(len(lat1D), len(lon1D)))
                            , 'count': (('lat', 'lon'), (cnt2D['emission'].values).reshape(len(lat1D), len(lon1D)))
                            , 'sd': (('lat', 'lon'), (sd2D['emission'].values).reshape(len(lat1D), len(lon1D)))
                            , 'sum': (('lat', 'lon'), (sum2D['emission'].values).reshape(len(lat1D), len(lon1D)))
                            , 'extndUncrt': (('lat', 'lon'), (extndUncrt['emission'].values).reshape(len(lat1D), len(lon1D)))
                            , 'rltvUncrt': (('lat', 'lon'), (rltvUncrt['emission'].values).reshape(len(lat1D), len(lon1D)))
                        }
                        , coords={
                            'lat': lat1D
                            , 'lon': lon1D
                        }
                    )

                    saveFile = '{}/{}_{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, keyInfo, 'statData', dtYear)
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    dsDataL2.to_netcdf(saveFile)
                    log.info('[CHECK] saveFile : {}'.format(saveFile))

                    dsDataL3 = dsDataL2.to_dataframe().reset_index()
                    dsDataL4 = dsDataL3.merge(posDataL1, how='left', left_on=['lat', 'lon'], right_on=['lat', 'lon'])

                    try:
                        rltvUncrtTotal = dsDataL4.mean()['rltvUncrt']
                        rltvUncrtLandSea = dsDataL4.groupby(by=['landSea']).mean()['rltvUncrt']
                        rltvUncrtCont = dsDataL4.groupby(by=['cont']).mean()['rltvUncrt']

                        emissionTotal = dsDataL4.mean()['mean']
                        emissionLandSea = dsDataL4.groupby(by=['landSea']).mean()['mean']
                        emissionCont = dsDataL4.groupby(by=['cont']).mean()['mean']

                        dict = {
                            'year': [dtYear]
                            , 'key': [keyInfo]
                            , 'rltvUncrt total': [rltvUncrtTotal]
                            , 'rltvUncrt land': [rltvUncrtLandSea['land']]
                            , 'rltvUncrt sea': [rltvUncrtLandSea['sea']]
                            , 'rltvUncrt Africa': [rltvUncrtCont['Africa']]
                            , 'rltvUncrt Antarctica': [rltvUncrtCont['Antarctica']]
                            , 'rltvUncrt Asia': [rltvUncrtCont['Asia']]
                            , 'rltvUncrt Australia': [rltvUncrtCont['Australia']]
                            , 'rltvUncrt Europe': [rltvUncrtCont['Europe']]
                            , 'rltvUncrt NorthAmerica': [rltvUncrtCont['NorthAmerica']]
                            , 'rltvUncrt SouthAmerica': [rltvUncrtCont['SouthAmerica']]

                            , 'emission total': [emissionTotal]
                            , 'emission land': [emissionLandSea['land']]
                            , 'emission sea': [emissionLandSea['sea']]
                            , 'emission Africa': [emissionCont['Africa']]
                            , 'emission Antarctica': [emissionCont['Antarctica']]
                            , 'emission Asia': [emissionCont['Asia']]
                            , 'Australia': [emissionCont['Australia']]
                            , 'emission Europe': [emissionCont['Europe']]
                            , 'emission NorthAmerica': [emissionCont['NorthAmerica']]
                            , 'emission SouthAmerica': [emissionCont['SouthAmerica']]
                        }

                        statData = statData.append(pd.DataFrame.from_dict(dict))
                    except Exception as e:
                        log.error("Exception : {}".format(e))

                    # 시각화
                    mainTitle = '[{}] {} {}'.format(dtYear, keyInfo, 'emission')
                    saveImg = '{}/{}_{}_{}-{}.png'.format(globalVar['figPath'], serviceName, keyInfo, 'emission', dtYear)
                    rtnInfo = makeMapPlot(lon2D, lat2D, mean2D['emission'].values, mainTitle, saveImg, True)
                    log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

                    mainTitle = '[{}] {} {}'.format(dtYear, keyInfo, 'relative uncertainty')
                    saveImg = '{}/{}_{}_{}-{}.png'.format(globalVar['figPath'], serviceName, keyInfo, 'relativeUncertainty', dtYear)
                    rtnInfo = makeMapPlot(lon2D, lat2D, rltvUncrt['emission'].values, mainTitle, saveImg, None)
                    log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

            saveXlsxFile = '{}/{}_{}.xlsx'.format(globalVar['outPath'], serviceName, 'statData')
            os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)
            statData.to_excel(saveXlsxFile, index=False)
            log.info("[CHECK] saveXlsxFile : {}".format(saveXlsxFile))

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
