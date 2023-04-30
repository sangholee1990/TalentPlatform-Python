# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import matplotlib.cm as cm
from scipy.stats import linregress
import traceback
import warnings
from datetime import datetime
import configparser
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj
import seaborn as sns
import rioxarray as rio
import cftime
import subprocess
from global_land_mask import globe
from matplotlib import font_manager, rc
import re
import json
from scipy.interpolate import Rbf

from datetime import datetime, timedelta
from turfpy.measurement import boolean_point_in_polygon
from geojson import Point, MultiPolygon, Feature

import pygrib
from scipy import spatial
import math

import geopandas as gpd
import rioxarray as rio
import xshape
from shapely.geometry import mapping
from shapely.geometry import Point, Polygon
import odc.geo.xr

# import xagg as xa
import rasterio
import odc.geo.xr
import xarray as xr
import matplotlib.font_manager as fm
from pandas.tseries.offsets import Day, Hour, Minute, Second

import urllib
from urllib import parse
from sspipe import p, px
import matplotlib.dates as mdates

# import sys
# sys.path.insert(0, DIR)
from importlib_metadata import version
# version()
# modulenames = set(sys.modules) & set(globals())
# modulenames = set(globals())
# allmodules = [sys.modules for name in modulenames]
# allmodules = [sys.modules[name] for name in modulenames]
# sys.modules['os']
# import statlib
from importlib.metadata import version
# # version('os')
# import pkg_resources
#
#
# print('rioxarray', pkg_resources.get_distribution('rioxarray').version)
# from importlib.metadata import version
# for i, module in enumerate(modulenames):
#     try:
#         print(module, pkg_resources.get_distribution(module).version)
#     except Exception as e:
#         continue

# import pkg_resources
# installed_packages = pkg_resources.working_set
# installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
#    for i in installed_packages])
# print(installed_packages_list)

# help("modules")

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


# plt.rcParams['axes.unicode_minus'] = False

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
        , 'sysCfg': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.json')
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

        # 글꼴 설정
        fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        plt.rcParams['font.family'] = fontName

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


def cartesian(latitude, longitude, elevation=0):
    # Convert to radians
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371  # 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)

    return (X, Y, Z)


def downCallBack(a, b, c):

    per = 100.0 * a * b / c

    if per > 100: per = 100

    log.info('[CHECK] percent : {:.2f}'.format(per))


# 빈도분포 2D 시각화
def makeUserHist2dPlot(prdVal, refVal, xlab, ylab, mainTitle, saveImg, minVal, maxVal, xIntVal, yIntVal, nbinVal, isSame):

    log.info('[START] {}'.format('makeUserHist2dPlot'))

    result = None

    try:

        # 그리드 설정
        plt.grid(True)

        # 결측값 마스킹
        mask = ~np.isnan(refVal)
        N = len(refVal[mask])

        # plt.scatter(prdVal, refVal)
        # nbins = 250
        hist2D, xEdge, yEdge = np.histogram2d(prdVal[mask], refVal[mask], bins=nbinVal)
        # hist2D, xEdge, yEdge = np.histogram2d(prdVal, refVal)

        # hist2D 전처리
        hist2D = np.rot90(hist2D)
        hist2D = np.flipud(hist2D)

        # 마스킹
        hist2DVal = np.ma.masked_where(hist2D == 0, hist2D)

        plt.pcolormesh(xEdge, yEdge, hist2DVal, cmap=cm.get_cmap('jet'), vmin=0, vmax=50)

        # 제목, x축, y축 설정
        plt.title(mainTitle)
        plt.xlabel(xlab)
        plt.ylabel(ylab)

        # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
        Bias = np.mean(prdVal - refVal)
        rBias = (Bias / np.mean(refVal)) * 100.0
        RMSE = np.sqrt(np.mean((prdVal - refVal) ** 2))
        rRMSE = (RMSE / np.mean(refVal)) * 100.0
        # MAPE = np.mean(np.abs((prdVal - refVal) / prdVal)) * 100.0

        # 선형회귀곡선에 대한 계산
        slope, intercept, rVal, pVal, stdErr = linregress(prdVal[mask], refVal[mask])

        lmfit = (slope * prdVal) + intercept
        # plt.plot(prdVal, lmfit, color='red', linewidth=2,linestyle="-")
        plt.plot([minVal, maxVal], [minVal, maxVal], color="black", linestyle="--", linewidth=2)
        plt.plot(prdVal, lmfit, color='red', linewidth=2, linestyle="-")

        # 컬러바
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('빈도수')

        # 라벨 추가
        plt.annotate('%s = %.2f x (%s) + %.2f' % (ylab, slope, xlab, intercept),
                     xy=(minVal + xIntVal, maxVal - yIntVal),
                     color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
        plt.annotate('R = %.2f  (p-value < %.2f)' % (rVal, pVal), xy=(minVal + xIntVal, maxVal - yIntVal * 2),
                     color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')

        if (isSame == True):
            # plt.axes().set_aspect('equal')
            # plt.axes().set_aspect(1)
            # plt.gca().set_aspect('equal')
            plt.xlim(minVal, maxVal)
            plt.ylim(minVal, maxVal)

            plt.annotate('Bias = %.2f  (%%Bias = %.2f %%)' % (Bias, rBias), xy=(minVal + xIntVal, maxVal - yIntVal * 3),
                         color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
            plt.annotate('RMSE = %.2f  (%%RMSE = %.2f %%)' % (RMSE, rRMSE), xy=(minVal + xIntVal, maxVal - yIntVal * 4),
                         color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
            # plt.annotate('MAPE = %.2f %%' % (MAPE), xy=(minVal + xIntVal, maxVal - yIntVal * 5),
            #              color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
            plt.annotate('N = %d' % N, xy=(minVal + xIntVal, maxVal - yIntVal * 5), color='black',
                         xycoords='data', horizontalalignment='left', verticalalignment='center')
        else:
            plt.annotate('N = %d' % N, xy=(minVal + xIntVal, maxVal - yIntVal * 3), color='black',
                         xycoords='data', horizontalalignment='left', verticalalignment='center')

        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
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
        log.info('[END] {}'.format('makeUserHist2dPlot'))


# 시계열 시각화
def makeUserTimeSeriesPlot(dtDate, mlPrdVal, dlPrdVal, refVal, mlPrdValLabel, dlPrdValLabel, refValLabel, xlab, ylab, mainTitle, saveImg, isFore):

    log.info('[START] {}'.format('makeUserTimeSeriesPlot'))

    result = None

    try:
        # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
        mlRMSE = np.sqrt(np.mean((mlPrdVal - refVal) ** 2))
        mlReRMSE = (mlRMSE / np.mean(refVal)) * 100.0

        dlRMSE = np.sqrt(np.mean((dlPrdVal - refVal) ** 2))
        dlReRMSE = (dlRMSE / np.mean(refVal)) * 100.0

        # 선형회귀곡선에 대한 계산
        mlFit = linregress(mlPrdVal, refVal)
        mlR = mlFit[2]

        dlFit = linregress(dlPrdVal, refVal)
        dlR = dlFit[2]

        prdValLabel_ml = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(mlPrdValLabel, mlR, mlReRMSE)
        prdValLabel_dnn = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(dlPrdValLabel, dlR, dlReRMSE)

        plt.grid(True)

        plt.plot(dtDate, mlPrdVal, label=prdValLabel_ml, marker='o')
        plt.plot(dtDate, dlPrdVal, label=prdValLabel_dnn, marker='o')
        plt.plot(dtDate, refVal, label=refValLabel, marker='o')

        # 제목, x축, y축 설정
        plt.title(mainTitle)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.ylim(0, 1000)

        if (isFore == True):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.gcf().autofmt_xdate()
            plt.xticks(rotation=45, ha='right')

        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.gcf().autofmt_xdate()
            plt.xticks(rotation=0, ha='right')

        plt.legend(loc='upper left')

        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
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
        log.info('[END] {}'.format('makeUserTimeSeriesPlot'))


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python 이용한 NetCDF 자료 읽기 그리고 도시별 Shp에서 폴리곤 결과 계산

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
    serviceName = 'LSH0349'

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
                    'srtDate': '2022-08-01'
                    , 'endDate': '2022-08-08'

                    # 수집 정보
                    # isOverWrite : True (덮어쓰기), False
                    , 'collect' : {
                        'um' : {
                            'isOverWrite': False
                        }
                        , 'asos': {
                            'isOverWrite': False
                        }
                        , 'aws': {
                            'isOverWrite': False
                        }
                    }
                }

                # 출력 정보
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2022-08-01'
                    , 'endDate': '2022-08-08'

                    # 수집 정보
                    # isOverWrite : True (덮어쓰기), False
                    , 'collect' : {
                        'um' : {
                            'isOverWrite': False
                        }
                        , 'asos': {
                            'isOverWrite': False
                        }
                        , 'aws': {
                            'isOverWrite': False
                        }
                    }
                }

                # 출력 정보
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # ****************************************************************************
            # 시작/종료일 설정
            # ****************************************************************************
            dtKst = timedelta(hours=9)

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtHourList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))
            dt6HourList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(6))
            dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))
            dt3HourList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(3))

            dtHourUtcList = dtHourList - dtKst
            dtDayUtcList = dtDayList - dtKst

            # ****************************************************************************
            # shp 파일 읽기
            # ****************************************************************************
            inpShpFile = '{}/{}/{}/{}.shp'.format(globalVar['inpPath'], serviceName, 'FLOOD-SHP', '*')
            fileShpList = sorted(glob.glob(inpShpFile))

            # if fileList is None or len(fileList) < 1:
            #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

            fileShpInfo = fileShpList[0]
            for i, fileShpInfo in enumerate(fileShpList):
                log.info("[CHECK] fileShpInfo : {}".format(fileShpInfo))

                # geoData = gpd.read_file(fileInfo, crs='epsg:4326')
                geoData = gpd.read_file(fileShpInfo, crs='epsg:3826')
                geoDataL1 = geoData.to_crs(crs=4326)
                geoDataL1['centroid'] = geoDataL1.centroid
                geoDataL1['lon'] = geoDataL1['centroid'].x
                geoDataL1['lat'] = geoDataL1['centroid'].y

            # ****************************************************************************
            # [동적] UM 설정 파일에서 최근접 화소 찾기
            # ****************************************************************************
            cfgFile = '{}/{}'.format(globalVar['cfgPath'], 'modelInfo/*.grb2')
            fileCfgList = sorted(glob.glob(cfgFile))
            umCfgData = xr.open_dataset(fileCfgList[0], decode_times=False, engine='pynio')

            umCfgDataL1 = umCfgData.get(['xgrid_0', 'ygrid_0', 'gridlat_0', 'gridlon_0'])
            umCfgDataL2 = umCfgDataL1.to_dataframe().reset_index(drop=False).rename(
                columns={
                    'gridlon_0': 'lon'
                    , 'gridlat_0': 'lat'
                    , 'xgrid_0': 'UM-nx'
                    , 'ygrid_0': 'UM-ny'
                }
            )

            # kdTree를 위한 초기 데이터
            posList = []
            for idx in range(0, len(umCfgDataL2)):
                coord = [umCfgDataL2.loc[idx, 'lat'], umCfgDataL2.loc[idx, 'lon']]
                posList.append(cartesian(*coord))

            tree = spatial.KDTree(posList)

            for i, posInfo in geoDataL1.iterrows():
                if (posInfo.isna()[['lon', 'lat']].any() == True): continue
                coord = cartesian(posInfo['lat'], posInfo['lon'])
                closest = tree.query([coord], k=1)
                cloDist = closest[0][0]
                cloIdx = closest[1][0]

                cloData = umCfgDataL2.iloc[cloIdx]

                geoDataL1.loc[i, 'UM-dist'] = cloDist
                geoDataL1.loc[i, 'UM-nx'] = cloData['UM-nx']
                geoDataL1.loc[i, 'UM-ny'] = cloData['UM-ny']
                geoDataL1.loc[i, 'UM-lon'] = cloData['lon']
                geoDataL1.loc[i, 'UM-lat'] = cloData['lat']

            # ****************************************************************************
            # [동적] KLAPS 설정 파일에서 최근접 화소 찾기
            # ****************************************************************************
            cfgFile = '{}/{}'.format(globalVar['cfgPath'], 'modelInfo/*.nc')
            fileCfgList = sorted(glob.glob(cfgFile))
            klapeCfgData = xr.open_dataset(fileCfgList[0], decode_times=False, engine='pynio')

            klapeCfgDataL1 = klapeCfgData.get(['x', 'y', 'lon', 'lat'])
            klapeCfgDataL2 = klapeCfgDataL1.to_dataframe().reset_index(drop=False).rename(
                columns={
                    'lon': 'lon'
                    , 'lat': 'lat'
                    , 'x': 'KLAPS-nx'
                    , 'y': 'KLAPS-ny'
                }
            )

            # kdTree를 위한 초기 데이터
            posList = []
            for idx in range(0, len(klapeCfgDataL2)):
                coord = [klapeCfgDataL2.loc[idx, 'lat'], klapeCfgDataL2.loc[idx, 'lon']]
                posList.append(cartesian(*coord))

            tree = spatial.KDTree(posList)

            for i, posInfo in geoDataL1.iterrows():
                if (posInfo.isna()[['lon', 'lat']].any() == True): continue
                coord = cartesian(posInfo['lat'], posInfo['lon'])
                closest = tree.query([coord], k=1)
                cloDist = closest[0][0]
                cloIdx = closest[1][0]

                cloData = klapeCfgDataL2.iloc[cloIdx]

                geoDataL1.loc[i, 'KLAPS-dist'] = cloDist
                geoDataL1.loc[i, 'KLAPS-nx'] = cloData['KLAPS-nx']
                geoDataL1.loc[i, 'KLAPS-ny'] = cloData['KLAPS-ny']
                geoDataL1.loc[i, 'KLAPS-lon'] = cloData['lon']
                geoDataL1.loc[i, 'KLAPS-lat'] = cloData['lat']

            # ****************************************************************************
            # [정적] UM, KLAPS 설정 저장/읽기
            # ****************************************************************************
            # UM, KLAPS 설정 저장
            geoDataL2 = geoDataL1.dropna()
            # saveFile = '{}/{}/{}.csv'.format(globalVar['cfgPath'], serviceName, 'UM-KLAPS_STN_DATA')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # geoDataL2.to_csv(saveFile, index=False)

            # UM, KLAPS 설정 읽기
            # geoDataL2 = pd.read_csv(saveFile)

            # ****************************************************************************
            # [정적] SHP 파일 및 UM/KLAPS
            # ****************************************************************************
            # klapsCfgMask = pd.DataFrame()
            # umCfgMask = pd.DataFrame()
            # gidList = geoDataL2['gid'].values
            #
            # for k, gid in enumerate(gidList):
            #     log.info("[CHECK] gid : {}".format(gid))
            #
            #     gidInfo = geoDataL2.loc[
            #         geoDataL2['gid'] == gid
            #         ]
            #
            #     jsonData = json.loads(gidInfo.geometry.to_json())
            #
            #     klapsDataAll = klapeCfgDataL2
            #     klapsDataAll['isMask'] = klapsDataAll.apply(
            #         lambda row: boolean_point_in_polygon(
            #             Feature(geometry=Point([row['lon'], row['lat']]))
            #             , jsonData['features'][0]
            #         )
            #         , axis=1
            #     )
            #
            #     klapsDataMask = klapsDataAll.loc[
            #         klapsDataAll['isMask'] == True
            #         ]
            #     klapsDataMask['gid'] = gid
            #
            #     umDataAll = umCfgDataL2
            #     umDataAll['isMask'] = umDataAll.apply(
            #         lambda row: boolean_point_in_polygon(
            #             Feature(geometry=Point([row['lon'], row['lat']]))
            #             , jsonData['features'][0]
            #         )
            #         , axis=1
            #     )
            #
            #     umDataMask = umDataAll.loc[
            #         umDataAll['isMask'] == True
            #         ]
            #     umDataMask['gid'] = gid
            #
            #     klapsCfgMask = pd.concat([klapsCfgMask, klapsDataMask], axis=0, ignore_index=True)
            #     umCfgMask = pd.concat([umCfgMask, umDataMask], axis=0, ignore_index=True)

            # UM, KLAPS 설정 저장
            saveUmFile = '{}/{}/{}.csv'.format(globalVar['cfgPath'], serviceName, 'UM_STN_MASK')
            # os.makedirs(os.path.dirname(saveUmFile), exist_ok=True)
            # umCfgMask.to_csv(saveUmFile, index=False)
            umCfgMask = pd.read_csv(saveUmFile)

            saveKlapsFile = '{}/{}/{}.csv'.format(globalVar['cfgPath'], serviceName, 'KLAPS_STN_MASK')
            # os.makedirs(os.path.dirname(saveKlapsFile), exist_ok=True)
            # klapsCfgMask.to_csv(saveKlapsFile, index=False)
            klapsCfgMask = pd.read_csv(saveKlapsFile)



            # ****************************************************************************
            # 홍수관리저수지 유역도에 따른 OBS 융합 관측소 영역 화소 시각화
            # ****************************************************************************
            # # ASOS/AWS 융합 관측소
            # inpAllStnFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/ALL_STN_INFO.csv')
            # allStnData = pd.read_csv(inpAllStnFile)
            # allStnDataL1 = allStnData[['STN', 'LON', 'LAT']]
            #
            # allStnDataL2 = allStnDataL1.rename(
            #     columns={'LON': 'lon', 'LAT': 'lat'}
            # )
            #
            # # gid = gidList[1]
            # gidList = geoDataL2['gid'].values
            # for k, gid in enumerate(gidList):
            #
            #     log.info("[CHECK] gid : {}".format(gid))
            #
            #     gidInfo = geoDataL2.loc[
            #         geoDataL2['gid'] == gid
            #         ]
            #
            #     jsonData = json.loads(gidInfo.geometry.to_json())
            #     # jsonData = json.loads(geoDataL2.geometry.to_json())
            #
            #
            #     allStnDataL2['isMask'] = allStnDataL2.apply(
            #         lambda row: boolean_point_in_polygon(
            #             Feature(geometry=Point([row['lon'], row['lat']]))
            #             , jsonData['features'][0]
            #         )
            #         , axis=1
            #     )
            #
            #     allStnDataL3 = allStnDataL2.loc[
            #         allStnDataL2['isMask'] == True
            #         ]
            #
            #     # res = 1.0
            #     # res = 0.10
            #
            #     mainTilte = '[{}-OBS 융합 관측소] {}, {}'.format(gidInfo['fac_name'].values[0], round(gidInfo['lon'].values[0], 2), round(gidInfo['lat'].values[0], 2))
            #     saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTilte)
            #     gidInfo.plot(color='lightgrey', legend=True)
            #     lonMin, lonMax, latMin, latMax = plt.axis()
            #
            #     plt.scatter(gidInfo['lon'], gidInfo['lat'], c='black', label='중심 ({})'.format(len(gidInfo)))
            #     plt.scatter(allStnDataL3['lon'], allStnDataL3['lat'], c='red', label='영역 OBS ({})'.format(len(allStnDataL3)))
            #     plt.title(mainTilte)
            #     plt.legend()
            #     # plt.axis((lonMin-res, lonMax+res, latMin-res, latMax+res))
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.show()
            #     plt.close()

            # ****************************************************************************
            # 홍수관리저수지 유역도에 따른 KLAPS, UM 최근접, 영역 화소 시각화
            # ****************************************************************************
            # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, '유역도에 따른 KLAPS 거리 차이')
            # geoDataL2.plot(column=geoDataL2['KLAPS-dist'], legend=True)
            # geoDataL2.apply(lambda x: plt.annotate(text=x['fac_name'], xy=[x.lon, x.lat], ha='center', va='top', color='black'), axis=1)
            # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            # plt.show()
            # plt.close()

            # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, '유역도에 따른 UM 거리 차이')
            # geoDataL2.plot(column=geoDataL2['UM-dist'], legend=True)
            # geoDataL2.apply(lambda x: plt.annotate(text=x['fac_name'], xy=[x.lon, x.lat], ha='center', va='top', color='black'), axis=1)
            # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            # plt.show()
            # plt.close()


            # gidList = geoDataL2['gid'].values
            # for k, gid in enumerate(gidList):
            #     log.info("[CHECK] gid : {}".format(gid))
            #
            #     gidInfo = geoDataL2.loc[
            #         geoDataL2['gid'] == gid
            #         ]
            #
            #     umCfgMaskL1 = umCfgMask.loc[
            #         umCfgMask['gid'] == gid
            #         ]
            #
            #     klapsCfgMaskL1 = klapsCfgMask.loc[
            #         klapsCfgMask['gid'] == gid
            #         ]
            #
            #     mainTilte = '[{}] {}, {}'.format(
            #         gidInfo['fac_name'].values[0]
            #         , round(gidInfo['lon'].values[0], 2)
            #         ,  round(gidInfo['lat'].values[0], 2)
            #     )
            #
            #     saveImg = '{}/{}/KLAPS-UM_{}.png'.format(globalVar['figPath'], serviceName, gidInfo['fac_name'].values[0])
            #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #
            #     gidInfo.plot(color='lightgrey')
            #     plt.scatter(gidInfo['lon'], gidInfo['lat'], c='black', label='중심 ({})'.format(len(gidInfo)))
            #     plt.scatter(gidInfo['KLAPS-lon'], gidInfo['KLAPS-lat'], c = 'red', label='최근접 KLAPS ({})'.format(len(gidInfo)))
            #     plt.scatter(gidInfo['UM-lon'], gidInfo['UM-lat'], c='blue', label='최근접 UM ({})'.format(len(gidInfo)))
            #     plt.scatter(klapsCfgMaskL1['lon'], klapsCfgMaskL1['lat'], alpha=0.25, c = 'red', label='영역 KLAPS ({})'.format(len(klapsCfgMaskL1)))
            #     plt.scatter(umCfgMaskL1['lon'], umCfgMaskL1['lat'], alpha=0.25, c='blue', label='영역 UM ({})'.format(len(umCfgMaskL1)))
            #     plt.title(mainTilte)
            #     plt.legend()
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.show()
            #     plt.close()

            # sorted(np.unique(umDataMask['UM-nx']))
            # sorted(np.unique(umDataMask['UM-ny']))

            # ****************************************************************************
            # URL API를 통해 UM LDAPS 수치모델 수집
            # ****************************************************************************
            cfgInfo = json.load(open(globalVar['sysCfg'], 'r'))

            apiUrl = cfgInfo['dataApi-um']['url']
            apikey = cfgInfo['dataApi-um']['key']

            for i, dt6HourInfo in enumerate(dt6HourList):
                for j, forHour in  enumerate(np.arange(0, 49, 1)):

                    filePathDateYmd = dt6HourInfo.strftime('%Y%m/%d')
                    filePathDateYmdH = dt6HourInfo.strftime('%Y%m/%d/%H')
                    fileNameDate = dt6HourInfo.strftime('%Y%m%d%H%M')

                    reqYmdHm = fileNameDate
                    reqForHour = forHour

                    saveFile = '{}/{}/MODEL/{}/UMKR_l015_unis_H{}_{}.grb2'.format(globalVar['inpPath'], serviceName, filePathDateYmdH, '{:03d}'.format(forHour), fileNameDate)
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    saveFileList = sorted(glob.glob(saveFile), reverse=True)

                    # 학습 모델이 없을 경우
                    if (sysOpt['collect']['um']['isOverWrite']) or (len(saveFileList) < 1):

                        try:
                            log.info("[CHECK] dt6HourInfo : {} / forHour : {}".format(dt6HourInfo, forHour))

                            # http://203.247.66.28/url/nwp_file_down.php?nwp=l015&sub=unis&tmfc=202208010000&hh_ef=0&authKey=6b89a8e82e7dc3a444ec0d0b0dbbd21ef86bc5c2151e15f9a5629a3ca147bb56cda47ea0a5305014474e1404bd68f65128fb2b80ce121bc85e4909c0b5c175a9
                            reqAtmosUmApi = (
                                    '{}nwp=l015&sub=unis&tmfc={}&hh_ef={}&authKey={}'.format(apiUrl, reqYmdHm, reqForHour, apikey)
                                    | p(parse.urlparse).query
                                    | p(parse.parse_qs)
                                    | p(parse.urlencode, doseq=True)
                                    | apiUrl + px
                            )

                            # http://203.247.66.126:8090/url/nwp_file_down.php?%3Fnwp=l015&sub=unis&tmfc=2022080100&ef=00&mode=I&authKey=sangho.lee.1990@gmail.com
                            # res = urllib.request.urlretrieve(reqAtmosUmApi, saveFile, downCallBack)
                            res = urllib.request.urlretrieve(reqAtmosUmApi, saveFile)

                            isFileExist = os.path.exists(saveFile)
                            log.info("[CHECK] {} : {}".format(isFileExist, saveFile))
                        except Exception as e:
                            log.error("Exception : {}".format(e))

            # ****************************************************************************
            # URL API를 통해 방재기상관측 (AWS) 수집
            # ****************************************************************************
            cfgInfo = json.load(open(globalVar['sysCfg'], 'r'))

            apiUrl = cfgInfo['dataApi-aws']['url']
            apikey = cfgInfo['dataApi-aws']['key']

            dt3HourInfo = dt3HourList[0]
            for i, dt3HourInfo in enumerate(dt3HourList):

                filePathDateYmd = dt3HourInfo.strftime('%Y%m/%d')
                fileNameDate = dt3HourInfo.strftime('%Y%m%d%H%M')

                reqSrtYmdHm = dt3HourInfo.strftime('%Y%m%d%H%M')
                reqEndYmdHm = (dt3HourInfo + timedelta(hours=3)).strftime('%Y%m%d%H%M')

                saveFile = '{}/{}/OBS/{}/AWS_OBS_{}.csv'.format(globalVar['inpPath'], serviceName, filePathDateYmd, fileNameDate)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                saveFileList = sorted(glob.glob(saveFile), reverse=True)

                # 학습 모델이 없을 경우
                if (sysOpt['collect']['aws']['isOverWrite']) or (len(saveFileList) < 1):

                    try:
                        log.info("[CHECK] dt3HourInfo : {}".format(dt3HourInfo))

                        # # YYMMDDHHMI   STN    WD1    WS1    WDS    WSS   WS10   WS10     TA     RE RN-15m RN-60m RN-12H RN-DAY     HM     PA     PS     TD
                        # #        KST    ID    deg    m/s    deg    m/s    deg    m/s      C      1     mm     mm     mm     mm      %    hPa    hPa     C
                        # 202208050000    90  350.1    0.7  358.1    0.8    9.8    0.6   26.8    0.0    0.0    0.0    0.0   11.3   89.1 1004.0 1006.0   24.9

                        # http://203.247.66.28/url/cgi-bin/url/nph-aws2_min?tm1=202208050000&tm2=202208050300&stn=0&disp=0&help=0&authKey=3e8e1ac56c218f12ceca29cc36438cb087b2605d898bcfe1387496d5387df25c51a31aa20bb9fbb0d0e13c860b6f49f012eebaebcadae9e5ff9e4eadf9f19036
                        reqAtmosAwsApi = (
                                '{}tm1={}&tm2={}&stn=0&disp=0&help=0&authKey={}'.format(apiUrl, reqSrtYmdHm, reqEndYmdHm, apikey)
                                | p(parse.urlparse).query
                                | p(parse.parse_qs)
                                | p(parse.urlencode, doseq=True)
                                | apiUrl + px
                        )

                        # res = urllib.request.urlretrieve(reqAtmosUmApi, saveFile, downCallBack)
                        res = urllib.request.urlretrieve(reqAtmosAwsApi, saveFile)

                        isFileExist = os.path.exists(saveFile)
                        log.info("[CHECK] {} : {}".format(isFileExist, saveFile))
                    except Exception as e:
                        log.error("Exception : {}".format(e))

            # ****************************************************************************
            # 오픈 API를 통해 종관기상관측 (ASOS) 수집
            # ****************************************************************************
            cfgInfo = json.load(open(globalVar['sysCfg'], 'r'))

            apiUrl = cfgInfo['dataApi-asos']['url']
            apikey = cfgInfo['dataApi-asos']['key']

            # ASOS 설정 정보
            inpAsosStnFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/ASOS_STN_INFO.csv')
            asosStnData = pd.read_csv(inpAsosStnFile)
            asosStnDataL1 = asosStnData[['STN', 'LON', 'LAT']]

            # dataL2 = pd.DataFrame()
            for i, dtDayInfo in enumerate(dtDayList):

                filePathDateYmd = dtDayInfo.strftime('%Y%m/%d')
                fileNameDate = dtDayInfo.strftime('%Y%m%d%H%M')

                reqSrtYmd = dtDayInfo.strftime('%Y%m%d')
                reqEndYmd = (dtDayInfo + timedelta(days=1)).strftime('%Y%m%d')

                saveFile = '{}/{}/OBS/{}/ASOS_OBS_{}.csv'.format(globalVar['inpPath'], serviceName, filePathDateYmd, fileNameDate)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                saveFileList = sorted(glob.glob(saveFile), reverse=True)

                # 학습 모델이 없을 경우
                if (sysOpt['collect']['asos']['isOverWrite']) or (len(saveFileList) < 1):

                    log.info("[CHECK] dtDayInfo : {}".format(dtDayInfo))

                    # posInfo = asosStnDataL1.iloc0[2]
                    dataL1 = pd.DataFrame()
                    for j, posInfo in asosStnDataL1.iterrows():
                        posId = int(posInfo['STN'])
                        posLat = posInfo['LAT']
                        posLon = posInfo['LON']

                        log.info("[CHECK] posId (posLon, posLat) : {} ({}. {})".format(posId, posLon, posLat))

                        # http://203.247.66.28/url/nwp_file_down.php?serviceKey=6b89a8e82e7dc3a444ec0d0b0dbbd21ef86bc5c2151e15f9a5629a3ca147bb56cda47ea0a5305014474e1404bd68f65128fb2b80ce121bc85e4909c0b5c175a9&pageNo=1&numOfRows=99&dataType=JSON&dataCd=ASOS&dateCd=HR&startDt=20220801&startHh=01&endDt=20220803&endHh=01&stnIds=295
                        reqAtmosObsApi = (
                                '{}serviceKey={}&pageNo=1&numOfRows=99&dataType=JSON&&dataCd=ASOS&dateCd=HR&startDt={}&startHh=01&endDt={}&endHh=01&stnIds={}'.format(apiUrl, apikey, reqSrtYmd, reqEndYmd, posId)
                                | p(parse.urlparse).query
                                | p(parse.parse_qs)
                                | p(parse.urlencode, doseq=True)
                                | apiUrl + px
                        )

                        res = urllib.request.urlopen(reqAtmosObsApi)
                        resCode = res.getcode()
                        if resCode != 200: continue

                        resData = json.loads(res.read().decode('utf-8'))
                        resultCode = resData['response']['header']['resultCode']
                        if (resultCode != '00'): continue

                        resBody = resData['response']['body']
                        resCnt = resBody['totalCount']
                        if (resCnt < 1): continue

                        itemList = resBody['items']['item']
                        if (len(itemList) < 1): continue

                        data = pd.DataFrame.from_dict(itemList)
                        dataL1 = pd.concat([dataL1, data], ignore_index=False)

                    dataL1.to_csv(saveFile, index=False)
                    log.info('[CHECK] saveFile : {}'.format(saveFile))
                # else:
                #     dataL1 = pd.read_csv(saveFile)
                #
                # dataL2 = pd.concat([dataL2, dataL1], ignore_index=False)


            # ****************************************************************************
            # 종관기상관측 (ASOS) 및 방재기상관측 (AWS) 융합 자료 처리
            # ****************************************************************************
            # ASOS/AWS 융합 관측소
            # inpAllStnFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/ALL_STN_INFO.csv')
            # allStnData = pd.read_csv(inpAllStnFile)
            # allStnDataL1 = allStnData[['STN', 'LON', 'LAT']]
            #
            # dtDayInfo = dtDayList[0]
            # asosDataL1 = pd.DataFrame()
            # for i, dtDayInfo in enumerate(dtDayList):
            #
            #     filePathDateYmd = dtDayInfo.strftime('%Y%m/%d')
            #     fileNameDate = dtDayInfo.strftime('%Y%m%d')
            #
            #     inpAsosFilePattern = 'OBS/{}/ASOS_OBS_{}*.csv'.format(filePathDateYmd, fileNameDate)
            #     inpAsosFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, inpAsosFilePattern)
            #     fileAsosList = sorted(glob.glob(inpAsosFile))
            #
            #     if (fileAsosList is None) or (len(fileAsosList) < 1): continue
            #     log.info("[CHECK] fileAsosList : {}".format(fileAsosList))
            #
            #     for fileAsosInfo in fileAsosList:
            #         asosData = pd.read_csv(fileAsosInfo)
            #
            #         asosData['tm'] = asosData['tm'].astype(str)
            #         asosData['tm'] = pd.to_datetime(asosData['tm'], format = '%Y-%m-%d %H:%M')
            #
            #         asosDataL1 = pd.concat([asosDataL1, asosData[['tm', 'stnId', 'ta', 'rn']]], ignore_index=False)
            #
            # awsDataL1 = pd.DataFrame()
            # for i, dt3HourInfo in enumerate(dt3HourList):
            #
            #     filePathDateYmd = dt3HourInfo.strftime('%Y%m/%d')
            #     fileNameDate = dt3HourInfo.strftime('%Y%m%d%H')
            #
            #     inpAwsFilePattern = 'OBS/{}/AWS_OBS_{}*.csv'.format(filePathDateYmd, fileNameDate)
            #     inpAwsFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, inpAwsFilePattern)
            #     fileAwsList = sorted(glob.glob(inpAwsFile))
            #
            #     if (fileAwsList is None) or (len(fileAwsList) < 1): continue
            #     log.info("[CHECK] fileAwsList : {}".format(fileAwsList))
            #
            #     for fileAwsInfo in fileAwsList:
            #         awsData = pd.read_csv(fileAwsInfo, skiprows=3, skipfooter=1, header=None, delimiter='\s+')
            #
            #         # 컬럼 설정
            #         awsData.columns = ['tm', 'stnId', 'wd', 'ws', 'wds', 'wss', 'wd10', 'ws10', 'ta', 're', 'rn-15m', 'rn', 'RN-12H', 'rn-day', 'hm', 'pa', 'ps', 'td']
            #         awsData['tm'] = awsData['tm'].astype(str)
            #         awsData['tm'] = pd.to_datetime(awsData['tm'], format = '%Y%m%d%H%M')
            #
            #         awsDataL1 = pd.concat([awsDataL1, awsData[['tm', 'stnId', 'ta', 'rn']]], ignore_index=False)
            #
            # asosDataL2 = asosDataL1[['tm', 'stnId', 'ta', 'rn']]
            # awsDataL2 = awsDataL1[['tm', 'stnId', 'ta', 'rn']]
            #
            # dataL1 = pd.merge(left=asosDataL2, right=awsDataL2, how='outer', left_on=['tm', 'stnId', 'ta', 'rn'], right_on=['tm', 'stnId', 'ta', 'rn'])
            # dataL2 = dataL1.drop_duplicates(['tm', 'stnId'], ignore_index=True)
            #
            # # TM 및 STN을 기준으로 중복 제거
            # dataL2['stnId'] = dataL2['stnId'].astype(int)
            # dataL2['ta'] = pd.to_numeric(dataL2['ta'])
            # dataL2['rn'] = pd.to_numeric(dataL2['rn'])
            # dataL2['dtDateKst'] = pd.to_datetime(dataL2['tm'], format='%Y-%m-%d %H:%M')
            # dataL2['dtDateUtc'] = dataL2['dtDateKst'] - dtKst
            #
            # # dataL2.describe()
            #
            # # 결측값 제거
            # dataL3 = dataL2
            # # dataL3['ta'] = np.where(dataL3['ta'] < -50.0, np.nan, dataL3['ta'])
            # # dataL3['rn'] = np.where(dataL3['rn'] < 0.0, np.nan, dataL3['rn'])
            # dataL3['ta'] = np.where(dataL3['ta'] > -50.0, dataL3['ta'], np.nan)
            # dataL3['rn'] = np.where(dataL3['rn'] > 0.0, dataL3['rn'], 0.0)
            #
            # actDataL2 = xr.Dataset()
            # dtHourInfo = dtHourList[3]
            # dtHourInfo = dtHourList[26]
            # for ii, dtHourInfo in enumerate(dtHourList):
            #
            #     dataL4 = dataL3.loc[
            #         dataL3['dtDateKst'] == dtHourInfo
            #         ]
            #
            #     if (len(dataL4) < 1): continue
            #
            #     log.info("[CHECK] dtHourInfo : {}".format(dtHourInfo))
            #
            #     # dataL5 = pd.merge(left=dataL4, right=asosStnDataL1, how='left', left_on='stnId', right_on='STN')
            #     dataL5 = pd.merge(left=dataL4, right=allStnDataL1, how='left', left_on='stnId', right_on='STN')
            #
            #     lat1D = sorted(set(allStnDataL1['LAT']))
            #     lon1D = sorted(set(allStnDataL1['LON']))
            #     lon2D, lat2D = np.meshgrid(lon1D, lat1D)
            #
            #     varList = {}
            #     colList = ['ta', 'rn']
            #     for colInfo in colList:
            #         # if (re.match('TM|STN|dtDate', colInfo)): continue
            #         dataL6 = dataL5[['dtDateKst', 'LON', 'LAT', colInfo]].dropna()
            #
            #         if (len(dataL6) < 1): continue
            #
            #         # varList[colInfo] = np.empty((len(lon1D), len(lat1D))) * np.nan
            #         # varList[colInfo] = np.full(shape=(len(lon1D), len(lat1D)), fill_value=None)
            #         varList[colInfo] = np.full(shape=(len(lon1D), len(lat1D)), fill_value=np.nan)
            #
            #         posLon = dataL6['LON'].values
            #         posLat = dataL6['LAT'].values
            #         posVar = dataL6[colInfo].values
            #
            #
            #         # # Radial basis function (RBF) interpolation in N dimensions.
            #         try:
            #             rbfModel = Rbf(posLon, posLat, posVar, function='linear')
            #             rbfRes = rbfModel(lon2D, lat2D)
            #             # rbfRes = rbfModel(lon2D, lat2D)
            #             # rbfRes = rbfModel(lonGrid, latGrid)
            #             varList[colInfo] = rbfRes
            #         except Exception as e:
            #             log.error("Exception : {}".format(e))
            #
            #     actData = xr.Dataset(
            #         {
            #             # ASOS 및 AWS 융합
            #             'ta': (('time', 'lat', 'lon'), (varList['ta']).reshape(1, len(lat1D), len(lon1D)))
            #             , 'rn': (('time', 'lat', 'lon'), (varList['rn']).reshape(1, len(lat1D), len(lon1D)))
            #         }
            #         , coords={
            #             'time': pd.date_range(dtHourInfo, periods=1)
            #             , 'lat': lat1D
            #             , 'lon': lon1D
            #         }
            #     )
            #
            #     actDataL1 = actData.interp(lon=geoDataL2['lon'], lat=geoDataL2['lat'], method='linear')
            #     actDataL2 = xr.merge([actDataL2, actDataL1])


                # geoDataL2.plot(column=geoDataL2['KLAPS-dist'], legend=True)
                # geoDataL2.apply(lambda x: plt.annotate(text=x['fac_name'], xy=[x.lon, x.lat], ha='center', va='top', color='black'), axis=1)
                # plt.scatter(posLon, posLat, c=posVar)
                # plt.colorbar()
                # plt.show()
                #
                # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                # plt.show()
                # plt.close()


            # actDataL2.isel(lon=0, lat=0)['rn'].plot()
            # plt.show()

            # actDataL2.to_dataframe().reset_index(drop=False).describe()

            # dd['rn'] = np.where(dd['rn'] < 0.0, np.nan, dd['rn'])
            # dd.describe()

            # ************************************************************************************************
            # 수치모델 (UM, KLAPS) 데이터
            # ************************************************************************************************
            geoDataL2 = geoDataL1.dropna()
            geoDataL2['UM-nx'] = geoDataL2['UM-nx'].astype(int)
            geoDataL2['UM-ny'] = geoDataL2['UM-ny'].astype(int)
            geoDataL2['KLAPS-nx'] = geoDataL2['KLAPS-nx'].astype(int)
            geoDataL2['KLAPS-ny'] = geoDataL2['KLAPS-ny'].astype(int)

            # UM
            dtHourUtcInfo = dtHourUtcList[9]
            # umDataL4 = xr.Dataset()
            # for j, dtHourUtcInfo in enumerate(dtHourUtcList):
            #
            #     filePathDateYmdH = dtHourUtcInfo.strftime('%Y%m/%d/%H')
            #     filePathDateYmd = dtHourUtcInfo.strftime('%Y%m/%d')
            #     fileName = dtHourUtcInfo.strftime('%Y%m%d%H%M')
            #
            #     inpUmFile = '{}/{}/MODEL/{}/UMKR_l015_unis_H*_{}.grb2'.format(globalVar['inpPath'], serviceName, filePathDateYmdH, fileName)
            #     # inpUmFile = '{}/{}/MODEL/202208/01/00/UMKR_l015_unis_H*_*.grb2'.format(globalVar['inpPath'], serviceName, filePathDateYmdH, fileName)
            #     fileUmList = sorted(glob.glob(inpUmFile))
            #
            #     if (len(fileUmList) < 1): continue
            #     log.info("[CHECK] dtHourUtcInfo : {}".format(dtHourUtcInfo))
            #
            #     # fileUmInfo = fileUmList[1]
            #     for k, fileUmInfo in enumerate(fileUmList):
            #         log.info("[CHECK] fileUmInfo : {}".format(fileUmInfo))
            #
            #         # validIdx = int(re.findall('_H\d{3}', fileUmInfo)[0].replace('_H', ''))
            #         # dtValidDate = grbInfo.validDate
            #         # dtAnalDate = grbInfo.analDate
            #
            #         try:
            #             grb = pygrib.open(fileUmInfo)
            #             grbInfo = grb.select(name='Temperature')[1]
            #             dtValidDate = pd.to_datetime(grbInfo.validDate) + dtKst
            #             dtAnalDate = pd.to_datetime(grbInfo.analDate) + dtKst
            #
            #             xEle = list(set(umCfgMask['UM-nx']) | set(geoDataL2['UM-nx']))
            #             yEle = list(set(umCfgMask['UM-ny']) | set(geoDataL2['UM-ny']))
            #
            #             # NCPCP_P8_L1_GLC0_acc1h, Large-scale precipitation (non-convective), kg m-2
            #             # LSPRATE_P8_L1_GLC0_avg1h, Large scale precipitation rate, kg m-2s-1
            #             # CPRAT_P8_L1_GLC0_acc1h, Convective precipitation rate, kg m-2 s-1
            #             # umData = xr.open_dataset(fileUmInfo, decode_times=False, engine='pynio')
            #             umData = xr.open_dataset(fileUmInfo, decode_times=True, engine='pynio')
            #             # umDataL1 = umData.get(['xgrid_0', 'ygrid_0', 'gridlat_0', 'gridlon_0', 'TMP_P0_L1_GLC0', 'NCPCP_P8_L1_GLC0_acc1h'])
            #             # umDataL1 = umData.get(['xgrid_0', 'ygrid_0', 'gridlat_0', 'gridlon_0', 'TMP_P0_L1_GLC0', 'NCPCP_P8_L1_GLC0_acc1h', 'LSPRATE_P8_L1_GLC0_avg1h'])
            #             umDataL1 = umData.get(['xgrid_0', 'ygrid_0', 'gridlat_0', 'gridlon_0', 'TMP_P0_L1_GLC0', 'NCPCP_P8_L1_GLC0_acc1h', 'LSPRATE_P8_L1_GLC0_avg1h'])
            #             # umDataL2 = umDataL1
            #             # umDataL2 = umDataL1.sel(xgrid_0 = geoDataL2['UM-nx'].tolist(), ygrid_0 = geoDataL2['UM-ny'].tolist())
            #             # umDataL2 = umDataL1.sel(xgrid_0=geoDataL2['UM-nx'].tolist(), ygrid_0=geoDataL2['UM-ny'].tolist())
            #             # umDataL2 = umDataL1.sel(xgrid_0=umCfgMask['UM-nx'].tolist(), ygrid_0=umCfgMask['UM-ny'].tolist())
            #             umDataL2 = umDataL1.sel(xgrid_0=xEle, ygrid_0=yEle)
            #
            #             # umDataL1 = umData.get(['TMP_P0_L1_GLC0', 'NCPCP_P8_L1_GLC0_acc1h', 'NCPCP_P8_L1_GLC0_acc'])
            #             # umDataL2 = umDataL1.interp(xgrid_0 = posInfo['UM-nx'], ygrid_0 = posInfo['UM-ny'], method = 'linear')
            #
            #             # umDataL2['rn'] = umDataL2['LSPRATE_P8_L1_GLC0_avg1h'] + umDataL2['NCPCP_P8_L1_GLC0_acc1h']
            #
            #
            #             nx1D = umDataL2['xgrid_0'].values
            #             ny1D = umDataL2['ygrid_0'].values
            #
            #             umDataL3 = xr.Dataset(
            #                 {
            #                     'UM-ta': (('anaTime', 'time', 'ny', 'nx'), (umDataL2['TMP_P0_L1_GLC0'].values).reshape(1, 1, len(ny1D), len(nx1D)))
            #                     # , 'UM-rn': (('anaTime', 'time', 'ny', 'nx'), (umDataL2['NCPCP_P8_L1_GLC0_acc1h'].values).reshape(1, 1, len(ny1D), len(nx1D)))
            #                     # , 'UM-rn': (('anaTime', 'time', 'ny', 'nx'), (umDataL2['rn'].values).reshape(1, 1, len(ny1D), len(nx1D)))
            #                     , 'UM-rn2': (('anaTime', 'time', 'ny', 'nx'), (umDataL2['LSPRATE_P8_L1_GLC0_avg1h'].values).reshape(1, 1, len(ny1D), len(nx1D)))
            #                     , 'UM-rn3': (('anaTime', 'time', 'ny', 'nx'), (umDataL2['NCPCP_P8_L1_GLC0_acc1h'].values).reshape(1, 1, len(ny1D), len(nx1D)))
            #                 }
            #                 , coords={
            #                     'anaTime': pd.date_range(dtAnalDate, periods=1)
            #                     , 'time': pd.date_range(dtValidDate, periods=1)
            #                     , 'ny': ny1D
            #                     , 'nx': nx1D
            #                 }
            #             )
            #
            #             umDataL4 = xr.merge([umDataL4, umDataL3])
            #         except Exception as e:
            #             log.error("Exception : {}".format(e))

            # # umDataL5 = umDataL4.sel(lon = geoDataL2['UM-nx'].tolist(), lat = geoDataL2['UM-ny'].tolist())
            # # umDataL4['time'].values
            # # umDataL4.isel(anaTime=0, nx=0, ny=0)['UM-ta'].plot()
            # # umDataL4.isel(anaTime=0, nx=0, ny=0)['UM-rn'].plot()
            # # plt.show()
            #
            # # from holoviews.plotting.util import list_cmaps
            # # cmap = cm.get_cmap("Spectral")
            # # colors = cmap(a / b)
            # #
            # # import matplotlib.pyplot as plt
            # # colors = plt.get_cmap('Spectral')(np.linspace(0, 1, 10))
            # # import proplot as pplt
            #
            # # .sel(nx=geoDataL2['UM-nx'].tolist(), ny=geoDataL2['UM-nx'].tolist())['UM-ta'].
            # # umDataL4.isel(anaTime=0, time=0)['UM-ta'].plot()
            # # plt.show()
            #
            # KLAPS
            # klapsDataL4 = xr.Dataset()
            # dtHourUtcInfo = dtHourUtcList[9]
            # for j, dtHourUtcInfo in enumerate(dtHourUtcList):
            #     filePathDateYmdH = dtHourUtcInfo.strftime('%Y%m/%d/%H')
            #     filePathDateYmd = dtHourUtcInfo.strftime('%Y%m/%d')
            #     fileName = dtHourUtcInfo.strftime('%Y%m%d%H%M')
            #
            #     inpNcFile = '{}/{}/MODEL/{}/klps_lc05_anal_{}.nc'.format(globalVar['inpPath'], serviceName, filePathDateYmd, fileName)
            #     fileNcList = sorted(glob.glob(inpNcFile))
            #
            #     if (len(fileNcList) < 1): continue
            #     log.info("[CHECK] dtHourUtcInfo : {}".format(dtHourUtcInfo))
            #
            #     fileNcInfo = fileNcList[0]
            #     for k, fileNcInfo in enumerate(fileNcList):
            #
            #         xEle = list(set(klapsCfgMask['KLAPS-nx']) | set(geoDataL2['KLAPS-nx']))
            #         yEle = list(set(klapsCfgMask['KLAPS-ny']) | set(geoDataL2['KLAPS-ny']))
            #
            #         try:
            #             dtKlapsDate = pd.to_datetime(dtHourUtcInfo) + dtKst
            #
            #             klapsData = xr.open_dataset(fileNcInfo, decode_times=False, engine='pynio')
            #             klapsDataL1 = klapsData.get(['x', 'y', 'lon', 'lat', 't', 'pc'])
            #             # klapsDataL2 = klapsDataL1.sel(record=0, levels_1=0, levels_23=0).interp(x=geoDataL2['KLAPS-nx'].tolist(), y=geoDataL2['KLAPS-ny'].tolist(), method='linear')
            #             # klapsDataL2 = klapsDataL1.sel(record=0, levels_1=0, levels_23=0).interp(x=klapsCfgMask['KLAPS-nx'].tolist(), y=klapsCfgMask['KLAPS-ny'].tolist(), method='linear')
            #             klapsDataL2 = klapsDataL1.sel(record=0, levels_1=0, levels_23=0).interp(x=xEle, y=yEle, method='linear')
            #
            #             nx1D = klapsDataL2['x'].values
            #             ny1D = klapsDataL2['y'].values
            #
            #             klapsDataL3 = xr.Dataset(
            #                 {
            #                     'KLAPS-ta': (('time', 'ny', 'nx'), (klapsDataL2['t'].values).reshape(1, len(ny1D), len(nx1D)))
            #                     , 'KLAPS-rn': (('time', 'ny', 'nx'), (klapsDataL2['pc'].values).reshape(1, len(ny1D), len(nx1D)))
            #                 }
            #                 , coords={
            #                     'time': pd.date_range(dtKlapsDate, periods=1)
            #                     , 'ny': ny1D
            #                     , 'nx': nx1D
            #                 }
            #             )
            #
            #             klapsDataL4 = xr.merge([klapsDataL4, klapsDataL3])
            #         except Exception as e:
            #             log.error("Exception : {}".format(e))
            #
            # # klapsDataL4.isel(nx=0, ny=0)['KLAPS-ta'].plot()
            # # plt.show()
            # # klapsDataL4.sel(nx=128, ny=106)['KLAPS-ta'].plot()
            # # plt.show()
            #
            # klapsDataL4['KLAPS-ta'] = klapsDataL4['KLAPS-ta'] - 273.15
            # klapsDataL4['KLAPS-rn'] = klapsDataL4['KLAPS-rn'] * 3600
            # umDataL4['UM-ta'] = umDataL4['UM-ta'] - 273.15
            # umDataL4['UM-rn'] = umDataL4['UM-rn2'] * 3600 + umDataL4['UM-rn3']

            saveFile = '{}/{}/asos_aws_obs_{}-{}.nc'.format(globalVar['inpPath'], serviceName, pd.to_datetime(dtHourUtcList.min()).strftime('%Y%m%d%H'), pd.to_datetime(dtHourUtcList.max()).strftime('%Y%m%d%H'))
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # actDataL2.to_netcdf(saveFile)
            log.info('[CHECK] saveFile : {}'.format(saveFile))
            actDataL2 = xr.open_dataset(saveFile)

            saveFile = '{}/{}/klps_lc05_anal_{}-{}.nc'.format(globalVar['inpPath'], serviceName, pd.to_datetime(dtHourUtcList.min()).strftime('%Y%m%d%H'),     pd.to_datetime(dtHourUtcList.max()).strftime('%Y%m%d%H'))
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # klapsDataL4.to_netcdf(saveFile)
            log.info('[CHECK] saveFile : {}'.format(saveFile))
            klapsDataL4 = xr.open_dataset(saveFile)

            klapsDataL4['KLAPS-ta'] = klapsDataL4['KLAPS-ta'] - 273.15
            klapsDataL4['KLAPS-rn'] = klapsDataL4['KLAPS-rn'] * 3600

            saveFile = '{}/{}/UMKR_l015_unis_{}-{}.nc'.format(globalVar['inpPath'], serviceName, pd.to_datetime(dtHourUtcList.min()).strftime('%Y%m%d%H'),     pd.to_datetime(dtHourUtcList.max()).strftime('%Y%m%d%H'))
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # umDataL4.to_netcdf(saveFile)
            log.info('[CHECK] saveFile : {}'.format(saveFile))
            umDataL4 = xr.open_dataset(saveFile)

            umDataL4['UM-ta'] = umDataL4['UM-ta'] - 273.15
            # umDataL4['UM-rn'] = umDataL4['UM-rn2'] * 3600 + umDataL4['UM-rn3']
            umDataL4['UM-rn'] = umDataL4['UM-rn3']

            # 분석 시간을 기준으로 0~5시간 내의 예보 시간 사용
            rtnData = pd.DataFrame()
            for j, anaTimeInfo in enumerate(umDataL4['anaTime'].values):
                for k, timeInfo in enumerate(umDataL4['time'].values):
                    hourDiff = (timeInfo - anaTimeInfo) / pd.Timedelta(hours=1)
                    if (0 > hourDiff) or (hourDiff > 5): continue

                    dict = {
                        'anaTime': [anaTimeInfo]
                        , 'time': [timeInfo]
                    }

                    rtnData = pd.concat([rtnData, pd.DataFrame.from_dict(dict)], ignore_index=True)

            # ************************************************************************************************
            # 융합 데이터
            # ************************************************************************************************
            # [103]: umDataL1.info()
            # xarray.Dataset {
            # dimensions:
            #  xgrid_0 = 602 ;
            #  ygrid_0 = 781 ;
            # variables:
            #  int64 xgrid_0(xgrid_0) ;
            #  int64 ygrid_0(ygrid_0) ;
            #  float32 gridlat_0(ygrid_0, xgrid_0) ;
            #     gridlat_0:corners = [32.256874 32.188087 42.93468  43.018272] ;
            #     gridlat_0:long_name = latitude ;
            #     gridlat_0:grid_type = Lambert Conformal (secant, tangent, conical or bipolar) ;
            #     gridlat_0:units = degrees_north ;
            #     gridlat_0:Latin2 = [60.] ;
            #     gridlat_0:Latin1 = [30.] ;
            #     gridlat_0:Dy = [1.5] ;
            #     gridlat_0:Dx = [1.5] ;
            #     gridlat_0:Lov = [126.] ;
            #     gridlat_0:Lo1 = [121.83443] ;
            #     gridlat_0:La1 = [32.256874] ;
            #  float32 gridlon_0(ygrid_0, xgrid_0) ;
            #     gridlon_0:corners = [121.83443  131.50974  132.53188  121.060295] ;
            #     gridlon_0:long_name = longitude ;
            #     gridlon_0:grid_type = Lambert Conformal (secant, tangent, conical or bipolar) ;
            #     gridlon_0:units = degrees_east ;
            #     gridlon_0:Latin2 = [60.] ;
            #     gridlon_0:Latin1 = [30.] ;
            #     gridlon_0:Dy = [1.5] ;
            #     gridlon_0:Dx = [1.5] ;
            #     gridlon_0:Lov = [126.] ;
            #     gridlon_0:Lo1 = [121.83443] ;
            #     gridlon_0:La1 = [32.256874] ;
            #  float32 TMP_P0_L1_GLC0(ygrid_0, xgrid_0) ;
            #     TMP_P0_L1_GLC0:center = Seoul ;
            #     TMP_P0_L1_GLC0:production_status = Operational products ;
            #     TMP_P0_L1_GLC0:long_name = Temperature ;
            #     TMP_P0_L1_GLC0:units = K ;
            #     TMP_P0_L1_GLC0:grid_type = Lambert Conformal can be secant or tangent, conical or bipolar ;
            #     TMP_P0_L1_GLC0:parameter_discipline_and_category = Meteorological products, Temperature ;
            #     TMP_P0_L1_GLC0:parameter_template_discipline_category_number = [0 0 0 0] ;
            #     TMP_P0_L1_GLC0:level_type = Ground or water surface ;
            #     TMP_P0_L1_GLC0:forecast_time = [48] ;
            #     TMP_P0_L1_GLC0:forecast_time_units = hours ;
            #     TMP_P0_L1_GLC0:initial_time = 08/02/2022 (12:00) ;
            #  float32 NCPCP_P8_L1_GLC0_acc1h(ygrid_0, xgrid_0) ;
            #     NCPCP_P8_L1_GLC0_acc1h:center = Seoul ;
            #     NCPCP_P8_L1_GLC0_acc1h:production_status = Operational products ;
            #     NCPCP_P8_L1_GLC0_acc1h:long_name = Large-scale precipitation (non-convective) ;
            #     NCPCP_P8_L1_GLC0_acc1h:units = kg m-2 ;
            #     NCPCP_P8_L1_GLC0_acc1h:grid_type = Lambert Conformal can be secant or tangent, conical or bipolar ;
            #     NCPCP_P8_L1_GLC0_acc1h:parameter_discipline_and_category = Meteorological products, Moisture ;
            #     NCPCP_P8_L1_GLC0_acc1h:parameter_template_discipline_category_number = [8 0 1 9] ;
            #     NCPCP_P8_L1_GLC0_acc1h:level_type = Ground or water surface ;
            #     NCPCP_P8_L1_GLC0_acc1h:type_of_statistical_processing = Accumulation ;
            #     NCPCP_P8_L1_GLC0_acc1h:statistical_process_duration = 1 hours (ending at forecast time) ;
            #     NCPCP_P8_L1_GLC0_acc1h:forecast_time = [48] ;
            #     NCPCP_P8_L1_GLC0_acc1h:forecast_time_units = hours ;
            #     NCPCP_P8_L1_GLC0_acc1h:initial_time = 08/02/2022 (12:00) ;
            #  float32 LSPRATE_P8_L1_GLC0_avg1h(ygrid_0, xgrid_0) ;
            #     LSPRATE_P8_L1_GLC0_avg1h:center = Seoul ;
            #     LSPRATE_P8_L1_GLC0_avg1h:production_status = Operational products ;
            #     LSPRATE_P8_L1_GLC0_avg1h:long_name = Large scale precipitation rate ;
            #     LSPRATE_P8_L1_GLC0_avg1h:units = kg m-2s-1 ;
            #     LSPRATE_P8_L1_GLC0_avg1h:grid_type = Lambert Conformal can be secant or tangent, conical or bipolar ;
            #     LSPRATE_P8_L1_GLC0_avg1h:parameter_discipline_and_category = Meteorological products, Moisture ;
            #     LSPRATE_P8_L1_GLC0_avg1h:parameter_template_discipline_category_number = [ 8  0  1 54] ;
            #     LSPRATE_P8_L1_GLC0_avg1h:level_type = Ground or water surface ;
            #     LSPRATE_P8_L1_GLC0_avg1h:type_of_statistical_processing = Average ;
            #     LSPRATE_P8_L1_GLC0_avg1h:statistical_process_duration = 1 hours (ending at forecast time) ;
            #     LSPRATE_P8_L1_GLC0_avg1h:forecast_time = [48] ;
            #     LSPRATE_P8_L1_GLC0_avg1h:forecast_time_units = hours ;
            #     LSPRATE_P8_L1_GLC0_avg1h:initial_time = 08/02/2022 (12:00) ;
            # // global attributes:
            # }


            # rn3 NCPCP_P8_L1_GLC0_acc1h, Large-scale precipitation (non-convective), kg m-2
            # rn2 LSPRATE_P8_L1_GLC0_avg1h, Large scale precipitation rate, kg m-2s-1
            # CPRAT_P8_L1_GLC0_acc1h, Convective precipitation rate, kg m-2 s-1


            posInfo = geoDataL2.iloc[1]
            validDataL1 = pd.DataFrame()
            validStatDataL2 = pd.DataFrame()
            for i, posInfo in geoDataL2.iterrows():

                # 최근접 위경도를 기준으로 ASOS, KLAPS, UM 데이터 가져오기
                actSelData = actDataL2.sel(lon = posInfo['lon'], lat = posInfo['lat']).to_dataframe().reset_index(drop=False).drop_duplicates(['time', 'lon', 'lat'], ignore_index=True)
                klapsSelData = klapsDataL4.sel(nx=posInfo['KLAPS-nx'], ny=posInfo['KLAPS-ny']).to_dataframe().reset_index(drop=False).drop_duplicates(['time', 'nx', 'ny'], ignore_index=True)
                umSelData = umDataL4.sel(nx = posInfo['UM-nx'], ny = posInfo['UM-ny']).to_dataframe().reset_index(drop=False).drop_duplicates(['anaTime', 'time', 'nx', 'ny'], ignore_index=True)
                umSelData = pd.merge(left=rtnData, right=umSelData, how='left', left_on=['time', 'anaTime'], right_on=['time', 'anaTime'])
                umSelData['anaDate'] = umSelData['anaTime'].dt.strftime("%Y%m%d%H").astype(str)

                validData = actSelData
                validData['gid'] = posInfo['gid']

                # KLAPS 데이터
                validData = pd.merge(left=validData, right=klapsSelData[['time', 'KLAPS-ta', 'KLAPS-rn']], how='left', left_on=['time'], right_on=['time'])

                # UM 데이터
                validData = pd.merge(left=validData, right=umSelData[['time', 'UM-ta', 'UM-rn']], how='left', left_on=['time'], right_on=['time'])
                # anaDateInfo = anaDateList[0]
                # anaDateList = set(umSelData['anaDate'])
                # for j, anaDateInfo in enumerate(anaDateList):
                #     umSelDataL1 = umSelData.loc[umSelData['anaDate'] == anaDateInfo]
                #     umSelDataL2 = umSelDataL1.reset_index(drop=True).rename(
                #         columns = { 'UM-ta' : 'UM-ta-' + anaDateInfo, 'UM-rn' : 'UM-rn-' + anaDateInfo}
                #         # columns = { 'UM-ta' : 'UM-ta-' + anaDateInfo, 'UM-rn3' : 'UM-rn-' + anaDateInfo}
                #         # columns = { 'UM-ta' : 'UM-ta-' + anaDateInfo, 'UM-rn3' : 'UM-rn-' + anaDateInfo}
                #     )
                #
                #     validData = pd.merge(left=validData, right=umSelDataL2[['time', 'UM-ta-' + anaDateInfo, 'UM-rn-' + anaDateInfo]], how='left', left_on=['time'], right_on=['time'])


                # 인접한 폴리곤을 기준으로 KLAPS 데이터 가져오기
                klapsAreaData = pd.DataFrame()
                klapsCfgMaskL1 = klapsCfgMask.loc[klapsCfgMask['gid'] == posInfo['gid']]
                for j, klapsInfo in klapsCfgMaskL1.iterrows():
                    klapsTmpData = klapsDataL4.sel(nx=klapsInfo['KLAPS-nx'], ny=klapsInfo['KLAPS-ny']).to_dataframe().reset_index(drop=False).drop_duplicates(['time', 'nx', 'ny'], ignore_index=True)
                    klapsAreaData = pd.concat([klapsAreaData, klapsTmpData], ignore_index=True)

                if (len(klapsAreaData) > 0):
                    klapsAreaDataL1 = klapsAreaData.groupby(by=['time']).mean().reset_index(drop=False).rename(
                        columns={'KLAPS-ta' : 'KLAPS-ta-area', 'KLAPS-rn' : 'KLAPS-rn-area'}
                    )

                    validData = pd.merge(left=validData, right=klapsAreaDataL1[['time', 'KLAPS-ta-area', 'KLAPS-rn-area']], how='left', left_on=['time'], right_on=['time'])

                # 인접한 폴리곤을 기준으로 UM 데이터 가져오기
                umAreaData = pd.DataFrame()
                umCfgMaskL1 = umCfgMask.loc[umCfgMask['gid'] == posInfo['gid']]
                for j, umInfo in umCfgMaskL1.iterrows():
                    umTmpData = umDataL4.sel(nx=umInfo['UM-nx'], ny=umInfo['UM-ny']).to_dataframe().reset_index(drop=False).drop_duplicates(['anaTime', 'time', 'nx', 'ny'], ignore_index=True)
                    umTmpData = pd.merge(left=rtnData, right=umTmpData, how='left', left_on=['time', 'anaTime'], right_on=['time', 'anaTime'])
                    umAreaData = pd.concat([umAreaData, umTmpData], ignore_index=True)

                if (len(umAreaData) > 0):
                    umAreaDataL1 = umAreaData.groupby(by=['anaTime', 'time']).mean().reset_index(drop=False).rename(
                        columns={'UM-ta' : 'UM-ta-area', 'UM-rn' : 'UM-rn-area'}
                        # columns={'UM-ta' : 'UM-area-ta', 'UM-rn3' : 'UM-area-rn'}
                    )
                    umAreaDataL1['anaDate'] = umAreaDataL1['anaTime'].dt.strftime("%Y%m%d%H").astype(str)

                    validData = pd.merge(left=validData, right=umAreaDataL1[['time', 'UM-ta-area', 'UM-rn-area']], how='left', left_on=['time'], right_on=['time'])

                    # # anaDateInfo = anaDateList[0]
                    # anaDateList = set(umAreaDataL1['anaDate'])
                    # for j, anaDateInfo in enumerate(anaDateList):
                    #     umAreaDataL2 = umAreaDataL1.loc[umAreaDataL1['anaDate'] == anaDateInfo]
                    #     umAreaDataL3 = umAreaDataL2.reset_index(drop=True).rename(
                    #         columns = { 'UM-area-ta' : 'UM-area-ta-' + anaDateInfo, 'UM-area-rn' : 'UM-area-rn-' + anaDateInfo}
                    #     )
                    #
                    #     validData = pd.merge(left=validData, right=umAreaDataL3[['time', 'UM-area-ta-' + anaDateInfo, 'UM-area-rn-' + anaDateInfo]], how='left', left_on=['time'], right_on=['time'])

                validDataL1 = pd.concat([validDataL1, validData], ignore_index=False)

                validStatData = validData
                validStatData['sDate'] = pd.to_datetime(validStatData['time']).dt.strftime('%Y%m%d')
                validStatData['date'] = pd.to_datetime(validStatData['sDate'], format='%Y%m%d')
                validStatDataL1 = validStatData.groupby(['date']).sum().reset_index(drop=False)

                validStatDataL2 = pd.concat([validStatDataL2, validStatDataL1], ignore_index=False)


                # validData.describe()

                minDate = pd.to_datetime(validData['time']).min().strftime("%Y%m%d")
                maxDate = pd.to_datetime(validData['time']).max().strftime("%Y%m%d")

                # 온도
                mainTitle = '[{}-{}] 기상 정보를 활용한 {} 시계열 ({})'.format(minDate, maxDate, '온도', posInfo['fac_name'])
                saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                plt.plot(validData['time'], validData['ta'], marker='o', label='OBS')

                colList = validData.loc[:, validData.columns.str.find('KLAPS-ta') >= 0].columns
                for j, colInfo in enumerate(sorted(set(colList))):
                    if (len(validData[colInfo]) < 1): continue
                    plt.plot(validData['time'], validData[colInfo], marker='o', label=colInfo)

                colList = validData.loc[:, validData.columns.str.find('UM-ta') >= 0].columns
                if (len(colList) > 0):
                    for j, colInfo in enumerate(sorted(set(colList))):
                        if (len(validData[colInfo]) < 1): continue
                        # if (not colInfo.split('-ta-')[1][8:10] == '21'): continue

                        plt.plot(validData['time'], validData[colInfo], label=colInfo)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%d %H'))
                plt.legend()
                plt.title(mainTitle)
                plt.xticks(rotation=45)
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                plt.show()
                plt.close()


                # 강수량
                mainTitle = '[{}-{}] 기상 정보를 활용한 {} 시계열 ({})'.format(minDate, maxDate, '강수량', posInfo['fac_name'])
                saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                plt.plot(validData['time'], validData['rn'], marker='o', label='OBS')

                colList = validData.loc[:, validData.columns.str.find('KLAPS-rn') >= 0].columns
                for j, colInfo in enumerate(sorted(set(colList))):
                    if (len(validData[colInfo]) < 1): continue
                    plt.plot(validData['time'], validData[colInfo], marker='o', label=colInfo)

                colList = validData.loc[:, validData.columns.str.find('UM-rn') >= 0].columns
                for j, colInfo in enumerate(sorted(set(colList))):
                    if (len(validData[colInfo]) < 1): continue
                    # if (not colInfo.split('-rn-')[1][8:10] == '21'): continue
                    plt.plot(validData['time'], validData[colInfo], label=colInfo)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%d %H'))
                plt.legend()
                plt.title(mainTitle)
                plt.xticks(rotation=45)
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                plt.show()
                plt.close()

                # 일별 강수량
                mainTitle = '[{}-{}] 기상 정보를 활용한 {} 일별 시계열 ({})'.format(minDate, maxDate, '강수량', posInfo['fac_name'])
                saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                plt.plot(validStatDataL1['date'], validStatDataL1['rn'], marker='o', label='OBS')

                colList = validStatDataL1.loc[:, validStatDataL1.columns.str.find('KLAPS-rn') >= 0].columns
                for j, colInfo in enumerate(sorted(set(colList))):
                    if (len(validStatDataL1[colInfo]) < 1): continue
                    plt.plot(validStatDataL1['date'], validStatDataL1[colInfo], marker='o', label=colInfo)

                colList = validStatDataL1.loc[:, validStatDataL1.columns.str.find('UM-rn') >= 0].columns
                for j, colInfo in enumerate(sorted(set(colList))):
                    if (len(validStatDataL1[colInfo]) < 1): continue
                    # if (not colInfo.split('-rn-')[1][8:10] == '21'): continue
                    plt.plot(validStatDataL1['date'], validStatDataL1[colInfo], label=colInfo)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%d %H'))
                plt.legend()
                plt.title(mainTitle)
                plt.xticks(rotation=45)
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                plt.show()
                plt.close()

            minDate = pd.to_datetime(validDataL1['time']).min().strftime("%Y%m%d")
            maxDate = pd.to_datetime(validDataL1['time']).max().strftime("%Y%m%d")

            # 온도
            colList = validDataL1.loc[:, validDataL1.columns.str.find('-ta') >= 0].columns
            for j, colInfo in enumerate(sorted(set(colList))):
                if (len(validData[colInfo]) < 1): continue

                validDataL2 = validDataL1[['time', 'ta', colInfo]].dropna()
                if (len(validDataL2) < 1): continue

                mainTitle = '[{}-{}] 기상 정보를 활용한 {} 2D 산점도 ({})'.format(minDate, maxDate, '온도', colInfo)
                saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                rtnInfo = makeUserHist2dPlot(validDataL2[colInfo], validDataL2['ta'], '수치모델 ({})'.format(colInfo.split('-')[0]), '실측 (OBS)', mainTitle, saveImg, 20, 40, 0.5, 1, 20, True)
                log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

            # 강수량
            colInfo = colList[0]
            colList = validDataL1.loc[:, validDataL1.columns.str.find('-rn') >= 0].columns
            for j, colInfo in enumerate(sorted(set(colList))):
                if (len(validData[colInfo]) < 1): continue

                validDataL2 = validDataL1[['time', 'rn', colInfo]].dropna()
                if (len(validDataL2) < 1): continue

                mainTitle = '[{}-{}] 기상 정보를 활용한 {} 2D 산점도 ({})'.format(minDate, maxDate, '강수량', colInfo)
                saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                rtnInfo = makeUserHist2dPlot(validDataL2[colInfo], validDataL2['rn'], '수치모델 ({})'.format(colInfo.split('-')[0]), '실측 (OBS)', mainTitle, saveImg, 0, 20, 0.5, 1.0, 30, True)
                log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

            # 일별 강수량
            validStatDataL3 = validStatDataL2

            colInfo = colList[0]
            colList = validStatDataL3.loc[:, validStatDataL3.columns.str.find('-rn') >= 0].columns
            for j, colInfo in enumerate(sorted(set(colList))):
                if (len(validStatDataL3[colInfo]) < 1): continue

                validStatDataL4 = validStatDataL3[['date', 'rn', colInfo]].dropna()
                if (len(validStatDataL4) < 1): continue

                # 강수 빈도
                # validStatDataL4['rn'] = np.where(validStatDataL4['rn'] > 0.0, 1.0, 0.0)
                # validStatDataL4[colInfo] = np.where(validStatDataL4[colInfo] > 0.0, 1.0, 0.0)

                mainTitle = '[{}-{}] 기상 정보를 활용한 {} 일별 2D 산점도 ({})'.format(minDate, maxDate, '강수량', colInfo)
                saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                rtnInfo = makeUserHist2dPlot(validStatDataL4[colInfo], validStatDataL4['rn'], '수치모델 ({})'.format(colInfo.split('-')[0]), '실측 (OBS)', mainTitle, saveImg, 0, 60, 1.0, 3.0, 30, True)
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
