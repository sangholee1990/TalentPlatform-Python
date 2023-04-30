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
# import ray
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

import xagg as xa
import rasterio
import odc.geo.xr
import xarray as xr
# import geopandas as gpd
import matplotlib.font_manager as fm
from pandas.tseries.offsets import Day, Hour, Minute, Second

import urllib
from urllib import parse
from sspipe import p, px
import matplotlib.dates as mdates


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


# def subAtmosActData(cfgInfo, posDataL1, dtIncDateList):
#     log.info('[START] {}'.format('subAtmosActData'))
#     result = None
#
#     try:
#         lat1D = np.array(posDataL1['LAT'])
#         lon1D = np.array(posDataL1['LON'])
#         lon2D, lat2D = np.meshgrid(lon1D, lat1D)
#
#         # ASOS 설정 정보
#         # inpAsosStnFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/ASOS_STN_INFO.csv')
#         # asosStnData = pd.read_csv(inpAsosStnFile)
#         # asosStnDataL1 = asosStnData[['STN', 'LON', 'LAT']]
#
#         # ALL 설정 정보
#         inpAllStnFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/ALL_STN_INFO.csv')
#         allStnData = pd.read_csv(inpAllStnFile)
#         allStnDataL1 = allStnData[['STN', 'LON', 'LAT']]
#         # allStnDataL1 = allStnDataL1.sort_values(by=['LON'], ascending=True).sort_values(by=['LAT'], ascending=True)
#
#         # GK2A 설정 정보
#         # cfgFile = '{}/{}'.format(globalVar['cfgPath'], 'satInfo/gk2a_ami_le2_cld_ko020lc_202009010000.nc')
#         # cfgData = xr.open_dataset(cfgFile)
#
#         # 위/경도 반환
#         # imgProjInfo = cfgData['gk2a_imager_projection'].attrs
#
#         # 1) ccrs 사용
#         # mapLccProj = ccrs.LambertConformal(
#         #     central_longitude=imgProjInfo['central_meridian']
#         #     , central_latitude=imgProjInfo['origin_latitude']
#         #     , secant_latitudes=(imgProjInfo['standard_parallel1'], imgProjInfo['standard_parallel2'])
#         #     , false_easting=imgProjInfo['false_easting']
#         #     , false_northing=imgProjInfo['false_northing']
#         # )
#         #
#         # try:
#         #     mapLccProjInfo = mapLccProj.to_proj4()
#         # except Exception as e:
#         #     log.error("Exception : {}".format(e))
#
#         # 2) Proj 사용
#         # mapLccProjInfo = '+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs'
#         # mapProj = pyproj.Proj(mapLccProjInfo)
#
#         # nx = imgProjInfo['image_width']
#         # ny = imgProjInfo['image_height']
#         # xOffset = imgProjInfo['lower_left_easting']
#         # yOffset = imgProjInfo['lower_left_northing']
#         #
#         # res = imgProjInfo['pixel_size']
#
#         # 직교 좌표
#         # rowEle = (np.arange(0, nx, 1) * res) + xOffset
#         # colEle = (np.arange(0, ny, 1) * res) + yOffset
#         # colEle = colEle[::-1]
#
#         # posRow, posCol = mapProj(lon1D, lat1D, inverse=False)
#
#         # dtIncDateInfo = dtIncDateList[0]
#         for ii, dtIncDateInfo in enumerate(dtIncDateList):
#             log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))
#
#             # if (dtIncDateInfo < pd.to_datetime('2021-10-01 09', format='%Y-%m-%d %H')): continue
#             # if (dtIncDateInfo < pd.to_datetime('2021-10-01 02:40', format='%Y-%m-%d %H:%M')): continue
#
#             dtAsosDatePath = dtIncDateInfo.strftime('%Y%m/%d')
#             # dtGk2aDatePath = dtIncDateInfo.strftime('%Y%m/%d/%H')
#             # dtH8DatePath = dtIncDateInfo.strftime('%Y%m/%d/%H')
#
#             dtAsosDateName = dtIncDateInfo.strftime('%Y%m%d')
#             # dtGk2aDateName = dtIncDateInfo.strftime('%Y%m%d%H%M')
#             # dtH8DateName = dtIncDateInfo.strftime('%Y%m%d_%H%M')
#
#             # ************************************************************************************************
#             # ASOS, AWS 융합 데이터
#             # ************************************************************************************************
#             inpAsosFilePattern = 'OBS/{}/ASOS_OBS_{}*.txt'.format(dtAsosDatePath, dtAsosDateName)
#             inpAsosFile = '{}/{}'.format(globalVar['inpPath'], inpAsosFilePattern)
#             fileList = sorted(glob.glob(inpAsosFile))
#
#             if fileList is None or len(fileList) < 1:
#                 log.error('[ERROR] inpAsosFile : {} / {}'.format(inpAsosFile, '입력 자료를 확인해주세요.'))
#                 continue
#
#             # log.info("[CHECK] fileList : {}".format(fileList))
#
#             dataL1 = pd.DataFrame()
#             for fileInfo in fileList:
#                 data = pd.read_csv(fileInfo, header=None, delimiter='\s+')
#
#                 # 컬럼 설정
#                 data.columns = ['TM', 'STN', 'WD', 'WS', 'GST_WD', 'GST_WS', 'GST_TM', 'PA', 'PS', 'PT', 'PR', 'TA',
#                                 'TD', 'HM', 'PV', 'RN', 'RN-DAY', 'TMP', 'RN_INT', 'SD_HR3', 'SD_DAY', 'SD_TOT', 'WC',
#                                 'WP', 'WW', 'CA_TOT', 'CA_MID', 'CH_MIN', 'CT', 'CT_TOP', 'CT_MID', 'CT_LOW', 'VS',
#                                 'SS', 'SI', 'ST_GD', 'TS', 'TE_005', 'TE_01', 'TE_02', 'TE_03', 'ST_SEA', 'WH', 'BF',
#                                 'IR', 'IX']
#                 # data = data[['TM', 'STN', 'WD', 'WS', 'PA', 'TA', 'TD', 'HM', 'CA_TOT']]
#                 data = data[['TM', 'STN', 'TA', 'RN-60m', 'RN-DAY']]
#                 dataL1 = pd.concat([dataL1, data], ignore_index=False)
#
#             inpAwsFilePattern = 'OBS/{}/AWS_OBS_{}*.txt'.format(dtAsosDatePath, dtAsosDateName)
#             inpAwsFile = '{}/{}'.format(globalVar['inpPath'], inpAwsFilePattern)
#             fileList = sorted(glob.glob(inpAwsFile))
#
#             if fileList is None or len(fileList) < 1:
#                 log.error('[ERROR] inpAwsFile : {} / {}'.format(inpAwsFile, '입력 자료를 확인해주세요.'))
#
#             # log.info("[CHECK] fileList : {}".format(fileList))
#
#             for fileInfo in fileList:
#                 data = pd.read_csv(fileInfo, header=None, delimiter='\s+')
#
#                 # 컬럼 설정
#                 data.columns = ['TM', 'STN', 'WD', 'WS', 'WDS', 'WSS', 'WD10', 'WS10', 'TA', 'RE', 'RN-15m', 'RN-60m',
#                                 'RN-12H', 'RN-DAY', 'HM', 'PA', 'PS', 'TD']
#                 data = data[['TM', 'STN', 'TA', 'RN-60m', 'RN-DAY']]
#
#                 dataL1 = pd.concat([dataL1, data], ignore_index=False)
#
#             dataL2 = dataL1
#
#             # TM 및 STN을 기준으로 중복 제거
#             dataL2['TM'] = dataL2['TM'].astype(str)
#             dataL2.drop_duplicates(subset=['TM', 'STN'], inplace=True)
#
#             # 결측값 제거
#             dataL3 = dataL2
#             dataL3['dtDate'] = pd.to_datetime(dataL3['TM'].astype(str), format='%Y%m%d%H%M')
#
#             dataL4 = dataL3.loc[
#                 dataL3['dtDate'] == dtIncDateInfo
#                 ]
#
#             # dataL4['WD'][dataL4['WD'] < 0] = np.nan
#             # dataL4['WS'][dataL4['WS'] < 0] = np.nan
#             # dataL4['PA'][dataL4['PA'] < 0] = np.nan
#             dataL4['TA'][dataL4['TA'] < -50.0] = np.nan
#             dataL4['RN-60m'][dataL4['RN-60m'] < 0.0] = np.nan
#             dataL4['RN-DAY'][dataL4['RN-DAY'] < 0.0] = np.nan
#             # dataL4['TD'][dataL4['TD'] < -50.0] = np.nan
#             # dataL4['HM'][dataL4['HM'] < 0] = np.nan
#             # dataL4['CA_TOT'][dataL4['CA_TOT'] < 0] = np.nan
#
#             # 풍향, 풍속을 이용해서 U,V 벡터 변환
#             # dataL4['uVec'], dataL4['vVec'] = wind_components(dataL4['WS'].values * units('m/s'), dataL4['WS'].values * units.deg)
#
#             # statData = dataL4.describe()
#
#             dataL5 = pd.merge(left=dataL4, right=allStnDataL1, how='left', left_on='STN', right_on='STN')
#
#             # colInfo = 'WD'
#             # colInfo = 'uVec'
#             # colInfo = 'CA_TOT'
#
#             varList = {}
#             colList = dataL4.columns
#             for colInfo in colList:
#                 if (re.match('TM|STN|dtDate', colInfo)): continue
#
#                 # varList[colInfo] = np.empty((len(lon1D), len(lat1D))) * np.nan
#                 # varList[colInfo] = np.full(shape=(len(lon1D), len(lat1D)), fill_value=None)
#                 varList[colInfo] = np.full(shape=(len(lon1D), len(lat1D)), fill_value=np.nan)
#
#                 dataL6 = dataL5[['dtDate', 'LON', 'LAT', colInfo]].dropna()
#
#                 if (len(dataL6) < 1): continue
#
#                 posLon = dataL6['LON'].values
#                 posLat = dataL6['LAT'].values
#                 posVar = dataL6[colInfo].values
#
#                 # Radial basis function (RBF) interpolation in N dimensions.
#                 try:
#                     rbfModel = Rbf(posLon, posLat, posVar, function='linear')
#                     rbfRes = rbfModel(lon2D, lat2D)
#                     varList[colInfo] = rbfRes
#                 except Exception as e:
#                     log.error("Exception : {}".format(e))
#
#             #  U,V 벡터를 이용해서 풍향, 풍속 변환
#             # varList['WD'] = wind_direction(varList['uVec']* units('m/s'), varList['vVec'] * units('m/s'), convention='from')
#             # varList['WS'] = wind_speed(varList['uVec']* units('m/s'), varList['vVec'] * units('m/s'))
#
#             # ************************************************************************************************
#             # GK2A 데이터
#             # ************************************************************************************************
#             # # inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/SAT/GK2A_*.nc')
#             # inpFilePattern = 'SAT/{}/gk2a_*_{}.nc'.format(dtGk2aDatePath, dtGk2aDateName)
#             # inpFile = '{}/{}'.format(globalVar['inpPath'], inpFilePattern)
#             # fileList = sorted(glob.glob(inpFile))
#             #
#             # if fileList is None or len(fileList) < 1:
#             #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
#             #     continue
#             #
#             # # log.info("[CHECK] fileList : {}".format(fileList))
#             #
#             # gk2aData = xr.open_mfdataset(fileList)
#             # gk2aDataL1 = gk2aData.assign_coords(
#             #     {"dim_x": ("dim_x", rowEle)
#             #         , "dim_y": ("dim_y", colEle)
#             #      }
#             # )
#             #
#             # selGk2aNearData = gk2aDataL1.sel(dim_x=posRow, dim_y=posCol, method='nearest')
#             # selGk2aIntpData = gk2aDataL1.interp(dim_x=posRow, dim_y=posCol)
#
#             # ************************************************************************************************
#             # Himawari-8 데이터
#             # ************************************************************************************************
#             # # inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/SAT/GK2A_*.nc')
#             # inpFilePattern = 'SAT/{}/H08_{}_*.nc'.format(dtH8DatePath, dtH8DateName)
#             # inpFile = '{}/{}'.format(globalVar['inpPath'], inpFilePattern)
#             # fileList = sorted(glob.glob(inpFile))
#             #
#             # if fileList is None or len(fileList) < 1:
#             #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
#             #     continue
#             #
#             # # log.info("[CHECK] fileList : {}".format(fileList))
#             #
#             # h8Data = xr.open_mfdataset(fileList)
#             #
#             # selH8NearData = h8Data.sel(latitude=lat1D, longitude=lon1D, method='nearest')
#             # selH8IntpData = h8Data.interp(latitude=lat1D, longitude=lon1D)
#
#             # ************************************************************************************************
#             # 융합 데이터
#             # ************************************************************************************************
#             try:
#                 actData = xr.Dataset(
#                     {
#                         # ASOS 및 AWS 융합
#                         'TA': (('time', 'lat', 'lon'), (varList['TA']).reshape(1, len(lat1D), len(lon1D)))
#                         , 'RN-60m': (('time', 'lat', 'lon'), (varList['RN-60m']).reshape(1, len(lat1D), len(lon1D)))
#                         , 'RN-DAY': (('time', 'lat', 'lon'), (varList['RN-DAY']).reshape(1, len(lat1D), len(lon1D)))
#
#                         # H8
#                         # , 'PAR': ( ('time', 'lat', 'lon'), (selH8NearData['PAR'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'SWR': ( ('time', 'lat', 'lon'), (selH8NearData['SWR'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'TAAE': (('time', 'lat', 'lon'), (selH8NearData['TAAE'].values).reshape(1, len(lat1D), len(lon1D)))
#                         # , 'TAOT_02': ( ('time', 'lat', 'lon'), (selH8NearData['TAOT_02'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'UVA': ( ('time', 'lat', 'lon'), (selH8NearData['UVA'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'UVB': ( ('time', 'lat', 'lon'), (selH8NearData['UVB'].values).reshape(1, len(lat1D), len(lon1D)) )
#
#                         # , 'PAR_intp': ( ('time', 'lat', 'lon'), (selH8IntpData['PAR'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'SWR_intp': ( ('time', 'lat', 'lon'), (selH8IntpData['SWR'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'TAAE_intp': (('time', 'lat', 'lon'), (selH8IntpData['TAAE'].values).reshape(1, len(lat1D), len(lon1D)))
#                         # , 'TAOT_02_intp': ( ('time', 'lat', 'lon'), (selH8IntpData['TAOT_02'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'UVA_intp': ( ('time', 'lat', 'lon'), (selH8IntpData['UVA'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'UVB_intp': ( ('time', 'lat', 'lon'), (selH8IntpData['UVB'].values).reshape(1, len(lat1D), len(lon1D)) )
#
#                         # GK2A
#                         # , 'CA': ( ('time', 'lat', 'lon'), (selGk2aNearData['CA'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'CF': ( ('time', 'lat', 'lon'), (selGk2aNearData['CF'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'CLD': (('time', 'lat', 'lon'), (selGk2aNearData['CLD'].values).reshape(1, len(lat1D), len(lon1D)))
#                         # , 'DSR': ( ('time', 'lat', 'lon'), (selGk2aNearData['DSR'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'ASR': ( ('time', 'lat', 'lon'), (selGk2aNearData['ASR'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'RSR': ( ('time', 'lat', 'lon'), (selGk2aNearData['RSR'].values).reshape(1, len(lat1D), len(lon1D)) )
#
#                         # , 'CA_intp': ( ('time', 'lat', 'lon'), (selGk2aIntpData['CA'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'CF_intp': ( ('time', 'lat', 'lon'), (selGk2aIntpData['CF'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'CLD_intp': (('time', 'lat', 'lon'), (selGk2aIntpData['CLD'].values).reshape(1, len(lat1D), len(lon1D)))
#                         # , 'DSR_intp': ( ('time', 'lat', 'lon'), (selGk2aIntpData['DSR'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'ASR_intp': ( ('time', 'lat', 'lon'), (selGk2aIntpData['ASR'].values).reshape(1, len(lat1D), len(lon1D)) )
#                         # , 'RSR_intp': ( ('time', 'lat', 'lon'), (selGk2aIntpData['RSR'].values).reshape(1, len(lat1D), len(lon1D)) )
#                     }
#                     , coords={
#                         'time': pd.date_range(dtIncDateInfo, periods=1)
#                         , 'lat': lat1D
#                         , 'lon': lon1D
#                     }
#                 )
#
#             except Exception as e:
#                 log.error("Exception : {}".format(e))
#
#             # plt.scatter(posLon, posLat, c=posVar)
#             # plt.colorbar()
#             # plt.show()
#
#             # actDataL1 = actData.isel(time=0)
#             # plt.scatter(lon1D, lat1D, c=np.diag(actDataL1['WD']))
#             # plt.colorbar()
#             # plt.show()
#
#             # actDataL1 = actData.isel(time=0)
#             # plt.scatter(lon1D, lat1D, c=np.diag(actDataL1['WS']))
#             # plt.colorbar()
#             # plt.show()
#
#             for kk, posInfo in posDataL1.iterrows():
#                 # log.info("[CHECK] posId (posLon, posLat) : {} ({}. {})".format(posId, posLon, posLat))
#                 if (len(actData) < 1): continue
#
#                 posId = int(posInfo['ID'])
#                 posLat = posInfo['LAT']
#                 posLon = posInfo['LON']
#                 # posSza = posInfo['STN_SZA']
#                 # posAza = posInfo['STN_AZA']
#
#                 try:
#                     actDataL2 = actData.sel(lat=posLat, lon=posLon)
#                     if (len(actDataL2) < 1): continue
#
#                     # actDataL3 = actDataL2.to_dataframe().dropna().reset_index(drop=True)
#                     actDataL3 = actDataL2.to_dataframe().reset_index(drop=True)
#                     # actDataL3['dtDate'] = pd.to_datetime(dtanaDateInfo) + (actDataL3.index.values * datetime.timedelta(hours=1))
#                     actDataL3['DATE_TIME'] = pd.to_datetime(dtIncDateInfo)
#                     # actDataL3['dtDateKst'] = actDataL3.index.tz_localize(tzUtc).tz_convert(tzKst)
#                     actDataL3['DATE_TIME_KST'] = actDataL3['DATE_TIME'] + dtKst
#                     actDataL4 = actDataL3
#                     # actDataL5 = actDataL4[[
#                     #     'DATE_TIME_KST', 'DATE_TIME', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS'
#                     #     , 'SWR', 'TAAE', 'UVA', 'UVB', 'SWR_intp', 'TAAE_intp', 'UVA_intp', 'UVB_intp'
#                     #     , 'CA', 'CF', 'CLD', 'DSR', 'ASR', 'RSR', 'CA_intp', 'CF_intp', 'CLD_intp', 'DSR_intp', 'ASR_intp', 'RSR_intp'
#                     # ]]
#
#                     actDataL5 = actDataL4[['DATE_TIME_KST', 'DATE_TIME', 'TA', 'TD', 'RN-60m', 'RN-DAY']]
#                     # actDataL5 = actDataL4[[
#                     #     'DATE_TIME_KST', 'DATE_TIME', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS'
#                     #     , 'SWR', 'TAAE', 'UVA', 'UVB', 'SWR_intp', 'TAAE_intp', 'UVA_intp', 'UVB_intp'
#                     #     , 'CA', 'CF', 'CLD', 'DSR', 'ASR', 'RSR', 'CA_intp', 'CF_intp', 'CLD_intp', 'DSR_intp', 'ASR_intp', 'RSR_intp'
#                     # ]]
#                     actDataL5['SRV'] = 'SRV{:05d}'.format(posId)
#                     # actDataL5['TA'] = actDataL5['TA'] - 273.15
#                     # actDataL5['TD'] = actDataL5['TD'] - 273.15
#                     # actDataL5['PA'] = actDataL5['PA'] / 100.0
#                     actDataL5['CA_TOT'] = actDataL5['CA_TOT'] / 10.0
#                     actDataL5['CA_TOT'] = np.where(actDataL5['CA_TOT'] < 0, 0, actDataL5['CA_TOT'])
#                     actDataL5['CA_TOT'] = np.where(actDataL5['CA_TOT'] > 1, 1, actDataL5['CA_TOT'])
#
#                     # solPosInfo = pvlib.solarposition.get_solarposition(pd.to_datetime(actDataL5['DATE_TIME'].values),
#                     #                                                    posLat, posLon,
#                     #                                                    pressure=actDataL5['PA'].values * 100.0,
#                     #                                                    temperature=actDataL5['TA'].values,
#                     #                                                    method='nrel_numpy')
#                     # actDataL5['SZA'] = solPosInfo['apparent_zenith'].values
#                     # actDataL5['AZA'] = solPosInfo['azimuth'].values
#                     # actDataL5['ET'] = solPosInfo['equation_of_time'].values
#                     #
#                     # # pvlib.location.Location.get_clearsky()
#                     # site = location.Location(posLat, posLon, tz='Asia/Seoul')
#                     # clearInsInfo = site.get_clearsky(pd.to_datetime(actDataL5['DATE_TIME'].values))
#                     # actDataL5['GHI_CLR'] = clearInsInfo['ghi'].values
#                     # actDataL5['DNI_CLR'] = clearInsInfo['dni'].values
#                     # actDataL5['DHI_CLR'] = clearInsInfo['dhi'].values
#                     #
#                     # poaInsInfo = irradiance.get_total_irradiance(
#                     #     surface_tilt=posSza,
#                     #     surface_azimuth=posAza,
#                     #     dni=clearInsInfo['dni'],
#                     #     ghi=clearInsInfo['ghi'],
#                     #     dhi=clearInsInfo['dhi'],
#                     #     solar_zenith=solPosInfo['apparent_zenith'].values,
#                     #     solar_azimuth=solPosInfo['azimuth'].values
#                     # )
#                     #
#                     # actDataL5['GHI_POA'] = poaInsInfo['poa_global'].values
#                     # actDataL5['DNI_POA'] = poaInsInfo['poa_direct'].values
#                     # actDataL5['DHI_POA'] = poaInsInfo['poa_diffuse'].values
#                     #
#                     # # 혼탁도
#                     # turbidity = pvlib.clearsky.lookup_linke_turbidity(pd.to_datetime(actDataL5['DATE_TIME'].values), posLat, posLon, interp_turbidity=True)
#                     # actDataL5['TURB'] = turbidity.values
#                     #
#                     # setAtmosActDataDB(cfgInfo, actDataL5)
#
#                 except Exception as e:
#                     log.error("Exception : {}".format(e))
#
#         result = {
#             'msg': 'succ'
#             , 'data': actDataL5
#         }
#
#         return result
#
#     except Exception as e:
#         log.error('Exception : {}'.format(e))
#         return result
#
#     finally:
#         # try, catch 구문이 종료되기 전에 무조건 실행
#         log.info('[END] {}'.format('subAtmosActData'))


def cartesian(latitude, longitude, elevation=0):
    # Convert to radians
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371  # 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)

    return (X, Y, Z)


# def initCfgInfo(sysPath):
#     log.info('[START] {}'.format('initCfgInfo'))
#     # log.info('[CHECK] sysPath : {}'.format(sysPath))
#
#     result = None
#
#     try:
#         # DB 연결 정보
#         # pymysql.install_as_MySQLdb()
#
#         # DB 정보
#         # dbUser = config.get('mariadb', 'user')
#         # dbPwd = config.get('mariadb', 'pwd')
#         # dbHost = config.get('mariadb', 'host')
#         # dbPort = config.get('mariadb', 'port')
#         # dbName = config.get('mariadb', 'dbName')
#
#         # dbEngine = create_engine('mariadb://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
#         # dbEngine = create_engine('mariadb+mariadbconnector://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
#         # sessMake = sessionmaker(bind=dbEngine)
#         # session = sessMake()
#
#         # API 정보
#         config = configparser.ConfigParser()
#         # config.read(sysPath, encoding='utf-8')
#         config.read(sysPath, encoding='utf-8')
#
#         apiUrl = config.get('dataApi-obs', 'url')
#         apiToken = config.get('dataApi-obs', 'key')
#
#         result = {
#             # 'dbEngine': dbEngine
#             # , 'session': session
#             'apiUrl': apiUrl
#             , 'apiToken': apiToken
#         }
#
#         return result
#
#     except Exception as e:
#         log.error('Exception : {}'.format(e))
#         return result
#
#     finally:
#         # try, catch 구문이 종료되기 전에 무조건 실행
#         log.info('[END] {}'.format('initCfgInfo'))


# def reqPvApi(cfgInfo, dtYmd, id):
#     # log.info('[START] {}'.format('subVisPrd'))
#     # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
#     # log.info('[CHECK] dtYmd : {}'.format(dtYmd))
#     # log.info('[CHECK] id : {}'.format(id))
#
#     result = None
#
#     try:
#
#         apiUrl = cfgInfo['apiUrl']
#         apiToken = cfgInfo['apiToken']
#         stnId = id
#         srvId = 'SRV{:05d}'.format(id)
#
#         reqUrl = '{}/{}/{}'.format(apiUrl, dtYmd, id)
#         reqHeader = {'Authorization': 'Bearer {}'.format(apiToken)}
#         res = requests.get(reqUrl, headers=reqHeader)
#
#         if not (res.status_code == 200): return result
#         resJson = res.json()
#
#         if not (resJson['success'] == True): return result
#         resInfo = resJson['pvs']
#
#         if (len(resInfo) < 1): return result
#         resData = pd.DataFrame(resInfo)
#
#         resData = resData.rename(
#             {
#                 'pv': 'PV'
#             }
#             , axis='columns'
#         )
#
#         resData['SRV'] = srvId
#         resData['DATE_TIME_KST'] = pd.to_datetime(resData['date'], format='%Y-%m-%d %H')
#         resData['DATE_TIME'] = resData['DATE_TIME_KST'] - dtKst
#         resData = resData.drop(['date'], axis='columns')
#
#         result = {
#             'msg': 'succ'
#             , 'resData': resData
#         }
#
#         return result
#
#     except Exception as e:
#         log.error('Exception : {}'.format(e))
#         return result
#
#     # finally:
#     #     # try, catch 구문이 종료되기 전에 무조건 실행
#     #     log.info('[END] {}'.format('reqPvApi'))


#
# def subPvData(cfgInfo, sysOpt, posDataL1, dtIncDateList):
#
#     log.info('[START] {}'.format('subPvData'))
#     # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
#     # log.info('[CHECK] sysOpt : {}'.format(sysOpt))
#     # log.info('[CHECK] posDataL1 : {}'.format(posDataL1))
#     # log.info('[CHECK] dtIncDateList : {}'.format(dtIncDateList))
#     result = None
#
#     try:
#         # dtIncDateInfo = dtIncDateList[0]
#         stnId = sysOpt.get('stnId')
#
#         for i, dtIncDateInfo in enumerate(dtIncDateList):
#             log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))
#
#             dtYmd = dtIncDateInfo.strftime('%Y/%m/%d')
#             dtYear = dtIncDateInfo.strftime('%Y')
#
#             isSearch = True if ((stnId == None) or (len(stnId) < 1)) else False
#             if (isSearch):
#                 for j, posInfo in posDataL1.iterrows():
#                     id = int(posInfo['ID'])
#
#                     result = reqPvApi(cfgInfo, dtYmd, id)
#                     # log.info("[CHECK] result : {}".format(result))
#
#                     resData = result['resData']
#                     if (len(resData) < 1): continue
#
#                     setPvDataDB(cfgInfo, resData, dtYear)
#             else:
#                 id = int(stnId)
#
#                 result = reqPvApi(cfgInfo, dtYmd, id)
#                 # log.info("[CHECK] result : {}".format(result))
#
#                 resData = result['resData']
#                 if (len(resData) < 1): continue
#
#                 setPvDataDB(cfgInfo, resData, dtYear)
#
#         result = {
#             'msg': 'succ'
#         }
#
#         return result
#
#     except Exception as e:
#         log.error('Exception : {}'.format(e))
#         return result
#
#     finally:
#         # try, catch 구문이 종료되기 전에 무조건 실행
#         log.info('[END] {}'.format('subPvData'))

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

        plt.pcolormesh(xEdge, yEdge, hist2DVal, cmap=cm.get_cmap('jet'), vmin=0, vmax=100)

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
                    # 'srtDate': '1990-01-01'
                    # , 'endDate': '2022-01-01'
                    # 'srtDate': '2022-08-01'
                    # , 'endDate': '2022-08-05'
                    # , 'endDate': '2022-08-04'

                    'srtDate': '2022-08-05'
                    , 'endDate': '2022-08-08'

                    # 경도 최소/최대/간격
                    , 'lonMin': 124
                    , 'lonMax': 131
                    , 'lonInv': 0.01

                    # 위도 최소/최대/간격
                    , 'latMin': 33
                    , 'latMax': 39
                    , 'latInv': 0.01

                    , 'collect' : {
                        'um' : {
                            'isOverWrite': False
                        }
                        , 'asos': {
                            'isOverWrite': False
                        }
                    }
                }

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
            # 홍수관리저수지 유역도에 따른 KLAPS, UM 최근접, 영역 화소 시각화
            # ****************************************************************************
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
            # 오픈 API를 통해 UM LDAPS 수치모델 수집
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
            # 오픈 API를 통해 종관기상관측 (ASOS) 수집
            # ****************************************************************************
            cfgInfo = json.load(open(globalVar['sysCfg'], 'r'))

            apiUrl = cfgInfo['dataApi-obs']['url']
            apikey = cfgInfo['dataApi-obs']['key']

            # ASOS 설정 정보
            inpAsosStnFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/ASOS_STN_INFO.csv')
            asosStnData = pd.read_csv(inpAsosStnFile)
            asosStnDataL1 = asosStnData[['STN', 'LON', 'LAT']]

            dataL2 = pd.DataFrame()
            for i, dtDayInfo in enumerate(dtDayList):
                log.info("[CHECK] dtDayInfo : {}".format(dtDayInfo))

                filePathDateYmd = dtDayInfo.strftime('%Y%m/%d')
                fileNameDate = dtDayInfo.strftime('%Y%m%d%H%M')

                reqSrtYmd = dtDayInfo.strftime('%Y%m%d')
                reqEndYmd = (dtDayInfo + timedelta(days=1)).strftime('%Y%m%d')

                saveFile = '{}/{}/OBS/{}/ASOS_OBS_{}.csv'.format(globalVar['inpPath'], serviceName, filePathDateYmd, fileNameDate)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                saveFileList = sorted(glob.glob(saveFile), reverse=True)

                # 학습 모델이 없을 경우
                if (sysOpt['collect']['asos']['isOverWrite']) or (len(saveFileList) < 1):
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
                else:
                    dataL1 = pd.read_csv(saveFile)

                dataL2 = pd.concat([dataL2, dataL1], ignore_index=False)


            # ****************************************************************************
            # 종관기상관측 (ASOS) 자료 처리
            # ****************************************************************************
            # TM 및 STN을 기준으로 중복 제거
            dataL2['tm'] = dataL2['tm'].astype(str)
            dataL2['stnId'] = dataL2['stnId'].astype(int)
            dataL2['ta'] = pd.to_numeric(dataL2['ta'])
            dataL2['rn'] = pd.to_numeric(dataL2['rn'])
            dataL2.drop_duplicates(subset=['tm', 'stnId'], inplace=True)

            # 결측값 제거
            dataL3 = dataL2
            dataL3['dtDateKst'] = pd.to_datetime(dataL3['tm'], format='%Y-%m-%d %H:%M')
            dataL3['dtDateUtc'] = dataL3['dtDateKst'] - dtKst

            dataL3['ta'] = np.where(dataL3['ta'] < -50.0, np.nan, dataL3['ta'])
            # dataL3['rn'] = np.where(dataL3['rn'] < 0.0, np.nan, dataL3['rn'])
            dataL3['rn'] = np.where(dataL3['rn'] > 0.0, dataL3['rn'], 0.0)


            # dsData = pd.merge(left=dataL3, right=asosStnDataL1, how='left', left_on='stnId', right_on='STN').rename(
            #     columns={
            #         'LON': 'lon'
            #         , 'LAT' : 'lat'
            #         , 'dtDateKst' : 'time'
            #     }
            # )
            # dsDataL1 = dsData[['lon', 'lat', 'time', 'ta', 'rn']].set_index(['lat', 'lon', 'time'])
            # dsDataL2 = dsDataL1.to_xarray()
            #
            # dsDataL2.isel(time = 2)['ta'].plot()
            # plt.show()

            # dataL4 = dataL3.loc[dataL3['stnId'] == 95]
            # plt.plot(dataL4['dtDateKst'], dataL4['ta'])
            # fig, ax = plt.subplots()
            # ax2 = ax.twinx()
            # ax.plot(dataL4['dtDateKst'], dataL4['ta'], label='기온', color='r')
            # ax2.plot(dataL4['dtDateKst'], dataL4['rn'], label='강수량', color='b')
            # ax.legend(loc=0)
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            # fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
            # ax.legend(loc=0)
            # ax2.legend(loc=0)
            # plt.legend()
            # plt.show()

            # dataL4.columns

            actDataL2 = xr.Dataset()
            dtHourInfo = dtHourList[1]
            for ii, dtHourInfo in enumerate(dtHourList):
                log.info("[CHECK] dtHourInfo : {}".format(dtHourInfo))

                # dataL4 = dataL3
                dataL4 = dataL3.loc[
                    dataL3['dtDateKst'] == dtHourInfo
                    ]

                if (len(dataL4) < 1): continue

                dataL5 = pd.merge(left=dataL4, right=asosStnDataL1, how='left', left_on='stnId', right_on='STN')

                lat1D = sorted(set(dataL5['LAT']))
                lon1D = sorted(set(dataL5['LON']))
                lon2D, lat2D = np.meshgrid(lon1D, lat1D)

                # lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
                # latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])
                # lonGrid, latGrid = np.meshgrid(lonList, latList)

                varList = {}
                colList = ['ta', 'rn']
                for colInfo in colList:
                    # if (re.match('TM|STN|dtDate', colInfo)): continue
                    dataL6 = dataL5[['dtDateKst', 'LON', 'LAT', colInfo]].dropna()
                    # dataL6 = dataL5[['dtDateKst', 'LON', 'LAT', colInfo]]

                    if (len(dataL6) < 1): continue

                    # varList[colInfo] = np.empty((len(lon1D), len(lat1D))) * np.nan
                    # varList[colInfo] = np.full(shape=(len(lon1D), len(lat1D)), fill_value=None)
                    varList[colInfo] = np.full(shape=(len(lon1D), len(lat1D)), fill_value=np.nan)

                    posLon = dataL6['LON'].values
                    posLat = dataL6['LAT'].values
                    posVar = dataL6[colInfo].values

                    # plt.scatter(posLon, posLat, c = posVar)
                    # # plt.legend()
                    # plt.show()

                    # Radial basis function (RBF) interpolation in N dimensions.
                    try:
                        rbfModel = Rbf(posLon, posLat, posVar, function='linear')
                        rbfRes = rbfModel(lon2D, lat2D)
                        # rbfRes = rbfModel(lon2D, lat2D)
                        # rbfRes = rbfModel(lonGrid, latGrid)
                        varList[colInfo] = rbfRes
                    except Exception as e:
                        log.error("Exception : {}".format(e))

                actData = xr.Dataset(
                    {
                        # ASOS 및 AWS 융합
                        'ta': (('time', 'lat', 'lon'), (varList['ta']).reshape(1, len(lat1D), len(lon1D)))
                        , 'rn': (('time', 'lat', 'lon'), (varList['rn']).reshape(1, len(lat1D), len(lon1D)))
                    }
                    , coords={
                        'time': pd.date_range(dtHourInfo, periods=1)
                        , 'lat': lat1D
                        , 'lon': lon1D
                    }
                )

                actDataL1 = actData.interp(lon=geoDataL2['lon'], lat=geoDataL2['lat'], method='linear')
                actDataL2 = xr.merge([actDataL2, actDataL1])

            # dd = actDataL2.to_dataframe().reset_index(drop=False)
            # actDataL2.isel(lon=2, lat=2)['rn'].plot()
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

            # rowUm2D, colUm2D = np.meshgrid(umCfgMask['UM-nx'], umCfgMask['UM-ny'])
            # rowKlaps2D, colKlaps2D = np.meshgrid(klapsCfgMask['KLAPS-nx'], klapsCfgMask['KLAPS-ny'])
            # klapsCfgMaskL1 = klapsCfgMask.sort_values(by=['KLAPS-nx', 'KLAPS-ny'])

            # # UM
            # dtHourUtcInfo = dtHourUtcList[9]
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
            #             umDataL1 = umData.get(['xgrid_0', 'ygrid_0', 'TMP_P0_L1_GLC0', 'NCPCP_P8_L1_GLC0_acc1h', 'LSPRATE_P8_L1_GLC0_avg1h'])
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
            #
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
            # # KLAPS
            # klapsDataL4 = xr.Dataset()
            # dtHourUtcInfo = dtHourUtcList[9]
            # # for j, dtHourUtcInfo in enumerate(dtHourList):
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

            saveFile = '{}/{}/klps_lc05_anal_{}-{}.nc'.format(globalVar['inpPath'], serviceName, pd.to_datetime(dtHourUtcList.min()).strftime('%Y%m%d%H'),     pd.to_datetime(dtHourUtcList.max()).strftime('%Y%m%d%H'))
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # klapsDataL4.to_netcdf(saveFile)
            log.info('[CHECK] saveFile : {}'.format(saveFile))
            klapsDataL4 = xr.open_dataset(saveFile)

            saveFile = '{}/{}/UMKR_l015_unis_{}-{}.nc'.format(globalVar['inpPath'], serviceName, pd.to_datetime(dtHourUtcList.min()).strftime('%Y%m%d%H'),     pd.to_datetime(dtHourUtcList.max()).strftime('%Y%m%d%H'))
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # umDataL4.to_netcdf(saveFile)
            log.info('[CHECK] saveFile : {}'.format(saveFile))
            umDataL4 = xr.open_dataset(saveFile)

            # ************************************************************************************************
            # 융합 데이터
            # ************************************************************************************************
            # }In [103]: umDataL1.info()
            # xarray.Dataset {
            # dimensions:
            # 	xgrid_0 = 602 ;
            # 	ygrid_0 = 781 ;
            # variables:
            # 	int64 xgrid_0(xgrid_0) ;
            # 	int64 ygrid_0(ygrid_0) ;
            # 	float32 gridlat_0(ygrid_0, xgrid_0) ;
            # 		gridlat_0:corners = [32.256874 32.188087 42.93468  43.018272] ;
            # 		gridlat_0:long_name = latitude ;
            # 		gridlat_0:grid_type = Lambert Conformal (secant, tangent, conical or bipolar) ;
            # 		gridlat_0:units = degrees_north ;
            # 		gridlat_0:Latin2 = [60.] ;
            # 		gridlat_0:Latin1 = [30.] ;
            # 		gridlat_0:Dy = [1.5] ;
            # 		gridlat_0:Dx = [1.5] ;
            # 		gridlat_0:Lov = [126.] ;
            # 		gridlat_0:Lo1 = [121.83443] ;
            # 		gridlat_0:La1 = [32.256874] ;
            # 	float32 gridlon_0(ygrid_0, xgrid_0) ;
            # 		gridlon_0:corners = [121.83443  131.50974  132.53188  121.060295] ;
            # 		gridlon_0:long_name = longitude ;
            # 		gridlon_0:grid_type = Lambert Conformal (secant, tangent, conical or bipolar) ;
            # 		gridlon_0:units = degrees_east ;
            # 		gridlon_0:Latin2 = [60.] ;
            # 		gridlon_0:Latin1 = [30.] ;
            # 		gridlon_0:Dy = [1.5] ;
            # 		gridlon_0:Dx = [1.5] ;
            # 		gridlon_0:Lov = [126.] ;
            # 		gridlon_0:Lo1 = [121.83443] ;
            # 		gridlon_0:La1 = [32.256874] ;
            # 	float32 TMP_P0_L1_GLC0(ygrid_0, xgrid_0) ;
            # 		TMP_P0_L1_GLC0:center = Seoul ;
            # 		TMP_P0_L1_GLC0:production_status = Operational products ;
            # 		TMP_P0_L1_GLC0:long_name = Temperature ;
            # 		TMP_P0_L1_GLC0:units = K ;
            # 		TMP_P0_L1_GLC0:grid_type = Lambert Conformal can be secant or tangent, conical or bipolar ;
            # 		TMP_P0_L1_GLC0:parameter_discipline_and_category = Meteorological products, Temperature ;
            # 		TMP_P0_L1_GLC0:parameter_template_discipline_category_number = [0 0 0 0] ;
            # 		TMP_P0_L1_GLC0:level_type = Ground or water surface ;
            # 		TMP_P0_L1_GLC0:forecast_time = [48] ;
            # 		TMP_P0_L1_GLC0:forecast_time_units = hours ;
            # 		TMP_P0_L1_GLC0:initial_time = 08/02/2022 (12:00) ;
            # 	float32 NCPCP_P8_L1_GLC0_acc1h(ygrid_0, xgrid_0) ;
            # 		NCPCP_P8_L1_GLC0_acc1h:center = Seoul ;
            # 		NCPCP_P8_L1_GLC0_acc1h:production_status = Operational products ;
            # 		NCPCP_P8_L1_GLC0_acc1h:long_name = Large-scale precipitation (non-convective) ;
            # 		NCPCP_P8_L1_GLC0_acc1h:units = kg m-2 ;
            # 		NCPCP_P8_L1_GLC0_acc1h:grid_type = Lambert Conformal can be secant or tangent, conical or bipolar ;
            # 		NCPCP_P8_L1_GLC0_acc1h:parameter_discipline_and_category = Meteorological products, Moisture ;
            # 		NCPCP_P8_L1_GLC0_acc1h:parameter_template_discipline_category_number = [8 0 1 9] ;
            # 		NCPCP_P8_L1_GLC0_acc1h:level_type = Ground or water surface ;
            # 		NCPCP_P8_L1_GLC0_acc1h:type_of_statistical_processing = Accumulation ;
            # 		NCPCP_P8_L1_GLC0_acc1h:statistical_process_duration = 1 hours (ending at forecast time) ;
            # 		NCPCP_P8_L1_GLC0_acc1h:forecast_time = [48] ;
            # 		NCPCP_P8_L1_GLC0_acc1h:forecast_time_units = hours ;
            # 		NCPCP_P8_L1_GLC0_acc1h:initial_time = 08/02/2022 (12:00) ;
            # 	float32 LSPRATE_P8_L1_GLC0_avg1h(ygrid_0, xgrid_0) ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:center = Seoul ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:production_status = Operational products ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:long_name = Large scale precipitation rate ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:units = kg m-2s-1 ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:grid_type = Lambert Conformal can be secant or tangent, conical or bipolar ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:parameter_discipline_and_category = Meteorological products, Moisture ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:parameter_template_discipline_category_number = [ 8  0  1 54] ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:level_type = Ground or water surface ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:type_of_statistical_processing = Average ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:statistical_process_duration = 1 hours (ending at forecast time) ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:forecast_time = [48] ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:forecast_time_units = hours ;
            # 		LSPRATE_P8_L1_GLC0_avg1h:initial_time = 08/02/2022 (12:00) ;
            # // global attributes:
            # }


            # rn3 NCPCP_P8_L1_GLC0_acc1h, Large-scale precipitation (non-convective), kg m-2
            # rn2 LSPRATE_P8_L1_GLC0_avg1h, Large scale precipitation rate, kg m-2s-1
            # CPRAT_P8_L1_GLC0_acc1h, Convective precipitation rate, kg m-2 s-1


            posInfo = geoDataL2.iloc[3]
            validDataL1 = pd.DataFrame()
            for i, posInfo in geoDataL2.iterrows():

                # 최근접 위경도를 기준으로 ASOS, KLAPS, UM 데이터 가져오기
                actSelData = actDataL2.sel(lon = posInfo['lon'], lat = posInfo['lat']).to_dataframe().reset_index(drop=False).drop_duplicates(['time', 'lon', 'lat'], ignore_index=True)
                klapsSelData = klapsDataL4.sel(nx=posInfo['KLAPS-nx'], ny=posInfo['KLAPS-ny']).to_dataframe().reset_index(drop=False).drop_duplicates(['time', 'nx', 'ny'], ignore_index=True)
                umSelData = umDataL4.sel(nx = posInfo['UM-nx'], ny = posInfo['UM-ny']).to_dataframe().reset_index(drop=False).drop_duplicates(['anaTime', 'time', 'nx', 'ny'], ignore_index=True)
                umSelData['anaDate'] = umSelData['anaTime'].dt.strftime("%Y%m%d%H").astype(str)

                validData = actSelData
                validData['gid'] = posInfo['gid']

                # KLAPS 데이터
                validData = pd.merge(left=validData, right=klapsSelData[['time', 'KLAPS-ta', 'KLAPS-rn']], how='left', left_on=['time'], right_on=['time'])

                # UM 데이터
                # anaDateInfo = anaDateList[0]
                anaDateList = set(umSelData['anaDate'])
                for j, anaDateInfo in enumerate(anaDateList):
                    umSelDataL1 = umSelData.loc[umSelData['anaDate'] == anaDateInfo]
                    umSelDataL2 = umSelDataL1.reset_index(drop=True).rename(
                        # columns = { 'UM-ta' : 'UM-ta-' + anaDateInfo, 'UM-rn' : 'UM-rn-' + anaDateInfo}
                        columns = { 'UM-ta' : 'UM-ta-' + anaDateInfo, 'UM-rn3' : 'UM-rn-' + anaDateInfo}
                        # columns = { 'UM-ta' : 'UM-ta-' + anaDateInfo, 'UM-rn3' : 'UM-rn-' + anaDateInfo}
                    )

                    validData = pd.merge(left=validData, right=umSelDataL2[['time', 'UM-ta-' + anaDateInfo, 'UM-rn-' + anaDateInfo]], how='left', left_on=['time'], right_on=['time'])


                # 인접한 폴리곤을 기준으로 KLAPS 데이터 가져오기
                klapsAreaData = pd.DataFrame()
                klapsCfgMaskL1 = klapsCfgMask.loc[klapsCfgMask['gid'] == posInfo['gid']]
                for j, klapsInfo in klapsCfgMaskL1.iterrows():
                    klapsTmpData = klapsDataL4.sel(nx=klapsInfo['KLAPS-nx'], ny=klapsInfo['KLAPS-ny']).to_dataframe().reset_index(drop=False).drop_duplicates(['time', 'nx', 'ny'], ignore_index=True)
                    klapsAreaData = pd.concat([klapsAreaData, klapsTmpData], ignore_index=True)

                if (len(klapsAreaData) > 0):
                    klapsAreaDataL1 = klapsAreaData.groupby(by=['time']).mean().reset_index(drop=False).rename(
                        columns={'KLAPS-ta' : 'KLAPS-area-ta', 'KLAPS-rn' : 'KLAPS-area-rn'}
                    )

                    validData = pd.merge(left=validData, right=klapsAreaDataL1[['time', 'KLAPS-area-ta', 'KLAPS-area-rn']], how='left', left_on=['time'], right_on=['time'])

                # 인접한 폴리곤을 기준으로 UM 데이터 가져오기
                umAreaData = pd.DataFrame()
                umCfgMaskL1 = umCfgMask.loc[umCfgMask['gid'] == posInfo['gid']]
                for j, umInfo in umCfgMaskL1.iterrows():
                    umTmpData = umDataL4.sel(nx=umInfo['UM-nx'], ny=umInfo['UM-ny']).to_dataframe().reset_index(drop=False).drop_duplicates(['anaTime', 'time', 'nx', 'ny'], ignore_index=True)
                    umAreaData = pd.concat([umAreaData, umTmpData], ignore_index=True)

                if (len(umAreaData) > 0):
                    umAreaDataL1 = umAreaData.groupby(by=['anaTime', 'time']).mean().reset_index(drop=False).rename(
                        columns={'UM-ta' : 'UM-area-ta', 'UM-rn3' : 'UM-area-rn'}
                    )
                    umAreaDataL1['anaDate'] = umAreaDataL1['anaTime'].dt.strftime("%Y%m%d%H").astype(str)

                    # anaDateInfo = anaDateList[0]
                    anaDateList = set(umAreaDataL1['anaDate'])
                    for j, anaDateInfo in enumerate(anaDateList):
                        umAreaDataL2 = umAreaDataL1.loc[umAreaDataL1['anaDate'] == anaDateInfo]
                        umAreaDataL3 = umAreaDataL2.reset_index(drop=True).rename(
                            columns = { 'UM-area-ta' : 'UM-area-ta-' + anaDateInfo, 'UM-area-rn' : 'UM-area-rn-' + anaDateInfo}
                        )

                        validData = pd.merge(left=validData, right=umAreaDataL3[['time', 'UM-area-ta-' + anaDateInfo, 'UM-area-rn-' + anaDateInfo]], how='left', left_on=['time'], right_on=['time'])

                validDataL1 = pd.concat([validDataL1, validData], ignore_index=False)

                # validData.describe()
                # 온도
                # plt.plot(validData['time'], validData['ta'], marker='o', label='ASOS')
                # if (len(validData['KLAPS-ta']) > 0):
                #     plt.plot(validData['time'], validData['KLAPS-ta'], marker='o', label='KLAPS')
                # if (len(validData['KLAPS-area-ta']) > 0):
                #     plt.plot(validData['time'], validData['KLAPS-area-ta'], marker='o', label='KLAPS-area')
                # # colList = validData.loc[:, validData.columns.str.find('-ta-') >= 0].columns
                # colList = validData.loc[:, validData.columns.str.find('-ta-') >= 0].columns
                # # colList = validData.loc[:, validData.columns.str.find('-area-ta-') > 0].columns
                # if (len(colList) > 0):
                #     for j, colInfo in enumerate(sorted(set(colList))):
                #         if (len(validData[colInfo]) < 1): continue
                #         if (not colInfo.split('-ta-')[1][8:10] == '21'): continue
                #         # log.info("[CHECK] colInfo : {}".format(colInfo))
                #         plt.plot(validData['time'], validData[colInfo], label=colInfo)
                # plt.legend()
                # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
                # plt.xticks(rotation=45)
                # plt.show()

                # 강수량
                # plt.plot(validData['time'], validData['rn'], marker='o', label='ASOS')
                # plt.plot(validData['time'], validData['KLAPS-rn'], marker='o', label='KLAPS')
                # plt.plot(validData['time'], validData['KLAPS-area-rn'], marker='o', label='KLAPS-area')
                # colList = validData.loc[:, validData.columns.str.find('-rn-') >= 0].columns
                # # colList = validData.loc[:, validData.columns.str.find('-area-rn-') >= 0].columns
                # for j, colInfo in enumerate(sorted(set(colList))):
                #     if (not colInfo.split('-rn-')[1][8:10] == '21'): continue
                #     # log.info("[CHECK] colInfo : {}".format(colInfo))
                #     plt.plot(validData['time'], validData[colInfo], label=colInfo)
                # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%d %H'))
                # plt.legend()
                # plt.xticks(rotation=45)
                # plt.show()

                # colList = validData.loc[:, validData.columns.str.find('-rn-') >= 0].columns
                # validDataL1 = validData
                # plt.scatter(validDataL1['KLAPS-rn'], validDataL1['rn'])
                # plt.show()


            # validDataL2 = validDataL1[['rn', 'UM-area-rn-2022080521']].dropna()
            # validDataL2 = validDataL1[['time', 'rn', 'KLAPS-rn']].dropna()
            validDataL2 = validDataL1[['time', 'rn', 'KLAPS-area-rn']].dropna()
            # plt.scatter(validDataL1['KLAPS-area-rn'], validDataL1['rn'])
            plt.scatter(validDataL2['KLAPS-rn'], validDataL2['rn'])
            # plt.scatter(validDataL1['UM-area-rn-2022080521'], validDataL1['rn'])
            # plt.scatter(validDataL1['UM-area-rn-2022080521'], validDataL1['rn'])
            plt.show()

            minAnaData = pd.to_datetime(validDataL2['time']).min().strftime("%Y%m%d")
            maxAnaData = pd.to_datetime(validDataL2['time']).max().strftime("%Y%m%d")

            # 온도
            mainTitle = '[{}-{}] {}'.format(minAnaData, maxAnaData, '기상 예보 정보 (수치모델)를 활용한 2D 산점도')
            saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            # rtnInfo = makeUserHist2dPlot(validDataL2['KLAPS-rn'], validDataL2['rn'], '머신러닝', '실측', mainTitle, saveImg, 0, 10, 0.2, 0.5, 30, True)
            rtnInfo = makeUserHist2dPlot(validDataL2['KLAPS-area-rn'], validDataL2['rn'], '머신러닝', '실측', mainTitle, saveImg, 0, 10, 0.2, 0.5, 20, True)
            log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

            validDataL2 = validDataL1[['time', 'ta', 'KLAPS-ta']].dropna()
            mainTitle = '[{}-{}] {}'.format(minAnaData, maxAnaData, '기상 예보 정보 (수치모델)를 활용한 2D 산점도')
            saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            rtnInfo = makeUserHist2dPlot(validDataL2['KLAPS-ta'], validDataL2['ta'], '실측 (ASOS)', '수치모델 (KLAPS)', mainTitle, saveImg, 20, 40, 0.5, 1, 20, True)
            log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

            validDataL2 = validDataL1[['time', 'ta', 'KLAPS-area-ta']].dropna()
            mainTitle = '[{}-{}] {}'.format(minAnaData, maxAnaData, '기상 예보 정보 (수치모델)를 활용한 2D 산점도')
            saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            rtnInfo = makeUserHist2dPlot(validDataL2['KLAPS-area-ta'], validDataL2['ta'], '실측 (ASOS)', '수치모델 (KLAPS)', mainTitle, saveImg, 20, 40, 0.5, 1, 20, True)
            log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

            validDataL2 = validDataL1[['time', 'ta', 'UM-area-ta-2022080521']].dropna()
            mainTitle = '[{}-{}] {}'.format(minAnaData, maxAnaData, '기상 예보 정보 (수치모델)를 활용한 2D 산점도')
            saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            rtnInfo = makeUserHist2dPlot(validDataL2['UM-area-ta-2022080521'], validDataL2['ta'], '실측 (ASOS)', '수치모델 (KLAPS)', mainTitle, saveImg, 20, 40, 0.5, 1, 20, True)
            log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

            # 화면

            geoDataL2.plot(column=geoDataL2['KLAPS-dist'], legend=True)
            geoDataL2.apply(lambda x: plt.annotate(text=x['fac_name'], xy=[x.lon, x.lat], ha='center', va='top', color='black'), axis=1)
            plt.show()

            geoDataL2.plot(column=geoDataL2['UM-dist'], legend=True)
            geoDataL2.apply(lambda x: plt.annotate(text=x['fac_name'], xy=[x.lon, x.lat], ha='center', va='top', color='black'), axis=1)
            plt.show()

            # gidList = validDataL1['gid'].values
            # timeList = validDataL1['time'].values
            # gid = gidList[2]
            # time = timeList[2]
            # for k, gid in enumerate(gidList):
            #     log.info("[CHECK] gid : {}".format(gid))
            #
            #     gidInfo = geoDataL2.loc[
            #         geoDataL2['gid'] == gid
            #         ]
            #
            #     validDataL3 = validDataL1.loc[
            #         (validDataL1['gid'] == gid)
            #         &  (validDataL1['time'] == time)
            #     ]
            #
            #
            #
            #     gidInfoL1 = pd.merge(left=gidInfo, right=validDataL3[validDataL3.columns[3:]], how='left', left_on=['gid'], right_on=['gid'])
            #
            #
            #     # plt.show()
            #
            #     # umCfgMaskL1 = umCfgMask.loc[
            #     #     umCfgMask['gid'] == gid
            #     #     ]
            #     #
            #     # klapsCfgMaskL1 = klapsCfgMask.loc[
            #     #     klapsCfgMask['gid'] == gid
            #     #     ]
            #     #
            #     # mainTilte = '[{}] {}, {}'.format(
            #     #     gidInfo['fac_name'].values[0]
            #     #     , round(gidInfo['lon'].values[0], 2)
            #     #     ,  round(gidInfo['lat'].values[0], 2)
            #     # )
            #
            #     saveImg = '{}/{}/KLAPS-UM_{}.png'.format(globalVar['figPath'], serviceName, gidInfo['fac_name'].values[0])
            #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #
            #     gidInfoL1.plot(column=gidInfoL1['ta'], legend=True)
            #     gidInfoL1.apply(lambda x: plt.annotate(text=x['fac_name'], xy=[x.lon, x.lat], ha='center', va='top', color='black'), axis=1)
            #
            #     # gidInfo.plot(color='lightgrey')
            #     # plt.scatter(gidInfoL1['lon'], gidInfoL1['lat'], c='black', label='중심 ({})'.format(len(gidInfo)))
            #     plt.scatter(gidInfoL1['KLAPS-ta'], gidInfoL1['KLAPS-ta'], c = 'red', label='최근접 KLAPS ({})'.format(len(gidInfo)))
            #     plt.scatter(gidInfoL1['UM-ta'], gidInfoL1['UM-ta'], c='blue', label='최근접 UM ({})'.format(len(gidInfo)))
            #     # plt.title(mainTilte)
            #     plt.legend()
            #     # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.show()
            #     # plt.close()




                # 60*60
                # # validDataL1 = validData.dropna()
                # plt.plot(validData['time'], validData['rn'], marker='o', label='ASOS')
                # # plt.plot(validDataL1['time'], validDataL1['UM-rn'] * 60, label='UM')
                # plt.plot(validData['time'], validData['KLAPS-rn'] * 3600, label='KLAPS')
                # # plt.plot(validDataL1['time'], validDataL1['UM-area-rn'] * 60, label='UM')
                # # plt.plot(validDataL1['time'], validDataL1['KLAPS-area-rn'] * 60, label='KLAPS')
                # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
                # plt.legend()
                # plt.show()

                # umSelDataL1 = umSelData
                # umSelDataL1
                #
                # umSelDataL1 = umSelData.pivot(index=['anaTime', 'time', 'nx', 'ny'], columns=['anaDate'], values=['UM-ta', 'UM-rn']).reset_index(drop=False)
                #
                # klapsSelData.columns
                # colListNew = []
                # for j, colInfo in enumerate(umSelDataL1.columns):
                #     if (len(colInfo[1]) < 1):
                #         colInfoNew = colInfo[0]
                #     else:
                #         colInfoNew =  colInfo[0]  + '-' + colInfo[1]
                #     print(colInfoNew)
                #     colListNew.append(colInfoNew)
                #
                # umSelDataL1.columns = colListNew
                #
                # validData.columns
                #
                # umSelDataL1[['time'], umSelDataL1.columns[1, 4:]]








                # validData = pd.merge(left=validData, right=umAreaDataL1, how='left', left_on=['time', 'anaTime'], right_on=['time', 'anaTime'])

                # validData['anaTimeC'].
                # validData['anaDate'] = validData['anaTime'].dt.strftime("%Y%m%d%H%M").astype(str)
                #
                # anaDateList = set(validData['anaDate'])
                #
                # plt.plot(validData['time'], validData['ta'], marker='o', label='ASOS')
                # plt.plot(validData['time'], validData['UM-ta'] - 273.15, label='UM')
                # plt.plot(validData['time'], validData['KLAPS-ta'] - 273.15, label='KLAPS')
                # plt.plot(validData['time'], validData['UM-area-ta'] - 273.15, label='UM')
                # plt.plot(validData['time'], validData['KLAPS-area-ta'] - 273.15, label='KLAPS')
                # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
                # plt.legend()
                # plt.show()
                #
                # # 60*60
                # validDataL1 = validData.dropna()
                # plt.plot(validDataL1['time'], validDataL1['rn'], marker='o', label='ASOS')
                # # plt.plot(validDataL1['time'], validDataL1['UM-rn'] * 60, label='UM')
                # plt.plot(validDataL1['time'], validDataL1['KLAPS-rn'] * 60, label='KLAPS')
                # # plt.plot(validDataL1['time'], validDataL1['UM-area-rn'] * 60, label='UM')
                # plt.plot(validDataL1['time'], validDataL1['KLAPS-area-rn'] * 60, label='KLAPS')
                # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
                # plt.legend()
                # plt.show()




                # validData
                #
                # validData.columns
                #
                # validDataL1 = validData[['time', 'ta', 'rn', 'KLPAS-ta', 'KLAPS-tn', 'UM-ta', 'UM-rn']]
                # validDataL1 = validData.melt(id_vars=['time'])
                # validDataL2 = validDataL1.pivot(index='time', columns='variable', values='value')

                # actDataL3 = actDataL2.to_dataframe().reset_index(drop=False)
                # klapsDataL5 = klapsDataL4.sel(nx = posInfo['KLAPS-nx'], ny = posInfo['KLAPS-ny'])


                # print(posInfo)
                # if (posInfo.isna()[['lon', 'lat']].any() == True): continue
                # coord = cartesian(posInfo['lat'], posInfo['lon'])
                # closest = tree.query([coord], k=1)

            # for i, itemInfo in enumerate(itemList):
            #     log.info("[CHECK] itemInfo : {}".format(itemInfo))
            # plt.scatter(posLon, posLat, c=posVar, vmin=20, vmax=38)
            # plt.colorbar()
            # plt.show()
            #
            # actDataL1 = actData.isel(time=0)
            # plt.scatter(lon1D, lat1D, c=np.diag(actDataL1['ta']))
            # plt.colorbar()
            # plt.show()
            # # np.max(actDataL1['ta'].values)
            # # np.min(actDataL1['ta'].values)

            # actDataL1['ta'].plot(vmin=22, vmax=38)
            # plt.scatter(posLon, posLat, edgecolors='white', c=posVar, vmin=22, vmax=38)
            # plt.show()

            # for i, itemInfo in enumerate(itemList):
            #     log.info("[CHECK] itemInfo : {}".format(itemInfo))
            #
            #     pd.DataFrame.from_records(itemList)

            # log.info("[CHECK] [{}] {}".format(i, addrInfo))
            #
            # posDataL1.loc[i, 'lat'] = responseBody['addresses'][0]['y']
            # posDataL1.loc[i, 'lon'] = responseBody['addresses'][0]['x']
            #
            #
            # with open(saveHwpFile, mode="wb") as f:
            #     f.write(res.read())

            # log.info('[CHECK] saveHwpFile : {} / {} / {}'.format(inCaSvephy, isFile, saveHwpFile))

            # reqUrl = '{}?serviceKey={}/{}'.format(apiUrl, dtYmd, id)

            # https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?serviceKey=bf9fH0KLgr65zXKT5D%2FdcgUBIj1znJKnUPrzDVZEe6g4gquylOjmt65R5cjivLPfOKXWcRcAWU0SN7KKXBGDKA%3D%3D&pageNo=1&numOfRows=10&dataType=JSON&dataCd=ASOS&dateCd=HR&startDt=20100101&startHh=01&endDt=20100601&endHh=01&stnIds=108
            # reqUrl = '{}/{}/{}'.format(apiUrl, dtYmd, id)
            # reqHeader = {'Authorization': 'Bearer {}'.format(apiToken) }
            # res = requests.get(reqUrl, headers=reqHeader)
            #
            # if not (res.status_code == 200): return result
            # resJson = res.json()
            #
            # if not (resJson['success'] == True): return result
            # resInfo = resJson['pvs']
            #
            # if (len(resInfo) < 1): return result
            # resData = pd.DataFrame(resInfo)
            #
            # resData = resData.rename(
            #     {
            #         'pv': 'PV'
            #     }
            #     , axis='columns'
            # )
            #
            # resData['SRV'] = srvId
            # resData['DATE_TIME_KST'] = pd.to_datetime(resData['date'], format='%Y-%m-%d %H')
            # resData['DATE_TIME'] = resData['DATE_TIME_KST'] - dtKst
            # resData = resData.drop(['date'], axis='columns')

            #
            #
            # # shp 파일 읽기
            # inpShpFile = '{}/{}/{}/{}.shp'.format(globalVar['inpPath'], serviceName, 'FLOOD-SHP', '*')
            # fileShpList = sorted(glob.glob(inpShpFile))
            #
            # # if fileList is None or len(fileList) < 1:
            # #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            # import regionmask
            #
            # fileShpInfo = fileShpList[0]
            # for i, fileShpInfo in enumerate(fileShpList):
            #     log.info("[CHECK] fileShpInfo : {}".format(fileShpInfo))
            #
            #     # geoData = gpd.read_file(fileInfo, crs='epsg:4326')
            #     geoData = gpd.read_file(fileShpInfo, crs='epsg:3826')
            #     geoDataL1 = geoData.to_crs(crs=4326)
            #     geoDataL1['centroid'] = geoDataL1.centroid
            #     geoDataL1['lon'] = geoDataL1['centroid'].x
            #     geoDataL1['lat'] = geoDataL1['centroid'].y
            #
            # mpl.use("TkAgg")
            # plt.scatter(geoDataL1['lon'].values, geoDataL1['lat'].values)
            # plt.show()

            # 장성호
            # 126.84696,35.39636
            # 35.4140574142
            # 126.8510679121

            #
            # geoDataL1.plot(legend=True)
            # # geoDataL1.apply(lambda x: plt.annotate(text=x['fac_name'], xy=(x.lon, x.lat), ha='center', color='black'), axis=1)
            # geoDataL1.apply(lambda x: plt.annotate(text=x['gid'], xy=(x.lon, x.lat), ha='center', color='black'), axis=1)
            # plt.show()

            # weightMap = xa.pixel_overlaps(ncData, geoData)
            # geoAggData = xa.aggregate(ncData, weightMap)

            # from pyproj import CRS, Transformer
            # def xyz_to_latlonalt(x, y, crs_in, crs_out):
            #     crs_from = CRS.from_user_input(crs_in)
            #     crs_to = CRS.from_user_input(crs_out)
            #     proj = Transformer.from_crs(crs_from, crs_to, always_xy=True)
            #     coordinates = proj.transform(x, y)
            #     return coordinates

            # geoData['lat'] = geoData.apply(lambda row: xyz_to_latlonalt(row['x'], row['y'], row['z'], 25832, 4326)[0], axis=1)

            # geoData['centroid'] = geoData.centroid

            # xyz_to_latlonalt(x=geoData.centroid.x, y=geoData.centroid.y, crs_in=3826, crs_out=4326)

            # df, dff = gpd.points_from_xy(geoData['centroid'].x, geoData['centroid'].y)
            # dff = gpd.GeoDataFrame(xyz_to_latlonalt(x=geoData.centroid.x, y=geoData.centroid.y))
            # dff = gpd.GeoDataFrame(xyz_to_latlonalt(x=geoData.geometry, y=geoData.geometry))
            # dff.iloc[0]
            # geoData['geometry'].to
            # geoData['centroid'] = geoData.centroid
            #
            # geoData.centroid.x
            #
            # geoData.centroid.x
            # geoData["lat"] = geoData.centroid.map(lambda p: p.x)
            # geoData["long"] = geoData.Center_point.map(lambda p: p.y)
            #
            # geoData.GeoSeries(geoData.centroid)

            # geoData['val'] = geoAggData.to_dataset()['val']

            # geoData.plot(column=geoData['val'], legend=True)
            # geoData.apply(lambda x: plt.annotate(text=x['NAME_2'], xy=x.geometry.centroid.coords[0], ha='center', color='white'), axis=1);
            # plt.show()

            # geoData.head()

            # CALCULATE MASK
            # maskPoly = regionmask.Regions(
            #     name='mask'
            #     , numbers=list(range(0, len(geoData)))
            #     , names=list(geoData.name)
            #     , abbrevs=list(geoData.name)
            #     , outlines=list(geoData.geometry.values[i] for i in range(0, len(geoData)))
            # )

            # mask = maskPoly.mask(ds.isel(time=0), lat_name='latitude', lon_name='longitude')
            # mask

            # 그림
            # geoData.plot()
            # plt.show()

            # http://203.247.66.126:8090/url/nwp_file_down.php?nwp=l015&sub=unis&tmfc=2022080100&ef=00&mode=I&authKey=sangho.lee.1990@gmail.com
            # http://203.247.66.126:8090/url/nwp_file_down.php?nwp=g128&sub=pres&tmfc=2022080100&ef=00&authKey=sangho.lee.1990@gmail.com

            #
            #
            # # posLon = posInfo['lon']
            # # posLat = posInfo['lat']
            # # lon1D = np.array(posLon).reshape(1)
            # # lat1D = np.array(posLat).reshape(1)
            #
            # inpCfgFile = '{}/{}'.format(globalVar['cfgPath'], 'modelInfo/UMKR_l015_unis_H000_202110010000.grb2')
            # fileCfgList = sorted(glob.glob(inpCfgFile))
            # log.info("[CHECK] fileCfgList : {}".format(fileCfgList))
            #
            # cfgInfo = pygrib.open(fileCfgList[0]).select(name='Temperature')[1]
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
            # for i, posInfo in geoDataL1.iterrows():
            #
            #     if (posInfo.isna()[['lon', 'lat']].any() == True): continue
            #
            #     coord = cartesian(posInfo['lat'], posInfo['lon'])
            #     closest = tree.query([coord], k=1)
            #     cloIdx = closest[1][0]
            #     row = int(cloIdx / lon2D.shape[0])
            #     col = int(cloIdx % lon2D.shape[1])
            #
            #     geoDataL1.loc[i, 'UM-nx'] = row
            #     geoDataL1.loc[i, 'UM-ny'] = col
            #
            #     print(posInfo['lat'], lat2D[row, col])
            #     print(posInfo['lon'], lon2D[row, col])
            #
            #     row1D.append(row)
            #     col1D.append(col)
            #
            # row2D, col2D = np.meshgrid(row1D, col1D)
            #
            # # KLAPS
            # inpNcFile = '{}/{}/{}/{}.nc'.format(globalVar['inpPath'], serviceName, 'MODEL/202208/01', 'klps_lc05_anal_202208010000')
            # fileNcList = sorted(glob.glob(inpNcFile))

            # KLAPS 데이터
            # 중심으로 사각형의 공간영역(31.344~44.283 N, 119.799~133.619 E)에 하나의층이 235×283의 수평 격자(약 5km의 공간해상도)로
            # ds2 = ds.get(['x', 'y', 'lon', 'lat', 'pc'])

            # 31.344~44.283 N, 119.799~133.619 E
            # mapInfo["klfs.extent"] = [-584831.24, 3864434.118, 585168.76, 5274434.118];

            ### result
            # 37.579871128849334, 126.98935225645432
            # 35.101148844565955, 129.02478725562108
            # 33.500946412305076, 126.54663058817043
            # import pyproj
            # import cartopy.crs as ccrs
            # # mapLccProjInfo = '+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs'
            # # mapLccProjInfo = '+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs'
            # # mapLccProjInfo = '+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
            # # mapLccProjInfo = '+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
            # mapLccProjInfo = '+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +a=6371229 +b=6371229 +ellps=WGS84 +units=m +no_defs'
            #
            # # La1, [31.344505]
            # # Lo1, [119.79932]
            # # La2, [44.28385]
            # # Lo2, [133.61935]
            # # LoV, [126.]
            #
            # # try:
            # #     mapLccProjInfo = mapLccProj.to_proj4()
            # # except Exception as e:
            # #     log.error("Exception : {}".format(e))
            #
            # mapProj = pyproj.Proj(mapLccProjInfo)
            #
            # lonGeo, latGeo = mapProj(-584831.24, 3864434.118, inverse=True)
            # lonGeo, latGeo = mapProj(585168.76, 5274434.118, inverse=True)
            #
            # posRow, posCol = mapProj(121.83443, 32.256874, inverse=False)
            #
            # lonGeo, latGeo = mapProj(posRow, posCol, inverse=True)
            # # posRow, posCol = mapProj(lon1D, lat1D, inverse=False)
            # # 602
            #
            # nx = 602
            # ny = 781
            # xOffset = -388785.88385362644
            # yOffset = -615447.218533753
            # # 1.5
            # res = 1500
            #
            # rowEle = (np.arange(0, nx, 1) * res) + xOffset
            # colEle = (np.arange(0, ny, 1) * res) + yOffset
            #
            # rowEle.min()
            # rowEle.max()
            # colEle.min()
            # colEle.max()
            #
            # lon, lat = mapProj(rowEle.min(), colEle.min(), inverse=True)
            # # (132.53188, 42.93468
            # lon, lat = mapProj(rowEle.max(), colEle.max(), inverse=True)
            #
            # #posRow, posCol = mapProj(121.83443, 32.256874, inverse=False)
            # # posRow, posCol = mapProj(131.50974, 32.188087, inverse=False)
            # # posRow, posCol = mapProj(132.53188, 42.93468, inverse=False)
            # # posRow, posCol = mapProj(121.060295, 43.018272, inverse=False)
            #
            # # lon, lat = mapProj(rowEle.min(), colEle.min(), inverse=True)
            # # lon, lat = mapProj(rowEle.min(), colEle.min(), inverse=True)
            # # posRow, posCol = mapProj(131.50974, 32.188087, inverse=False)
            # # posRow, posCol = mapProj(132.53188, 42.93468, inverse=False)
            # # posRow, posCol = mapProj(121.060295, 43.018272, inverse=False)
            #
            # #     nx = imgProjInfo['image_width']
            # #         ny = imgProjInfo['image_height']
            # #         xOffset = imgProjInfo['lower_left_easting']
            # #         yOffset = imgProjInfo['lower_left_northing']
            # #
            # #         res = imgProjInfo['pixel_size']
            # #
            # #         # 직교 좌표
            # #         rowEle = (np.arange(0, nx, 1) * res) + xOffset
            # #         colEle = (np.arange(0, ny, 1) * res) + yOffset
            # #         colEle = colEle[::-1]
            #
            # # 		gridlat_0:corners = [32.256874 32.188087 42.93468  43.018272] ;
            # # 		gridlat_0:long_name = latitude ;
            # # 		gridlat_0:grid_type = Lambert Conformal (secant, tangent, conical or bipolar) ;
            # # 		gridlat_0:units = degrees_north ;
            # # 		gridlat_0:Latin2 = [60.] ;
            # # 		gridlat_0:Latin1 = [30.] ;
            # # 		gridlat_0:Dy = [1.5] ;
            # # 		gridlat_0:Dx = [1.5] ;
            # # 		gridlat_0:Lov = [126.] ;
            # # 		gridlat_0:Lo1 = [121.83443] ;
            # # 		gridlat_0:La1 = [32.256874] ;
            #
            # posRow, posCol = mapProj(121.83443, 32.256874, inverse=False)
            # # posRow, posCol = mapProj(131.50974, 32.188087, inverse=False)
            # # posRow, posCol = mapProj(132.53188, 42.93468, inverse=False)
            # # posRow, posCol = mapProj(121.060295, 43.018272, inverse=False)
            #
            #
            # # 		gridlon_0:corners = [121.83443  131.50974  132.53188  121.060295] ;
            # # 		gridlon_0:long_name = longitude ;
            # # 		gridlon_0:grid_type = Lambert Conformal (secant, tangent, conical or bipolar) ;
            # # 		gridlon_0:units = degrees_east ;
            # # 		gridlon_0:Latin2 = [60.] ;
            # # 		gridlon_0:Latin1 = [30.] ;
            # # 		gridlon_0:Dy = [1.5] ;
            # # 		gridlon_0:Dx = [1.5] ;
            # # 		gridlon_0:Lov = [126.] ;
            # # 		gridlon_0:Lo1 = [121.83443] ;
            # # 		gridlon_0:La1 = [32.256874] ;
            #
            #
            #
            #
            #
            #
            #
            #
            # import xarray as xr
            # import eccodes
            # import pygrib
            # # import cfgrib
            #
            # # ds = xr.open_dataset(fileNcList[0], decode_times=False, engine='pynio')
            #
            # grb = pygrib.open(fileNcList[0])
            #
            # # grb.select()
            # grbInfo = grb.select(name='Temperature')[1]
            # lat2D, lon2D = grbInfo.latlons()
            #
            # #
            # # data = xr.open_dataset(fileNcList[0], decode_times=False )
            # # data = xr.open_dataset(fileNcList[0], engine='cfgrib')
            # # data = xr.open_dataset(fileNcList[0], engine='rasterio')
            # ds = xr.open_dataset(fileNcList[0], decode_times=False, engine='pynio')
            # ds = xr.open_dataset(fileNcList[0], decode_times=False)
            # data.plot(data.band)

            #
            # for v in ds:
            #     print("{}, {}, {}".format(v, ds[v].attrs["long_name"], ds[v].attrs["units"]))
            #
            # bb = ds.head()
            #
            # # ds.info()
            # # for v in ds:
            # #     print("{}, {}".format(v, ds[v].values))
            #
            # # ds['La1'].values
            #
            # # NCPCP_P8_L1_GLC0_acc1h, Large-scale precipitation (non-convective), kg m-2
            # # LSPRATE_P8_L1_GLC0_avg1h, Large scale precipitation rate, kg m-2s-1
            # # CPRAT_P8_L1_GLC0_acc1h, Convective precipitation rate, kg m-2 s-1
            #
            # # UM 데이터
            # ds2 = ds.get(['xgrid_0', 'ygrid_0', 'gridrot_0', 'TMP_P0_L1_GLC0', 'NCPCP_P8_L1_GLC0_acc1h'])
            # ds2.info()
            #
            # ds2['xgrid_0'].values
            # ds2['ygrid_0'].values
            #
            # # 121.83443  131.50974  132.53188  121.060295
            #
            # ds2['TMP_P0_L1_GLC0'].plot()
            # plt.show()
            #
            # # ds2['DPT_P0_L103_GLC0'].plot()
            # # plt.show()
            #
            # ds2['NCPCP_P8_L1_GLC0_acc1h'].plot()
            # plt.show()
            #
            # # KLAPS
            # # RN1
            # # pc : 1시간 누적 강수량
            # # 중심으로 사각형의 공간영역(31.344~44.283 N, 119.799~133.619 E)에 하나의층이 235×283의 수평 격자(약 5km의 공간해상도)로
            # ds2 = ds.get(['x', 'y', 'lon', 'lat', 't', 'pc'])
            # # ds2 = ds.get(['lon', 'lat', 'pc'])
            # ds2.info()
            #
            # # ds2['x']
            #
            # # ds2.sel(lon = 31.34)
            #
            # # dd = ds.get(['lat'])
            # # dd['lat']
            #
            # # import matplotlib
            # # matplotlib.use('TkAgg')
            # # import matplotlib.pyplot as plt
            #
            # ds2['levels_1'].values
            # ds2['record'].values
            # ds2['levels_23'].values

            # ds3 = ds2.isel(record=0, levels_1=0, levels_23=0)
            # ds3['t'].plot()
            # plt.show()
            # ds3['pc'].plot()
            # plt.show()
            #
            # # # ds3.sel(x = 10, y = 10)
            #
            # dsDataL3 = ds3.to_dataframe().reset_index(drop=False)
            #
            #
            #
            # # CSV to NetCDF 변환
            # dsDataL4 = dsDataL3.set_index(['lat', 'lon'])
            # dsDataL5 = dsDataL4.to_xarray()
            # # fileNameNoExt = os.path.basename(fileNcList[0]).split('.')[0]
            #
            # # dsDataL5.sel(lat = 35, lon = 125)
            #
            #
            # # saveFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, fileNameNoExt)
            # # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # # dsDataL5.to_netcdf(saveFile)
            # # log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # # dsDataL5['pc'].plot()
            # # plt.show()
            #
            # # ds3.plot()
            # plt.imshow(ds3['pc'].values)
            # plt.colorbar()
            # plt.show()
            # import pyproj
            # import cartopy.crs as ccrs
            # # map_proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=45)
            #
            # # mapLccProj = ccrs.LambertConformal(
            # #     central_longitude=126
            # #     , central_latitude=38
            # #     , secant_latitudes=(30, 60)
            # #     # , false_easting=imgProjInfo['false_easting']
            # #     # , false_northing=imgProjInfo['false_northing']
            # # )
            # #
            # # mapLccProjInfo = '+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs'
            # mapLccProjInfo = '+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs'
            # # mapLccProjInfo = '+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=5000m +no_defs +type=crs'
            #
            # # try:
            # #     mapLccProjInfo = mapLccProj.to_proj4()
            # # except Exception as e:
            # #     log.error("Exception : {}".format(e))
            #
            # mapProj = pyproj.Proj(mapLccProjInfo)
            #
            # lonGeo, latGeo = mapProj(rowEle1D, colEle1D, inverse=True)
            # posRow, posCol = mapProj(0, 0, inverse=False)
            #
            # lonGeo, latGeo = mapProj(posRow, posCol, inverse=True)
            #
            # # p = air.plot(
            # #     transform=ccrs.PlateCarree(),  # the data's projection
            # #     col="time",
            # #     col_wrap=1,  # multiplot settings
            # #     aspect=ds.dims["lon"] / ds.dims["lat"],  # for a sensible figsize
            # #     subplot_kws={"projection": map_proj},
            # # )  # the plot's projection
            #
            # # We have to set the map's options on all axes
            # # for ax in p.axes.flat:
            # #     ax.coastlines()
            # #     ax.set_extent([-160, -30, 5, 75])
            #
            #
            # # ds2.to_netcdf()
            #
            # # ds.get('TMP_P0_L103_GLL0')
            #
            # df = ds.to_dataframe()
            #
            #
            # # data.to_netcdf('netcdf_file.nc')
            #
            # grb = pygrib.open(fileNcList[0])
            # grbInfo = grb.select(name='Temperature')[1]
            #
            # dtValidDate = grbInfo.validDate
            # dtAnalDate = grbInfo.analDate
            #
            # TA = grbInfo.values[300, 300]
            #
            #
            #     # geoData.geometry
            #
            # # # 도법 설정
            # # proj4326 = 'epsg:4326'
            # # mapProj4326 = Proj(proj4326)
            #
            # # NetCDF 파일 읽기
            # # inpNcFile = '{}/{}/{}.nc'.format(globalVar['outPath'], 'LSH0344', '*')
            # # # inpNcFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, '*')
            # # fileNcList = sorted(glob.glob(inpNcFile))
            # #
            # # ncData = xr.open_dataset(fileNcList[0])
            # #
            # # # shp 파일 읽기
            # # inpFile = '{}/{}/{}.shp'.format(globalVar['inpPath'], serviceName, '*')
            # # fileList = sorted(glob.glob(inpFile))
            # #
            # # if fileList is None or len(fileList) < 1:
            # #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            # #
            # # # fileInfo = '/DATA/INPUT/LSH0347/eThekwini.shp'
            # # for i, fileInfo in enumerate(fileList):
            # #     log.info("[CHECK] fileInfo : {}".format(fileInfo))
            # #
            # #     # geoData = gpd.read_file(fileInfo, crs='epsg:4326')
            # #     geoData = gpd.read_file(fileInfo)
            # #     # weightMap = xa.pixel_overlaps(ncData, geoData)
            # #     # geoAggData = xa.aggregate(ncData, weightMap)
            # #
            # #     # 126.0, 33.96, 128.0, 38.33
            # #     # <Derived Projected CRS: EPSG:5181>
            # #     # Name: Korea 2000 / Central Belt
            # #     geoData.crs
            # #
            # #     geoDataL1 = geoData.to_crs(4326)
            # #     geoDataL1['lon'] = geoDataL1.centroid.x
            # #     geoDataL1['lat'] = geoDataL1.centroid.y
            # #
            # #
            # #
            # #     from shapely import wkt
            # #     # geoData['Coordinates'] = gpd.GeoSeries.from_wkt(geoData['geometry'])
            # #
            # #     geoDataL1['geometry'].plot()
            # #     plt.show()
            # #
            # #
            # #
            # #     geoData['centroid'] = geoData.centroid
            # #     geoData['val'] = geoAggData.to_dataset()['val']
            # #
            # #     geoData.plot(column=geoData['val'], legend=True)
            # #     geoData.apply(lambda x: plt.annotate(text=x['NAME_2'], xy=x.geometry.centroid.coords[0], ha='center', color='white'), axis=1);
            # #     plt.show()

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
