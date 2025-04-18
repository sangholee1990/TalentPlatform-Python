# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import xarray as xr
from pandas.tseries.offsets import Day, Hour, Minute, Second
import re
import tempfile
import shutil
import pymannkendall as mk
from dask.distributed import Client
import dask
import ssl
import cartopy.crs as ccrs
from matplotlib import font_manager, rc

import geopandas as gpd
import cartopy.feature as cfeature

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

# SSL 인증 모듈
ssl._create_default_https_context = ssl._create_unverified_context

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
        , 'resPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'output', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'mapInfo')
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

        # 글꼴 설정
        fontList = font_manager.findSystemFonts(fontpaths=globalVar['fontPath'])
        for fontInfo in fontList:
            font_manager.fontManager.addfont(fontInfo)
            fontName = font_manager.FontProperties(fname=fontInfo).get_name()
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

def calcMannKendall(data, colName):
    try:
        # trend 추세, p 유의수준, Tau 상관계수, z 표준 검정통계량, s 불일치 개수, slope 기울기
        result = mk.original_test(data)
        return getattr(result, colName)
    except Exception:
        return np.nan


def calcMaxContDay(isMask):
    arr = isMask.astype(int)
    sumCum = np.where(arr, arr.cumsum(axis=0), 0)

    diff = np.diff(sumCum, axis=0, prepend=0)
    sumCumData = diff * arr

    result = sumCumData.max(axis=0) - sumCumData.min(axis=0)

    return result

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 연간 HDF5 위성 자료로부터 특정 영역 SHP 파일 내 마스킹

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
    serviceName = 'LSH0549'

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
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            # 옵션 설정
            sysOpt = {
                # 시작일, 종료일, 시간 간격 (연 1y, 월 1h, 일 1d, 시간 1h)
                'srtDate': '2012-10-01'
                , 'endDate': '2022-01-01'
                , 'invDate': '1y'

                # 수행 목록
                , 'modelList': ['SAT-VNP']

                , 'SAT-VNP': {
                    'filePath': '/DATA/INPUT/LSH0549/VNP46A3*'
                    , 'fileName': 'VNP46A3.A%Y*.*.*.*.h5'
                    , 'varList': ['AllAngle_Composite_Snow_Covered']

                    # 가공 변수
                    , 'procList': ['AllAngle_Composite_Snow_Covered']

                    # 가공 파일 정보
                    , 'procPath': '/DATA/OUTPUT/LSH0549'
                    , 'procName': '{}_{}-{}_{}-{}.nc'

                    # , 'figPath': '/DATA/FIG/LSH0549'
                    # , 'figName': '{}_{}-{}_{}-{}.png'
                }
                , 'SHP': {
                    'filePath': '/DATA/INPUT/LSH0549/gadm41_KOR_shp'
                    # , 'fileName': 'gadm41_KOR_0.shp'
                    , 'fileName': 'gadm41_KOR_1.shp'
                }
            }


            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

            # ===================================================================================
            # SHP 파일 읽기
            # ===================================================================================
            shpFile = '{}/{}'.format(sysOpt['SHP']['filePath'], sysOpt['SHP']['fileName'])
            shpData = gpd.read_file(shpFile, encoding='EUC-KR').to_crs(epsg=4326)

            # shpData.plot(color=None, edgecolor='k', facecolor='none')
            # for idx, row in shpData.iterrows():
            #     centroid = row.geometry.centroid
            #     plt.annotate(text=row['NAME_1'], xy=(centroid.x, centroid.y), horizontalalignment='center', verticalalignment='center')
            # plt.show()

            # ===================================================================================
            # 가공 파일 생산
            # ===================================================================================
            for modelType in sysOpt['modelList']:
                log.info(f'[CHECK] modelType : {modelType}')

                modelInfo = sysOpt.get(modelType)
                if modelInfo is None: continue

                # 시작일/종료일에 따른 데이터 병합
                mrgData = xr.Dataset()
                for dtDateInfo in dtDateList:
                    log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                    inpFilePattern = '{}/{}'.format(modelInfo['filePath'], modelInfo['fileName'])
                    inpFile = dtDateInfo.strftime(inpFilePattern)
                    fileList = sorted(glob.glob(inpFile))

                    if fileList is None or len(fileList) < 1:
                        # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                        continue

                    fileInfo = fileList[0]
                    # data = xr.open_dataset(fileInfo)

                    # import rasterio
                    # from rasterio import Affine, windows
                    #
                    # gg2 = rasterio.open(fileInfo).read(0)
                    # gg2.crs
                    # gg2.transform
                    # gg2.meta


                    data = xr.open_dataset(fileInfo, engine='rasterio')
                    # data = xr.open_dataset(fileInfo, engine='pynio')
                    # data = xr.open_dataset(fileInfo, engine='cfgrib')
                    log.info(f'[CHECK] fileInfo : {fileInfo}')

                    data['x'].values


                    dataL1 = data.isel(band = 0)
                    # dataL1 = data.sel(g0_lon_1 = slice(sysOpt['roi']['minLon'], sysOpt['roi']['maxLon']), g0_lat_0 = slice(sysOpt['roi']['minLat'], sysOpt['roi']['maxLat']))
                    # dataL1 = data.sel(g0_lon_2 = slice(sysOpt['roi']['minLon'], sysOpt['roi']['maxLon']), g0_lat_1 = slice(sysOpt['roi']['minLat'], sysOpt['roi']['maxLat']))

                    # ff = data.to_netcdf()
                    # dataL1.coords

                    from rasterio.warp import transform

                    # Compute the lon/lat coordinates with rasterio.warp.transform
                    ny, nx = len(dataL1['y']), len(dataL1['x'])
                    x, y = np.meshgrid(dataL1['x'], dataL1['y'])

                    # data.attrs
                    # Rasterio works with 1D arrays
                    lon, lat = transform("epsg:4326", {'init': 'EPSG:4326'}, x.flatten(), y.flatten())
                    lon = np.asarray(lon).reshape((ny, nx))
                    lat = np.asarray(lat).reshape((ny, nx))
                    da.coords['lon'] = (('y', 'x'), lon)
                    da.coords['lat'] = (('y', 'x'), lat)



                    gg = dataL1['HDFEOS_GRIDS_VIIRS_Grid_DNB_2d_Data_Fields_AllAngle_Composite_Snow_Covered']
                    lat = data['lat']

                    # data.data_vars
                    data.variables
                    # gg.plot()
                    plt.show()
                    ff = dataL1['x']

                    # 동적 NetCDF 생선
                    lon1D = dataL1['g0_lon_1'].values
                    lat1D = dataL1['g0_lat_0'].values
                    # lon1D = dataL1['g0_lon_2'].values
                    # lat1D = dataL1['g0_lat_1'].values
                    # time1D = dataL1['initial_time0_hours'].values
                    time1D = dtDateInfo

                    dataL2 = xr.Dataset(
                        coords={
                            'time': pd.date_range(time1D, periods=1)
                            # 'time': pd.to_datetime(time1D)
                            , 'lat': lat1D
                            , 'lon': lon1D
                        }
                    )

                    for varInfo in modelInfo['varList']:
                        try:
                            dataL2[varInfo] = (('time', 'lat', 'lon'), (dataL1[varInfo].values).reshape(1, len(lat1D), len(lon1D)))
                            # dataL2[varInfo] = (('time', 'lat', 'lon'), (dataL1[varInfo].values).reshape(len(time1D), len(lat1D), len(lon1D)))
                        except Exception as e:
                            pass

                    # 변수 삭제
                    selList = ['expver']
                    for selInfo in selList:
                        try:
                            dataL2 = dataL2.isel(expver=1).drop_vars([selInfo])
                        except Exception as e:
                            pass

                    mrgData = xr.merge([mrgData, dataL2])

                if len(mrgData) < 1: continue

                # 7월 추출
                # mrgData = mrgData.sel(time = (mrgData['time'].dt.month == 7))

                # shp 영역 내 자료 추출
                # import rioxarray

                # roiData = mrgData['2T_GDS0_SFC'].rio.write_crs("epsg:4326")
                # roiDataL1 = roiData.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
                # roiDataL2 = roiDataL1.rio.clip(shpData.geometry, shpData.crs, from_disk=True)
                #
                # roiDataL2.isel(time = 0).plot()
                # plt.show()

                for varIdx, varInfo in enumerate(modelInfo['varList']):
                    procInfo = modelInfo['procList'][varIdx]
                    log.info(f'[CHECK] varInfo : {varInfo} / procInfo : {procInfo}')

                    if re.search('t2m', procInfo, re.IGNORECASE):
                        # 0 초과 필터, 그 외 결측값 NA
                        varData = mrgData[varInfo]
                        varDataL1 = varData.where(varData > 0)
                        varDataL2 = varDataL1 - 273.15
                    else:
                        continue

                    timeList = varDataL2['time'].values
                    minDate = pd.to_datetime(timeList).min().strftime("%Y%m%d")
                    maxDate = pd.to_datetime(timeList).max().strftime("%Y%m%d")

                    # 통계 분석
                    # timeList = varDataL2['time'].values
                    # for timeInfo in timeList:
                    #     selData = varDataL2.sel(time = timeInfo)
                    #
                    #     preDate = pd.to_datetime(timeInfo).strftime("%Y")
                    #
                    #     meanVal = np.nanmean(selData)
                    #     maxVal = np.nanmax(selData)
                    #     minVal = np.nanmin(selData)
                    #
                    #     log.info(f'[CHECK] preDate : {preDate} / meanVal : {meanVal} / maxVal : {maxVal} / minVal : {minVal}')

                    # ******************************************************************************************************
                    # 관측소 시계열 검정
                    # ******************************************************************************************************
                    for i, posInfo in pd.DataFrame(sysOpt['posData']).iterrows():

                        posName = f"{posInfo['GU']}-{posInfo['NAME']}"

                        saveFilePattern = '{}/{}'.format(modelInfo['figPath'], modelInfo['figName'])
                        saveImg = saveFilePattern.format(modelType, 'orgTime', posName, minDate, maxDate)
                        os.makedirs(os.path.dirname(saveImg), exist_ok=True)

                        posData = varDataL2.interp({'lon': posInfo['LON'], 'lat': posInfo['LAT']}, method='linear')
                        posData.plot(marker='o')
                        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                        log.info(f'[CHECK] saveImg : {saveImg}')
                        plt.show()
                        plt.close()

                        # 5년 이동평균
                        saveFilePattern = '{}/{}'.format(modelInfo['figPath'], modelInfo['figName'])
                        saveImg = saveFilePattern.format(modelType, 'movTime', posName, minDate, maxDate)
                        os.makedirs(os.path.dirname(saveImg), exist_ok=True)

                        movData = posData.rolling(time=5, center=True).mean()
                        movData.plot(marker='o')
                        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                        log.info(f'[CHECK] saveImg : {saveImg}')
                        plt.show()
                        plt.close()


                    # ******************************************************************************************************
                    # Mann Kendall 검정
                    # ******************************************************************************************************
                    colName = 'slope'
                    mkData = xr.apply_ufunc(
                        calcMannKendall,
                        varDataL2,
                        kwargs={'colName': colName},
                        input_core_dims=[['time']],
                        output_core_dims=[[]],
                        vectorize=True,
                        dask='parallelized',
                        output_dtypes=[np.float64],
                        dask_gufunc_kwargs={'allow_rechunk': True}
                    ).compute()

                    mkName = f'{procInfo}-{colName}'
                    mkData.name = mkName


                    # 시각화
                    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

                    ax.coastlines()
                    gl = ax.gridlines(draw_labels=True)
                    gl.top_labels = False
                    gl.right_labels = False

                    mkData.plot(ax=ax, transform=ccrs.PlateCarree())

                    shpData.plot(ax=ax, edgecolor='k', facecolor='none')
                    for idx, row in shpData.iterrows():
                        centroid = row.geometry.centroid
                        ax.annotate(text=row['gu'], xy=(centroid.x, centroid.y), horizontalalignment='center', verticalalignment='center')

                    minVal = np.nanmin(mkData)
                    maxVal = np.nanmax(mkData)
                    meanVal = np.nanmean(mkData)
                    plt.title(f'minVal = {minVal:.2f} / meanVal = {meanVal:.2f} / maxVal = {maxVal:.2f}')

                    saveFilePattern = '{}/{}'.format(modelInfo['figPath'], modelInfo['figName'])
                    saveImg = saveFilePattern.format(modelType, mkName, 'MK', minDate, maxDate)
                    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                    log.info(f'[CHECK] saveImg : {saveImg}')
                    plt.show()
                    plt.close()

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
