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
import seaborn as sns
from pandas.tseries.offsets import Day, Hour, Minute, Second
import re
import tempfile
import shutil

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

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================

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
    serviceName = 'LSH0543'

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
                'srtDate': '2023-10-01'
                , 'endDate': '2024-01-01'
                , 'invDate': '1m'

                # 수행 목록
                , 'modelList': ['REANALY-ECMWF-1D']

                # 일 평균
                , 'REANALY-ECMWF-1D': {
                    # 원본 파일 정보
                    'filePath': '/DATA/INPUT/LSH0544/%Y%m'
                    , 'fileName': 'data.nc'
                    , 'comVar': {'longitude': 'lon', 'latitude': 'lat', 'Value': '{}'}
                    , 'varList': ['t2m', 'tp', 'tp']

                    # 가공 파일 덮어쓰기 여부
                    , 'isOverWrite': True
                    # , 'isOverWrite': False

                    # 가공 변수
                    , 'procList': ['TXX', 'R10', 'CDD']

                    # 가공 파일 정보
                    , 'procPath': '/DATA/OUTPUT/LSH0543'
                    , 'procName': '{}_{}_ClimStatAnomaly-{}_{}_{}-{}.csv'
                }
            }

            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'permaice.shp')
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1:
                log.error(f'[ERROR] inpFile : {inpFile} / 입력 자료를 확인해주세요.')


            # permaice.shp

            import geopandas as gpd
            import xarray as xr
            from pyproj import CRS

            shpData = gpd.read_file(fileList[0])
            # /DATA/INPUT/LSH0543/permaice.shp
            # shp1 = gpd.read_file(fileList[2])
            # shp1.plot(color='None', edgecolor='black', linewidth=3)



            # 헤더 데이터
            # shpData.info()
            # <class 'geopandas.geodataframe.GeoDataFrame'>
            # RangeIndex: 13671 entries, 0 to 13670
            # Data columns (total 7 columns):
            #  #   Column    Non-Null Count  Dtype
            # ---  ------    --------------  -----
            #  0   NUM_CODE  13671 non-null  object
            #  1   COMBO     13671 non-null  object
            #  2   RELICT    13 non-null     object
            #  3   EXTENT    7191 non-null   object
            #  4   CONTENT   7191 non-null   object
            #  5   LANDFORM  7032 non-null   object
            #  6   geometry  13671 non-null  geometry
            # dtypes: geometry(1), object(6)
            # memory usage: 747.8+ KB

            # 좌표계
            # shpData.crs

            # Polar Stereographic 좌표계로 변환 (여기서는 북극 스테레오그래픽 예시, EPSG:3995 사용)
            polar_crs = CRS("EPSG:3995")  # 북극 스테레오그래픽
            # gdf_polar = shpData.to_crs(polar_crs)
            gdf_polar = shpData.to_crs(epsg=4326)

            gdf_polar.plot()
            saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, '테스트')
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            # plt.show()
            log.info(f'[CHECK] saveImg : {saveImg}')

            # bb = gdf_polar.to_xarray()

            g = gdf_polar.iloc[[1, ]]
            g.plot()
            plt.show()

            g['geometry']

            g.plot(column='NUM_CODE', legend=True)
            plt.show()

            #  0   NUM_CODE  13671 non-null  object
            #  1   COMBO     13671 non-null  object
            #  2   RELICT    13 non-null     object
            #  3   EXTENT    7191 non-null   object
            #  4   CONTENT   7191 non-null   object
            #  5   LANDFORM  7032 non-null   object
            #  6   geometry  13671 non-null  geometry



            h = shpData.iloc[[1, ]]
            # h.plot()
            # plt.show()

            # gdf_polar['geometry'][0].plot()

            # gdf_polar['coords'] = gdf_polar['geometry'].apply(lambda x: x.representative_point().coords[:])
            # gdf_polar['coords'] = [coords[0] for coords in gdf_polar['coords']]

            # gdf_polar
            print('sadfasdfasdf')


            #   fig, ax = plt.subplots(figsize=(10, 8))
            #
            #         crs = {'init': 'epsg:4162'}
            #         geometry = [Point(xy) for xy in zip(result_data["longitude"], result_data["latitude"])]
            #         geodata1 = gpd.GeoDataFrame(result_data, crs=crs, geometry=geometry)
            #
            #         data_seoul['coords'] = data_seoul['geometry'].apply(lambda x: x.representative_point().coords[:])
            #         data_seoul['coords'] = [coords[0] for coords in data_seoul['coords']]
            #
            #         # 컬러바 표시
            #         # gplt.kdeplot(geodata1, cmap='rainbow', zorder=0, cbar=True, shade=True, alpha=0.5, ax=ax)
            #         gplt.kdeplot(geodata1, cmap='rainbow', shade=True, alpha=0.5, ax=ax)
            #         gplt.polyplot(data_seoul, ax=ax)
            #
            #         # 서울 시군구 표시
            #         for i, row in data_seoul.iterrows():
            #             ax.annotate(size=10, text=row['sigungu_nm'], xy=row['coords'], horizontalalignment='center')
            #
            #         plt.gcf()
            #         plt.savefig(saveImg, width=10, height=8, dpi=600, bbox_inches='tight', transparent = True)
            #         plt.show()


            shpData.plot()
            plt.show()
            #
            # proj = ccrs.NorthPolarStereo(central_longitude=90)
            # fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': proj})

            # import cartopy.crs as ccrs
            # import matplotlib.pyplot as plt
            #
            # ax = plt.axes(projection=ccrs.PlateCarree())  # PlateCarreeは正距円筒図法
            # ax.coastlines()
            #
            # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            # plt.show()
            # # plt.show()

            print('setset')

            # shp1.to_xarray()

            # shp1['EXTENT']
            # shp1.exterior

            # # ===================================================================================
            # # 가공 파일 생산
            # # ===================================================================================
            # for modelType in sysOpt['modelList']:
            #     log.info(f'[CHECK] modelType : {modelType}')
            #
            #     modelInfo = sysOpt.get(modelType)
            #     if modelInfo is None: continue
            #
            #     # 시작일/종료일에 따른 데이터 병합
            #     mrgData = xr.Dataset()
            #     for dtDateInfo in dtDateList:
            #         log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
            #
            #         inpFilePattern = '{}/{}'.format(modelInfo['filePath'], modelInfo['fileName'])
            #         inpFile = dtDateInfo.strftime(inpFilePattern)
            #         fileList = sorted(glob.glob(inpFile))
            #
            #         if fileList is None or len(fileList) < 1:
            #             # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
            #             continue
            #
            #         fileInfo = fileList[0]
            #         data = xr.open_dataset(fileInfo)
            #         log.info(f'[CHECK] fileInfo : {fileInfo}')
            #
            #         dataL1 = data
            #
            #         # 변수 삭제
            #         selList = ['expver']
            #         for selInfo in selList:
            #             try:
            #                 dataL1 = dataL1.isel(expver=1).drop_vars([selInfo])
            #             except Exception as e:
            #                 pass
            #
            #         mrgData = xr.merge([mrgData, dataL1])
            #
            #
            #     # ******************************************************************************************************
            #     # 1) 월간 데이터에서 격자별 값을 추출하고 새로운 3장(각각 1장)의 netCDF 파일로 생성
            #     # ******************************************************************************************************
            #     for varIdx, varInfo in enumerate(modelInfo['varList']):
            #         procInfo = modelInfo['procList'][varIdx]
            #         log.info(f'[CHECK] varInfo : {varInfo} / procInfo : {procInfo}')
            #
            #         # TXX: Montly maximum value of daily maximum temperature
            #         if re.search('TXX', procInfo, re.IGNORECASE):
            #             # 0 초과 필터, 그 외 결측값 NA
            #             varData = mrgData[varInfo]
            #
            #             varDataL1 = varData.where(varData > 0).resample(time='1D').max()
            #             varDataL2 = varDataL1.resample(time='1M').max()
            #
            #         # R10: Number of heavy precipitation days(precipitation > 10mm)
            #         elif re.search('R10', procInfo, re.IGNORECASE):
            #             # 단위 변환 (m/hour -> mm/day)
            #             varData = mrgData['tp'] * 24 * 1000
            #
            #             varDataL1 = varData.resample(time='1D').sum()
            #             varDataL2 = varDataL1.where(varDataL1 > 10.0).resample(time='1M').count()
            #
            #         # CDD: The largests No. of consecutive days with, 1mm of precipitation
            #         elif re.search('CDD', procInfo, re.IGNORECASE):
            #
            #             # 단위 변환 (m/hour -> mm/day)
            #             varData = mrgData['tp'] * 24 * 1000
            #
            #             varDataL1 = varData.resample(time='1D').sum()
            #
            #             # True: 1 mm 이상 강수량 / False: 그 외
            #             varDataL1 = varDataL1 >= 1.0
            #             varDataL2 = varDataL1.resample(time='1M').apply(calcMaxContDay)
            #         else:
            #             continue
            #
            #         timeList = varDataL2['time'].values
            #         minDate = pd.to_datetime(timeList).min().strftime("%Y%m%d")
            #         maxDate = pd.to_datetime(timeList).max().strftime("%Y%m%d")
            #
            #         saveFile = '{}/{}/{}_{}_{}-{}.nc'.format(globalVar['outPath'], serviceName, modelType, procInfo, minDate, maxDate)
            #         os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #         varDataL2.to_netcdf(saveFile)
            #         log.info(f'[CHECK] saveFile : {saveFile}')
            #
            #         # ******************************************************************************************************
            #         # 2) 각 격자별 trend를 계산해서 지도로 시각화/ Mann Kendall 검정
            #         # (2개월 데이터로만 처리해주셔도 됩니다. 첨부사진처럼 시각화하려고 합니다. )
            #         # ******************************************************************************************************
            #         colName = 'slope'
            #
            #         mkData = xr.apply_ufunc(
            #             calcMannKendall,
            #             varDataL2,
            #             kwargs={'colName': colName},
            #             input_core_dims=[['time']],
            #             output_core_dims=[[]],
            #             vectorize=True,
            #             dask='parallelized',
            #             output_dtypes=[np.float64],
            #             dask_gufunc_kwargs={'allow_rechunk': True}
            #         ).compute()
            #
            #         mkName = f'{procInfo}-{colName}'
            #         mkData.name = mkName
            #
            #         saveFile = '{}/{}/{}_{}-{}_{}-{}.nc'.format(globalVar['outPath'], serviceName, modelType, mkName, 'MK', minDate, maxDate)
            #         os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #         mkData.to_netcdf(saveFile)
            #         log.info(f'[CHECK] saveFile : {saveFile}')
            #
            #         # mkData.plot()
            #         # plt.show()

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