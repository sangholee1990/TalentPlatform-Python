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
from _ast import expr
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc

import seaborn as sns

from scipy.interpolate import Rbf
from xarray.util.generate_aggregations import skipna

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
        # fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        # fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        # plt.rcParams['font.family'] = fontName

    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        globalVar[key] = val

    # 전역 변수
    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)

        globalVar[key] = val.replace('\\', '/')

        log.info(f"[CHECK] {key} : {val}")

    return globalVar

# 날짜형을 10진수 변환
def convDateToDeci(dtDate):
    srt = datetime(year=dtDate.year, month=1, day=1)
    end = datetime(year=dtDate.year + 1, month=1, day=1)
    return dtDate.year + ((dtDate - srt) / (end - srt))

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 컬럼비아 인구 데이터 UN 총인구 가중치 계산 및 NetCDF 가공

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'LSH0575'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info('[START] {}'.format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("init"))

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

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': '2019-01-01'
                , 'endDate': '2023-01-01'

                # 경도 최소/최대/간격
                # , 'lonMin': 0
                # , 'lonMax': 360
                , 'lonMin': 130
                , 'lonMax': 140
                , 'lonInv': 1

                # 위도 최소/최대/간격
                # , 'latMin': -90
                # , 'latMax': 90
                , 'latMin': 30
                , 'latMax': 40
                , 'latInv': 1
            }

            import xarray as xr
            import geopandas as gpd

            # 위경도 설정
            lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
            latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])
            log.info(f'[CHECK] len(lonList) : {len(lonList)}')
            log.info(f'[CHECK] len(latList) : {len(latList)}')


            # =========================================================
            # 엑셀 파일 읽기
            # =========================================================
            xlsxFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx')
            xlsxList = sorted(glob.glob(xlsxFile))
            xlsxInfo = xlsxList[0]
            xlsxData = pd.read_excel(xlsxInfo, skiprows=16, engine='openpyxl')

            # xlsxData.columns
            xlsxColList = ['ISO3 Alpha-code', 'Location code', 'Year', 'Total Population, as of 1 January (thousands)']

            locCodeList = np.unique(xlsxData['Location code'])
            xlsxDataL5 = pd.DataFrame()
            for locCode in locCodeList:
                xlsxDataL1 = xlsxData.loc[
                    (xlsxData['Location code'] == locCode)
                    & (2019 <= xlsxData['Year']) & (xlsxData['Year'] <= 2025)
                    ]
                if len(xlsxDataL1) < 1: continue

                log.info(f'[CHECK] locCode : {locCode}')

                xlsxDataL2 = xlsxDataL1.copy().rename(columns={'Total Population, as of 1 January (thousands)': 'orgVal'})
                xlsxDataL2['dtDate'] = pd.to_datetime(xlsxDataL2['Year'], format='%Y')
                # xlsxDataL2.set_index('dtDate', inplace=True)
                # xlsxDataL3['newVal'] = xlsxDataL3['orgVal'].interpolate(method='linear')

                xlsxDataL2['dtXran'] = pd.to_numeric(xlsxDataL2['dtDate'].apply(lambda x: convDateToDeci(x)), errors='coerce')
                xlsxDataL2['orgVal'] = pd.to_numeric(xlsxDataL2['orgVal'], errors='coerce')

                xlsxDataL3 = xlsxDataL2.copy().set_index('dtDate').resample('1D').asfreq()
                xlsxDataL3['dtXran'] = pd.to_numeric(xlsxDataL3.index.to_series().apply(lambda x: convDateToDeci(x)), errors='coerce')

                # Radial basis function (RBF) interpolation in N dimensions.
                try:
                    rbfModel = Rbf(xlsxDataL2['dtXran'].values, xlsxDataL2['orgVal'].values, function='linear')
                    rbfRes = rbfModel(xlsxDataL3['dtXran'].values)
                    xlsxDataL3['newVal'] = rbfRes
                except Exception as e:
                    log.error(f"Exception : {e}")

                xlsxDataL4 = xlsxDataL3.reset_index()[['dtDate', 'orgVal', 'newVal']]
                xlsxDataL4['code'] = xlsxDataL2['ISO3 Alpha-code'].values[0]
                xlsxDataL4['locCode'] = xlsxDataL2['Location code'].values[0]

                xlsxDataL5 = pd.concat([xlsxDataL5, xlsxDataL4], axis=0)

            # =========================================================
            # gpw 데이터 읽기/병합
            # =========================================================
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*/gpw_v4_national_identifier_grid_rev11_30_sec.tif')
            fileList = sorted(glob.glob(inpFile))
            # if (len(fileList) < 1): continue
            fileInfo = fileList[0]
            gpwNatData = xr.open_rasterio(fileInfo)
            gpwNatDataL1 = gpwNatData.sel(band=1)

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*/gpw_v4_population_count_rev11_2000_30_sec.tif')
            fileList = sorted(glob.glob(inpFile))
            # if (len(fileList) < 1): continue
            fileInfo = fileList[0]
            gpwPopData = xr.open_rasterio(fileInfo)
            gpwPopDataL1 = gpwPopData.sel(band=1).to_dataset(name='POP')

            # 데이터 병합
            gpwData = gpwPopDataL1.copy()
            gpwData.coords['NAT'] = (('y', 'x'), gpwNatDataL1.values)

            # gpwDataL1 = gpwData
            gpwDataL1 = gpwData.sel(x=slice(130, 131)).isel(y=slice(5500, 6000))

            gpwDataL1['NAT'] = xr.where((gpwDataL1['NAT'] == gpwNatData.nodatavals), np.nan, gpwDataL1['NAT'])
            gpwDataL1['POP'] = xr.where((gpwDataL1['POP'] == gpwPopData.nodatavals), np.nan, gpwDataL1['POP'])

            # NAT에 따른 합계
            sumData = gpwDataL1.groupby(gpwDataL1['NAT']).sum()


            # natList = np.unique(gpwDataL1['NAT'].values)
            # for nat in natList:
            #     if pd.isna(nat): continue
            #     log.info(f'[CHECK] nat : {nat}')
            #
            #     gpwDataL3 = gpwDataL2.sel(NAT = nat)
            #
            #     sumDataL1 = sumData.sel(NAT = nat)
            #     sumVal = sumDataL1['POP'].values

            gpwDataL2 = gpwDataL1.copy()
            locCodeList = np.unique(xlsxDataL5['locCode'])
            for locCode in locCodeList:
                log.info(f'[CHECK] locCode : {locCode}')

                gpwDataL3 = gpwDataL2.where(gpwDataL2['NAT'] == locCode, drop=True)


                sumDataL1 = sumData.sel(NAT = locCode)
                sumVal = sumDataL1['POP'].values

            # gpwNatDataL1['x'].values

            # gpwNatDataL1['NAT'].plot()
            # plt.show()

            # gpwPopDataL1 = gpwPopData.to_dataset(name='POP')
            # gpwPopDataL1 = gpwPopData.to_dataset(name='POP').isel(x=slice(120, 150), y=slice(40, 50))
            # gpwPopDataL1 = gpwPopData.sel(band = 1).to_dataset(name='POP').sel(x=slice(130, 131)).isel(y=slice(5500, 6000))

            # gpwNatDataL1['NAT'].plot()
            # gpwPopDataL1['POP'].plot()
            # plt.show()


            # gpwData = xr.merge([gpwNatDataL1, gpwPopDataL1])

            # gpwData

            # isMask = gpwNatDataL1['NAT'] == 156
            # gpwPopDataL1['POP'] * isMask.values
            #
            # gpwPopDataL1.coords['NAT'] = (('y', 'x'), gpwNatDataL1['NAT'].values)
            #
            # isMask = gpwPopDataL1['NAT'] == 156
            # gpwPopDataL1['POP'] * isMask
            #
            # # isMask = (gpwData['NAT'] == 156)
            #
            # gg = gpwPopDataL1['POP'].groupby(gpwPopDataL1['NAT']).sum()
            # gg.sel(NAT = 156).values



            # gpwData['NAT'].where(isMask, drop=True).values

            #
            # gpwData['NAT'] = xr.where((gpwData['NAT'] == gpwNatData.nodatavals), np.nan, gpwData['NAT'])
            # gpwData['POP'] = xr.where((gpwData['POP'] == gpwPopData.nodatavals), np.nan, gpwData['POP'])
            #
            # # gpwDataL1 = gpwData.sel(band=1)
            # gpwDataL1 = gpwData
            #
            # # np.unique(gpwDataL1['NAT'].values)
            # # np.unique(gpwDataL1['POP'].values)
            #
            # # array([156., 408., 643.,  nan])
            # # isMask = (gpwNatData.sel(band=1) == 156)
            # isMask = (gpwDataL1['NAT'] == 156)
            # # gpwNatData[isFlag]




            # gpwNatData.where(isMask)
            # gpwDataL1.where(isMask)

            # b = gpwDataL1['NAT'].where(isMask, drop=False)
            #
            # # pop_sum = gpwDataL1.groupby(gpwDataL1['NAT']).sum()
            #
            # # gpwDataL1.groupby(gpwDataL1['NAT']).sum( skipna=True)
            #
            # # gpwDataL1.sel(NAT = 156)
            #
            # isMask = (gpwDataL1['NAT'] == 156)
            # gpwDataL1['NAT'].where(isMask, drop=True).values
            #
            # # gpwData['x'].values
            #
            # # gpwDataL2 = gpwDataL1.to_dataframe().reset_index(drop=False)
            # gpwDataL2 = gpwData.to_dataframe().reset_index(drop=False)
            # gpwDataL2.dropna().groupby(['NAT']).sum().reset_index()
            #


            # =========================================================
            # gpw 데이터 읽기/병합
            # =========================================================
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*/gpw_v4_national_identifier_grid_rev11_30_sec.shp')
            # fileList = sorted(glob.glob(inpFile))
            # fileInfo = fileList[0]
            # shpData = gpd.read_file(fileInfo)
            #
            # # shpData.head()
            # # shpData.info()
            # # shpData.describe()
            #
            # # shpData.plot()
            # # plt.show()
            #
            # # [CHECK] shpCode : KOR / shpVal : 410 / shpName : Republic of Korea / shpGeoCen : POINT (127.75250070905732 36.30623116907186)
            # for idx, item in shpData.iterrows():
            #     shpCode = item.ISOCODE
            #     shpName = item.NAME0
            #     shpVal = item.Value
            #     shpGeoCen = item.geometry.centroid
            #     log.info(f'[CHECK] shpCode : {shpCode} / shpVal : {shpVal} / shpName : {shpName} / shpGeoCen : {shpGeoCen}')
            #


            # xlsxDataL2['dtDate'].apply(lambda x: x.timestamp())


            # 1950.0, 19757.089
            # 2023.0, 51759.392




            # lon1D = gpwNatDataL3['x'].values
            # lat1D = gpwNatDataL3['y'].values
            #
            # data = xr.Dataset(
            #     {
            #         'NAT': (('time', 'lat', 'lon'), (gpwNatDataL3.values).reshape(1, len(lat1D), len(lon1D)))
            #         , 'POP': (('time', 'lat', 'lon'), (gpwPopDataL3.values).reshape(1, len(lat1D), len(lon1D)))
            #     }
            #     , coords={
            #         'time': pd.date_range(pd.to_datetime(2020, format='%Y'), periods=1)
            #         , 'lat': lat1D
            #         , 'lon': lon1D
            #     }
            # )

            # print(data)

            # data['NAT'].plot()
            # plt.show()
            #
            # data['POP'].plot()
            # plt.show()

            # if (len(dataL5) < 1):
            #     dataL5 = dataL4
            # else:
            #     dataL5 = xr.concat([dataL5, dataL4], "time")

            # gpwDataL1 = gpwData
            # gpwDataL1['NAT'] = xr.where((gpwDataL1['NAT'] == gpwNatData.nodatavals), np.nan, gpwDataL1['NAT'])
            # gpwDataL1['POP'] = xr.where((gpwDataL1['POP'] == gpwPopData.nodatavals), np.nan, gpwDataL1['POP'])
            #
            # gpwDataL1['POP'].plot()
            # plt.show()

            # gpwPopDataL2.plot()
            # plt.show()

            # gpwPopData.attrs
            # dataL3 = dataL2.interp(x=lonList, y=latList, method='nearest')

            # 결측값 처리
            # dataL3 = xr.where((dataL3 < 0), np.nan, dataL3)

            # gpwNatDataL1 = gpwNatData.sel(band=1)
            # gpwNatDataL2 = gpwNatDataL1.interp({'x': lonList, 'y': latList}, method='nearest')
            # gpwNatDataL2 = gpwNatDataL1.isel(x=slice(14000, 14100), y=slice(100, 200))
            # gpwNatDataL2 = gpwNatDataL1.sel(x=slice(119.999, 121.001), y=slice(88.999, 90.001))
            # gpwNatDataL1['x'].values
            #
            # gpwNatDataL1['x'].values

            # gpwNatDataL2.plot()
            # plt.show()

            # gpwNatDataL2.plot()
            # plt.show()
            # gpwNatDataL1['x'].values

            # gpwNatDataL1.plot()

            # dataL1 = data.to_dataset('band').rename({1: 'val'})


        except Exception as e:
            log.error(f"Exception : {str(e)}")
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