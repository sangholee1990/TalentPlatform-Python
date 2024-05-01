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
import rioxarray

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
from dask.distributed import Client

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


def makePlot(data, saveImg, shpData, opt=None):

    # log.info('[START] {}'.format('makePlot'))

    result = None

    try:
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

        ax.coastlines()
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

        if opt is None:
            data.plot(ax=ax, transform=ccrs.PlateCarree())
        else:
            data.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=opt['vmin'], vmax=opt['vmax'])

        shpData.plot(ax=ax, edgecolor='k', facecolor='none')
        for idx, row in shpData.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(text=row['gu'], xy=(centroid.x, centroid.y), horizontalalignment='center', verticalalignment='center')

        minVal = np.nanmin(data)
        maxVal = np.nanmax(data)
        meanVal = np.nanmean(data)

        plt.title(f'minVal = {minVal:.3f} / meanVal = {meanVal:.3f} / maxVal = {maxVal:.3f}')
        plt.suptitle(os.path.basename(saveImg).split('.')[0])

        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
        log.info(f'[CHECK] saveImg : {saveImg}')
        # plt.show()
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

    # finally:
        # log.info('[END] {}'.format('makePlot'))

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 시간별 재분석 ERA5 모델 (Grib)로부터 통계 분석 그리고 MK 검정 (Mann-Kendall)

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
    serviceName = 'LSH0547'

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
                'srtDate': '1980-10-01'
                , 'endDate': '2024-01-01'
                , 'invDate': '1y'

                # 영역 설정
                , 'roi': {
                    'gl': {'minLon': -180, 'maxLon': 180, 'minLat': -90, 'maxLat': 90}
                    , 'as': {'minLon': 80, 'maxLon': 180, 'minLat': 10, 'maxLat': 60}
                    , 'ko': {'minLon': 123, 'maxLon': 133, 'minLat': 31, 'maxLat': 44}
                    , 'gw': {'minLon': 126.638, 'maxLon': 127.023, 'minLat': 35.0069, 'maxLat': 35.3217}
                }

                , 'seasonList': {
                    'MAM': [3, 4, 5]
                    , 'JJA': [6, 7, 8]
                    , 'SON': [9, 10, 11]
                    , 'DJF': [12, 1, 2]
                }

                # 관측 지점
                , 'posData': [
                    {"GU": "광산구", "NAME": "광산 관측지점", "ENGNAME": "AWS GwangSan", "ENGSHORTNAME": "St. GwangSan", "LAT": 35.12886, "LON": 126.74525, "INFO": "AWS", "MARK": "\u23F3"}
                    , {"GU": "북구", "NAME": "과학기술원", "ENGNAME": "AWS Gwangju Institute of Science and Technology", "ENGSHORTNAME": "St. GIST", "LAT": 35.23026, "LON": 126.84076, "INFO": "AWS", "MARK": "\u23F3"}
                    , {"GU": "서구", "NAME": "풍암 관측지점", "ENGNAME": "AWS PungArm", "ENGSHORTNAME": "St. PungArm", "LAT": 35.13159, "LON": 126.88132, "INFO": "AWS", "MARK": "\u23F3"}
                    , {"GU": "동구", "NAME": "조선대 관측지점", "ENGNAME": "AWS Chosun University", "ENGSHORTNAME": "St. Chosun", "LAT": 35.13684, "LON": 126.92875, "INFO": "AWS", "MARK": "\u23F3"}
                    , {"GU": "동구", "NAME": "무등산 관측지점", "ENGNAME": "AWS Mudeung Mountain", "ENGSHORTNAME": "St. M.T Mudeung", "LAT": 35.11437, "LON": 126.99743, "INFO": "AWS", "MARK": "\u23F3"}
                    , {"GU": "남구", "NAME": "광주 남구 관측지점", "ENGNAME": "AWS Nam-gu", "ENGSHORTNAME": "St. Nam-gu", "LAT": 35.100807, "LON": 126.8985, "INFO": "AWS", "MARK": "\u23F3"}
                    , {"GU": "북구", "NAME": "광주지방기상청", "ENGNAME": "GwangJuKMA", "ENGSHORTNAME": "GMA", "LAT": 35.17344444, "LON": 126.8914639, "INFO": "LOCATE", "MARK": "\u23F3"}
                ]

                # 수행 목록
                , 'modelList': ['REANALY-ECMWF-1M-GW-IDX2']

                , 'analyList': ['1981-2020', '1981-1990', '1991-2000', '2001-2010', '2011-2020']

                , 'REANALY-ECMWF-1M-GW-IDX2': {
                    # 'filePath': '/DATA/INPUT/LSH0547/era5_monthly_gwangju/%Y'
                    'filePath': '/DATA/INPUT/LSH0547/gwangju_monthly_new/monthly/%Y'
                    , 'fileName': 'era5_merged_monthly_mean.grib'
                    , 'varList': ['SD_GDS0_SFC']
                    , 'procList': ['sd']

                    # 가공 파일 정보
                    , 'procPath': '/DATA/OUTPUT/LSH0547'
                    , 'procName': '{}_{}-{}_{}-{}.nc'

                    , 'figPath': '/DATA/FIG/LSH0547'
                    , 'figName': '{}_{}-{}_{}-{}.png'
                }

                , 'POS': {
                    'filePath': '/DATA/INPUT/LSH0547/POS'
                    , 'fileName': 'STCS_폭염일수_20240417212144.csv'
                }
                , 'SHP-GW': {
                    'filePath': '/DATA/INPUT/LSH0547/shp'
                    , 'fileName': '002_gwj_gu.shp'
                }
                , 'SHP-DTL-GW': {
                    'filePath': '/DATA/INPUT/LSH0547/shp'
                    , 'fileName': '002_gwj_gu_dong_5179.shp'
                }
            }

            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

            # ===================================================================================
            # SHP 파일 읽기
            # ===================================================================================
            shpFile = '{}/{}'.format(sysOpt['SHP-GW']['filePath'], sysOpt['SHP-GW']['fileName'])
            shpData = gpd.read_file(shpFile, encoding='EUC-KR').to_crs(epsg=4326)

            shpDtlFile = '{}/{}'.format(sysOpt['SHP-DTL-GW']['filePath'], sysOpt['SHP-DTL-GW']['fileName'])
            shpDtlData = gpd.read_file(shpDtlFile, encoding='EUC-KR').to_crs(epsg=4326)

            # shpData.plot(color=None, edgecolor='k', facecolor='none')
            # for idx, row in shpData.iterrows():
            #     centroid = row.geometry.centroid
            #     plt.annotate(text=row['gu'], xy=(centroid.x, centroid.y), horizontalalignment='center', verticalalignment='center')
            # plt.show()

            # ===================================================================================
            # POS 파일 읽기
            # # ===================================================================================
            # csvFile = '{}/{}'.format(sysOpt['POS']['filePath'], sysOpt['POS']['fileName'])
            # # csvData = pd.read_csv(csvFile, encoding='EUC-KR')
            # csvData = pd.read_csv(csvFile, encoding='UTF-8')
            #
            # # csvData['time'] = pd.to_datetime(csvData['년월'].str.strip())
            # csvData['time'] = pd.to_datetime(csvData['연도'], format='%Y')
            # csvData['kma-org'] = csvData['연합계'].astype(float)

            # ===================================================================================
            # 가공 파일 생산
            # ===================================================================================
            for modelType in sysOpt['modelList']:
                log.info(f'[CHECK] modelType : {modelType}')

                modelInfo = sysOpt.get(modelType)
                if modelInfo is None: continue

                # mrgData = xr.open_dataset('/DATA/OUTPUT/LSH0547/REANALY-ECMWF-1M-GW_proc-mrg_19810101-20221201.nc', engine='pynio')
                # mrgData = xr.open_dataset('/DATA/OUTPUT/LSH0547/REANALY-ECMWF-1M-GW-IDX_proc-mrg_19811231-20221231.nc', engine='pynio')
                mrgData = xr.open_dataset('/DATA/OUTPUT/LSH0547/REANALY-ECMWF-1M-GW-IDX2_proc-mrg_19811231-20231231.nc', engine='pynio')

                # shp 영역 내 자료 추출
                roiData = mrgData.rio.write_crs("epsg:4326")
                roiDataL1 = roiData.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
                roiDataL2 = roiDataL1.rio.clip(shpData.geometry, shpData.crs, from_disk=True)

                # roiDataL2.isel(time = 0)['cd'].plot()
                # plt.show()

                # roiDataL2.isel(time=1)['id'].plot()
                # plt.show()

                meanData = pd.DataFrame()
                valData = pd.DataFrame()
                for varIdx, varInfo in enumerate(modelInfo['varList']):
                    procInfo = modelInfo['procList'][varIdx]
                    log.info(f'[CHECK] varInfo : {varInfo} / procInfo : {procInfo}')

                    # ******************************************************************************************************
                    # 관측소 시계열 검정
                    # ******************************************************************************************************
                    mrgDataL1 = mrgData
                    mrgDataL2 = mrgDataL1
                    mrgDataL3 = mrgDataL2

                    # mrgDataL1.isel(time = 0).plot()
                    # plt.show()

                    # timeList = mrgData['time'].values
                    # minDate = pd.to_datetime(timeList).min().strftime("%Y%m%d")
                    # maxDate = pd.to_datetime(timeList).max().strftime("%Y%m%d")
                    # for i, posInfo in pd.DataFrame(sysOpt['posData']).iterrows():
                    #     # if not re.search('hd', procInfo, re.IGNORECASE): continue
                    #
                    #     posName = f"{posInfo['GU']}-{posInfo['NAME']}"
                    #
                    #     saveFilePattern = '{}/{}'.format(modelInfo['figPath'], modelInfo['figName'])
                    #     saveImg = saveFilePattern.format(modelType, procInfo, 'org', posName, f'{minDate}-{maxDate}')
                    #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                    #
                    #     mrgDataL3['time'] = pd.to_datetime(mrgDataL3['time'].dt.strftime('%Y'))
                    #
                    #     # posData = mrgDataL3.interp({'lon': posInfo['LON'], 'lat': posInfo['LAT']}, method='linear')
                    #     posData = mrgDataL3[procInfo].interp({'lon': posInfo['LON'], 'lat': posInfo['LAT']}, method='nearest')
                    #     posData.plot(marker='o')
                    #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                    #     log.info(f'[CHECK] saveImg : {saveImg}')
                    #     plt.show()
                    #     plt.close()
                    #
                    #     # 5년 이동평균
                    #     saveFilePattern = '{}/{}'.format(modelInfo['figPath'], modelInfo['figName'])
                    #     saveImg = saveFilePattern.format(modelType, procInfo, 'mov', posName, f'{minDate}-{maxDate}')
                    #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                    #
                    #     movData = posData.rolling(time=5, center=True).mean()
                    #     movData.plot(marker='o')
                    #     plt.title(posName)
                    #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                    #     log.info(f'[CHECK] saveImg : {saveImg}')
                    #     plt.show()
                    #     plt.close()
                    #
                    #     # posDataL1 = posData.to_dataframe().reset_index(drop=False).rename({'2T_GDS0_SFC': 'ecmwf-org'}, axis='columns')
                    #     # movDataL1 = movData.to_dataframe().reset_index(drop=False).rename({'2T_GDS0_SFC': 'ecmwf-mov'}, axis='columns')
                    #     posDataL1 = posData.to_dataframe().reset_index(drop=False).rename({procInfo: 'ecmwf-org'}, axis='columns')
                    #     movDataL1 = movData.to_dataframe().reset_index(drop=False).rename({procInfo: 'ecmwf-mov'}, axis='columns')
                    #
                    #     # posDataL2 = pd.merge(left=posDataL1, right=movDataL1, how='left', left_on=['time', 'lon', 'lat'], right_on=['time', 'lon', 'lat'])
                    #     # posDataL2 = posDataL1.merge(movDataL1, how='left', on=['time', 'lon', 'lat'])
                    #     posDataL2 = (posDataL1.merge(movDataL1, how='left', on = ['time', 'lon', 'lat'])
                    #                  .merge(csvData, how='left', on=['time']))
                    #
                    #     # 엑셀 저장
                    #     saveXlsxFile = '{}/{}/{}-{}-{}-{}.xlsx'.format(globalVar['outPath'], serviceName, modelType, procInfo, 'pos', posName)
                    #     os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)
                    #     with pd.ExcelWriter(saveXlsxFile, engine='xlsxwriter', options={'use_zip64': True}) as writer:
                    #         posDataL2.to_excel(writer, sheet_name='meanData', index=True)
                    #     log.info(f'[CHECK] saveXlsxFile : {saveXlsxFile}')


                    for roiName in sysOpt['roi']:
                        if not re.search('gw', roiName, re.IGNORECASE): continue
                        log.info(f'[CHECK] roiName : {roiName}')

                        roi = sysOpt['roi'][roiName]

                        # lonList = [x for x in varDataL1['lon'].values if roi['minLon'] <= x <= roi['maxLon']]
                        # latList = [x for x in varDataL1['lat'].values if roi['minLat'] <= x <= roi['maxLat']]

                        # ****************************************************
                        # 평균 데이터
                        # ****************************************************
                        # varDataL3 = varDataL2.sel(lat=latList, lon=lonList)
                        # varDataL3 = roiDataL2[procInfo]
                        varDataL3 = roiDataL2.get(procInfo)
                        if varDataL3 is None: continue

                        varDataL4 = varDataL3.mean('time')
                        # varDataL3.isel(time = 0).plot()
                        # plt.show()

                        meanVal = np.nanmean(varDataL4)
                        maxVal = np.nanmax(varDataL4)
                        minVal = np.nanmin(varDataL4)
                        # log.info(f'[CHECK] timeInfo : all / meanVal : {meanVal} / maxVal : {maxVal} / minVal : {minVal}')
                        # log.info(f'[CHECK] timeInfo : all / meanVal : {meanVal:.2f}')

                        saveFilePattern = '{}/{}'.format(modelInfo['figPath'], modelInfo['figName'])
                        saveImg = saveFilePattern.format(modelType, procInfo, roiName, 'all', 'mean')
                        result = makePlot(varDataL4, saveImg, shpData, opt={'vmin': minVal, 'vmax': maxVal})
                        log.info(f'[CHECK] result : {result}')

                        meanDict = [{
                            'roiName': roiName
                            , 'type': 0
                            , 'procInfo': procInfo
                            , 'meanVal': meanVal
                        }]

                        meanData = pd.concat([meanData, pd.DataFrame.from_dict(meanDict)], ignore_index=True)

                        for analyInfo in sysOpt['analyList']:
                            # if re.search('1981-2020', analyInfo, re.IGNORECASE): continue

                            log.info(f'[CHECK] analyInfo : {analyInfo}')
                            analySrtDate, analyEndDate = analyInfo.split('-')

                            varDataL4 = varDataL3.sel(time=slice(analySrtDate, analyEndDate)).mean('time')

                            saveFilePattern = '{}/{}'.format(modelInfo['figPath'], modelInfo['figName'])
                            saveImg = saveFilePattern.format(modelType, procInfo, roiName, analyInfo, 'mean')
                            result = makePlot(varDataL4, saveImg, shpData, opt={'vmin': minVal, 'vmax': maxVal})
                            log.info(f'[CHECK] result : {result}')

                            # varDataL3.isel(time = 0).plot()
                            # plt.show()

                            meanVal = np.nanmean(varDataL4)
                            # maxVal = np.nanmax(varDataL2)
                            # minVal = np.nanmin(varDataL2)
                            # log.info(f'[CHECK] timeInfo : all / meanVal : {meanVal} / maxVal : {maxVal} / minVal : {minVal}')
                            # log.info(f'[CHECK] timeInfo : all / meanVal : {meanVal:.2f}')

                            meanDict = [{
                                'roiName': roiName
                                , 'type': analyInfo
                                , 'procInfo': procInfo
                                , 'meanVal': meanVal
                            }]

                            meanData = pd.concat([meanData, pd.DataFrame.from_dict(meanDict)], ignore_index=True)

                        # ****************************************************
                        # 기울기
                        # ****************************************************
                        for analyInfo in sysOpt['analyList']:
                            log.info(f'[CHECK] analyInfo : {analyInfo}')
                            analySrtDate, analyEndDate = analyInfo.split('-')

                            # inpFile = '/DATA/OUTPUT/LSH0547/REANALY-ECMWF-1M-GW-IDX_{}-slope-MK{}-{}_*.nc'.format(procInfo, analySrtDate, analyEndDate)
                            inpFile = '/DATA/OUTPUT/LSH0547/REANALY-ECMWF-1M-GW-IDX2_{}-slope-MK{}-{}_*.nc'.format(procInfo, analySrtDate, analyEndDate)
                            fileList = sorted(glob.glob(inpFile), reverse=True)

                            if fileList is None or len(fileList) < 1: continue
                            slopeData = xr.open_dataset(fileList[0], engine='pynio')

                            # shp 영역 내 자료 추출
                            slopeDataL1 = slopeData.rio.write_crs("epsg:4326")
                            slopeDataL2 = slopeDataL1.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
                            slopeDataL3 = slopeDataL2.rio.clip(shpData.geometry, shpData.crs, from_disk=True)

                            # slopeDataL1 = slopeData[f't2m-slope'].sel(lat=latList, lon=lonList)
                            slopeDataL4 = slopeDataL3[f'{procInfo}-slope']

                            # if re.search('1981-2020', analyInfo, re.IGNORECASE):
                            #     vmin = np.nanmin(slopeDataL4)
                            #     vmax = np.nanmax(slopeDataL4)

                            meanVal = np.nanmean(slopeDataL4)

                            key = f'{analyInfo}-all'
                            saveFilePattern = '{}/{}'.format(modelInfo['figPath'], modelInfo['figName'])
                            saveImg = saveFilePattern.format(modelType, procInfo, roiName, key, 'slope')
                            # result = makePlot(slopeDataL4, saveImg, shpData, opt={'vmin': vmin, 'vmax': vmax})
                            result = makePlot(slopeDataL4, saveImg, shpData, None)
                            log.info(f'[CHECK] result : {result}')

                            dict = [{
                                'roiName': roiName
                                , 'analyInfo': analyInfo
                                , 'procInfo': procInfo
                                , 'meanVal': meanVal
                                , 'meanVal10': meanVal * 10
                            }]

                            valData = pd.concat([valData, pd.DataFrame.from_dict(dict)], ignore_index=True)

                # 엑셀 저장
                saveXlsxFile = '{}/{}/{}-{}.xlsx'.format(globalVar['outPath'], serviceName, modelType, 'all')
                os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)
                with pd.ExcelWriter(saveXlsxFile, engine='xlsxwriter', options={'use_zip64': True}) as writer:
                    meanData.to_excel(writer, sheet_name='meanData', index=True)
                    valData.to_excel(writer, sheet_name='valData', index=True)
                log.info(f'[CHECK] saveXlsxFile : {saveXlsxFile}')

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
