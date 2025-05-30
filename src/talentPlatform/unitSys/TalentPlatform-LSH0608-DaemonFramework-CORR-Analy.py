# ================================================
# 요구사항
# ================================================

# cd /data2/hzhenshao/EMI
# /data2/hzhenshao/EMI/py38/bin/python3 TalentPlatform-LSH0608-DaemonFramework-CORR-Analy.py
# nohup /data2/hzhenshao/EMI/py38/bin/python3 TalentPlatform-LSH0608-DaemonFramework-CORR-Analy.py &
# tail -f nohup.out

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
from datetime import datetime
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj
import pymannkendall as mk

# Xarray
import xarray as xr
# Dask stuff
import dask.array as da
from dask.diagnostics import ProgressBar
from xarrayMannKendall import *
# import dask.array as da
import dask
from dask.distributed import Client

from scipy.stats import kendalltau
from plotnine import ggplot, aes, geom_boxplot
import gc
import xarray as xr
import pandas as pd
import numpy as np
import fiona
from shapely.geometry import shape
from rasterio import features
from affine import Affine
import matplotlib.pyplot as plt

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
        , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    return globalVar

#  초기 전달인자 설정
def initArgument(globalVar):
    parser = argparse.ArgumentParser()

    for i, argv in enumerate(sys.argv[1:]):
        if not argv.__contains__('--'): continue
        parser.add_argument(argv)

    inParInfo = vars(parser.parse_args())
    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        if env not in 'local' and key.__contains__('Path'):
            os.makedirs(val, exist_ok=True)
        globalVar[key] = val

    return globalVar

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0608'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info("[START] __init__ : {}".format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

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

            if platform.system() == 'Windows':
                pass
            else:
                pass
                # globalVar['inpPath'] = '/DATA/INPUT'
                # globalVar['outPath'] = '/DATA/OUTPUT'
                # globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': '2000-01-01'
                , 'endDate': '2019-12-31'

                # 경도 최소/최대/간격
                , 'lonMin': -180
                , 'lonMax': 180
                , 'lonInv': 0.1

                # 위도 최소/최대/간격
                , 'latMin': -90
                , 'latMax': 90
                , 'latInv': 0.1

                , 'dateList': {
                    '2000-2019': {
                        'srtDate': '2000-01-01',
                        'endDate': '2019-12-31',
                    },
                    '2000-2009': {
                        'srtDate': '2000-01-01',
                        'endDate': '2009-12-31',
                    },
                    '2010-2019': {
                        'srtDate': '2010-01-01',
                        'endDate': '2019-12-31',
                    },
                }
                # , 'typeList': ['EC', 'GDP', 'landscan', 'Land_Cover_Type_1_Percent']
                , 'typeList': ['landscan', 'Land_Cover_Type_1_Percent']
                , 'keyList': ['N2O', 'GHG', 'CO2', 'CO2bio', 'CH4']
                # , 'shpInfo': '/HDD/DATA/INPUT/LSH0608/COFIG/ne_10m_admin_0_countries_chn/ne_10m_admin_0_countries_chn.shp'
                # , 'xlsxInfo': '/HDD/DATA/INPUT/LSH0608/COFIG/HDR25_Statistical_Annex_HDI_Table.xlsx'
                , 'shpInfo': '/data2/hzhenshao/EMI/LSH0608/COFIG/ne_10m_admin_0_countries_chn/ne_10m_admin_0_countries_chn.shp'
                , 'xlsxInfo': '/data2/hzhenshao/EMI/LSH0608/COFIG/HDR25_Statistical_Annex_HDI_Table.xlsx'
            }

            with fiona.open(sysOpt['shpInfo'], 'r') as src:
                gdf_attributes = pd.DataFrame([f['properties'] for f in src])

            undp_df = pd.read_excel(sysOpt['xlsxInfo'], skiprows=5)
            undp_df = undp_df.rename(columns={'Country': 'CountryName'})
            undp_df['Category'] = undp_df['CountryName']
            undp_df['CountryName'] = undp_df['CountryName'].where(undp_df['HDI rank'].notna())
            undp_df['Category'] = undp_df['Category'].where(undp_df['HDI rank'].isna()).ffill()
            undp_cleaned = undp_df.dropna(subset=['CountryName'])
            undp_cleaned = undp_cleaned[['CountryName', 'Category']]
            category_map = {
                'Very high human development': 'Developed',
                'High human development': 'Developing',
                'Medium human development': 'Developing',
                'Low human development': 'LDC'
            }
            undp_cleaned['DevClass'] = undp_cleaned['Category'].map(category_map)

            merged_df = gdf_attributes.merge(undp_cleaned, left_on='ADMIN', right_on='CountryName', how='left')
            iso3_to_devclass = dict(zip(merged_df['ISO_A3'], merged_df['DevClass']))

            iso3_list = list(iso3_to_devclass.keys())
            iso3_to_int = {iso3: i + 1 for i, iso3 in enumerate(iso3_list)}
            int_to_iso3 = {v: k for k, v in iso3_to_int.items()}

            with fiona.open(sysOpt['shpInfo'], 'r') as src:
                shapes = [
                    (shape(feature['geometry']), iso3_to_int[feature['properties']['ISO_A3']])
                    for feature in src
                    if feature['properties']['ISO_A3'] in iso3_to_int
                ]

            for dateInfo in sysOpt['dateList']:
                for i, typeInfo in enumerate(sysOpt['typeList']):
                    for j, keyInfo in enumerate(sysOpt['keyList']):
                        log.info(f'[CHECK] dateInfo : {dateInfo} / typeInfo : {typeInfo} / keyInfo : {keyInfo}')

                        saveImg = '{}/{}/{}/{}_{}_{}_{}.png'.format(globalVar['outPath'], serviceName, 'CORR-ANALY', dateInfo, 'corr-analy', typeInfo, keyInfo)
                        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                        if len(glob.glob(saveImg)) > 0: continue

                        inpFile = '{}/{}/{}/{}_{}_{}_{}.nc'.format(globalVar['inpPath'], serviceName, 'CORR', dateInfo, 'corr', typeInfo, keyInfo)
                        fileList = sorted(glob.glob(inpFile))

                        if fileList is None or len(fileList) < 1:
                            log.error(f"파일 없음 : {inpFile}")
                            continue

                        fileInfo = fileList[0]
                        ds = xr.open_dataset(fileInfo)
                        corr = ds[f"{typeInfo}_{keyInfo}"].values
                        lat = ds['lat'].values
                        lon = ds['lon'].values

                        res_lat = lat[1] - lat[0]
                        res_lon = lon[1] - lon[0]
                        transform = Affine.translation(lon[0] - res_lon / 2, lat[0] - res_lat / 2) * Affine.scale(res_lon, res_lat)

                        country_id_raster = features.rasterize(
                            shapes=shapes,
                            out_shape=corr.shape,
                            transform=transform,
                            fill=0,
                            dtype='int16'
                        )
                        country_iso3_raster = np.vectorize(int_to_iso3.get)(country_id_raster)

                        corr_flat = corr.flatten()
                        iso3_flat = country_iso3_raster.flatten()
                        devclass_flat = np.array([iso3_to_devclass.get(code, None) for code in iso3_flat])

                        valid_mask = (~np.isnan(corr_flat)) & (devclass_flat != None)
                        corr_valid = corr_flat[valid_mask]
                        devclass_valid = devclass_flat[valid_mask]

                        all_values = corr_valid
                        developed_values = corr_valid[devclass_valid == 'Developed']
                        developing_values = corr_valid[devclass_valid == 'Developing']
                        ldc_values = corr_valid[devclass_valid == 'LDC']

                        plt.figure(figsize=(10, 6))
                        plt.boxplot(
                            [all_values, developed_values, developing_values, ldc_values],
                            labels=['All', 'Developed', 'Developing', 'LDC'],
                            patch_artist=True
                        )
                        plt.title(f"Distribution of {keyInfo}-{keyInfo} Correlation by Development Class")
                        plt.ylabel('Correlation Coefficient')
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                        # plt.show()
                        plt.close()
                        log.info(f'[CHECK] saveImg : {saveImg}')

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
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
