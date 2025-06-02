# ================================================
# 요구사항
# ================================================
# cd /data2/hzhenshao/EMI
# /data2/hzhenshao/EMI/py38/bin/python3 TalentPlatform-LSH0608-DaemonFramework-statLdmi.py
# nohup /data2/hzhenshao/EMI/py38/bin/python3 TalentPlatform-LSH0608-DaemonFramework-statLdmi.py &
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
import statsmodels.api as sm

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

def statOls(ln_eg, ln_pop, ln_gdp, ln_ur, ln_ec):
    valid_indices = ~np.isnan(ln_eg) & ~np.isnan(ln_pop) & ~np.isnan(ln_gdp) & ~np.isnan(ln_ur) & ~np.isnan(ln_ec)
    if np.sum(valid_indices) < 5:
        return np.array([np.nan] * 5)

    y = ln_eg[valid_indices]
    X_df = pd.DataFrame({
        'ln_pop': ln_pop[valid_indices],
        'ln_gdp': ln_gdp[valid_indices],
        'ln_ur': ln_ur[valid_indices],
        'ln_ec': ln_ec[valid_indices]
    })

    # 절편(상수항) 추가
    X = sm.add_constant(X_df)

    try:
        model = sm.OLS(y, X)
        results = model.fit()
        # return np.array([results.params['const'], results.params['ln_pop'], results.params['ln_gdp'], results.params['ln_ur'], results.params['ln_ec']])
        return np.array([np.exp(results.params['const']), results.params['ln_pop'], results.params['ln_gdp'], results.params['ln_ur'], results.params['ln_ec']])
    except Exception as e:
        return np.array([np.nan] * 5)

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

            if (platform.system() == 'Windows'):
                pass
            else:
                # pass
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

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
                    # '2000-2019': {
                    #     'srtDate': '2000-01-01',
                    #     'endDate': '2019-12-31',
                    # },
                    # '2000-2009': {
                    #     'srtDate': '2000-01-01',
                    #     'endDate': '2009-12-31',
                    # },
                    '2010-2019': {
                        'srtDate': '2010-01-01',
                        'endDate': '2019-12-31',
                    },

                }
                , 'typeList': ['EC', 'GDP', 'landscan', 'Land_Cover_Type_1_Percent']
                # , 'typeList': ['EC']
                # , 'typeList': ['landscan']
                # , 'typeList': ['Land_Cover_Type_1_Percent']

                # , 'keyList': ['CH4', 'CO2_excl', 'CO2_org', 'N2O', 'NH3', 'NMVOC', 'OC', 'NH3', 'SO2']
                # , 'keyList': ['emi_co', 'emi_n2o', 'emi_nh3', 'emi_nmvoc', 'emi_nox', 'emi_oc', 'emi_so2']
                # , 'keyList': ['emi_nmvoc']
                # , 'keyList': ['N2O', 'GHG', 'CO2', 'CO2bio', 'CH4']
                , 'keyList': ['SO2', 'N2O', 'CH4', 'NMVOC', 'NOx', 'NH3', 'CO', 'PM10', 'PM2.5', 'OC', 'BC']

            #     /data2/hzhenshao/EMI/LSH0608/STAT
            }


            for dateInfo in sysOpt['dateList']:
                inpFile = '{}/{}/{}.nc'.format(globalVar['inpPath'], serviceName, 'EDGAR2-*')
                fileList = sorted(glob.glob(inpFile))

                if fileList is None or len(fileList) < 1:
                    log.error(f"파일 없음 : {inpFile}")
                    continue

                log.info(f'[CHECK] dateInfo : {dateInfo}')
                srtDate = sysOpt['dateList'][dateInfo]['srtDate']
                endDate = sysOpt['dateList'][dateInfo]['endDate']

                data = xr.open_mfdataset(fileList).sel(time=slice(srtDate, endDate))
                log.info(f'[CHECK] fileList : {fileList}')
                log.info(f'[CHECK] srtDate : {srtDate}')
                log.info(f'[CHECK] endDate : {endDate}')

                # 회귀계수
                keyInfo = 'BC'
                inpFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, f"STAT/{dateInfo}_*{keyInfo}")
                fileList = sorted(glob.glob(inpFile))

                if fileList is None or len(fileList) < 1:
                    log.error(f"파일 없음 : {inpFile}")
                    continue

                coefData = xr.open_mfdataset(fileList)
                coefData = coefData.rename({'__xarray_dataarray_variable__': 'coefVar'})

                # coefData.isel(period=0, isel=)
                # coefData.sel(period=dateInfo, coef='c')['coefVar'].plot()
                # plt.show()

                # np.nanmin(coefData.sel(period=dateInfo, coef='c')['coefVar'])
                # np.nanmax(coefData.sel(period=dateInfo, coef='c')['coefVar'])
                # np.nanmean(coefData.sel(period=dateInfo, coef='c')['coefVar'])


                # b_coeff_data = ds[data_var_name].sel(period='2010-2019', coef='b')
                coefData['coef']

                # <xarray.DataArray (period: 1, lat: 2, lon: 2, coef_alias: 4)>

                # **********************************************************************************************************
                # 화소별 회귀계수 계산
                # **********************************************************************************************************
                for keyInfo in sysOpt['keyList']:
                    log.info(f'[CHECK] keyInfo : {keyInfo}')

                    # saveFile = '{}/{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, 'STAT', dateInfo, 'statOls', keyInfo)
                    # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    # if len(glob.glob(saveFile)) > 0: continue

                    # coreNum = int(os.cpu_count() * 0.50)
                    # log.info(f'[CHECK] coreNum : {coreNum}')
                    # client = Client(n_workers=coreNum)
                    # dask.config.set(scheduler='processes')
                    #
                    # statOlsRes = xr.apply_ufunc(
                    #     statOls,
                    #     data[keyInfo],
                    #     data['landscan'],
                    #     data['GDP'],
                    #     data['Land_Cover_Type_1_Percent'],
                    #     data['EC'],
                    #     input_core_dims=[['time'], ['time'], ['time'], ['time'], ['time']],
                    #     output_core_dims=[['coef']],
                    #     exclude_dims=set(('time',)),
                    #     output_sizes={'coef': 5},
                    #     # vectorize=False,
                    #     vectorize=True,
                    #     dask="parallelized",
                    #     output_dtypes=[float],
                    # )
                    #
                    # # 계수 이름 지정
                    # statOlsRes = statOlsRes.assign_coords(coef=['a', 'b', 'c', 'd', 'e'])
                    # statOlsRes = statOlsRes.expand_dims(period=[dateInfo])
                    #
                    # # alpha 값 계산
                    # # statOlsRes['alpha'] = np.exp(statOlsRes.sel(coef='lnAlpha'))
                    # # statOlsRes['alpha'] = np.exp(statOlsRes.sel(coef='lnAlpha', drop=True))
                    # log.info(f'[CHECK] statOlsRes : {statOlsRes}')
                    #
                    # # 파일 저장
                    # statOlsRes.to_netcdf(saveFile)
                    # log.info(f'[CHECK] saveFile : {saveFile}')

                    # client.close()


            # --- 개념적인 위경도 기반 데이터 및 공간 탄력성 준비 ---
            # 실제로는 NetCDF 파일 등에서 데이터를 불러옵니다.
            years = [2010, 2020]
            # lats = np.array([35.0, 35.1])  # 예시 위도
            # lons = np.array([127.0, 127.1])  # 예시 경도
            lats = np.array([35.0, 35.1])  # 예시 위도
            lons = np.array([127.0, 127.1])  # 예시 경도
            factors_lmdi = ['POP', 'GDP', 'UR', 'EC']
            # STIRPAT 모델의 계수 'b', 'c', 'd', 'e' 가 각각 POP, GDP, UR, EC의 탄력성이라고 가정
            elasticity_coef_names = ['b', 'c', 'd', 'e']  # 탄력성 계수 이름 (Dataset 내 coef 차원과 일치)

            # 매핑: 분석 요인명 -> 탄력성 계수 이름
            factor_to_elasticity_coef_map = dict(zip(factors_lmdi, elasticity_coef_names))

            example_coords_base = {'year': years, 'lat': lats, 'lon': lons}
            example_coords_elasticity = {'period': ['2010-2019'], 'lat': lats, 'lon': lons, 'coef_alias': elasticity_coef_names}

            # 기본 데이터 (POP, GDP, EG 등)
            ds_spatial = xr.Dataset(
                {
                    'POP': (('year', 'lat', 'lon'), np.random.rand(len(years), len(lats), len(lons)) * 1 + 1),
                    'GDP': (('year', 'lat', 'lon'), np.random.rand(len(years), len(lats), len(lons)) * 10 + 5),
                    'UR': (('year', 'lat', 'lon'), np.random.rand(len(years), len(lats), len(lons)) * 20 + 40),
                    'EC': (('year', 'lat', 'lon'), np.random.rand(len(years), len(lats), len(lons)) * 0.5 + 0.3),
                    'EG': (('year', 'lat', 'lon'), np.random.rand(len(years), len(lats), len(lons)) * 5 + 5)
                },
                coords=example_coords_base
            )

            # 공간적으로 변화하는 탄력성 데이터 (예시)
            # 실제로는 이 데이터가 ds_spatial에 이미 포함되어 있거나 별도로 로드될 수 있습니다.
            # 여기서는 예시로 생성합니다.
            elasticity_data_values = np.random.rand(1, len(lats), len(lons), len(elasticity_coef_names)) * 0.5 + 0.1
            # 'EC' 탄력성은 음수일 수 있으므로, 예시에서는 e 계수(마지막 인덱스)를 음수로 만듭니다.
            elasticity_data_values[:, :, :, elasticity_coef_names.index('e')] = - (np.random.rand(1, len(lats), len(lons)) * 0.5 + 0.1)

            # 탄력성 데이터를 별도의 DataArray 또는 Dataset 변수로 준비했다고 가정
            # 여기서는 ds_spatial에 'spatial_elasticities'라는 이름으로 추가합니다.
            ds_spatial['spatial_elasticities'] = xr.DataArray(
                elasticity_data_values,
                coords=example_coords_elasticity,
                dims=('period', 'lat', 'lon', 'coef_alias')
            )

            # 분석 시작/종료 시점 데이터 선택
            data_t0_spatial = ds_spatial.sel(year=2010)
            data_tT_spatial = ds_spatial.sel(year=2020)

            # 분석 기간 선택 (탄력성 데이터용)
            current_period = '2010-2019'  # 사용자의 Dataset 구조에 맞게 조정

            # --- 단계 1: 대수평균 L(EG) 계산 (격자 셀별) ---
            eg_t0_spatial_da = data_t0_spatial['EG']
            eg_tT_spatial_da = data_tT_spatial['EG']
            l_eg_spatial = xr.where(
                eg_tT_spatial_da == eg_t0_spatial_da,
                eg_t0_spatial_da,
                (eg_tT_spatial_da - eg_t0_spatial_da) / (np.log(eg_tT_spatial_da) - np.log(eg_t0_spatial_da))
            )
            print(f"## 공간 데이터 기반 STIRPAT LMDI 분해 분석 (공간적 탄력성 적용)\n")
            print(f"단계 1: 대수평균 L(EG) 계산 완료.\n L(EG) 예시 (첫번째 셀): {l_eg_spatial.isel(lat=0, lon=0).item():.2f}\n")

            # --- 단계 2: 초기 요인별 기여도 계산 (격자 셀별, 공간적 탄력성 사용) ---
            delta_E_factors_spatial_calculated = {}
            print("단계 2: 초기 요인별 기여도 계산 (격자 셀별, 공간적 탄력성 사용)")

            for factor_name in factors_lmdi:
                factor_t0_da = data_t0_spatial[factor_name]
                factor_tT_da = data_tT_spatial[factor_name]

                # 해당 요인의 공간적 탄력성 계수 DataArray 추출
                # factor_to_elasticity_coef_map에서 실제 'coef_alias' 이름을 가져옴
                coef_alias_for_factor = factor_to_elasticity_coef_map[factor_name]

                # 공간적 탄력성 값 (DataArray[lat, lon])
                # .squeeze()는 period 차원(크기가 1인 경우)을 제거하여 (lat, lon, coef_alias) 로 만듭니다.
                # 실제 데이터 구조에 따라 squeeze() 사용 여부나 sel 조건이 달라질 수 있습니다.
                elasticity_val_spatial = ds_spatial['spatial_elasticities'].sel(
                    period=current_period,
                    coef_alias=coef_alias_for_factor
                )
                #.squeeze(dim='period', drop=True))  # period 차원이 있으면 제거, 없으면 에러 방지 위해 drop=True

                log_ratio_factor_da = xr.where(
                    factor_tT_da == factor_t0_da,
                    0.0,
                    xr.where(
                        (factor_t0_da > 0) & (factor_tT_da > 0),
                        np.log(factor_tT_da / factor_t0_da),
                        np.nan
                    )
                )

                # 초기 기여도 계산 (요소별)
                # 이제 elasticity_val_spatial도 (lat, lon) 차원의 DataArray임
                calculated_delta_E = elasticity_val_spatial * l_eg_spatial * log_ratio_factor_da
                delta_E_factors_spatial_calculated[factor_name] = calculated_delta_E
                print(f"  ΔEG_{factor_name} (첫번째 셀 값 예시): {calculated_delta_E.isel(lat=0, lon=0).item():.2f}")
                print(f"    사용된 탄력성 (첫번째 셀 값 예시): {elasticity_val_spatial.isel(lat=0, lon=0).item():.2f}")

            print("")
            # 이후 단계 3, 4, 5 (분해 검증, 실질 기여도, 백분율 기여도)는
            # delta_E_factors_spatial_calculated 딕셔너리에 저장된 DataArray들을 사용하여
            # 이전 코드와 동일한 로직으로 요소별 연산을 수행하면 됩니다.
            # 예를 들어, sum_delta_E_factors_da는 이제 delta_E_factors_spatial_calculated의 DataArray들을 합산합니다.

            # --- 단계 3 예시 (격자 셀별 합산) ---
            sum_delta_E_factors_da_spatial = xr.DataArray(
                np.zeros_like(l_eg_spatial.values),  # 초기화할 배열 (l_eg_spatial과 동일한 형태)
                coords=l_eg_spatial.coords,
                dims=l_eg_spatial.dims
            )
            for factor_name in factors_lmdi:
                # NaN 값을 0으로 처리하여 합산 (필요에 따라 다른 처리 가능)
                sum_delta_E_factors_da_spatial += delta_E_factors_spatial_calculated[factor_name].fillna(0)

            actual_delta_EG_da_spatial = eg_tT_spatial_da - eg_t0_spatial_da

            print(f"단계 3: 분해 검증 (격자 셀별)")
            print(f"  초기 기여도 합계 (첫번째 셀 값 예시) = {sum_delta_E_factors_da_spatial.isel(lat=0, lon=0).item():.2f}")
            print(f"  실제 EG 변화량 (첫번째 셀 값 예시) = {actual_delta_EG_da_spatial.isel(lat=0, lon=0).item():.2f}\n")

            # (단계 4, 5는 이전과 유사하게 delta_Real_EG_factors_spatial, percentage_EG_factors_spatial 딕셔너리에
            #  공간 DataArray 결과를 저장하도록 수정 가능)

            # for dateInfo in sysOpt['dateList']:
            #     inpFile = '{}/{}/{}.nc'.format(globalVar['inpPath'], serviceName, 'EDGAR2-*')
            #     fileList = sorted(glob.glob(inpFile))
            #
            #     if fileList is None or len(fileList) < 1:
            #         log.error(f"파일 없음 : {inpFile}")
            #         continue
            #
            #     log.info(f'[CHECK] dateInfo : {dateInfo}')
            #     srtDate = sysOpt['dateList'][dateInfo]['srtDate']
            #     endDate = sysOpt['dateList'][dateInfo]['endDate']
            #
            #     data = xr.open_mfdataset(fileList).sel(time=slice(srtDate, endDate))
            #     log.info(f'[CHECK] fileList : {fileList}')
            #     log.info(f'[CHECK] srtDate : {srtDate}')
            #     log.info(f'[CHECK] endDate : {endDate}')
            #
            #     # **********************************************************************************************************
            #     # 화소별 회귀계수 계산
            #     # **********************************************************************************************************
            #     for keyInfo in sysOpt['keyList']:
            #
            #         saveFile = '{}/{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, 'STAT', dateInfo, 'statOls', keyInfo)
            #         os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #         if len(glob.glob(saveFile)) > 0: continue
            #
            #         coreNum = int(os.cpu_count() * 0.50)
            #         log.info(f'[CHECK] coreNum : {coreNum}')
            #         client = Client(n_workers=coreNum)
            #         dask.config.set(scheduler='processes')
            #
            #         statOlsRes = xr.apply_ufunc(
            #             statOls,
            #             data[keyInfo],
            #             data['landscan'],
            #             data['GDP'],
            #             data['Land_Cover_Type_1_Percent'],
            #             data['EC'],
            #             input_core_dims=[['time'], ['time'], ['time'], ['time'], ['time']],
            #             output_core_dims=[['coef']],
            #             exclude_dims=set(('time',)),
            #             output_sizes={'coef': 5},
            #             # vectorize=False,
            #             vectorize=True,
            #             dask="parallelized",
            #             output_dtypes=[float],
            #         )
            #
            #         # 계수 이름 지정
            #         statOlsRes = statOlsRes.assign_coords(coef=['a', 'b', 'c', 'd', 'e'])
            #         statOlsRes = statOlsRes.expand_dims(period=[dateInfo])
            #
            #         # alpha 값 계산
            #         # statOlsRes['alpha'] = np.exp(statOlsRes.sel(coef='lnAlpha'))
            #         # statOlsRes['alpha'] = np.exp(statOlsRes.sel(coef='lnAlpha', drop=True))
            #         log.info(f'[CHECK] statOlsRes : {statOlsRes}')
            #
            #         # 파일 저장
            #         statOlsRes.to_netcdf(saveFile)
            #         log.info(f'[CHECK] saveFile : {saveFile}')
            #
            #         # client.close()

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
