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
import faulthandler
faulthandler.enable()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy.ext.declarative import declarative_base
from xclim import sdba
from xclim.sdba.adjustment import QuantileDeltaMapping
import pandas as pd
import xarray as xr
from multiprocessing import Pool
import multiprocessing as mp
from dask.distributed import Client, LocalCluster

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

Base = declarative_base()

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

def add_leap_day_mrg(modDataL3, var_name='pr'):
    start_year = 1980
    end_year = 1982

    # 윤년 확인
    leap_years = [year for year in range(start_year, end_year + 1) if
                  (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)]

    # MrgData에 있는 윤년확인
    existing_leap_days_years = set(
        modDataL3.time.dt.year.where((modDataL3.time.dt.month == 2) & (modDataL3.time.dt.day == 29), drop=True).values)

    # mrgData에서 윤년이 누락된 경우만 처리
    missing_leap_years = [year for year in leap_years if year not in existing_leap_days_years]

    if not missing_leap_years:
        # 모든 윤년이 존재하면 mrgData 반환
        return modDataL3

    # 윤년이 누락된 경우, 윤년 추가
    # 전체 기간에 대한 날짜 추가
    new_times = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 기본 day_range 설정
            day_range = 30 if month in [4, 6, 9, 11] else 31

            # 2월의 경우 윤년 확인
            if month == 2:
                # 윤년에서 누락된 해에만 2월 29일을 추가
                day_range = 29 if year in missing_leap_years else 28

            for day in range(1, day_range + 1):
                new_times.append(f'{year}-{month:02d}-{day:02d}')

    # 새로운 시간 인덱스를 기반으로 데이터셋을 재색인합니다.
    new_times_sorted = sorted(pd.to_datetime(new_times))
    full_time_ds = modDataL3.reindex(time=new_times_sorted, method='nearest', tolerance='1D')

    # 누락된 윤년의 2월 29일에 대해 데이터 보간을 수행합니다.
    for year in missing_leap_years:
        feb28 = f'{year}-02-28'
        mar01 = f'{year}-03-01'
        feb29 = f'{year}-02-29'
        if np.datetime64(feb28) in full_time_ds.time.values and np.datetime64(mar01) in full_time_ds.time.values:
            feb28_data = full_time_ds.sel(time=feb28)[var_name]
            mar01_data = full_time_ds.sel(time=mar01)[var_name]
            feb29_vals = (feb28_data + mar01_data) / 2
            full_time_ds[var_name].loc[{'time': np.datetime64(feb29)}] = feb29_vals

    return full_time_ds.sortby('time')


def remove_leap_days_xarray(mod, obs_start_year=1980, obs_end_year=2014, sim_start_year=2015, sim_end_year=2100):
    def is_leap_year(year):
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    obs_leap_years = [year for year in range(obs_start_year, obs_end_year + 1) if is_leap_year(year)]
    sim_leap_years = [year for year in range(sim_start_year, sim_end_year + 1) if is_leap_year(year)]
    total_leap_years = obs_leap_years + sim_leap_years

    leap_days_to_remove = pd.to_datetime([f'{year}-02-29' for year in total_leap_years])

    # xarray Dataset을 대상으로 윤년의 2월 29일을 제거합니다.
    mod_filtered = mod.sel(time=~mod.time.dt.date.isin(leap_days_to_remove.date))

    return mod_filtered

def makeSbckProc(method, contDataL4, mrgDataProcessed, simDataL3Processed, keyInfo):

    log.info('[START] {}'.format("makeSbckProc"))

    result = None

    try:
        procInfo = mp.current_process()
        log.info(f'[CHECK] method : {method} / pid : {procInfo.pid}')

        # dayofyear 추가
        # mrgDataProcessed['dayofyear'] = mrgDataProcessed['time'].dt.dayofyear

        # ***********************************************************************************
        # 학습 데이터 (ref 실측, hist 관측/학습) : 일 단위로 가중치 보정
        # ***********************************************************************************
        # methodList = {
        #     # Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
        #     'QDM': lambda: sdba.QuantileDeltaMapping.train(ref=mrgDataProcessed['rain'], hist=mrgDataProcessed['pr'], nquantiles=1000, group='time.dayofyear')
        #
        #     # Dequé, M. (2007). Frequency of precipitation and temperature extremes over France in an anthropogenic scenario: Model results and statistical correction according to observed values. Global and Planetary Change, 57(1–2), 16–26. https://doi.org/10.1016/j.gloplacha.2006.11.030
        #     , 'EQM': lambda: sdba.EmpiricalQuantileMapping.train(ref=mrgDataProcessed['rain'], hist=mrgDataProcessed['pr'], nquantiles=1000, group='time.dayofyear')
        #
        #     # Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
        #     , 'DQM': lambda: sdba.DetrendedQuantileMapping.train(ref=mrgDataProcessed['rain'], hist=mrgDataProcessed['pr'], nquantiles=1000, group='time.dayofyear')
        #     , 'Scaling': lambda: sdba.Scaling.train(ref=mrgDataProcessed['rain'], hist=mrgDataProcessed['pr'], group='time.dayofyear')
        # }

        # 2024.04.11 nquantiles 설정
        methodList = {
            # Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
            'QDM': lambda: sdba.QuantileDeltaMapping.train(ref=mrgDataProcessed['rain'], hist=mrgDataProcessed['pr'], nquantiles=20, group='time.dayofyear')

            # Dequé, M. (2007). Frequency of precipitation and temperature extremes over France in an anthropogenic scenario: Model results and statistical correction according to observed values. Global and Planetary Change, 57(1–2), 16–26. https://doi.org/10.1016/j.gloplacha.2006.11.030
            , 'EQM': lambda: sdba.EmpiricalQuantileMapping.train(ref=mrgDataProcessed['rain'], hist=mrgDataProcessed['pr'], nquantiles=20, group='time.dayofyear')

            # Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
            , 'DQM': lambda: sdba.DetrendedQuantileMapping.train(ref=mrgDataProcessed['rain'], hist=mrgDataProcessed['pr'], nquantiles=20, group='time.dayofyear')

            , 'Scaling': lambda: sdba.Scaling.train(ref=mrgDataProcessed['rain'], hist=mrgDataProcessed['pr'], group='time.dayofyear')
        }

        if method not in methodList:
            log.error("주어진 학습 모형 (QDM, EQM, DQM, Scaling)을 선택해주세요.")
            return result

        prd = methodList[method]()

        # ***********************************************************************************
        # 보정 결과
        # ***********************************************************************************
        corData = prd.adjust(sim=mrgDataProcessed['pr'], interp="linear")
        # corData = prd.adjust(sim=mrgDataProcessed['pr'].drop_vars(['lat', 'lon']), interp="linear")
        corDataL1 = xr.merge([corData, contDataL4])

        # 음수의 경우 0으로 대체
        corDataL1['scen'] = xr.where((corDataL1['scen'] < 0), 0.0, corDataL1['scen'])

        # ***********************************************************************************
        # 격자에 따른 검증지표
        # ***********************************************************************************
        lonList = corDataL1['lon'].values
        latList = corDataL1['lat'].values

        valData = pd.DataFrame()
        for lon in lonList:
            for lat in latList:
                # log.info(f'[CHECK] lon : {lon} / lat : {lat}')

                yList = mrgDataProcessed['rain'].sel({'lon': lon, 'lat': lat}).values.flatten()
                yhatList = corDataL1['scen'].sel({'lon': lon, 'lat': lat}).values.flatten()

                mask = (yList > 0) & ~np.isnan(yList) & (yhatList > 0) & ~np.isnan(yhatList)

                y = yList[mask]
                yhat = yhatList[mask]

                if (len(y) == 0) or (len(yhat) == 0): continue

                # 검증 지표 계산
                dict = {
                    'lon': [lon]
                    , 'lat': [lat]
                    , 'cnt': [len(y)]
                    , 'bias': [np.nanmean(yhat - y)]
                    , 'rmse': [np.sqrt(np.nanmean((yhat - y) ** 2))]
                    , 'corr': [np.corrcoef(yhat, y)[0, 1]]
                }

                valData = pd.concat([valData, pd.DataFrame.from_dict(dict)], ignore_index=True)

        saveFile = '{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], keyInfo, 'PAST-VALID-ByGeoCont', method, keyInfo)
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        valData.to_csv(saveFile, index=False)
        log.info(f'[CHECK] saveFile : {saveFile}')

        # ***********************************************************************************
        # 대륙 및 격자에 따른 검증지표
        # ***********************************************************************************
        # lonList = corDataL1['lon'].values
        # latList = corDataL1['lat'].values
        #
        # valData = pd.DataFrame()
        # for lon in lonList:
        #     for lat in latList:
        #         # log.info(f'[CHECK] lon : {lon} / lat : {lat}')
        #
        #         yList = mrgData['rain'].sel({'lon': lon, 'lat': lat}).values.flatten()
        #         yhatList = corDataL1['scen'].sel({'lon': lon, 'lat': lat}).values.flatten()
        #
        #         # isLand = contDataL4['isLand'].sel({'lon': lon, 'lat': lat}).values.item()
        #         contIdx = contDataL4['contIdx'].sel({'lon': lon, 'lat': lat}).values.item()
        #
        #         mask = (yList > 0) & ~np.isnan(yList) & (yhatList > 0) & ~np.isnan(yhatList)
        #
        #         y = yList[mask]
        #         yhat = yhatList[mask]
        #
        #         if (len(y) == 0) or (len(yhat) == 0): continue
        #
        #         # 검증 지표 계산
        #         dict = {
        #             'lon': [lon]
        #             , 'lat': [lat]
        #             # , 'isLand': [isLand]
        #             , 'contIdx': [contIdx]
        #             , 'cnt': [len(y)]
        #             , 'bias': [np.nanmean(yhat - y)]
        #             , 'rmse': [np.sqrt(np.nanmean((yhat - y) ** 2))]
        #             , 'corr': [np.corrcoef(yhat, y)[0, 1]]
        #         }
        #
        #         valData = pd.concat([valData, pd.DataFrame.from_dict(dict)], ignore_index=True)
        #
        # saveFile = '{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, 'PAST-VALID-ByGeoLandCont', method, keyInfo)
        # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        # valData.to_csv(saveFile, index=False)
        # log.info(f'[CHECK] saveFile : {saveFile}')


        # ***********************************************************************************
        # 과거 기간 보정 NetCDF 저장
        # ***********************************************************************************
        # lat1D = corDataL1['lat'].values
        # lon1D = corDataL1['lon'].values
        # time1D = corDataL1['time'].values
        #
        # corDataL2 = xr.Dataset(
        #     {
        #         'OBS': (('time', 'lat', 'lon'), (mrgDataProcessed['rain'].values).reshape(len(time1D), len(lat1D), len(lon1D)))
        #         , 'ERA': (('time', 'lat', 'lon'), (mrgDataProcessed['pr'].values).reshape(len(time1D), len(lat1D), len(lon1D)))
        #         , method: (('time', 'lat', 'lon'), (corDataL1['scen'].transpose('time', 'lat', 'lon').values).reshape(len(time1D), len(lat1D), len(lon1D)))
        #         # , 'isLand': (('time', 'lat', 'lon'), np.tile(contDataL4['isLand'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(len(time1D), len(lat1D), len(lon1D)))
        #         # , 'isLand': (('time', 'lat', 'lon'), np.tile(contDataL4['isLand'].values[np.newaxis, :, :], (1, 1, 1)).reshape(1, len(lat1D), len(lon1D)))
        #         , 'contIdx': (('time', 'lat', 'lon'), np.tile(contDataL4['contIdx'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(len(time1D), len(lat1D), len(lon1D)))
        #         # , 'contIdx': (('lat', 'lon'), (contDataL4['contIdx'].values).reshape(1, len(lat1D), len(lon1D)))
        #         # , 'contIdx': (('lat', 'lon'), contDataL4['contIdx'].values)
        #     }
        #     , coords={
        #         'time': time1D
        #         , 'lat': lat1D
        #         , 'lon': lon1D
        #     }
        # )

        mrgDataProcL1 =  mrgDataProcessed.rename({'rain': 'OBS', 'pr': 'ERA'}).drop_vars (['contIdx'])
        corDataL2 = xr.merge([corDataL1, mrgDataProcL1])

        # corDataL2['OBS'].isel(time = 10).plot()
        # corDataL2[method].isel(time = 10).plot()
        # corDataL2['contIdx'].isel(time=10).plot()
        # plt.show()

        saveFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], keyInfo, 'PAST-MBC', method, keyInfo)
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        corDataL2.to_netcdf(saveFile)
        log.info(f'[CHECK] saveFile : {saveFile}')

        # ***********************************************************************************
        # 과거 기간 보정 NetCDF에서 세부 저장
        # ***********************************************************************************
        varList = ['OBS', 'ERA', method]
        contIdxList = np.unique(corDataL2['contIdx'])
        corDataL3 = corDataL2.to_dataframe().reset_index(drop=False)
        # corDataL3 = corDataL2.chunk({'time': 100, 'lat': 50, 'lon': 50}).to_dataframe().reset_index(drop=False)
        for varInfo in varList:
            selCol = ['time', 'lon', 'lat', varInfo]
            # corDataL4 = corDataL3[selCol].pivot(index=['time'], columns=['lon', 'lat'])
            # corDataL4 = corDataL3[selCol].pivot(index=['lon', 'lat'], columns=['time'])
            corDataL4 = corDataL3[selCol].dropna().pivot(index=['lon', 'lat'], columns=['time'])
            # corDataL4 = corDataL3[selCol].dropna().pivot(index=['time'], columns=['lon', 'lat'])

            # 엑셀 저장
            #saveXlsxFile = '{}/{}/{}-{}_{}_{}.xlsx'.format(globalVar['outPath'], keyInfo, 'PAST-MBC', varInfo, method, keyInfo)
            #os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)

            #with pd.ExcelWriter(saveXlsxFile, engine='xlsxwriter', options={'use_zip64': True}) as writer:
                #corDataL4.to_excel(writer, index=True)
                #log.info(f'[CHECK] saveXlsxFile : {saveXlsxFile}')

            #saveCsvFile = '{}/{}/{}-{}_{}_{}.csv'.format(globalVar['outPath'], keyInfo, 'PAST-MBC', varInfo, method, keyInfo)
            #os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)

            #csvData = corDataL4.reset_index(drop=False)
            #csvData.columns = [col[0] if pd.isna(col[1]) else col[1] for col in csvData.columns]
            #csvData.to_csv(saveCsvFile, index=False)
            #log.info(f'[CHECK] saveCsvFile : {saveCsvFile}')

            # 대륙별로 엑셀 저장
            for contIdx in contIdxList:
                if pd.isna(contIdx): continue

                corDataL5 = corDataL3.loc[corDataL3['contIdx'] == contIdx].reset_index(drop=True)
                if len(corDataL5) < 0: continue

                sheetName = str(int(contIdx))

                # corDataL6 = corDataL5[selCol].pivot(index=['lon', 'lat'], columns=['time'])
                # corDataL6 = corDataL5[selCol].dropna().pivot(index=['lon', 'lat'], columns=['time'])
                corDataL6 = corDataL5[selCol].dropna().pivot(index=['time'], columns=['lon', 'lat'])

                # 엑셀 저장
                #saveXlsxFile = '{}/{}/{}-{}_{}-{}_{}.xlsx'.format(globalVar['outPath'], keyInfo, 'PAST-MBC', varInfo, method, sheetName, keyInfo)
                #os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)

                # corDataL6.to_excel(writer, sheet_name=str(int(contIdx)), index=True)
                # corDataL6.to_excel(writer, sheet_name=sheetName, index=True)
                #with pd.ExcelWriter(saveXlsxFile, engine='xlsxwriter', options={'use_zip64': True}) as writer2:
                    #corDataL6.to_excel(writer2, sheet_name=sheetName, index=True)
                    #log.info(f'[CHECK] saveXlsxFile : {saveXlsxFile}')

                saveCsvFile = '{}/{}/{}-{}_{}-{}_{}.csv'.format(globalVar['outPath'], keyInfo, 'PAST-MBC', varInfo, method, sheetName, keyInfo)
                os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)

                csvData = corDataL6.reset_index(drop=False)
                # csvData.columns = [col[0] if pd.isna(col[1]) else col[1] for col in csvData.columns]
                csvData.to_csv(saveCsvFile, index=False)
                log.info(f'[CHECK] saveCsvFile : {saveCsvFile}')


            # corDataL4.to_excel(saveXlsxFile, index=True)
            #
            # for contIdx in contIdxList:
            #     if np.isnan(contIdx): continue
            #
            #     corDataL5 = corDataL3.loc[(corDataL3['contIdx'] == contIdx)].reset_index(drop=True)
            #     if len(corDataL5) < 0: continue
            #
            #     corDataL6 = corDataL5[selCol].pivot(index=['lon', 'lat'], columns=['time'])
            #     with pd.ExcelWriter(saveXlsxFile, engine='openpyxl', mode='a') as writer:
            #         corDataL6.to_excel(writer, sheet_name=str(int(contIdx)), index=True)

        # ***********************************************************************************
        # 과거 기간 95% 이상 분위수 계산
        # ***********************************************************************************
        # 95% 이상 분위수 계산
        corDataL3 = corDataL2.quantile(0.95, dim='time')

        # NetCDF 자료 저장
        saveFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], keyInfo, 'PAST-RES95', method, keyInfo)
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        corDataL3.to_netcdf(saveFile)
        log.info(f'[CHECK] saveFile : {saveFile}')

        # CSV 자료 저장
        saveFile = '{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], keyInfo, 'PAST-RES95', method, keyInfo)
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        corDataL3.to_dataframe().reset_index(drop=False).to_csv(saveFile, index=False)
        log.info(f'[CHECK] saveFile : {saveFile}')

        # ***********************************************************************************
        # 시뮬레이션 예측 (sim 관측)
        # ***********************************************************************************
        prdData = prd.adjust(sim=simDataL3Processed['pr'], interp="linear")
        prdDataL1 = xr.merge([prdData, contDataL4])

        # 음수의 경우 0으로 대체
        prdDataL1['scen'] = xr.where((prdDataL1['scen'] < 0), 0.0, prdDataL1['scen'])

        # obsDataL2['rain'].isel(time=10).plot(x='lon', y='lat', vmin=0, vmax=100, cmap='viridis')
        # qdmDataL1['contIdx'].plot(x='lon', y='lat', cmap='viridis')

        # prdData.isel(time=10).plot(x='lon', y='lat', vmin=0, vmax=100, cmap='viridis')
        # plt.show()

        # prdDataL1['pr'].isel(time=10).plot(x='lon', y='lat', vmin=0, vmax=100, cmap='viridis')
        # plt.show()

        # ***********************************************************************************
        # 미래 기간 예측 NetCDF 저장
        # ***********************************************************************************
        # lat1D = simDataL3Processed['lat'].values
        # lon1D = simDataL3Processed['lon'].values
        # time1D = simDataL3Processed['time'].values

        # mrgDataL1 = xr.Dataset(
        #     {
        #         'SIM': (('time', 'lat', 'lon'), (simDataL3Processed['pr'].values).reshape(len(time1D), len(lat1D), len(lon1D)))
        #         , method: (('time', 'lat', 'lon'), (prdDataL1['scen'].transpose('time', 'lat', 'lon').values).reshape(len(time1D), len(lat1D), len(lon1D)))
        #         # , 'isLand': (('time', 'lat', 'lon'), np.tile(contDataL4['isLand'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(len(time1D), len(lat1D), len(lon1D)))
        #         , 'contIdx': (('time', 'lat', 'lon'), np.tile(contDataL4['contIdx'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(len(time1D), len(lat1D), len(lon1D)))
        #     }
        #     , coords={
        #         'time': time1D
        #         , 'lat': lat1D
        #         , 'lon': lon1D
        #     }
        # )

        mrgDataProcL1 =  simDataL3Processed.rename({'pr': 'SIM'}).drop_vars (['contIdx'])
        mrgDataL1 = xr.merge([prdDataL1, mrgDataProcL1])

        # mrgDataL1['SIM'].isel(time = 10).plot()
        # mrgDataL1[method].isel(time = 10).plot()
        # plt.show()

        saveFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC', method, keyInfo)
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        mrgDataL1.to_netcdf(saveFile)
        log.info(f'[CHECK] saveFile : {saveFile}')

        # ***********************************************************************************
        # 미래 기간 보정 NetCDF에서 세부 저장
        # ***********************************************************************************
        varList = ['SIM', method]
        contIdxList = np.unique(mrgDataL1['contIdx'])
        mrgDataL2 = mrgDataL1.to_dataframe().reset_index(drop=False)
        for varInfo in varList:
            selCol = ['time', 'lon', 'lat', varInfo]
            # mrgDataL3 = mrgDataL2[selCol].pivot(index=['time'], columns=['lon', 'lat'])
            # mrgDataL3 = mrgDataL2[selCol].pivot(index=['lon', 'lat'], columns=['time'])
            mrgDataL3 = mrgDataL2[selCol].dropna().pivot(index=['lon', 'lat'], columns=['time'])

            # 엑셀 저장
            #saveXlsxFile = '{}/{}/{}-{}_{}_{}.xlsx'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC', varInfo, method, keyInfo)
            #os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)

            #with pd.ExcelWriter(saveXlsxFile, engine='xlsxwriter', options={'use_zip64': True}) as writer:
                #mrgDataL3.to_excel(writer, index=True)
                #log.info(f'[CHECK] saveXlsxFile : {saveXlsxFile}')

            #saveCsvFile = '{}/{}/{}-{}_{}_{}.csv'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC', varInfo, method, keyInfo)
            #os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)

            #csvData = mrgDataL3.reset_index(drop=False)
            #csvData.columns = [col[0] if pd.isna(col[1]) else col[1] for col in csvData.columns]
            #csvData.to_csv(saveCsvFile, index=False)
            #log.info(f'[CHECK] saveCsvFile : {saveCsvFile}')

            # 대륙별로 엑셀 저장
            for contIdx in contIdxList:
                if np.isnan(contIdx): continue

                mrgDataL5 = mrgDataL2.loc[mrgDataL2['contIdx'] == contIdx].reset_index(drop=True)
                if len(mrgDataL5) < 0: continue

                sheetName = str(int(contIdx))

                # mrgDataL6 = mrgDataL5[selCol].pivot(index=['lon', 'lat'], columns=['time'])
                # mrgDataL6 = mrgDataL5[selCol].dropna().pivot(index=['lon', 'lat'], columns=['time'])
                mrgDataL6 = mrgDataL5[selCol].dropna().pivot(index=['time'], columns=['lon', 'lat'])

                # 엑셀 저장
                # mrgDataL6.to_excel(writer, sheet_name=sheetName, index=True)
                #saveXlsxFile = '{}/{}/{}-{}_{}-{}_{}.xlsx'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC', varInfo, method, sheetName, keyInfo)
                #os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)

                #with pd.ExcelWriter(saveXlsxFile, engine='xlsxwriter', options={'use_zip64': True}) as writer2:
                    #mrgDataL6.to_excel(writer2, sheet_name=sheetName, index=True)
                    #log.info(f'[CHECK] saveXlsxFile : {saveXlsxFile}')

                saveCsvFile = '{}/{}/{}-{}_{}-{}_{}.csv'.format(globalVar['outPath'], keyInfo, 'FUTURE-MBC', varInfo, method, sheetName, keyInfo)
                os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)

                csvData = mrgDataL6.reset_index(drop=False)
                # csvData.columns = [col[0] if pd.isna(col[1]) else col[1] for col in csvData.columns]
                csvData.to_csv(saveCsvFile, index=False)
                log.info(f'[CHECK] saveCsvFile : {saveCsvFile}')

        # ***********************************************************************************
        # 과거 기간 95% 이상 분위수 계산
        # ***********************************************************************************
        # 95% 이상 분위수 계산
        mrgDataL2 = mrgDataL1.quantile(0.95, dim='time')

        # NetCDF 자료 저장
        saveFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], keyInfo, 'FUTURE-RES95', method, keyInfo)
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        mrgDataL2.to_netcdf(saveFile)
        log.info(f'[CHECK] saveFile : {saveFile}')

        # CSV 자료 저장
        saveFile = '{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], keyInfo, 'FUTURE-RES95', method, keyInfo)
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        mrgDataL2.to_dataframe().reset_index(drop=False).to_csv(saveFile, index=False)
        log.info(f'[CHECK] saveFile : {saveFile}')


        # ***********************************************************************************
        # 대륙 및 시간에 따른 검증 지표
        # ***********************************************************************************
        # valData = pd.DataFrame()
        # contIdxList = np.unique(prdDataL1['contIdx'].values)
        # timeList = prdDataL1['time'].values
        # for contIdx in contIdxList:
        #     if pd.isna(contIdx): continue
        #     # log.info(f'[CHECK] contIdx : {contIdx}')
        #
        #     for time in timeList:
        #         # log.info(f'[CHECK] time : {time}')
        #
        #         yList = simDataL3.sel(time=time).where(simDataL3['contIdx'] == contIdx, drop=True)['pr'].values.flatten()
        #         yhatList = prdDataL1.sel(time=time).where(prdDataL1['contIdx'] == contIdx, drop=True)['scen'].values.flatten()
        #
        #         # mask = ~np.isnan(x) & (x > 0) & (y > 0) & ~np.isnan(y) & (yhat > 0) & ~np.isnan(yhat)
        #         mask = (yList > 0) & ~np.isnan(yList) & (yhatList > 0) & ~np.isnan(yhatList)
        #
        #         # X = x[mask]
        #         y = yList[mask]
        #         yhat = yhatList[mask]
        #
        #         if (len(y) == 0) or (len(yhat) == 0): continue
        #
        #         # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
        #         dict = {
        #             'contIdx': [contIdx]
        #             , 'time': [time]
        #             , 'cnt': [len(y)]
        #             , 'bias': [np.nanmean(yhat - y)]
        #             , 'rmse': [np.sqrt(np.nanmean((yhat - y) ** 2))]
        #             , 'corr': [np.corrcoef(yhat, y)[0, 1]]
        #         }
        #
        #         valData = pd.concat([valData, pd.DataFrame.from_dict(dict)], ignore_index=True)
        #
        # # 대륙에 따른 CSV 자료 저장
        # saveFile = '{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, 'FUTURE-VALID-ByContTime', method, keyInfo)
        # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        # valData.to_csv(saveFile, index=False)
        # log.info(f'[CHECK] saveFile : {saveFile}')

        # ***********************************************************************************
        # 대륙에 따른 XLSX 자료 저장
        # ***********************************************************************************
        # contIdxList = np.unique(prdDataL1['contIdx'])
        # for contIdx in contIdxList:
        #     if pd.isna(contIdx): continue
        #     # log.info(f'[CHECK] contIdx : {contIdx}')
        #
        #     selData = prdDataL1.where(prdDataL1['contIdx'] == contIdx, drop=True)
        #
        #     selDataL1 = selData['scen'].to_dataframe().reset_index(drop=False)
        #     if len(selDataL1['scen'].dropna()) < 0: continue
        #
        #     # selDataL2 = selDataL1.pivot(index=['time'], columns=['lon', 'lat'])
        #     selDataL2 = selDataL1.pivot(index=['lon', 'lat'], columns=['time'])
        #
        #     # 엑셀 저장
        #     saveXlsxFile = '{}/{}/{}_{}-{}_{}.xlsx'.format(globalVar['outPath'], serviceName, 'FUTURE-ORG', method, int(contIdx), keyInfo)
        #     os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)
        #
        #     with pd.ExcelWriter(saveXlsxFile, engine='xlsxwriter', options={'use_zip64': True}) as writer:
        #         selDataL2.to_excel(writer, index=True)
        #
        #     log.info(f'[CHECK] saveXlsxFile : {saveXlsxFile}')

    except Exception as e:
        log.error(f'Exception in makeSbckProc: {str(e)}')
        traceback.print_exc()
        return result
    finally:
        log.info(f'[CHECK] method : {method} / pid : {procInfo.pid}')
        log.info('[END] {}'.format("makeSbckProc"))


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 전 지구 규모의 일 단위 강수량 편의보정 및 성능평가

    # 편향 조정 관련 API 명세
    # https://xclim.readthedocs.io/en/v0.30.1/sdba_api.html#bias-adjustment-algos
    # https://xclim.readthedocs.io/en/stable/notebooks/sdba.html
    # https://xclim.readthedocs.io/en/stable/apidoc/xclim.sdba.html#xclim.sdba.processing.jitter_under_thresh
    # https://xclim.readthedocs.io/en/stable/notebooks/sdba.html

    # cd /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/unitSys
    # conda activate py38
    # nohup python TalentPlatform-LSH0466.py &

    # python TalentPlatform-LSH0466-Fversion.py --keyList "INM-CM5-0" --methodList "QDM"

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
    serviceName = 'LSH0466'

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
                globalVar['inpPath'] = 'E:/Global bias/Regridding'
                globalVar['outPath'] = 'L:/Global bias results/OUTPUT'
                globalVar['figPath'] = 'L:/Global bias results/FIG'

            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 학습 시작/종료 시간
                'srtDate': '1980-01-01'
                , 'endDate': '1982-12-31'

                # 예측 시작/종료 시간
                # , 'srtDate2': '2015-01-01'
                # , 'endDate2': '2020-12-31'
                , 'srtDate2': '2015-01-01'
                # , 'endDate2': '2015-12-31'
                , 'endDate2': '2015-02-01'

                # 경도 최소/최대/간격
                , 'lonMin': 0
                , 'lonMax': 30
                # , 'lonMax': 360
                , 'lonInv': 1

                # 위도 최소/최대/간격
                # , 'latMin': -90
                , 'latMin': 60
                , 'latMax': 90
                , 'latInv': 1

                #, 'keyList' : ['GFDL-ESM4','INM-CM4-8','INM-CM5-0','IPSL-CM6A-LR','MIROC6','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NorESM2-LM','NorESM2-MM','TaiESM1']
                # , 'keyList': ['MRI-ESM2-0', 'INM-CM5-0']
                # , 'keyList': ['MRI-ESM2-0']
                #, 'keyList': ['MRI-ESM2-0','ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','CanESM5','CESM2-WACCM','CMCC-CM2-SR5','CMCC-ESM2','CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3-Veg-LR']
                # , 'keyList': ['INM-CM5-0']
                , 'keyList': [globalVar['keyList']]

                # 메서드 목록
                # , 'methodList': ['QDM', 'EQM', 'DQM', 'Scaling']
                , 'methodList': [globalVar['methodList']]

                # 비동기 다중 프로세스 개수
                # , 'cpuCoreNum': 4
                # , 'cpuCoreDtlNum': 4
                # , 'cpuCoreDtlNum': 2
                , 'cpuCoreDtlNum': 1

                # 분산 처리
                # Exception in makeSbckProc: Multiple chunks along the main adjustment dimension time is not supported.
                # , 'chunks': {'time': 1}
                # , 'chunks': {'lat': 5, 'lon': 5, 'time': -1}
            }

            # ********************************************************************
            # 주요 설정
            # ********************************************************************
            # 날짜 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            # dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1D')
            # dtMonthList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1M')

            # 위경도 설정
            lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
            latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])
            log.info(f'[CHECK] len(lonList) : {len(lonList)}')
            log.info(f'[CHECK] len(latList) : {len(latList)}')

            # 비동기 다중 프로세스 개수
            # pool = Pool(sysOpt['cpuCoreNum'])

            # 분산 처리
            # client = Client(processes = True, n_workers=1, threads_per_worker=4, memory_limit="4GB")
            # client = Client(cluster)

            # cluster = LocalCluster(processes=True, n_workers=4,threads_per_worker=1)

            # ********************************************************************
            # 대륙별 분류 전처리
            # ********************************************************************
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'TTL4.csv')
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], 'Historical', 'TTL4.csv')
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1: raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

            contData = pd.read_csv(fileList[0]).rename(columns={'type': 'contIdx', 'Latitude': 'lat', 'Longitude': 'lon'})
            # contDataL1 = contData[['lon', 'lat', 'isLand', 'contIdx']]
            contDataL1 = contData[['lon', 'lat', 'contIdx']]

            # 경도 변환 (-180~180 to 0~360)
            # contDataL1['lon'] = np.where(contDataL1['lon'] < 0, (contDataL1['lon']) % 360, contDataL1['lon'])

            contDataL2 = contDataL1.set_index(['lat', 'lon'])
            contDataL3 = contDataL2.to_xarray()
            contDataL4 = contDataL3.interp({'lon': lonList, 'lat': latList}, method='nearest')

            # contDataL3['contIdx'].plot()
            # contDataL4['contIdx'].plot()
            # plt.show()

            # ********************************************************************
            # 강수량 데이터 전처리
            # ********************************************************************
            # 실측 데이터
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ERA5_1979_2020.nc')
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], 'Historical', 'ERA5_1979_2020.nc')
            fileList = sorted(glob.glob(inpFile))
            if fileList is None or len(fileList) < 1: raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

            obsData = xr.open_dataset(fileList[0]).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
            # obsData = xr.open_dataset(fileList[0], chunks=sysOpt['chunks']).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))

            # 경도 변환 (-180~180 to 0~360)
            obsDataL1 = obsData
            #obsDataL1.coords['lon'] = (obsDataL1.coords['lon']) % 360
            #obsDataL1 = obsDataL1.sortby(obsDataL1.lon)

            #obsDataL2 = obsDataL1.interp({'lon': lonList, 'lat': latList}, method='linear')
            #obsDataL3 = xr.merge([obsDataL2['rain'], contDataL4])
            obsDataL3 = remove_leap_days_xarray(obsDataL1)
            obsDataL3 = xr.merge([obsDataL3['rain'], contDataL4])
            time_index = pd.to_datetime(obsDataL3.time.values)
            normalized_time_index = time_index.normalize()
            obsDataL3['time'] = ('time', normalized_time_index)

            # obsDataL2.attrs
            # obsDataL2['rain'].attrs

            keyList = sysOpt['keyList']
            for keyInfo in keyList:
                log.info(f"[CHECK] keyInfo : {keyInfo}")

                # 관측/학습 데이터
                inpFile = '{}/{}/*{}*{}*.nc'.format(globalVar['inpPath'], serviceName, keyInfo, 'historical')
                # inpFile = '{}/{}/*{}*{}*.nc'.format(globalVar['inpPath'], 'Historical', keyInfo, 'historical')
                fileList = sorted(glob.glob(inpFile))
                if fileList is None or len(fileList) < 1: raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

                # fileInfo = fileList[0]
                # fileInfo = fileList[1]
                modDataL2 = xr.Dataset()
                for fileInfo in fileList:
                    log.info(f"[CHECK] fileInfo : {fileInfo}")

                    # fileNameNoExt = os.path.basename(fileInfo).split('.')[0]

                    modData = xr.open_dataset(fileInfo).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
                    # modData = xr.open_dataset(fileInfo, chunks=sysOpt['chunks']).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
                    if (len(modData['time']) < 1): continue

                    # 필요없는 변수 삭제
                    selList = ['lat_bnds', 'lon_bnds', 'time_bnds']
                    for i, selInfo in enumerate(selList):
                        try:
                            modData = modData.drop([selInfo])
                        except Exception as e:
                            pass
                            # log.error("Exception : {}".format(e))

                    modDataL1 = modData.interp({'lon': lonList, 'lat': latList}, method='linear')

                    # 일 강수량 단위 환산 : 60 * 60 * 24
                    modDataL1['pr'] = modDataL1['pr']
                    modDataL1['pr'].attrs["units"] = "mm d-1"

                    modDataL2 = xr.merge([modDataL2, modDataL1])

                modDataL3 = xr.merge([modDataL2['pr'], contDataL4])
                # modDataL3C = add_leap_day_mrg(modDataL3.drop_vars('contIdx'), var_name='pr')
                modDataL3C = remove_leap_days_xarray(modDataL3)
                modDataL3C['time'] = ('time', normalized_time_index)

                # 병합 데이터 : 실측 + 관측/학습
                mrgData = xr.merge([obsDataL3, modDataL3C])

                # 예측 데이터
                inpFile = '{}/{}/*{}*{}*.nc'.format(globalVar['inpPath'], serviceName, keyInfo, 'ssp126')
                # inpFile = '{}/{}/*{}*{}*.nc'.format(globalVar['inpPath'], 'Future', keyInfo, 'ssp126')
                fileList = sorted(glob.glob(inpFile))
                if fileList is None or len(fileList) < 1: raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

                # fileInfo = fileList[0]
                simDataL2 = xr.Dataset()
                for fileInfo in fileList:
                    log.info(f"[CHECK] fileInfo : {fileInfo}")

                    # fileNameNoExt = os.path.basename(fileInfo).split('.')[0]

                    simData = xr.open_dataset(fileInfo).sel(time=slice(sysOpt['srtDate2'], sysOpt['endDate2']))
                    # simData = xr.open_dataset(fileInfo, chunks=sysOpt['chunks']).sel(time=slice(sysOpt['srtDate2'], sysOpt['endDate2']))
                    if (len(simData['time']) < 1): continue

                    # 필요없는 변수 삭제
                    selList = ['lat_bnds', 'lon_bnds', 'time_bnds']

                    for i, selInfo in enumerate(selList):
                        try:
                            simData = simData.drop([selInfo])
                        except Exception as e:
                            pass
                            # log.error("Exception : {}".format(e))

                    simDataL1 = simData.interp({'lon': lonList, 'lat': latList}, method='linear')

                    # 일 강수량 단위 환산 : 60 * 60 * 24
                    simDataL1['pr'] = simDataL1['pr']
                    simDataL1['pr'].attrs["units"] = "mm d-1"

                    simDataL2 = xr.merge([simDataL2, simDataL1])

                simDataL3 = xr.merge([simDataL2['pr'], contDataL4])
                normalized_time_index2 = pd.to_datetime(simDataL3.time.values)
                # simDataL3 = remove_leap_days_xarray(simDataL3)
                simDataL3 = simDataL3.convert_calendar("noleap")

                # 윤년을 추가한 데이터셋 생성
                #mrgDataProcessed = remove_feb29(mrgData.drop_vars(['contIdx','rain']))

                # mrgDataProcessed = add_leap_day_mrg(mrgData.drop_vars('contIdx'), var_name='pr')
                mrgDataProcessed = mrgData.drop_vars('contIdx')

                #simDataL3Processed = remove_feb29(simDataL3.drop_vars('contIdx'))
                #simDataL3Processed = add_leap_day_L3(simDataL3.drop_vars('contIdx'), var_name='pr')
                simDataL3Processed = simDataL3
                #simDataL3Processed['time'] = simDataL3Processed['time'].dt.floor('D')

                # 윤년이 추가된 데이터셋에 'contIdx' 다시 결합
                mrgDataProcessed = xr.merge([mrgDataProcessed, mrgData['contIdx']]).transpose('lat','lon','time')
                simDataL3Processed = xr.merge([simDataL3Processed, simDataL3['contIdx']]).transpose('lat','lon','time')
                simDataL3Processed['time'] = simDataL3Processed['time'].dt.floor('D')
                #mrgDataProcessed = fix_dayofyear(mrgDataProcessed)
                #simDataL3Processed = fix_dayofyear(simDataL3Processed)

                # makeSbckProc(method='QDM', contDataL4=contDataL4, mrgDataProcessed=mrgDataProcessed, simDataL3Processed=simDataL3Processed, keyInfo=keyInfo)
                # makeSbckProc(method='EQM', contDataL4=contDataL4, mrgDataProcessed=mrgDataProcessed, simDataL3Processed=simDataL3Processed, keyInfo=keyInfo)
                # makeSbckProc(method='DQM', contDataL4=contDataL4, mrgDataProcessed=mrgDataProcessed, simDataL3Processed=simDataL3Processed, keyInfo=keyInfo)
                # makeSbckProc(method='Scaling', contDataL4=contDataL4, mrgDataProcessed=mrgDataProcessed, simDataL3=simDataL3Processed, keyInfo=keyInfo)

                # 윤년 제거
                mrgDataProcessed = mrgDataProcessed.convert_calendar("noleap")

                # 단일 다중 프로세스 수행
                for method in sysOpt['methodList']:
                    log.info(f"[CHECK] method : {method}")
                    result = makeSbckProc(method, contDataL4, mrgDataProcessed, simDataL3Processed, keyInfo)
                    log.info(f"[CHECK] result : {result}")

                # 비동기 다중 프로세스 수행
                # poolDtl = Pool(sysOpt['cpuCoreDtlNum'])
                # for method in sysOpt['methodList']:
                #     if not method is sysOpt['method']: continue
                #     log.info(f"[CHECK] method : {method}")
                #     poolDtl.apply_async(makeSbckProc, args=(method, contDataL4, mrgDataProcessed, simDataL3Processed, keyInfo))
                # poolDtl.close()
                # poolDtl.join()

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