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
from datetime import timedelta
from urllib.parse import quote_plus

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import xarray as xr
from pandas.tseries.offsets import Hour
import yaml
from multiprocessing import Pool
import multiprocessing as mp
from retrying import retry
# import cdsapi
import shutil

import requests
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime
import subprocess

from remssHelper.ssmis.ssmis_daily_v7 import SSMISdaily
from remssHelper.gmi.gmi_daily_v8 import GMIdaily
from remssHelper.ascat.ascat_daily import ASCATDaily
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
import matplotlib.colors as colors
from viresclient import AeolusRequest
from netCDF4 import Dataset, num2date
import cartopy.crs as ccrs
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

# plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')
dtKst = timedelta(hours=9)

# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}_{}_{}.log".format(
        os.path.join(contextPath, 'log') if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
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
        # , 'resPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        # , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        # , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        # , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        # , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        # , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        # , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        # , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        # , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        # , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        # , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
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

    return globalVar

def format_date(x, pos=None):
    dt_obj = num2date(x, units="s since 2000-01-01", only_use_cftime_datetimes=False)
    return dt_obj.strftime("%H:%M:%S")

def plotParam2D(parameter="wind_result_wind_velocity", channel="rayleigh", obs_type="clear", QC_filter=True, error_estimate_threshold=800, start_bin=0, end_bin=-1, ds=None):
    # define necessary parameters for plotting
    X0 = ds[channel + "_wind_result_start_time"].values
    X1 = ds[channel + "_wind_result_stop_time"].values

    Y0 = ds[channel + "_wind_result_bottom_altitude"].values / 1000.0
    Y1 = ds[channel + "_wind_result_top_altitude"].values / 1000.0
    Z = ds[channel + "_" + parameter].values

    # create a mask out of different filters which can be applied to the different parameters
    mask = np.zeros(len(Z), dtype=bool)

    # mask dependent on start and end bin given as parameter to the plot function
    mask[0:start_bin] = True
    mask[end_bin:-1] = True

    # mask where validity flag is 0
    if QC_filter:
        mask = mask | (ds[channel + "_wind_result_validity_flag"] == 0)

    # mask dependent on observation type
    if obs_type == "cloudy":
        mask = mask | (ds[channel + "_wind_result_observation_type"] != 1)
    elif obs_type == "clear":
        mask = mask | (ds[channel + "_wind_result_observation_type"] != 2)

    # mask where wind results have error estimates larger than a given threshold
    mask = mask | (ds[channel + "_wind_result_HLOS_error"] > error_estimate_threshold)

    # mask all necessary parameters for plotting
    # tilde before mask inverts the boolean mask array
    X0 = X0[~mask]
    X1 = X1[~mask]
    Y0 = Y0[~mask]
    Y1 = Y1[~mask]
    Z = Z[~mask]

    patches = []
    for x0, x1, y0, y1 in zip(X0, X1, Y0, Y1):
        patches.append(((x0, y0), (x0, y1), (x1, y1), (x1, y0)))

    # define min and max value for the colorbar
    if parameter == "wind_result_wind_velocity":
        Z_vmax = np.amax(np.abs(np.asarray([np.nanpercentile(Z, 2), np.nanpercentile(Z, 98)])))
        Z_vmin = -Z_vmax
    else:
        Z_vmax = np.nanpercentile(Z, 99)
        Z_vmin = np.nanpercentile(Z, 1)

    # fig, (axis, axis2) = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)
    # fig, (axis, axis2) = plt.subplots(1, 2, figsize=(20, 5), constrained_layout=True)
    fig, (axis, axis2) = plt.subplots(2, 1, figsize=(9, 10), constrained_layout=True)

    # axis = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
    axis = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
    axis.stock_img()
    axis.gridlines(draw_labels=True, linewidth=0.3, color="black", alpha=0.5, linestyle="-")
    axis.scatter(
        ds[channel + "_wind_result_COG_longitude"],
        ds[channel + "_wind_result_COG_latitude"],
        marker="o",
        c="k",
        s=3,
        label='wind result COG',
        transform=ccrs.Geodetic(),
    )
    axis.scatter(
        ds[channel + "_wind_result_COG_longitude"][0],
        ds[channel + "_wind_result_COG_latitude"][0],
        marker="o",
        c="g",
        edgecolor="g",
        s=40,
        label="start",
        transform=ccrs.Geodetic(),
    )
    axis.scatter(
        ds[channel + "_wind_result_COG_longitude"][-1],
        ds[channel + "_wind_result_COG_latitude"][-1],
        marker="o",
        c="r",
        edgecolor="r",
        s=40,
        label="stop",
        transform=ccrs.Geodetic(),
    )
    axis.legend()
    axis.set_title(channel.title())

    coll = PolyCollection(
        patches,
        array=Z,
        cmap=cm.RdBu_r,
        norm=colors.Normalize(
            vmin=Z_vmin,
            vmax=Z_vmax,
            clip=False,
        ),
    )

    axis2.add_collection(coll)

    axis2.scatter(
        ds[channel + "_wind_result_COG_time"][~mask],
        ds[channel + "_wind_result_alt_of_DEM_intersection"][~mask] / 1000.0,
        marker='o',
        c='r',
        s=5,
        label='DEM altitude',
    )

    # ax.set_ylim(-1, 30)
    axis2.set_xlabel("Date [UTC]")
    axis2.set_ylabel("Altitude [km]")
    axis2.set_title("{} - {} \n {} wind results".format(channel.title(), parameter, len(Z)))
    axis2.grid()
    axis2.legend()

    axis2.xaxis.set_major_formatter(format_date)
    axis2.autoscale()
    fig.colorbar(coll, ax=axis2, aspect=50, pad=0.01)
    fig.autofmt_xdate()

def visProc(modelType, modelInfo, dtDateInfo):
    try:
        visFunList = {
            'SSMIS': visSSMIS
            , 'AMSR2': visAMSR2
            , 'GMI': visGMI
            , 'SMAP': visSMAP
            , 'ASCAT-B': visASCAT
            , 'ASCAT-C': visASCAT
            , 'AEOLUS-RAY': visAEOLUS
            , 'AEOLUS-MIE': visAEOLUS
        }

        visFun = visFunList.get(modelType)
        visFun(modelInfo, dtDateInfo)
    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e

@retry(stop_max_attempt_number=10)
def visSSMIS(modelInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        inpFile = dtDateInfo.strftime(modelInfo['fileInfo'])
        fileList = sorted(glob.glob(inpFile))
        if len(fileList) < 1: return

        for fileInfo in fileList:
            data = SSMISdaily(fileInfo)

            dim = data.dimensions
            var = data.variables
            dataL1 = xr.Dataset(
                coords={
                    'orbit': np.arange(dim['orbit_segment'])
                    , 'lat': var['latitude']
                    , 'lon': var['longitude']
                }
            )

            for key, val in var.items():
                if re.search('longitude|latitude', key, re.IGNORECASE): continue
                # Time:  7.10 = fractional hour GMT, NOT local time,  valid data range=0 to 24.0,  255 = land
                # Wind: 255=land, 253=bad data,  251=no wind calculated, other data <=50.0 is 10-meter wind speed
                # Water Vapor:  255=land, 253=bad data, other data <=75 is water vapor (mm)
                # Cloud Liquid Water:  255=land,  253=bad data, other data <=2.5 is cloud (mm)
                # Rain:  255=land, 253=bad data, other data <= 25 is rain (mm/hr)
                if re.search('time', key, re.IGNORECASE):
                    val2 = xr.where((0 <= val) & (val <= 24), val, np.nan)
                elif re.search('wspd_mf', key, re.IGNORECASE):
                    val2 = xr.where((val <= 50), val, np.nan)
                elif re.search('vapor', key, re.IGNORECASE):
                    val2 = xr.where((val <= 75), val, np.nan)
                elif re.search('cloud', key, re.IGNORECASE):
                    val2 = xr.where((val <= 2.5), val, np.nan)
                elif re.search('rain', key, re.IGNORECASE):
                    val2 = xr.where((val <= 25), val, np.nan)
                else:
                    val2 = val

                try:
                    dataL1[key] = (('orbit', 'lat', 'lon'), (val2))
                except Exception as e:
                    pass

            # NetCDF 저장
            fileName, fileExt = os.path.splitext(fileInfo)
            procFile = fileInfo.replace(fileExt, '.nc')
            if re.search('.gz', fileExt, re.IGNORECASE) and not os.path.isfile(procFile):
                os.makedirs(os.path.dirname(procFile), exist_ok=True)
                dataL1.to_netcdf(procFile)
                log.info(f'[CHECK] procFile : {procFile}')

            # 영상 생산
            for i, orgVar in enumerate(modelInfo['orgVar']):
                newVar = modelInfo['newVar'][i]

                saveImg = dtDateInfo.strftime(modelInfo['figInfo']).format(newVar)
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)

                # 파일 검사
                fileList = sorted(glob.glob(saveImg))
                if len(fileList) > 0: continue

                fig, ax = plt.subplots(figsize=(10, 4), subplot_kw={'projection': ccrs.PlateCarree()})

                ax.set_global()
                ax.add_feature(cfeature.LAND.with_scale('110m'), edgecolor='k', lw=0.5)
                ax.add_feature(cfeature.OCEAN.with_scale('110m'), edgecolor='k', facecolor='white', lw=0.5)
                ax.add_feature(cfeature.LAKES.with_scale('110m'), edgecolor='k', facecolor='white', lw=0.5)
                ax.add_feature(cfeature.BORDERS.with_scale('110m'), lw=0.5, edgecolor='k')
                ax.add_feature(cfeature.RIVERS.with_scale('110m'), lw=0.5, edgecolor='k')

                meanData = dataL1[orgVar].mean(dim=['orbit'])
                meanData.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cm.get_cmap('jet'), vmin=0, vmax=50)

                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False
                gl.right_labels = False

                plt.title(os.path.basename(saveImg).split('.')[0])

                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
                plt.close()
                log.info(f'[CHECK] saveImg : {saveImg}')

            log.info(f'[END] visSSMIS : {dtDateInfo} / pid : {procInfo.pid}')

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e

@retry(stop_max_attempt_number=10)
def visAMSR2(modelInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        inpFile = dtDateInfo.strftime(modelInfo['fileInfo'])
        fileList = sorted(glob.glob(inpFile))
        if len(fileList) < 1: return

        for fileInfo in fileList:
            data = xr.open_dataset(fileInfo)
            dataL1 = data

            # 영상 생산
            for i, orgVar in enumerate(modelInfo['orgVar']):
                newVar = modelInfo['newVar'][i]

                saveImg = dtDateInfo.strftime(modelInfo['figInfo']).format(newVar)
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)

                # 파일 검사
                fileList = sorted(glob.glob(saveImg))
                if len(fileList) > 0: continue

                fig, ax = plt.subplots(figsize=(10, 4), subplot_kw={'projection': ccrs.PlateCarree()})

                ax.set_global()
                ax.add_feature(cfeature.LAND.with_scale('110m'), edgecolor='k', lw=0.5)
                ax.add_feature(cfeature.OCEAN.with_scale('110m'), edgecolor='k', facecolor='white', lw=0.5)
                ax.add_feature(cfeature.LAKES.with_scale('110m'), edgecolor='k', facecolor='white', lw=0.5)
                ax.add_feature(cfeature.BORDERS.with_scale('110m'), lw=0.5, edgecolor='k')
                ax.add_feature(cfeature.RIVERS.with_scale('110m'), lw=0.5, edgecolor='k')

                meanData = dataL1[orgVar].mean(dim=['pass'])
                meanData.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cm.get_cmap('jet'), vmin=0, vmax=50)

                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False
                gl.right_labels = False

                plt.title(os.path.basename(saveImg).split('.')[0])

                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
                plt.close()
                log.info(f'[CHECK] saveImg : {saveImg}')

            log.info(f'[END] visAMSR2 : {dtDateInfo} / pid : {procInfo.pid}')

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e

@retry(stop_max_attempt_number=10)
def visGMI(modelInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        inpFile = dtDateInfo.strftime(modelInfo['fileInfo'])
        fileList = sorted(glob.glob(inpFile))
        if len(fileList) < 1: return

        for fileInfo in fileList:
            data = GMIdaily(fileInfo)

            dim = data.dimensions
            var = data.variables
            dataL1 = xr.Dataset(
                coords={
                    'orbit': np.arange(dim['orbit_segment'])
                    , 'lat': var['latitude']
                    , 'lon': var['longitude']
                }
            )

            for key, val in var.items():
                if re.search('longitude|latitude', key, re.IGNORECASE): continue
                # gmt time, valid data range 0 to 1440 (in minutes)
                # sea surface temperature, valid data range -3 to 34.5 (degree C)
                # wind speed low frequency, valid data range 0 to 50.0 (meters/second)
                # wind speed medium frequency, valid data range 0 to 50.0 (meters/second)
                # water vapor, valid data range 0 to 75 (millimeters)
                # cloud, valid data range -0.05 to 2.45 (millimeters)
                # rain rate, valid data range 0 to 25 (millimeters/hour)
                if re.search('time', key, re.IGNORECASE):
                    val2 = xr.where((0 <= val) & (val <= 1440), val, np.nan)
                elif re.search('sst', key, re.IGNORECASE):
                    val2 = xr.where((-3 <= val) & (val <= 34.5), val, np.nan)
                elif re.search('windLF|windMF', key, re.IGNORECASE):
                    val2 = xr.where((0 <= val) & (val <= 50.0), val, np.nan)
                elif re.search('vapor', key, re.IGNORECASE):
                    val2 = xr.where((0 <= val) & (val <= 75), val, np.nan)
                elif re.search('cloud', key, re.IGNORECASE):
                    val2 = xr.where((-0.05 <= val) & (val <= 2.45), val, np.nan)
                elif re.search('rain', key, re.IGNORECASE):
                    val2 = xr.where((0 <= val) & (val <= 25), val, np.nan)
                else:
                    val2 = val

                try:
                    dataL1[key] = (('orbit', 'lat', 'lon'), (val2))
                except Exception as e:
                    pass

            # NetCDF 저장
            fileName, fileExt = os.path.splitext(fileInfo)
            procFile = fileInfo.replace(fileExt, '.nc')
            if re.search('.gz', fileExt, re.IGNORECASE) and not os.path.isfile(procFile):
                os.makedirs(os.path.dirname(procFile), exist_ok=True)
                dataL1.to_netcdf(procFile)
                log.info(f'[CHECK] procFile : {procFile}')

            # 영상 생산
            for i, orgVar in enumerate(modelInfo['orgVar']):
                newVar = modelInfo['newVar'][i]

                saveImg = dtDateInfo.strftime(modelInfo['figInfo']).format(newVar)
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)

                # 파일 검사
                fileList = sorted(glob.glob(saveImg))
                if len(fileList) > 0: continue

                fig, ax = plt.subplots(figsize=(10, 4), subplot_kw={'projection': ccrs.PlateCarree()})

                ax.set_global()
                ax.add_feature(cfeature.LAND.with_scale('110m'), edgecolor='k', lw=0.5)
                ax.add_feature(cfeature.OCEAN.with_scale('110m'), edgecolor='k', facecolor='white', lw=0.5)
                ax.add_feature(cfeature.LAKES.with_scale('110m'), edgecolor='k', facecolor='white', lw=0.5)
                ax.add_feature(cfeature.BORDERS.with_scale('110m'), lw=0.5, edgecolor='k')
                ax.add_feature(cfeature.RIVERS.with_scale('110m'), lw=0.5, edgecolor='k')

                meanData = dataL1[orgVar].mean(dim=['orbit'])
                meanData.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cm.get_cmap('jet'), vmin=0, vmax=50)

                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False
                gl.right_labels = False

                plt.title(os.path.basename(saveImg).split('.')[0])

                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
                plt.close()
                log.info(f'[CHECK] saveImg : {saveImg}')

            log.info(f'[END] visSSMIS : {dtDateInfo} / pid : {procInfo.pid}')

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e


@retry(stop_max_attempt_number=10)
def visSMAP(modelInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        inpFile = dtDateInfo.strftime(modelInfo['fileInfo'])
        fileList = sorted(glob.glob(inpFile))
        if len(fileList) < 1: return

        for fileInfo in fileList:
            data = xr.open_dataset(fileInfo)
            dataL1 = data

            # 영상 생산
            for i, orgVar in enumerate(modelInfo['orgVar']):
                newVar = modelInfo['newVar'][i]

                saveImg = dtDateInfo.strftime(modelInfo['figInfo']).format(newVar)
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)

                # 파일 검사
                fileList = sorted(glob.glob(saveImg))
                if len(fileList) > 0: continue

                fig, ax = plt.subplots(figsize=(10, 4), subplot_kw={'projection': ccrs.PlateCarree()})

                ax.set_global()
                ax.add_feature(cfeature.LAND.with_scale('110m'), edgecolor='k', lw=0.5)
                ax.add_feature(cfeature.OCEAN.with_scale('110m'), edgecolor='k', facecolor='white', lw=0.5)
                ax.add_feature(cfeature.LAKES.with_scale('110m'), edgecolor='k', facecolor='white', lw=0.5)
                ax.add_feature(cfeature.BORDERS.with_scale('110m'), lw=0.5, edgecolor='k')
                ax.add_feature(cfeature.RIVERS.with_scale('110m'), lw=0.5, edgecolor='k')

                meanData = dataL1[orgVar].mean(dim=['node'])
                # meanData.plot()
                meanData.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cm.get_cmap('jet'), vmin=0, vmax=50)

                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False
                gl.right_labels = False

                plt.title(os.path.basename(saveImg).split('.')[0])

                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
                plt.close()
                log.info(f'[CHECK] saveImg : {saveImg}')

            log.info(f'[END] visAMSR2 : {dtDateInfo} / pid : {procInfo.pid}')

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e


@retry(stop_max_attempt_number=10)
def visASCAT(modelInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        inpFile = dtDateInfo.strftime(modelInfo['fileInfo'])
        fileList = sorted(glob.glob(inpFile))
        if len(fileList) < 1: return

        for fileInfo in fileList:
            data = ASCATDaily(fileInfo)

            dim = data.dimensions
            var = data.variables
            dataL1 = xr.Dataset(
                coords={
                    'orbit': np.arange(dim['orbit_segment'])
                    , 'lat': var['latitude']
                    , 'lon': var['longitude']
                }
            )

            for key, val in var.items():
                if re.search('longitude|latitude', key, re.IGNORECASE): continue
                val2 = xr.where((val != -999.0), val, np.nan)

                try:
                    dataL1[key] = (('orbit', 'lat', 'lon'), (val2))
                except Exception as e:
                    pass

            # NetCDF 저장
            fileName, fileExt = os.path.splitext(fileInfo)
            procFile = fileInfo.replace(fileExt, '.nc')
            if re.search('.gz', fileExt, re.IGNORECASE) and not os.path.isfile(procFile):
                os.makedirs(os.path.dirname(procFile), exist_ok=True)
                dataL1.to_netcdf(procFile)
                log.info(f'[CHECK] procFile : {procFile}')

            # 영상 생산
            for i, orgVar in enumerate(modelInfo['orgVar']):
                newVar = modelInfo['newVar'][i]

                saveImg = dtDateInfo.strftime(modelInfo['figInfo']).format(newVar)
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)

                # 파일 검사
                fileList = sorted(glob.glob(saveImg))
                if len(fileList) > 0: continue

                fig, ax = plt.subplots(figsize=(10, 4), subplot_kw={'projection': ccrs.PlateCarree()})

                ax.set_global()
                ax.add_feature(cfeature.LAND.with_scale('110m'), edgecolor='k', lw=0.5)
                ax.add_feature(cfeature.OCEAN.with_scale('110m'), edgecolor='k', facecolor='white', lw=0.5)
                ax.add_feature(cfeature.LAKES.with_scale('110m'), edgecolor='k', facecolor='white', lw=0.5)
                ax.add_feature(cfeature.BORDERS.with_scale('110m'), lw=0.5, edgecolor='k')
                ax.add_feature(cfeature.RIVERS.with_scale('110m'), lw=0.5, edgecolor='k')

                meanData = dataL1[orgVar].mean(dim=['orbit'])
                meanData.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cm.get_cmap('jet'), vmin=0, vmax=50)

                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False
                gl.right_labels = False

                plt.title(os.path.basename(saveImg).split('.')[0])

                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
                plt.close()
                log.info(f'[CHECK] saveImg : {saveImg}')

            log.info(f'[END] visSSMIS : {dtDateInfo} / pid : {procInfo.pid}')

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e


@retry(stop_max_attempt_number=10)
def visAEOLUS(modelInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        inpFile = dtDateInfo.strftime(modelInfo['fileInfo'])
        fileList = sorted(glob.glob(inpFile))
        if len(fileList) < 1: return

        for fileInfo in fileList:
            request = AeolusRequest(url=modelInfo['request']['url'], token=modelInfo['request']['token'])
            data = request.get_from_file(fileInfo)
            dataL1 = data.as_xarray()

            isRay = re.search('_wind-ray_', fileInfo, re.IGNORECASE)

            if len(dataL1) < 1: continue
            if len(dataL1["rayleigh_wind_data" if isRay else "mie_wind_data"]) < 1: continue

            match = re.search(r"\d{12}", fileInfo)
            if not match: continue
            getDateTime = pd.to_datetime(match.group(0), format='%Y%m%d%H%M')

            saveImg = getDateTime.strftime(modelInfo['figInfo'])
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)

            # 파일 검사
            fileList = sorted(glob.glob(saveImg))
            if len(fileList) > 0: continue

            plotParam2D(
                parameter="wind_result_wind_velocity",
                channel="rayleigh" if isRay else "mie",
                obs_type="clear" if isRay else "cloudy",
                QC_filter=True,
                error_estimate_threshold=800 if isRay else 500,
                start_bin=0,
                end_bin=-1,
                ds=dataL1
            )

            plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
            plt.close()
            log.info(f'[CHECK] saveImg : {saveImg}')

            log.info(f'[END] visAEOLUS : {dtDateInfo} / pid : {procInfo.pid}')

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 위성측정기반 바람 산출물 수집

    # cd /home/hanul/SYSTEMS/KIER/PROG/PYTHON/colct
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-remssAMSR2.py --modelList SAT-AMSR2 --cpuCoreNum 5 --srtDate 2024-08-01 --endDate 2024-08-15 &

    # ps -ef | grep "TalentPlatform-INDI2024-colct-remssSMAP.py" | awk '{print $2}' | xargs kill -9
    # ps -ef | grep "RunShell-get-gfsncep2.sh" | awk '{print $2}' | xargs kill -9
    # ps -ef | egrep "RunShell|Repro" | awk '{print $2}' | xargs kill -9

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        # contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'
        contextPath = os.getcwd() if env in 'local' else '/vol01/SYSTEMS/KIER/PROG/PYTHON'

    prjName = 'test'
    serviceName = 'INDI2024'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info(f"[START] __init__ : init")

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info(f"[END] __init__ : init")

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info(f"[START] exec")

        try:

            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            # 옵션 설정
            sysOpt = {
                # 시작일, 종료일, 시간 간격 (연 1y, 월 1h, 일 1d, 시간 1h)
                'srtDate': '2023-01-01'
                , 'endDate': '2023-01-03'
                # 'srtDate': globalVar['srtDate']
                # , 'endDate': globalVar['endDate']
                , 'invDate': '1d'

                # 수행 목록
                , 'modelList': ['SSMIS', 'AMSR2', 'GMI', 'SMAP', 'ASCAT-B', 'ASCAT-C', 'AEOLUS-RAY', 'AEOLUS-MIE']
                # 'modelList': [globalVar['modelList']]

                # 비동기 다중 프로세스 개수
                , 'cpuCoreNum': '5'
                # , 'cpuCoreNum': globalVar['cpuCoreNum']

                , 'SSMIS': {
                    'fileInfo': '/HDD/DATA/data1/SAT/SSMIS/%Y%m/%d/f18_%Y%m%dv*.gz'
                    , 'orgVar': ['wspd_mf']
                    , 'newVar': ['wind-mf']
                    , 'figInfo': '/HDD/DATA/data1/IMG/SSMIS/%Y%m/%d/ssmis_{}-1D_%Y%m%d%H%M.png'
                }
                , 'AMSR2': {
                    'fileInfo': '/HDD/DATA/data1/SAT/AMSR2/%Y%m/%d/RSS_AMSR2_ocean_L3_daily_%Y-%m-%d_v*.*.nc'
                    , 'orgVar': ['wind_speed_LF', 'wind_speed_MF', 'wind_speed_AW']
                    , 'newVar': ['wind-lf', 'wind-mf', 'wind-aw']
                    , 'figInfo': '/HDD/DATA/data1/IMG/AMSR2/%Y%m/%d/amsr2_{}-1D_%Y%m%d%H%M.png'
                }
                , 'GMI': {
                    'fileInfo': '/HDD/DATA/data1/SAT/GMI/%Y%m/%d/f35_%Y%m%dv*.*.gz'
                    , 'orgVar': ['windLF', 'windMF']
                    , 'newVar': ['wind-lf', 'wind-mf']
                    , 'figInfo': '/HDD/DATA/data1/IMG/GMI/%Y%m/%d/gmi_{}-1D_%Y%m%d%H%M.png'
                }
                , 'SMAP': {
                    'fileInfo': '/HDD/DATA/data1/SAT/SMAP/%Y%m/%d/RSS_smap_wind_daily_%Y_%m_%d_NRT_v*.*.nc'
                    , 'orgVar': ['wind']
                    , 'newVar': ['wind']
                    , 'figInfo': '/HDD/DATA/data1/IMG/SMAP/%Y%m/%d/smap_{}-1D_%Y%m%d%H%M.png'
                }
                , 'ASCAT-B': {
                    'fileInfo': '/HDD/DATA/data1/SAT/ASCAT/%Y%m/%d/ascatb_%Y%m%d_v*.*.gz'
                    , 'orgVar': ['windspd']
                    , 'newVar': ['wind']
                    , 'figInfo': '/HDD/DATA/data1/IMG/ASCAT/%Y%m/%d/ascatb_{}-1D_%Y%m%d%H%M.png'
                }
                , 'ASCAT-C': {
                    'fileInfo': '/HDD/DATA/data1/SAT/ASCAT/%Y%m/%d/ascatc_%Y%m%d_v*.*.gz'
                    , 'orgVar': ['windspd']
                    , 'newVar': ['wind']
                    , 'figInfo': '/HDD/DATA/data1/IMG/ASCAT/%Y%m/%d/ascatc_{}-1D_%Y%m%d%H%M.png'
                }
                , 'AEOLUS-RAY': {
                    'fileInfo': '/HDD/DATA/data1/SAT/AEOLUS/%Y%m/%d/aeolus_wind-ray_%Y%m%d*.nc'
                    , 'request': {
                        'url': 'https://aeolus.services/ows'
                        , 'token': ''
                    }
                    , 'figInfo': '/HDD/DATA/data1/IMG/AEOLUS/%Y%m/%d/aeolus_wind-ray_%Y%m%d%H%M.png'
                }
                , 'AEOLUS-MIE': {
                    'fileInfo': '/HDD/DATA/data1/SAT/AEOLUS/%Y%m/%d/aeolus_wind-mie_%Y%m%d*.nc'
                    , 'request': {
                        'url': 'https://aeolus.services/ows'
                        , 'token': ''
                    }
                    , 'figInfo': '/HDD/DATA/data1/IMG/AEOLUS/%Y%m/%d/aeolus_wind-mie_%Y%m%d%H%M.png'
                }
            }

            # **************************************************************************************************************
            # 비동기 다중 프로세스 수행
            # **************************************************************************************************************
            # 비동기 다중 프로세스 개수
            pool = Pool(int(sysOpt['cpuCoreNum']))

            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])
            for dtDateInfo in dtDateList:
                log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                for modelType in sysOpt['modelList']:
                    # log.info(f'[CHECK] modelType : {modelType}')

                    modelInfo = sysOpt.get(modelType)
                    if modelInfo is None: continue

                    pool.apply_async(visProc, args=(modelType, modelInfo, dtDateInfo))

            pool.close()
            pool.join()

        except Exception as e:
            log.error(f"Exception : {e}")
            raise e

        finally:
            log.info(f"[END] exec")


# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print(f'[START] main')

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
        print('[END] main')
