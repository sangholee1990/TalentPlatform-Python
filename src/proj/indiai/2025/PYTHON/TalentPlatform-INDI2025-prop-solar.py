# ================================================
# 요구사항
# ================================================
# Python을 이용한

# ps -ef | grep "TalentPlatform-INDI2025-colct-kmaApiHub.py" | awk '{print $2}' | xargs kill -9

# cd /vol01/SYSTEMS/INDIAI/PROG/PYTHON
# /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-colct-kmaApiHub.py --modelList 'UMKR' --cpuCoreNum '5' --srtDate '2019-01-01' --endDate '2020-01-01'

# nohup /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-colct-kmaApiHub.py --modelList 'UMKR' --cpuCoreNum '5' --srtDate '2019-01-01' --endDate '2020-01-01' &

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
from isodate import parse_duration
from pandas.tseries.offsets import DateOffset
from sklearn.neighbors import BallTree

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

def propProc(modelType, modelInfo, dtDateInfo):
    try:
        propFunList = {
            'UMKR': propNwp,
        }

        propFun = propFunList.get(modelType)
        propFun(modelInfo, dtDateInfo)

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e

def parseDateOffset(invDate):
    unit = invDate[-1]
    value = int(invDate[:-1])

    if unit == 'y':
        return DateOffset(years=value)
    elif unit == 'm':
        return DateOffset(months=value)
    elif unit == 'd':
        return DateOffset(days=value)
    elif unit == 'h':
        return DateOffset(hours=value)
    elif unit == 't':
        return DateOffset(minutes=value)
    elif unit == 's':
        return DateOffset(seconds=value)
    else:
        raise ValueError(f"날짜 파싱 오류 : {unit}")

@retry(stop_max_attempt_number=10)
def propNwp(dataInfo, dtDateInfo):
    try:
        procInfo = mp.current_process()

        searchFileInfo = dtDateInfo.strftime(dataInfo['searchFileList'])

        # 파일 검사
        fileList = sorted(glob.glob(searchFileInfo))
        if len(fileList) > 0: return

        # reqUrl = dtDateInfo.strftime(f"{modelInfo['request']['url']}").format(
        #     tmfc=dtDateInfo.strftime('%Y%m%d%H%M'),
        #     tmfc2=(dtDateInfo + parseDateOffset(modelInfo['request']['invDate']) - parseDateOffset('1s')).strftime('%Y%m%d%H%M'),
        #     authKey=modelInfo['request']['authKey']
        # )

        # res = requests.get(reqUrl)
        # if not (res.status_code == 200): return

        # os.makedirs(os.path.dirname(tmpFileInfo), exist_ok=True)
        # os.makedirs(os.path.dirname(updFileInfo), exist_ok=True)
        #
        # if os.path.exists(tmpFileInfo):
        #     os.remove(tmpFileInfo)
        #
        # cmd = f"curl -s -C - '{reqUrl}' --retry 10 -o {tmpFileInfo}"
        # log.info(f'[CHECK] cmd : {cmd}')
        #
        # try:
        #     subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
        # except subprocess.CalledProcessError as e:
        #     raise ValueError(f'[ERROR] 실행 프로그램 실패 : {str(e)}')
        #
        # if os.path.exists(tmpFileInfo):
        #     if os.path.getsize(tmpFileInfo) > 1000:
        #         shutil.move(tmpFileInfo, updFileInfo)
        #         log.info(f'[CHECK] CMD : mv -f {tmpFileInfo} {updFileInfo}')
        #     else:
        #         os.remove(tmpFileInfo)
        #         log.info(f'[CHECK] CMD : rm -f {tmpFileInfo}')

        log.info(f'[END] colctKmaApiHub : {dtDateInfo} / pid : {procInfo.pid}')

    except Exception as e:
        log.error(f'Exception : {str(e)}')
        raise e


def matchStnFor(subOpt, subData):
    
    try:
        # ==========================================================================================================
        # 지상 관측소을 기준으로 최근접 가공파일 화소 찾기 (posRow, posCol, posLat, posLon, posDistKm)
        # ==========================================================================================================
        umkrFileInfo = subOpt['umkrFileInfo']

        if umkrFileInfo is None or len(umkrFileInfo) < 1:
            log.error(f"[ERROR] umkrFileInfo : {umkrFileInfo} / 가공파일을 확인해주세요.")
            return

        # 레이더 가공 파일 일부
        fileInfo = umkrFileInfo
        cfgData = xr.open_dataset(fileInfo, decode_times=True, engine='pynio')
        cfgDataL1 = cfgData[['TMP_P0_L1_GLC0']]
        cfgDataL2 = cfgDataL1.to_dataframe().reset_index(drop=False)
        cfgDataL3 = cfgDataL2.rename(
            columns={
                'ygrid_0': 'row'
                , 'xgrid_0': 'col'
                , 'gridlat_0': 'lat'
                , 'gridlon_0': 'lon'
            }
        )

        # 지상관측소
        allStnData = subData
        allStnDataL1 = subData[['Id', 'Latitude', 'Longitude']]

        # 2024.10.24 수정
        # allStnDataL2 = allStnDataL1[allStnDataL1['STN'].isin(sysOpt['stnInfo']['list'])]
        allStnDataL2 = allStnDataL1

        # 지상 관측소을 기준으로 최근접 가공파일 화소 찾기 (posRow, posCol, posLat, posLon, posDistKm)
        #           Id   Latitude   Longitude  ...     posLat      posLon  posDistKm
        # 0    당진수상태양광  37.050753  126.510299  ...  37.054657  126.508148   0.074440
        # 1  당진자재창고태양광  37.050753  126.510299  ...  37.054657  126.508148   0.074440
        # 2      당진태양광  37.050753  126.510299  ...  37.054657  126.508148   0.074440
        # 3      울산태양광  35.477651  129.380778  ...  35.480633  129.386398   0.095255
        baTree = BallTree(np.deg2rad(cfgDataL3[['lat', 'lon']].values), metric='haversine')
        for i, posInfo in allStnDataL2.iterrows():
            if (pd.isna(posInfo['Latitude']) or pd.isna(posInfo['Longitude'])): continue

            closest = baTree.query(np.deg2rad(np.c_[posInfo['Latitude'], posInfo['Longitude']]), k=1)
            cloDist = closest[0][0][0] * 1000.0
            cloIdx = closest[1][0][0]
            cfgInfo = cfgDataL3.loc[cloIdx]

            allStnDataL2.loc[i, 'posRow'] = cfgInfo['row']
            allStnDataL2.loc[i, 'posCol'] = cfgInfo['col']
            allStnDataL2.loc[i, 'posLat'] = cfgInfo['lat']
            allStnDataL2.loc[i, 'posLon'] = cfgInfo['lon']
            allStnDataL2.loc[i, 'posDistKm'] = cloDist

        return allStnDataL2

    except Exception as e:
        log.error(f'Exception : {str(e)}')

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

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'INDI2025'

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
                # 예보시간 시작일, 종료일, 시간 간격 (연 1y, 월 1m, 일 1d, 시간 1h, 분 1t, 초 1s)
                'srtDate': '2019-01-01',
                'endDate': '2019-01-04',
                # 'invDate': '1d',
                # 'srtDate': globalVar['srtDate'],
                # 'endDate': globalVar['endDate'],

                # 수행 목록
                # 'modelList': ['ACT', 'FOR'],
                'modelList': ['UMKR'],
                # 'modelList': [globalVar['modelList']],

                # 비동기 다중 프로세스 개수
                'cpuCoreNum': '5',
                # 'cpuCoreNum': globalVar['cpuCoreNum'],
                # 설정 파일
                'CFG': {
                    'siteInfo': '/DATA/PROP/SAMPLE/site_info.csv',
                    'umkrFileInfo': '/DATA/COLCT/UMKR/201901/01/UMKR_l015_unis_H00_201901011200.grb2',
                    # 'energy': {
                    #     'filePath': '/DATA/PROP/SAMPLE',
                    #     'fileName': 'energy.csv',
                    # },
                    # 'ulsanObs': {
                    #     'filePath': '/DATA/PROP/SAMPLE',
                    #     'fileName': 'ulsan_obs_data.csv',
                    # },
                    # 'ulsanFcst': {
                    #     'filePath': '/DATA/PROP/SAMPLE',
                    #     'fileName': 'ulsan_fcst_data.csv',
                    # },
                },

                'UMKR': {
                    'fileList': f"/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d*.grb2",
                    'type': 'FOR',
                    'invDate': '1d',
                    'varList': ['gridlat_0', 'gridlon_0', 'TMP_P0_L1_GLC0', 'RH_P0_L103_GLC0', 'UGRD_P0_L103_GLC0', 'VGRD_P0_L103_GLC0', 'VBDSF_P8_L1_GLC0_avg1h', 'LCDC_P0_L200_GLC0', 'MCDC_P0_L200_GLC0', 'HCDC_P0_L200_GLC0'],
                    'procList': ['lat', 'lon', 'TM', 'RH', 'U', 'V', 'SW_D', 'LCDC', 'MCDC', 'HCDC'],
                },

                # 'ACT': {
                #     'ASOS': {
                #         'searchFileList': f"/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d%H%M.grb2",
                #         'invDate': '6h',
                #     },
                #     'AWS': {
                #         'searchFileList': f"/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d%H%M.grb2",
                #         'invDate': '6h',
                #     },
                # },
            }

            # **************************************************************************************************************
            # 설정 파일 읽기
            # **************************************************************************************************************
            filePattern = sysOpt['CFG']['siteInfo']
            fileList = sorted(glob.glob(filePattern))
            if fileList is None or len(fileList) < 1:
                log.error(f"[ERROR] filePattern : {filePattern} / 파일을 확인해주세요.")
                raise Exception(f"[ERROR] filePattern : {filePattern} / 파일을 확인해주세요.")
            cfgData = pd.read_csv(fileList[0])
            cfgDataL1 = matchStnFor(sysOpt['CFG'], cfgData)

            # print('asdfasdf')

            # cfgDataL1['posRow']
            # cfgDataL1['posCol']

            # posDataL2 = cfgDataL1
            # for kk, posInfo in posDataL2.iterrows():
            #     posId = posInfo['Id']
            #     posLat = posInfo['Latitude']
            #     posLon = posInfo['Longitude']



            for modelType in sysOpt['modelList']:
                modelInfo = sysOpt.get(modelType)
                if modelInfo is None: continue

                # 시작일/종료일 설정
                dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
                dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
                dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=modelInfo['invDate'])

                for dtDateInfo in dtDateList:
                    log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                    filePattern = dtDateInfo.strftime(modelInfo['fileList'])
                    fileList = sorted(glob.glob(filePattern))
                    if len(fileList) < 1: continue

                    dsData = xr.Dataset()
                    for fileInfo in fileList:
                        try:
                            umData = xr.open_dataset(fileInfo, engine='pynio')
                            if len(umData) < 1: continue
                            log.info(f'[CHECK] fileInfo : {fileInfo}')

                            attrInfo = umData[list(umData.dtypes)[0]].attrs
                            anaDate = pd.to_datetime(attrInfo['initial_time'], format="%m/%d/%Y (%H:%M)")
                            forDate = anaDate + pd.DateOffset(hours=int(attrInfo['forecast_time'][0]))

                            import pygrib
                            grb = pygrib.open(fileInfo)



                            umDataL1 = umData.isel(ygrid_0=cfgDataL1['posRow'].astype(int).tolist(), xgrid_0=cfgDataL1['posCol'].astype(int).tolist())
                            umDataL2 = umDataL1[modelInfo['varList']]

                            umDataL3 = umDataL2.rename_dims({'ygrid_0': 'row', 'xgrid_0': 'col'})
                            renameItem = dict(zip(modelInfo['varList'], modelInfo['procList']))
                            umDataL4 = umDataL3.rename_vars(renameItem)

                            umDataL1['ygrid_0'].values
                            umDataL1['xgrid_0'].values
                            umDataL4['row'].values
                            umDataL4['col'].values

                            dsDataL1 = xr.Dataset(
                                {
                                    'uVec': (
                                    ('anaDate', 'time', 'lat', 'lon'), (uVec).reshape(1, 1, len(lat1D), len(lon1D)))
                                    , 'vVec': (
                                ('anaDate', 'forDate', 'lat', 'lon'), (vVec).reshape(1, 1, len(lat1D), len(lon1D)))
                                    , 'WD': (
                                ('anaDate', 'forDate', 'lat', 'lon'), (WD).reshape(1, 1, len(lat1D), len(lon1D)))
                                    , 'WS': (
                                ('anaDate', 'forDate', 'lat', 'lon'), (WS).reshape(1, 1, len(lat1D), len(lon1D)))
                                    , 'PA': (
                                ('anaDate', 'forDate', 'lat', 'lon'), (PA).reshape(1, 1, len(lat1D), len(lon1D)))
                                    , 'TA': (
                                ('anaDate', 'forDate', 'lat', 'lon'), (TA).reshape(1, 1, len(lat1D), len(lon1D)))
                                    , 'TD': (
                                ('anaDate', 'forDate', 'lat', 'lon'), (TD).reshape(1, 1, len(lat1D), len(lon1D)))
                                    , 'HM': (
                                ('anaDate', 'forDate', 'lat', 'lon'), (HM).reshape(1, 1, len(lat1D), len(lon1D)))
                                    , 'lowCA': (
                                ('anaDate', 'forDate', 'lat', 'lon'), (lowCA).reshape(1, 1, len(lat1D), len(lon1D)))
                                    , 'medCA': (
                                ('anaDate', 'forDate', 'lat', 'lon'), (medCA).reshape(1, 1, len(lat1D), len(lon1D)))
                                    , 'higCA': (
                                ('anaDate', 'forDate', 'lat', 'lon'), (higCA).reshape(1, 1, len(lat1D), len(lon1D)))
                                    , 'CA_TOT': (
                                ('anaDate', 'forDate', 'lat', 'lon'), (CA_TOT).reshape(1, 1, len(lat1D), len(lon1D)))
                                    , 'SS': (
                                ('anaDate', 'forDate', 'lat', 'lon'), (SS).reshape(1, 1, len(lat1D), len(lon1D)))
                                }
                                , coords={
                                    'anaDate': pd.date_range(anaDate, periods=1)
                                    , 'forDate': pd.date_range(forDate, periods=1)
                                    , 'lat': lat1D
                                    , 'lon': lon1D
                                }
                            )

                            dsData = xr.merge([dsData, umDataL4])
                        except Exception as e:
                            log.error(f'[ERROR] Exception : {str(e)}')


                        # for varName, procName in zip(modelInfo['varList'], modelInfo['procList']):
                        #     log.info(f'[CHECK] varName: {varName}, / procName : {procName}')

                        # varList = list(umDataL1.data_vars)
                        # for varInfo in varList:
                        #     log.info(f'[CHECK] varInfo : {varInfo} / {umDataL1[varInfo].long_name}')









                        # umDataL2['UGRD_P0_L103_GLC0'].plot()
                        # plt.show()

                        # umDataL2 = umData.sel(ygrid_0= cfgDataL1['posRow'].tolist(), xgrid_0= cfgDataL1['posCol'].tolist())
                        # umDataL3 = umDataL2.to_dataframe().dropna().reset_index(drop=True)


                    #   LDAPS-1.5K_UNIS:
                    #     varName:
                    #       - name: 'UGRD_P0_L103_GLC0'
                    #         level: [ '-1' ]
                    #         colName: [ 'U' ]
                    #       - name: 'VGRD_P0_L103_GLC0'
                    #         level: [ '-1' ]
                    #         colName: [ 'V' ]
                    #       - name: 'DSWRF_P8_L1_GLC0_avg1h'
                    #         level: [ '-1' ]
                    #         colName: [ 'SW_NET' ]
                    #       - name: 'CSUSF_P8_L103_GLC0_avg1h'
                    #         level: [ '-1' ]
                    #         colName: [ 'SW_DDNI' ]
                    #       - name: 'VBDSF_P8_L1_GLC0_avg1h'
                    #         level: [ '-1' ]
                    #         colName: [ 'SW_D' ]
                    #       - name: 'CFNSF_P8_L103_GLC0_avg1h'
                    #         level: [ '-1' ]
                    #         colName: [ 'SW_DDIF' ]
                    #       - name: 'TMP_P0_L1_GLC0'
                    #         level: [ '-1' ]
                    #         colName: [ 'TM' ]
                    #       - name: 'RH_P0_L103_GLC0'
                    #         level: [ '-1' ]
                    #         colName: [ 'RH' ]
                    #       - name: 'PRES_P0_L1_GLC0'
                    #         level: [ '-1' ]
                    #         colName: [ 'PRE' ]
                    #       - name: 'SNFALB_P0_L103_GLC0'
                    #         level: [ '-1' ]
                    #         colName: [ 'ALB' ]
                    #
                    #
                    # dbData['LON_SFC'] = data['XLONG'].values.tolist() if len(data['XLONG'].values) > 0 else None
                    # dbData['LAT_SFC'] = data['XLAT'].values.tolist() if len(data['XLAT'].values) > 0 else None
                    #
                    #
                    #
                    #         tmpADate = gribApp.getAttrValue(self.varNameLists[0]['name'], 'initial_time')
                    #         analDate = datetime.datetime.strptime(tmpADate, '%m/%d/%Y (%H:%M)')
                    #         tmpFDate = gribApp.getAttrValue(self.varNameLists[0]['name'], 'forecast_time')
                    #         tmpFHour = tmpFDate[0]
                    #         forcDate = analDate + datetime.timedelta(hours=int(tmpFHour))
                    #         common.logger.info(f'[CHECK] anaDate : {analDate} / forDate : {forcDate}')
                    #
                    #         # DB 등록/수정
                    #         self.dbData['ANA_DT'] = analDate
                    #         self.dbData['FOR_DT'] = forcDate
                    #         self.dbData['MODEL_TYPE'] = self.modelName
                    #
                    #         for vlist in self.varNameLists:
                    #             for idx, level in enumerate(vlist['level'], 0):
                    #                 try:
                    #                     if level == '-1':
                    #                         if len(gribApp.getVariable(vlist['name'])) < 1: continue
                    #                         self.dbData[vlist['colName'][idx]] = self.convFloatToIntList(gribApp.getVariable(vlist['name']))
                    #                     else:
                    #                         if len(gribApp.getVariable31(vlist['name'], idx)) < 1: continue
                    #                         self.dbData[vlist['colName'][idx]] = self.convFloatToIntList(gribApp.getVariable31(vlist['name'], idx))
                    #
                    #                 except Exception as e:
                    #                     common.logger.error(f'Exception : {e}')



                    # reqUrl = dtDateInfo.strftime(f"{modelInfo['request']['url']}").format(
                    #     tmfc=dtDateInfo.strftime('%Y%m%d%H'), ef=ef, authKey=modelInfo['request']['authKey'])
                    # res = requests.get(reqUrl)




                #     pool.apply_async(colctProc, args=(modelType, modelInfo, dtDateInfo))



            # # 시작일/종료일 설정
            # dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            # dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            # dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=modelInfo['invDate'])
            #
            # for dtDateInfo in dtDateList:
            #     for time in range(0, 88):
            #         ftime = f'{time:03}'
            #         log.info(f'[CHECK] dtDateInfo : {dtDateInfo} / ftime : {ftime}')
            #         pool.apply_async(subColct, args=(modelInfo, dtDateInfo, ftime))
            #
            #
            #
            # # log.info("[CHECK] posId (posLon, posLat) : {} ({}. {})".format(posId, posLon, posLat))
            # umData = dsDataL1
            # dtanaDateInfo = umData['anaDate'].values
            # # log.info("[CHECK] dtanaDateInfo : {}".format(dtanaDateInfo))
            #
            # try:
            #     umDataL2 = umData.sel(lat=posLat, lon=posLon, anaDate=dtanaDateInfo)
            #     umDataL3 = umDataL2.to_dataframe().dropna().reset_index(drop=True)
            #     # umDataL3['dtDate'] = pd.to_datetime(dtanaDateInfo) + (umDataL3.index.values * datetime.timedelta(hours=1))
            #     umDataL3['DATE_TIME'] = pd.to_datetime(dtanaDateInfo) + (validIdx * datetime.timedelta(hours=1))
            #     # umDataL3['dtDateKst'] = umDataL3.index.tz_localize(tzUtc).tz_convert(tzKst)
            #     umDataL3['DATE_TIME_KST'] = umDataL3['DATE_TIME'] + dtKst
            #     umDataL4 = umDataL3.rename({'SS': 'SWR'}, axis='columns')
            #     umDataL5 = umDataL4[
            #         ['DATE_TIME_KST', 'DATE_TIME', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR']]
            #     umDataL5['SRV'] = 'SRV{:05d}'.format(posId)
            #     umDataL5['TA'] = umDataL5['TA'] - 273.15
            #     umDataL5['TD'] = umDataL5['TD'] - 273.15
            #     umDataL5['PA'] = umDataL5['PA'] / 100.0
            #     umDataL5['CA_TOT'] = np.where(umDataL5['CA_TOT'] < 0, 0, umDataL5['CA_TOT'])
            #     umDataL5['CA_TOT'] = np.where(umDataL5['CA_TOT'] > 1, 1, umDataL5['CA_TOT'])
            #
            #     solPosInfo = pvlib.solarposition.get_solarposition(pd.to_datetime(umDataL5['DATE_TIME'].values),
            #                                                        posLat, posLon,
            #                                                        pressure=umDataL5['PA'].values * 100.0,
            #                                                        temperature=umDataL5['TA'].values,
            #                                                        method='nrel_numpy')
            #     umDataL5['SZA'] = solPosInfo['apparent_zenith'].values
            #     umDataL5['AZA'] = solPosInfo['azimuth'].values
            #     umDataL5['ET'] = solPosInfo['equation_of_time'].values
            #     umDataL5['ANA_DATE'] = pd.to_datetime(dtanaDateInfo)
            #
            #     # pvlib.location.Location.get_clearsky()
            #     site = location.Location(posLat, posLon, tz='Asia/Seoul')
            #     clearInsInfo = site.get_clearsky(pd.to_datetime(umDataL5['DATE_TIME'].values))
            #     umDataL5['GHI_CLR'] = clearInsInfo['ghi'].values
            #     umDataL5['DNI_CLR'] = clearInsInfo['dni'].values
            #     umDataL5['DHI_CLR'] = clearInsInfo['dhi'].values
            #
            #     # poaInsInfo = irradiance.get_total_irradiance(
            #     #     surface_tilt=posSza,
            #     #     surface_azimuth=posAza,
            #     #     dni=clearInsInfo['dni'],
            #     #     ghi=clearInsInfo['ghi'],
            #     #     dhi=clearInsInfo['dhi'],
            #     #     solar_zenith=solPosInfo['apparent_zenith'].values,
            #     #     solar_azimuth=solPosInfo['azimuth'].values
            #     # )
            #
            #     # umDataL5['GHI_POA'] = poaInsInfo['poa_global'].values
            #     # umDataL5['DNI_POA'] = poaInsInfo['poa_direct'].values
            #     # umDataL5['DHI_POA'] = poaInsInfo['poa_diffuse'].values
            #
            #     # 혼탁도
            #     turbidity = pvlib.clearsky.lookup_linke_turbidity(pd.to_datetime(umDataL5['DATE_TIME'].values),
            #                                                       posLat, posLon, interp_turbidity=True)
            #     umDataL5['TURB'] = turbidity.values





            #   LDAPS-1.5K_UNIS:
            #     varName:
            #       - name: 'UGRD_P0_L103_GLC0'
            #         level: [ '-1' ]
            #         colName: [ 'U' ]
            #       - name: 'VGRD_P0_L103_GLC0'
            #         level: [ '-1' ]
            #         colName: [ 'V' ]
            #       - name: 'DSWRF_P8_L1_GLC0_avg1h'
            #         level: [ '-1' ]
            #         colName: [ 'SW_NET' ]
            #       - name: 'CSUSF_P8_L103_GLC0_avg1h'
            #         level: [ '-1' ]
            #         colName: [ 'SW_DDNI' ]
            #       - name: 'VBDSF_P8_L1_GLC0_avg1h'
            #         level: [ '-1' ]
            #         colName: [ 'SW_D' ]
            #       - name: 'CFNSF_P8_L103_GLC0_avg1h'
            #         level: [ '-1' ]
            #         colName: [ 'SW_DDIF' ]
            #       - name: 'TMP_P0_L1_GLC0'
            #         level: [ '-1' ]
            #         colName: [ 'TM' ]
            #       - name: 'RH_P0_L103_GLC0'
            #         level: [ '-1' ]
            #         colName: [ 'RH' ]
            #       - name: 'PRES_P0_L1_GLC0'
            #         level: [ '-1' ]
            #         colName: [ 'PRE' ]
            #       - name: 'SNFALB_P0_L103_GLC0'
            #         level: [ '-1' ]
            #         colName: [ 'ALB' ]

            # **************************************************************************************************************
            # 비동기 다중 프로세스 수행
            # **************************************************************************************************************
            # # 비동기 다중 프로세스 개수
            # pool = Pool(int(sysOpt['cpuCoreNum']))
            #
            # for modelType in sysOpt['modelList']:
            #     modelInfo = sysOpt.get(modelType)
            #     if modelInfo is None: continue
            #
            #     for dataType in modelInfo.keys():
            #         dataInfo = modelInfo.get(dataType)
            #
            #         # 시작일/종료일 설정
            #         dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            #         dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            #         dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=dataInfo['invDate'])
            #
            #         for dtDateInfo in dtDateList:
            #             # log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
            #             pool.apply_async(propProc, args=(dataType, dataInfo, dtDateInfo))
            #
            # pool.close()
            # pool.join()

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
