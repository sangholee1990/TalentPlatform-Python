# ================================================
# 요구사항
# ================================================
# bash
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
                pass
                # globalVar['inpPath'] = '/DATA/INPUT'
                # globalVar['outPath'] = '/DATA/OUTPUT'
                # globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                'dateList': {
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
                , 'typeList': ['landscan', 'GDP', 'Land_Cover_Type_1_Percent', 'EC']
                , 'coefList': ['b', 'c', 'd', 'e']
                , 'keyList': ['SO2', 'N2O', 'CH4', 'NMVOC', 'NOx', 'NH3', 'CO', 'PM10', 'PM2.5', 'OC', 'BC']
            }

            for dateInfo in sysOpt['dateList']:
                try:
                    inpFile = '{}/{}/{}.nc'.format(globalVar['inpPath'], serviceName, 'EDGAR2-*')
                    fileList = sorted(glob.glob(inpFile))

                    if fileList is None or len(fileList) < 1:
                        log.error(f"파일 없음 : {inpFile}")
                        continue

                    log.info(f'[CHECK] dateInfo : {dateInfo}')
                    srtDate = sysOpt['dateList'][dateInfo]['srtDate']
                    endDate = sysOpt['dateList'][dateInfo]['endDate']
                    data = xr.open_mfdataset(fileList).sel(time=slice(srtDate, endDate))

                    # log.info(f'[CHECK] fileList : {fileList}')
                    # log.info(f'[CHECK] srtDate : {srtDate}')
                    # log.info(f'[CHECK] endDate : {endDate}')

                    # 회귀계수
                    for keyInfo in sysOpt['keyList']:
                        inpFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, f"STAT/{dateInfo}_statOls_{keyInfo}")
                        fileList = sorted(glob.glob(inpFile))

                        if fileList is None or len(fileList) < 1:
                            log.error(f"파일 없음 : {inpFile}")
                            continue

                        log.info(f'[CHECK] keyInfo : {keyInfo}')
                        coefData = xr.open_mfdataset(fileList)
                        # coefData = xr.open_dataset(fileList)
                        coefData = coefData.rename({'__xarray_dataarray_variable__': 'coefVar'})

                        # 테스트
                        # 'landscan', 'GDP', 'Land_Cover_Type_1_Percent', 'EC'
                        # coefData['coefVar'].loc[dict(period='2010-2019', coef='b')] = 0.5
                        # coefData['coefVar'].loc[dict(period='2010-2019', coef='c')] = 0.8
                        # coefData['coefVar'].loc[dict(period='2010-2019', coef='d')] = 0.3
                        # coefData['coefVar'].loc[dict(period='2010-2019', coef='e')] = -0.6
                        #
                        # data['landscan'].loc[dict(time='2010')] = 1.2
                        # data['GDP'].loc[dict(time='2010')] = 10
                        # data['Land_Cover_Type_1_Percent'].loc[dict(time='2010')] = 50
                        # data['EC'].loc[dict(time='2010')] = 0.6
                        # data['BC'].loc[dict(time='2010')] = 7.2
                        #
                        # data['landscan'].loc[dict(time='2019')] = 1.5
                        # data['GDP'].loc[dict(time='2019')] = 15
                        # data['Land_Cover_Type_1_Percent'].loc[dict(time='2019')] = 60
                        # data['EC'].loc[dict(time='2019')] = 0.5
                        # data['BC'].loc[dict(time='2019')] = 9.0

                        dataL1 = xr.merge([data, coefData])
                        srtYear = pd.to_datetime(np.min(dataL1['time'].values)).strftime('%Y')
                        endYear = pd.to_datetime(np.max(dataL1['time'].values)).strftime('%Y')

                        srtData = dataL1.sel(time=srtYear).isel(time=0)
                        endData = dataL1.sel(time=endYear).isel(time=0)

                        # 1단계 대수평균 계산
                        srtKeyData = srtData[keyInfo]
                        endKeyData = endData[keyInfo]

                        calcKeyData = xr.where(
                            endKeyData == srtKeyData,
                            np.nan,
                            (endKeyData - srtKeyData) / (np.log(endKeyData) - np.log(srtKeyData))
                        )
                        # calcKeyData = (endKeyData - srtKeyData) / (np.log(endKeyData) - np.log(srtKeyData))
                        # calcKeyData.isel(lat=0, lon=0).values

                        # 2단계 초기 요인별 기여도 계산
                        factorList = sysOpt['typeList']
                        coefList = sysOpt['coefList']
                        matList = dict(zip(factorList, coefList))

                        mrgDict = {}
                        for factor in factorList:
                            srtFacData = srtData[factor]
                            endFacData = endData[factor]

                            coefVal  = matList[factor]
                            coefDataL1 = coefData.sel(coef=coefVal, period=dateInfo)['coefVar']

                            ratio = xr.where(
                                (srtFacData > 0) & (endFacData > 0),
                                np.log(endFacData / srtFacData),
                                np.nan
                            )
                            mrgDict[factor] = coefDataL1 * calcKeyData * ratio

                        # 3단계 기여도 합산
                        sumData = xr.concat(list(mrgDict.values()), dim='factor').sum(dim='factor', skipna=True)
                        diffKeyData = endKeyData - srtKeyData
                        # sumData.isel(lat=0, lon=0).values
                        # diffKeyData.isel(lat=0, lon=0).values

                        # 실질 기여도 및 백분율 기여도 계산
                        residRatio = xr.where(sumData != 0, diffKeyData / sumData, 0)
                        comData = {}
                        for factor, calcDelta in mrgDict.items():

                            con = calcDelta * residRatio
                            delList = ['coef', 'period', 'time']
                            for delInfo in delList:
                                try:
                                    con = con.drop_vars(delInfo)
                                except Exception as e:
                                    pass
                            comData[f"con-{keyInfo}-{factor}"] = con

                            percentage_delta = xr.where(srtKeyData != 0, (con / srtKeyData) * 100, 0)
                            delList = ['coef', 'period', 'time']
                            for delInfo in delList:
                                try:
                                    percentage_delta = percentage_delta.drop_vars(delInfo)
                                except Exception as e:
                                    pass
                            comData[f"per-{keyInfo}-{factor}"] = percentage_delta

                        comDataL1 = xr.Dataset(comData)
                        # comDataL1.isel(lat=0, lon=0).values
                        # comDataL1['con-landscan'].isel(lat=0, lon=0).values
                        # comDataL1['per-landscan'].isel(lat=0, lon=0).values

                        saveFile = '{}/{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, 'LDMI', 'statLdmi', keyInfo, dateInfo)
                        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                        comDataL1.to_netcdf(saveFile)
                        log.info(f'[CHECK] saveFile : {saveFile}')
                except Exception as e:
                    log.error(f"Exception : {e}")
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
