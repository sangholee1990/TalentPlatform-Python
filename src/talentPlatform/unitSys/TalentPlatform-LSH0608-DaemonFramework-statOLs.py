# ================================================
# 요구사항
# ================================================
# cd /data2/hzhenshao/EMI
# /data2/hzhenshao/EMI/py38/bin/python3 TalentPlatform-LSH0608-DaemonFramework-statOLs.py
# nohup /data2/hzhenshao/EMI/py38/bin/python3 TalentPlatform-LSH0608-DaemonFramework-statOLs.py &
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
                , 'typeList': ['EC', 'GDP', 'landscan', 'Land_Cover_Type_1_Percent']
                # , 'typeList': ['EC']
                # , 'typeList': ['landscan']
                # , 'typeList': ['Land_Cover_Type_1_Percent']

                # , 'keyList': ['CH4', 'CO2_excl', 'CO2_org', 'N2O', 'NH3', 'NMVOC', 'OC', 'NH3', 'SO2']
                # , 'keyList': ['emi_co', 'emi_n2o', 'emi_nh3', 'emi_nmvoc', 'emi_nox', 'emi_oc', 'emi_so2']
                # , 'keyList': ['emi_nmvoc']
                # , 'keyList': ['N2O', 'GHG', 'CO2', 'CO2bio', 'CH4']
                , 'keyList': ['SO2', 'N2O', 'CH4', 'NMVOC', 'NOx', 'NH3', 'CO', 'PM10', 'PM2.5', 'OC', 'BC']
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

                # **********************************************************************************************************
                # 화소별 회귀계수 계산
                # **********************************************************************************************************
                for keyInfo in sysOpt['keyList']:

                    saveFile = '{}/{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, 'STAT', dateInfo, 'statOls', keyInfo)
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    if len(glob.glob(saveFile)) > 0: continue

                    coreNum = int(os.cpu_count() * 0.50)
                    log.info(f'[CHECK] coreNum : {coreNum}')
                    client = Client(n_workers=coreNum)
                    dask.config.set(scheduler='processes')

                    statOlsRes = xr.apply_ufunc(
                        statOls,
                        data[keyInfo],
                        data['landscan'],
                        data['GDP'],
                        data['Land_Cover_Type_1_Percent'],
                        data['EC'],
                        input_core_dims=[['time'], ['time'], ['time'], ['time'], ['time']],
                        output_core_dims=[['coef']],
                        exclude_dims=set(('time',)),
                        output_sizes={'coef': 5},
                        # vectorize=False,
                        vectorize=True,
                        dask="parallelized",
                        output_dtypes=[float],
                    )

                    # 계수 이름 지정
                    statOlsRes = statOlsRes.assign_coords(coef=['a', 'b', 'c', 'd', 'e'])
                    statOlsRes = statOlsRes.expand_dims(period=[dateInfo])

                    # alpha 값 계산
                    # statOlsRes['alpha'] = np.exp(statOlsRes.sel(coef='lnAlpha'))
                    # statOlsRes['alpha'] = np.exp(statOlsRes.sel(coef='lnAlpha', drop=True))
                    log.info(f'[CHECK] statOlsRes : {statOlsRes}')

                    # 파일 저장
                    statOlsRes.to_netcdf(saveFile)
                    log.info(f'[CHECK] saveFile : {saveFile}')

                    # client.close()

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
