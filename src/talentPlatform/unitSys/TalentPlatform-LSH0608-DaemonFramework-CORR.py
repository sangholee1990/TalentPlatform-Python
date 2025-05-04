# ================================================
# 요구사항
# ================================================
# Python 이용한 CO2 및 CH4 자료 처리 및 연도별 저장

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
import dask.array as da
import dask
from dask.distributed import Client

from scipy.stats import kendalltau
from plotnine import ggplot, aes, geom_boxplot
import gc

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

def calcMannKendall(x):
    try:
        result = mk.original_test(x)
        return result.Tau
        # return result.trend, result.p, result.Tau

    except Exception:
        return np.nan
        # return np.nan, np.nan, np.nan

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

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
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': '2001-01-01'
                , 'endDate': '2018-01-01'

                # 경도 최소/최대/간격
                , 'lonMin': -180
                , 'lonMax': 180
                , 'lonInv': 0.1

                # 위도 최소/최대/간격
                , 'latMin': -90
                , 'latMax': 90
                , 'latInv': 0.1

                , 'typeList': ['EC', 'GDP', 'Land_Cover_Type_1_Percent', 'landscan']
                # , 'typeList': ['EC']
                # , 'typeList': ['landscan']
                # , 'typeList': ['Land_Cover_Type_1_Percent']

                # , 'keyList': ['CH4', 'CO2_excl', 'CO2_org', 'N2O', 'NH3', 'NMVOC', 'OC', 'NH3', 'SO2']
                , 'keyList': ['emi_co', 'emi_n2o', 'emi_nh3', 'emi_nmvoc', 'emi_nox', 'emi_oc', 'emi_so2']
                # , 'keyList': ['emi_nmvoc']
            }


            # for i, keyInfo in enumerate(sysOpt['keyList']):
            #     log.info("[CHECK] keyInfo : {}".format(keyInfo))
            #
            #     inpFile = '{}/{}/*{}*.nc'.format(globalVar['outPath'], serviceName, keyInfo)
            #     fileList = sorted(glob.glob(inpFile))
            #
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #         continue

            # fileInfo = fileList[0]
            # log.info('[CHECK] fileInfo : {}'.format(fileInfo))

            # data = xr.open_mfdataset(fileInfo)

            # inpFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, '*_1990-2021')
            # inpFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, '*')
            inpFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, '*')
            fileList = sorted(glob.glob(inpFile))

            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                # continue

            # data = xr.open_mfdataset(fileList, chunks={'time': 10, 'lat': 10, 'lon': 10}).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
            data = xr.open_mfdataset(fileList).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))

            # **********************************************************************************************************
            # 피어슨 상관계수 계산
            # **********************************************************************************************************
            # for i, typeInfo in enumerate(sysOpt['typeList']):
            #     for j, keyInfo in enumerate(sysOpt['keyList']):
            #         log.info(f'[CHECK] typeInfo : {typeInfo} / keyInfo : {keyInfo}')
            #
            #         saveFile = '{}/{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, 'CORR', 'corr', typeInfo, keyInfo)
            #         fileChkList = glob.glob(saveFile)
            #         # if (len(fileChkList) > 0): continue
            #
            #         var1 = data[typeInfo]
            #         var2 = data[keyInfo]
            #
            #         # np.nanmin(var1)
            #         # np.nanmax(var1)
            #
            #         # np.nanmin(var2)
            #         # np.nanmax(var2)
            #
            #         cov = ((var1 - var1.mean(dim='time', skipna=True)) * (var2 - var2.mean(dim='time', skipna=True))).mean(dim='time', skipna=True)
            #         stdVar1 = var1.std(dim='time', skipna=True)
            #         stdVar2 = var2.std(dim='time', skipna=True)
            #
            #         # 0값일 경우 결측값 처리
            #         stdVar1 = xr.where((stdVar1 == 0), np.nan, stdVar1)
            #         stdVar2 = xr.where((stdVar2 == 0), np.nan, stdVar2)
            #
            #         peaCorr = cov / (stdVar1 * stdVar2)
            #         peaCorr = peaCorr.rename(f'{typeInfo}_{keyInfo}')
            #
            #         # 0값일 경우 결측값 처리
            #         peaCorr = xr.where(peaCorr > 1, np.nan, peaCorr)
            #         peaCorr = xr.where(peaCorr < -1, np.nan, peaCorr)
            #
            #         # log.info(f'min ~ max : {np.nanmin(peaCorr)} ~ {np.nanmax(peaCorr)}')
            #
            #         # EC, CO
            #         # 6480000만개 중에서 134개 발생
            #         # dd = peaCorr.to_dataframe().reset_index(drop=False)
            #         # filtered_df = dd[(dd['EC_emi_co'] > 1) | (dd['EC_emi_co'] < -1)]
            #         # -16.20000,123.60000,-1.80950
            #
            #         saveImg = '{}/{}/{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, 'CORR', 'corr', typeInfo, keyInfo)
            #         os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #         peaCorr.plot(vmin=-1.0, vmax=1.0)
            #         plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #         plt.tight_layout()
            #         # plt.show()
            #         plt.close()
            #         log.info(f'[CHECK] saveImg : {saveImg}')
            #
            #         os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #         peaCorr.to_netcdf(saveFile)
            #         log.info(f'[CHECK] saveFile : {saveFile}')
            #
            #         # 데이터셋 닫기 및 메모리에서 제거
            #         var1.close(), var2.close(), cov.close(), stdVar1.close(), stdVar2.close(), peaCorr.close()
            #         del var1, var2, cov, stdVar1, stdVar2, peaCorr
            #
            #         # 가비지 수집기 강제 실행
            #         # gc.collect()

            # **********************************************************************************************************
            # 온실가스 배출량 계산
            # **********************************************************************************************************
            # for i, keyInfo in enumerate(sysOpt['keyList']):
            #     log.info(f'[CHECK] keyInfo : {keyInfo}')
            #
            #     var = data[keyInfo]
            #
            #     meanData = var.mean(dim=('time'), skipna=True)
            #     # meanData = meanData.where(meanData > 0)
            #     meanData = meanData.where(meanData != 0)
            #
            #     meanDataL1 = np.log10(meanData)
            #
            #     saveImg = '{}/{}/{}/{}.png'.format(globalVar['figPath'], serviceName, 'EMI', keyInfo)
            #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #     meanDataL1.plot()
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.tight_layout()
            #     # plt.show()
            #     plt.close()
            #     log.info(f'[CHECK] saveImg : {saveImg}')
            #
            #     saveFile = '{}/{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'EMI', keyInfo)
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     meanDataL1.to_netcdf(saveFile)
            #     log.info(f'[CHECK] saveFile : {saveFile}')

            # **********************************************************************************************************
            # Mann-Kendall 계산
            # **********************************************************************************************************
            # for i, keyInfo in enumerate(sysOpt['keyList']):
            #     log.info(f'[CHECK] keyInfo : {keyInfo}')
            #
            #     var = data[keyInfo]
            #
            #     client = Client(n_workers=os.cpu_count(), threads_per_worker=os.cpu_count())
            #     dask.config.set(scheduler='processes')
            #
            #     mannKendall = xr.apply_ufunc(
            #         calcMannKendall,
            #         var,
            #         input_core_dims=[['time']],
            #         output_core_dims=[[]],
            #         vectorize=True,
            #         dask='parallelized',
            #         output_dtypes=[np.float64],
            #         dask_gufunc_kwargs={'allow_rechunk': True}
            #     ).compute()
            #
            #     saveImg = '{}/{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, 'MANN', 'mann', keyInfo)
            #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #     mannKendall.plot(vmin=-1.0, vmax=1.0)
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.tight_layout()
            #     # plt.show()
            #     plt.close()
            #     log.info(f'[CHECK] saveImg : {saveImg}')
            #
            #     saveFile = '{}/{}/{}/{}_{}.nc'.format(globalVar['outPath'], serviceName, 'MANN', 'mann', keyInfo)
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     mannKendall.to_netcdf(saveFile)
            #     log.info(f'[CHECK] saveFile : {saveFile}')
            #
            #     client.close()

            # **********************************************************************************************************
            # Mann Kendall 상자 그림
            # **********************************************************************************************************
            # inpFile = '{}/{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'MANN', '*')
            # fileList = sorted(glob.glob(inpFile))
            #
            # if fileList is None or len(fileList) < 1:
            #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            # data = xr.open_mfdataset(fileList)
            # dataL1 = data.to_dataframe().reset_index(drop=True)
            # dataL1.columns = dataL1.columns.str.replace('emi_', '')
            #
            # dataL2 = pd.melt(dataL1, id_vars=[], var_name='key', value_name='val')
            #
            # mainTitle = '{}'.format('EDGAR Mann-Kendall Trend (2001~2018)')
            # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #
            # sns.set_style("whitegrid")
            # sns.set_palette(sns.color_palette("husl", len(dataL1.columns)))
            # sns.boxplot(x='key', y='val', data=dataL2, dodge=False, hue='key')
            # plt.xlabel(None)
            # plt.ylabel('Mann-Kendall Trend')
            # plt.title(mainTitle)
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, title=None)
            # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            # plt.tight_layout()
            # plt.show()
            # plt.close()
            # log.info(f'[CHECK] saveImg : {saveImg}')

            # **********************************************************************************************************
            # typeList에 따른 상자 그림
            # **********************************************************************************************************
            for i, typeInfo in enumerate(sysOpt['typeList']):
                log.info(f'[CHECK] typeInfo : {typeInfo}')

                inpFile = '{}/{}/{}/*{}*.nc'.format(globalVar['outPath'], serviceName, 'CORR', typeInfo)
                fileList = sorted(glob.glob(inpFile))

                if fileList is None or len(fileList) < 1:
                    log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

                data = xr.open_mfdataset(fileList)
                dataL1 = data.to_dataframe().reset_index(drop=True)
                dataL1.columns = dataL1.columns.str.replace(f'{typeInfo}-emi_', '').str.replace(f'{typeInfo}_emi_', '')

                dataL2 = pd.melt(dataL1, id_vars=[], var_name='key', value_name='val')

                mainTitle = f'EDGAR Pearson-Corr {typeInfo} (2001~2018)'
                saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)

                sns.set_style("whitegrid")
                sns.set_palette(sns.color_palette("husl", len(dataL1.columns)))
                sns.boxplot(x='key', y='val', data=dataL2, dodge=False, hue='key')
                plt.xlabel(None)
                plt.ylabel('Pearson-Corr')
                plt.title(mainTitle)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, title=None)
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                plt.tight_layout()
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