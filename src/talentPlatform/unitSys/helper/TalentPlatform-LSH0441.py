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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns


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


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 3차원 극한가뭄 빈도수 및 상자그림

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
    serviceName = 'LSH0441'

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

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'dateList': {
                        'all': ('2000-01-01', '2100-12-31')
                        , 'case': ('2031-01-01', '2065-12-31')
                        , 'case2': ('2066-01-01', '2100-12-31')
                    }

                    # 목록
                    , 'keyList': ['ACCESS-CM2']

                    # 가뭄 목록
                    , 'drgCondList': {
                        'EW': (2.0, 4.0)
                        , 'VW': (1.50, 1.99)
                        , 'MW': (1.00, 1.49)
                        , 'NN': (-0.99, 0.99)
                        , 'MD': (-1.49, -1.00)
                        , 'SD': (-1.99, -1.50)
                        , 'ED': (-4.00, -2.00)
                    }

                    # 극한 가뭄값
                    , 'extDrgVal': -2
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'dateList': {
                        'all': ('2000-01-01', '2100-12-31')
                        , 'case': ('2031-01-01', '2065-12-31')
                        , 'case2': ('2066-01-01', '2100-12-31')
                    }

                    # 목록
                    , 'keyList': ['ACCESS-CM2']

                    # 가뭄 목록
                    , 'drgCondList': {
                        'EW': (2.0, 4.0)
                        , 'VW': (1.50, 1.99)
                        , 'MW': (1.00, 1.49)
                        , 'NN': (-0.99, 0.99)
                        , 'MD': (-1.49, -1.00)
                        , 'SD': (-1.99, -1.50)
                        , 'ED': (-4.00, -2.00)
                    }

                    # 극한 가뭄값
                    , 'extDrgVal': -2
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # ********************************************************************
            # 대륙별 분류 전처리
            # ********************************************************************
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'detailCont.csv')
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(fileList, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(fileList, '입력 자료를 확인해주세요.'))

            contData = pd.read_csv(fileList[0]).rename(columns={'type': 'idx'})
            contDataL1 = contData[['lon', 'lat', 'isLand', 'idx']]
            contDataL2 = contDataL1.set_index(['lat', 'lon'])
            contDataL3 = contDataL2.to_xarray()

            # ********************************************************************
            # 가뭄 전처리
            # ********************************************************************
            for i, keyInfo in enumerate(sysOpt['keyList']):
                log.info(f"[CHECK] keyInfo : {keyInfo}")

                inpFile = '{}/{}/{}/*{}*.nc'.format(globalVar['inpPath'], serviceName, keyInfo, keyInfo)
                fileList = sorted(glob.glob(inpFile))

                if fileList is None or len(fileList) < 1:
                    log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                    continue

                log.info(f'[CHECK] fileList : {fileList}')

                orgData = xr.open_mfdataset(fileList)

                # ***************************************************************************
                # 극한 가뭄에 따른 빈도수 계산
                # ***************************************************************************
                for dateInfo, (srtDate, endDate) in sysOpt['dateList'].items():
                    log.info(f"[CHECK] dateInfo : {dateInfo}")

                    data = orgData.sel(time=slice(srtDate, endDate))

                    # 극한 가뭄
                    dataL1 = data.where(data <= sysOpt['extDrgVal']).count(dim='time')
                    dataL1 = dataL1.where(dataL1 != 0)

                    # dataL2 = dataL1
                    dataL2 = xr.merge([dataL1, contDataL3])

                    for k, varInfo in enumerate(dataL2.data_vars.keys()):
                        # log.info(f'[CHECK] varInfo : {varInfo}')
                        saveImg = '{}/{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, keyInfo, dateInfo, 'ins', varInfo)
                        # if os.path.exists(saveImg): continue
                        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                        dataL2[varInfo].plot()
                        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                        plt.tight_layout()
                        # plt.show()
                        plt.close()
                        log.info(f'[CHECK] saveImg : {saveImg}')

                    # NetCDF 자료 저장
                    saveFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, keyInfo, dateInfo, 'ins')
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    dataL2.to_netcdf(saveFile)
                    log.info(f'[CHECK] saveFile : {saveFile}')

                    # CSV 자료 저장
                    saveFile = '{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, keyInfo, dateInfo, 'ins')
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    dataL2.to_dataframe().reset_index(drop=False).to_csv(saveFile, index=False)
                    log.info(f'[CHECK] saveFile : {saveFile}')

                    dataL3 = dataL2
                    dataL4 = dataL3.to_dataframe().reset_index(drop=False)
                    dataL4.columns = dataL4.columns.to_series().replace(
                        {
                            'spei_gamma_': 'gam'
                            , 'spei_pearson_': 'pea'
                        }
                        , regex=True
                    )

                    dataL5 = pd.melt(dataL4, id_vars=['lat', 'lon'], var_name='key', value_name='val')

                    # CSV 자료 저장
                    saveFile = '{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, keyInfo, dateInfo, 'ins')
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    dataL5.to_csv(saveFile, index=False)
                    log.info(f'[CHECK] saveFile : {saveFile}')

                    dataL6 = dataL5

                    # 시각화 저장
                    saveImg = '{}/{}/{}_{}_{}_{}.png'.format(globalVar['figPath'], serviceName, keyInfo, dateInfo, 'ins', 'boxplot')
                    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                    sns.boxplot(
                        x='key', y='val', hue='key', data=dataL6, showmeans=True, width=0.5, dodge=False
                        , meanprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 3}
                    )
                    plt.xticks(rotation=45, ha='right')
                    plt.xlabel(None)
                    plt.ylabel(None)
                    plt.legend([], [], frameon=False, title=None)
                    plt.tight_layout()
                    plt.savefig(saveImg, dpi=600, transparent=True)
                    # plt.show()
                    plt.close()
                    log.info(f'[CHECK] saveImg : {saveImg}')

                # ***************************************************************************
                # 구간에 따른 빈도수 계산
                # ***************************************************************************
                for dateInfo, (srtDate, endDate) in sysOpt['dateList'].items():
                    log.info(f"[CHECK] dateInfo : {dateInfo}")

                    data = orgData.sel(time=slice(srtDate, endDate))

                    # 동적으로 생성
                    lat1D = data['lat'].values
                    lon1D = data['lon'].values

                    for drgCond, (srtVal, endVal) in sysOpt['drgCondList'].items():
                        log.info(f"[CHECK] drgCond : {drgCond}")
                        dataL1 = data.where((data >= srtVal) & (data <= endVal)).count(dim='time')
                        dataL1 = dataL1.where(dataL1 != 0)

                        dataL2 = xr.Dataset(
                            {
                                'spei_gamma_03': (('type', 'lat', 'lon'), (dataL1['spei_gamma_03'].values.reshape(1, len(lat1D), len(lon1D))))
                                , 'spei_gamma_06': (('type', 'lat', 'lon'), (dataL1['spei_gamma_06'].values.reshape(1, len(lat1D), len(lon1D))))
                                , 'spei_gamma_09': (('type', 'lat', 'lon'), (dataL1['spei_gamma_09'].values.reshape(1, len(lat1D), len(lon1D))))
                                , 'spei_gamma_12': (('type', 'lat', 'lon'), (dataL1['spei_gamma_12'].values.reshape(1, len(lat1D), len(lon1D))))
                                , 'spei_gamma_18': (('type', 'lat', 'lon'), (dataL1['spei_gamma_18'].values.reshape(1, len(lat1D), len(lon1D))))
                                , 'spei_gamma_24': (('type', 'lat', 'lon'), (dataL1['spei_gamma_24'].values.reshape(1, len(lat1D), len(lon1D))))
                                , 'spei_pearson_03': (('type', 'lat', 'lon'), (dataL1['spei_pearson_03'].values.reshape(1, len(lat1D), len(lon1D))))
                                , 'spei_pearson_06': (('type', 'lat', 'lon'), (dataL1['spei_pearson_06'].values.reshape(1, len(lat1D), len(lon1D))))
                                , 'spei_pearson_09': (('type', 'lat', 'lon'), (dataL1['spei_pearson_09'].values.reshape(1, len(lat1D), len(lon1D))))
                                , 'spei_pearson_12': (('type', 'lat', 'lon'), (dataL1['spei_pearson_12'].values.reshape(1, len(lat1D), len(lon1D))))
                                , 'spei_pearson_18': (('type', 'lat', 'lon'), (dataL1['spei_pearson_18'].values.reshape(1, len(lat1D), len(lon1D))))
                                , 'spei_pearson_24': (('type', 'lat', 'lon'), (dataL1['spei_pearson_24'].values.reshape(1, len(lat1D), len(lon1D))))
                            }
                            , coords={
                                'lon': lon1D
                                , 'lat': lat1D
                                , 'type': [drgCond]
                            }
                        )

                        dataL2 = xr.merge([dataL2, contDataL3])

                        # 각 변수마다 시각화
                        for k, varInfo in enumerate(dataL2.data_vars.keys()):
                            # log.info(f'[CHECK] varInfo : {varInfo}')
                            saveImg = '{}/{}/{}_{}_{}_{}.png'.format(globalVar['figPath'], serviceName, keyInfo, dateInfo, drgCond, varInfo)
                            # if os.path.exists(saveImg): continue
                            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                            dataL2.sel(type = drgCond)[varInfo].plot()
                            plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                            plt.tight_layout()
                            # plt.show()
                            plt.close()
                            log.info(f'[CHECK] saveImg : {saveImg}')

                        # NetCDF 자료 저장
                        saveFile = '{}/{}/{}_{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, keyInfo, dateInfo, drgCond, 'cnt')
                        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                        dataL2.to_netcdf(saveFile)
                        log.info(f'[CHECK] saveFile : {saveFile}')

                        # CSV 자료 저장
                        saveFile = '{}/{}/{}_{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, keyInfo, dateInfo, drgCond, 'cnt')
                        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                        dataL2.to_dataframe().reset_index(drop=False).to_csv(saveFile, index=False)
                        log.info(f'[CHECK] saveFile : {saveFile}')

                for dateInfo, (srtDate, endDate) in sysOpt['dateList'].items():
                    log.info(f"[CHECK] dateInfo : {dateInfo}")

                    inpFile = '{}/{}/{}*{}*{}*.nc'.format(globalVar['outPath'], serviceName, keyInfo, dateInfo, 'cnt')
                    fileList2 = sorted(glob.glob(inpFile))

                    if fileList2 is None or len(fileList2) < 1:
                        log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                        continue

                    log.info(f'[CHECK] fileList2 : {fileList2}')

                    dataL3 = xr.open_mfdataset(fileList2)

                    dataL4 = dataL3.to_dataframe().reset_index(drop=False)
                    dataL4.columns = dataL4.columns.to_series().replace(
                        {
                            'spei_gamma_': 'gam'
                            , 'spei_pearson_': 'pea'
                        }
                        , regex=True
                    )

                    dataL5 = pd.melt(dataL4, id_vars=['type', 'lat', 'lon'], var_name='key', value_name='val')

                    # CSV 자료 저장
                    saveFile = '{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, keyInfo, dateInfo, 'cnt')
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    dataL5.to_csv(saveFile, index=False)
                    log.info(f'[CHECK] saveFile : {saveFile}')

                    typeList = set(dataL5['type'])
                    for k, typeInfo in enumerate(typeList):
                        log.info(f'[CHECK] typeInfo : {typeInfo}')

                        dataL6 = dataL5.loc[dataL5['type'] == typeInfo]

                        # 시각화 저장
                        saveImg = '{}/{}/{}_{}_{}_{}.png'.format(globalVar['figPath'], serviceName, keyInfo, dateInfo, typeInfo, 'boxplot')
                        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                        sns.boxplot(
                            x='key', y='val', hue='key', data=dataL6, showmeans=True, width=0.5, dodge=False
                            , meanprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 3}
                        )
                        plt.xticks(rotation=45, ha='right')
                        plt.xlabel(None)
                        plt.ylabel(None)
                        plt.legend([], [], frameon=False, title=None)
                        plt.tight_layout()
                        plt.savefig(saveImg, dpi=600, transparent=True)
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
