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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

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

def getCellArea(lat):
    # https://gis.stackexchange.com/questions/29734/how-to-calculate-area-of-1-x-1-degree-cells-in-a-raster

    # 지구 반지름 m
    R = 6371000

    # 1도 to 라디안
    delLonRad = np.radians(1.0)

    f0_rad = np.radians(lat - 0.5)
    f1_rad = np.radians(lat + 0.5)

    area = (np.sin(f1_rad) - np.sin(f0_rad)) * delLonRad * (R ** 2)

    return area

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 NetCDF 파일을 읽고 히스토그램 시각화

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
    serviceName = 'LSH0507'

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

            # ********************************************************************
            # 파일 읽기
            # ********************************************************************
            dataL1 = {}
            typeList = ['GRACED 2019', 'GRACED 2020', 'GRACED 2021', 'GRACED 2022', 'EDGAR 2019', 'GCP GridFED 2019', 'ODIAC 2019']
            for type in typeList:
                log.info(f'[CHECK] type : {type}')

                inpFileNamePattern = f'MODEL_TYPE/{type}/*.nc'
                inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, inpFileNamePattern)
                fileList = sorted(glob.glob(inpFile))

                for fileInfo in fileList:
                    log.info(f"[CHECK] fileInfo : {fileInfo}")

                    data = xr.open_dataset(fileInfo, group=None)

                    # 컬럼 선택 및 단위 변환 (kgC/year)
                    if re.search('GRACED', type, re.IGNORECASE):
                        # kgC/d -> kgC/year
                        val = data['emission'].values * 365
                    elif re.search('EDGAR', type, re.IGNORECASE):
                        # tonCo2/year -> kgC/year
                        # val = data['emissions'].values / 1000
                        # 2024.02.28
                        val = data['emissions'].values * 1000 * (12/44)
                    elif re.search('ODIAC', type, re.IGNORECASE):
                        sumData = data.sum(dim='month')

                        # 위경도를 통해 면적 계산
                        lon1D = sumData['lon'].values
                        lat1D = sumData['lat'].values

                        cellAreaByLat = np.vectorize(getCellArea)(lat1D)

                        cellAreaByGeo = np.tile(cellAreaByLat, (len(lon1D), 1)).T

                        cellAreaData = xr.Dataset(
                            {
                                'area': (('lat', 'lon'), cellAreaByGeo)
                            }
                            , coords={
                                'lon': lon1D
                                , 'lat': lat1D
                            }
                        )

                        sumDataL1 = xr.merge([sumData, cellAreaData])

                        # gC/m2/d -> kgC/year
                        # val = sumData['land'].values * (365 / 1000)
                        # val = (sumData['land'] + sumData['intl_bunker']).values * (365 / 1000)
                        val = (sumDataL1['land'] + sumDataL1['intl_bunker']).values * sumDataL1['area'].values * (365 / 1000)

                    elif re.search('GCP GridFED', type, re.IGNORECASE):
                        dataGrp = xr.open_dataset(fileInfo, group='CO2')
                        sumData = dataGrp.sum(dim='time')
                        val = (sumData['OIL'] + sumData['GAS'] + sumData['COAL'] + sumData['CEMENT'] + sumData['BUNKER']).values
                    else:
                        continue

                    logVal = np.log(val)
                    dataL1[type] = logVal[np.isfinite(logVal)]

                    log.info(f'[CHECK] min : {int(np.nanmin(val))} / max : {int(np.nanmax(val))} / mean : {int(np.nanmean(val))} / sum : {int(np.nansum(val))}')
                    log.info(f'[CHECK] min : {int(np.nanmin(dataL1[type]))} / max : {int(np.nanmax(dataL1[type]))} / mean : {int(np.nanmean(dataL1[type]))} / sum : {int(np.nansum(dataL1[type]))}')

            # ******************************************************************************
            # 시각화
            # ******************************************************************************
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(hspace=0.3)

            # x축 눈금의 범위 설정
            binEdgeList = np.arange(-24, 32, 4)
            # valList = list(dataL1.values())
            valList = [np.ravel(value) for value in dataL1.values()]
            keyList = list(dataL1.keys())

            n, bins, patches = ax.hist(valList, bins=binEdgeList, alpha=1.0, label=keyList, zorder=3)

            colors = plt.cm.coolwarm(np.linspace(0, 1, len(dataL1)))
            for patch, color in zip(patches, colors):
                for rect in patch:
                    rect.set_facecolor(color)

            # x축 눈금 레이블 설정
            binLabelList = [f"{int(bins[j])}~{int(bins[j + 1])}" for j in range(len(bins) - 1)]
            ax.set_xticks(bins[:-1])
            ax.set_xticklabels(binLabelList, rotation=45, ha='right', fontsize=10)

            ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
            # plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

            # 그래프 제목과 축 레이블 추가
            mainTitle = f'TOTAL LN EMISSION HIST'
            ax.set_title(mainTitle, loc='left')
            ax.legend(loc='upper left')
            ax.grid(True, color='lightgrey', linestyle='-', linewidth=0.5, zorder=0)

            # plt.xlabel('Ln Emission [kg/C/year]')
            # plt.ylabel('Number of Grids')

            fig.text(0.5, 0.02, 'Ln Emission [kg/C/year]', ha='center', va='center', fontsize=12)
            fig.text(0.08, 0.5, 'Number of Grids', ha='center', va='center', rotation='vertical', fontsize=12)

            # mainTitle = f'Ln Emission'
            # plt.suptitle(mainTitle)
            saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, 'TOTAL_EMI_HIST')
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
            plt.tight_layout()
            plt.show()
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
