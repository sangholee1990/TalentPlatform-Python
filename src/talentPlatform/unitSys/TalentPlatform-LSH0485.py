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
import pyart
import xarray as xr

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
    # Python을 이용한 대한민국 기상청 레이더 자료처리 및 다양한 자료 저장

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
    serviceName = 'LSH0485'

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

            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간, 5분 간격
                'srtDate': '2023-02-09 13:00'
                , 'endDate': '2023-02-09 13:30'
                , 'invDate': '5T'

                # 수행 목록
                , 'nameList': ['radar']

                # 수행 정보 : 파일 경로, 파일명, 변수 선택
                , 'nameInfo': {
                    'radar': {
                        'filePath': '/DATA/INPUT/LSH0485/GDK_230209-10'
                        , 'fileName': 'RDR_GDK_FQC_%Y%m%d%H%M.uf'
                        # 전체 변수
                        # , 'varList': ['reflectivity', 'velocity', 'spectrum_width', 'corrected_reflectivity', 'corrected_differential_reflectivity', 'cross_correlation_ratio', 'differential_phase', 'specific_differential_phase']
                        # 특정 변수
                        , 'varList': ['reflectivity', 'velocity']
                    }
                }
            }

            # ======================================================================================
            # 테스트 파일
            # ======================================================================================
            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d %H:%M')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d %H:%M')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

            dataL1 = xr.Dataset()
            for dtDateIdx, dtDateInfo in enumerate(dtDateList):
                log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                for nameIdx, nameType in enumerate(sysOpt['nameList']):
                    # log.info(f'[CHECK] nameType : {nameType}')

                    modelInfo = sysOpt['nameInfo'].get(nameType)
                    if modelInfo is None: continue

                    inpFile = '{}/{}'.format(modelInfo['filePath'], modelInfo['fileName'])
                    inpFileDate = dtDateInfo.strftime(inpFile)
                    fileList = sorted(glob.glob(inpFileDate))

                    if fileList is None or len(fileList) < 1:
                        log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                        continue

                    for j, fileInfo in enumerate(fileList):

                        # 파일 읽기
                        data = pyart.io.read_uf(fileInfo)
                        log.info(f'[CHECK] fileInfo : {fileInfo}')

                        # 메타 정보
                        # data.info()

                        # 위경도 추출
                        # data.init_gate_altitude()
                        # data.init_gate_longitude_latitude()
                        lon2D = data.gate_longitude["data"]
                        lat2D = data.gate_latitude["data"]

                        # x, y축 배열 정보
                        xdim = lon2D.shape[0]
                        ydim = lon2D.shape[1]

                        for field in data.fields.keys():
                            if not (field in modelInfo['varList']): continue
                            log.info(f'[CHECK] field : {field}')

                            val2D = data.fields[field]['data']


                            dsData = xr.Dataset(
                                {
                                    field: (('time', 'row', 'col'), (data.fields[field]['data']).reshape(1, xdim, ydim))
                                }
                                , coords={
                                    'row': np.arange(val2D.shape[0])
                                    , 'col': np.arange(val2D.shape[1])
                                    , 'lon': (('time', 'row', 'col'), (lon2D).reshape(1, xdim, ydim))
                                    , 'lat': (('time', 'row', 'col'), (lat2D).reshape(1, xdim, ydim))
                                    , 'time': pd.date_range(dtDateInfo, periods=1)
                                }
                            )

                            dataL1 = xr.merge([dataL1, dsData])

            # NetCDF 저장
            timeStrList = pd.to_datetime(dataL1['time'].values).strftime('%Y%m%d%H%M')

            saveNcFile = '{}/{}/{}_{}-{}.nc'.format(globalVar['outPath'], serviceName, 'RDR_GDK_FQC', timeStrList.min(), timeStrList.max())
            os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
            dataL1.to_netcdf(saveNcFile)
            log.info('[CHECK] saveNcFile : {}'.format(saveNcFile))

            # CSV 저장
            saveCsvFile = '{}/{}/{}_{}-{}.csv'.format(globalVar['outPath'], serviceName, 'RDR_GDK_FQC', timeStrList.min(), timeStrList.max())
            os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)
            dataL1.to_dataframe().reset_index(drop=False).to_csv(saveCsvFile, index=False)
            log.info('[CHECK] saveCsvFile : {}'.format(saveCsvFile))

            # 시각화 저장
            timeList = dataL1['time'].values
            for timeInfo in timeList:
                # 날짜 제한
                if (timeInfo != timeList[0]): continue
                log.info(f'[CHECK] timeInfo : {timeInfo}')

                selData = dataL1.sel(time = timeInfo)
                for field in list(selData.data_vars.keys()):
                    lon2D = selData['lon'].values
                    lat2D = selData['lat'].values
                    val2D = selData[field].values

                    mainTitle = '{}_{}_{}'.format('RDR_GDK_FQC', field, pd.to_datetime(timeInfo).strftime('%Y%m%d%H%M'))
                    saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                    os.makedirs(os.path.dirname(saveImg), exist_ok=True)

                    plt.pcolormesh(lon2D, lat2D, val2D)
                    plt.colorbar()
                    plt.title(mainTitle)
                    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
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
