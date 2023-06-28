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
from pyproj import Proj
# import rioxarray as rio
import cftime
import subprocess
from global_land_mask import globe

import re
# import ray

import dask.dataframe as ds

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


def exeSubProc(cmd):
    log.info('[SRT] {}'.format('exeSubProc'))

    log.info("[CHECK] cmd : {}".format(cmd))

    result = None

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True, encoding='utf-8')

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        log.info('[END] {}'.format('exeSubProc'))


def makeLandMask(data):

    log.info('[START] {}'.format('makeLandMask'))
    result = None

    try:

        lon1D = sorted(set(data['lon'].values))
        lat1D = sorted(set(data['lat'].values))
        time1D = sorted(set(data['time'].values))

        # 경도 변환 (0~360 to -180~180)
        convLon1D = []
        for i, lon in enumerate(lon1D):
            convLon1D.append((lon + 180) % 360 - 180)

        lon2D, lat2D = np.meshgrid(convLon1D, lat1D)
        isLand2D = globe.is_land(lat2D, lon2D)
        isLand3D = np.tile(isLand2D, (len(time1D), 1, 1))

        landData = xr.Dataset(
            {
                'isLand': (('time', 'lat', 'lon'), (isLand3D).reshape(len(time1D), len(lat1D), len(lon1D)))
            }
            , coords={
                'lat': lat1D
                , 'lon': lon1D
                , 'time': time1D
            }
        )

        # 육해상에 대한 강수량
        dataL1 = xr.merge([data, landData])

        result = {
            'msg': 'succ'
            , 'lon1D': lon1D
            , 'lat1D': lat1D
            , 'time1D': time1D
            , 'isLand2D': isLand2D
            , 'isLand3D': isLand3D
            , 'resData': dataL1
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeLandMask'))

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python 이용한 NetCDF 파일 처리 및 3종 증발산량 (Penman, Hargreaves, Thornthwaite) 계산

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/pycharm_project_83'

    prjName = 'test'
    serviceName = 'LSH0346'

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
                    'srtDate': '1990-01-01'
                    , 'endDate': '2022-01-01'

                    # 경도 최소/최대/간격
                    , 'lonMin': -180
                    , 'lonMax': 180
                    , 'lonInv': 0.1
                    # , 'lonInv': 5

                    # 위도 최소/최대/간격
                    , 'latMin': -90
                    , 'latMax': 90
                    , 'latInv': 0.1
                    # , 'latInv': 5
                }

                globalVar['inpPath'] = 'G:/Climate variables/PET'
                globalVar['outPath'] = 'G:/Climate variables/PET/OUTPUT1'
                # globalVar['figPath'] = '/DATA/FIG'
                # globalVar['fortranPath'] = '/SYSTEMS/PROG/FORTRAN/REA'

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': globalVar['srtDate']
                    # , 'endDate': globalVar['endDate']
                    'srtDate': '1990-01-01'
                    , 'endDate': '2022-01-01'

                    # 경도 최소/최대/간격
                    , 'lonMin': -180
                    , 'lonMax': 180
                    , 'lonInv': 0.1

                    # 위도 최소/최대/간격
                    , 'latMin': -90
                    , 'latMax': 90
                    , 'latInv': 0.1
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['fortranPath'] = '/SYSTEMS/PROG/FORTRAN/REA'

            # # 도법 설정
            proj4326 = 'epsg:4326'
            mapProj4326 = Proj(proj4326)


            # ************************************************************************************************
            # 자료 병합
            # ************************************************************************************************
            # inpFile = '{}/{}/*.nc'.format(globalVar['inpPath'], serviceName)
            inpFile = '{}/{}/{}/{}.nc'.format(globalVar['inpPath'], '*', '*', '*')
            fileList = sorted(glob.glob(inpFile))

            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                # continue

            dataL3 = xr.Dataset()
            for j, fileInfo in enumerate(fileList):
                fileNameNoExt = os.path.basename(fileInfo).split('.nc')[0]
                # if (fileNameNoExt.__contains__('ssp126')): continue
                # if (fileNameNoExt.__contains__('OBS')): continue
                if (fileNameNoExt.__contains__('Aski OBS')): continue
                if (fileInfo.__contains__('OUTPUT')): continue

                log.info('[CHECK] fileInfo : {}'.format(fileInfo))

                data = xr.open_mfdataset(fileInfo)
                # data = xr.open_dataset(fileInfo)

                # 삭제
                dataL1 = data.drop(['thornthwaite'])

                # 이름 변경
                dataL2 = dataL1.rename(
                    {
                        'hargreaves' : '{}_{}'.format('har', fileNameNoExt)
                        , 'penman-monteith' : '{}_{}'.format('pen', fileNameNoExt)
                    }
                )

                dataL3 = xr.merge([dataL3, dataL2])

            # saveFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'clmData')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # dataL3.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

            # ************************************************************************************************
            # 육/해상 마스크 생성
            # ************************************************************************************************
            # 자료 읽기
            # inpFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'clmData')
            # fileList = sorted(glob.glob(inpFile))

            # data = xr.open_mfdataset(fileList)
            data = dataL3

            # 육/해상 마스크 생성
            dataL1 = makeLandMask(data)['resData']

            # 육지에 대한 강수량
            dataL2 = dataL1.where(dataL1['isLand'] == True).sel(lat = slice(-60, 90))

            # ************************************************************************************************
            # pen 변수 속성 추출
            # 과거/미래에 대한 csv 생성
            # ************************************************************************************************
            penData = dataL2
            for i, varInfo in enumerate(penData.data_vars.keys()):
                if (not re.match('pen', varInfo)):
                    penData = penData.drop([varInfo])
                    continue

                log.info('[CHECK] varInfo : {}'.format(varInfo))

            # 과거
            # prevPenDataL1 = penData[['pen_ACCESS-CM2_eto', 'pen_IPSL-CM6A-LR_eto', 'pen_MIROC6_eto', 'pen_ACCESS-ESM1-5_eto', 'pen_OBS_eto']].sel(time = slice('1948', '2014'))
            # prevPenDataL2 = prevPenDataL1.to_dataframe().reset_index(drop=False)
            # prevPenDataL3 = prevPenDataL2.dropna()
            #
            # NetCDF 파일 저장
            for i, varInfo in enumerate(penData.data_vars.keys()):
                saveFile = '{}/{}/{}-{}.nc'.format(globalVar['outPath'], serviceName, 'prevPenData', varInfo)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                penData[varInfo].to_netcdf(saveFile)
                log.info('[CHECK] saveFile : {}'.format(saveFile))

            # saveFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'prevPenData')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # prevPenDataL1.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # saveFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'prevPenData-NA')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # prevPenDataL3.to_csv(saveFile, index=False)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

            # 미래
            # nextPenDataL1 = penData[['pen_ACCESS-CM2 ssp245_eto', 'pen_IPSL-CM6A-LR ssp245_eto', 'pen_MIROC6 ssp245_eto', 'pen_ACCESS-ESM1-5 ssp245_eto']].sel(time=slice('2015', '2100'))
            # nextPenDataL2 = nextPenDataL1.to_dataframe().reset_index(drop=False)
            # nextPenDataL3 = nextPenDataL2.dropna()
            #
            # saveFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'nextPenData')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # nextPenDataL1.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # # saveFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'nextPenData')
            # # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # # penDataL2.to_csv(saveFile, index=False)
            # # log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # saveFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'nextPenData-NA')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # nextPenDataL3.to_csv(saveFile, index=False)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))


            # 위도/경도
            # inpFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'prevPenData-NA')
            # prevPenDataL3 = ds.read_csv(sorted(glob.glob(inpFile))[0])
            #
            # inpFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'nextPenData-NA')
            # nextPenDataL3 = ds.read_csv(sorted(glob.glob(inpFile))[0])
            #
            # lonList = sorted(set(prevPenDataL3['lon'].unique()) | set(nextPenDataL3['lon'].unique()))
            # log.info("[CHECK] lonList : {}".format(lonList))
            #
            # saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, serviceName, 'unique_pen_lon')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # pd.DataFrame({'lon': lonList}).to_csv(saveFile, index=False)
            # log.info("[CHECK] saveFile : {}".format(saveFile))
            #
            # latList = sorted(set(prevPenDataL3['lat'].unique()) | set(nextPenDataL3['lat'].unique()))
            # log.info("[CHECK] latList : {}".format(latList))
            #
            # saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, serviceName, 'unique_pen_lat')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # pd.DataFrame({'lat': latList}).to_csv(saveFile, index=False)
            # log.info("[CHECK] saveFile : {}".format(saveFile))

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
        inParams = { }

        print("[CHECK] inParams : {}".format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
