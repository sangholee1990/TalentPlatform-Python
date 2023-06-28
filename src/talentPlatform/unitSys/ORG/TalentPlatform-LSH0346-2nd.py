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
import rioxarray as rio
import cftime
import subprocess
from global_land_mask import globe

import re
# import ray
import dask.dataframe as ds

import f90nml
import tempfile



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

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

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

            # saveFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'prevPenData')
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # penDataL1.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

            inpFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'prevPenData-NA')
            # inpFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'prevPenData')
            fileList = sorted(glob.glob(inpFile))

            prevPenData = ds.read_csv(fileList[0])
            # prevPenData = pd.read_csv(fileList[0])
            # prevPenData.head()
            # len(prevPenData)

            prevPenData['geo'] = prevPenData['lon'].astype(str) + '-' +  prevPenData['lat'].astype(str)

            inpFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'nextPenData-NA')
            # inpFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'nextPenData')
            fileList = sorted(glob.glob(inpFile))

            nextPenData = ds.read_csv(fileList[0])
            # nextPenData = pd.read_csv(fileList[0])
            # nextPenData.head()

            nextPenData['geo'] = nextPenData['lon'].astype(str) + '-' + nextPenData['lat'].astype(str)

            # latList = set(prevPenData['lat']) & set(nextPenData['lon'])
            latList = sorted(set(prevPenData['lat'].unique()) | set(nextPenData['lat'].unique()))
            lonList = sorted(set(prevPenData['lon'].unique()) | set(nextPenData['lon'].unique()))
            log.info("[CHECK] latList : {}".format(latList))

            geoList = sorted(set(prevPenData['geo'].unique()) | set(nextPenData['geo'].unique()))
            log.info("[CHECK] geoList : {}".format(geoList))
            print(len(geoList))

            # 메모리 해제
            prevPenData = prevPenData.compute()
            nextPenData = nextPenData.compute()

            for i, lonInfo in enumerate(lonList):
                for j, latInfo in enumerate(latList):

                    log.info("[CHECK] lonInfo : {} / latInfo : {}".format(lonInfo, latInfo))

                    prevData = prevPenData.loc[
                        (prevPenData['lon'] == lonInfo)
                        & (prevPenData['lat'] == latInfo)
                        ].reset_index(drop=True)

                    nextData = nextPenData.loc[
                        (nextPenData['lon'] == lonInfo)
                        & (nextPenData['lat'] == latInfo)
                        ].reset_index(drop=True)

                    if (len(prevData) < 1): continue
                    if (len(nextData) < 1): continue

                    prevData.drop(['time', 'lat', 'lon'], axis='columns', inplace=True)
                    nextData.drop(['time', 'lat', 'lon'], axis='columns', inplace=True)

                    # prevData.drop(['pen_MIROC6_eto'], axis='columns', inplace=True)

                    log.info("[CHECK] prevData.columns : {}".format(prevData.columns))
                    log.info("[CHECK] nextData.columns : {}".format(nextData.columns))

                    prevData.columns = ['ACCESS-CM2', 'IPSL-CM6A-LR', 'MIROC6', 'ACCESS-ESM1-5', 'OBS']
                    nextData.columns = ['ACCESS-CM2', 'IPSL-CM6A-LR', 'MIROC6', 'ACCESS-ESM1-5']

                    # prevData.columns
                    # nextData.columns

                    # inpFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'Historical')
                    # fileList = sorted(glob.glob(inpFile))
                    #
                    # if fileList is None or len(fileList) < 1:
                    #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                    #
                    # prevData = pd.read_csv(fileList[0])

                    #
                    # inpFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'LSTM SSP245 21-25')
                    # fileList = sorted(glob.glob(inpFile))
                    #
                    # if fileList is None or len(fileList) < 1:
                    #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                    #
                    # nextData = pd.read_csv(fileList[0])

                    # prevData.drop(['Period'], axis='columns', inplace=True)
                    # nextData.drop(['Period'], axis='columns', inplace=True)

                    try:

                        modelNum = len(prevData.columns) - 1
                        # indices = 1:60

                        prevData['Ensemble'] = None

                        delMatData = pd.DataFrame()
                        for i, colInfo in enumerate(prevData.columns):
                            if (re.match('OBS', colInfo)): continue
                            if (re.match('Ensemble', colInfo)): continue

                            # 각 모형에 따라 -절대값 (미래의 평균 - 현재의 평균) 차이
                            delMatData.loc[0, colInfo] = -1.0 * np.abs(np.nanmean(nextData[colInfo]) - np.nanmean(prevData[colInfo]))

                        # 앙살블 평균 수행
                        delMatData.loc[0, 'Ensemble'] = np.nanmean(delMatData)

                        # B1, 2, 3, 4 calculation
                        # 동일한 컬럼에 대한 평균 (예측 - 실측) 차이

                        rMatData = pd.DataFrame()
                        for i, colInfo in enumerate(prevData.columns):
                            if (re.match('OBS', colInfo)): continue
                            if (re.match('Ensemble', colInfo)): continue

                            # 동일한 컬럼에 대한 평균 (예측 - 실측) 차이
                            rMatData.loc[0, colInfo] = np.nanmean(prevData[colInfo] - np.nanmean(prevData['OBS']))

                        # 이동평균
                        movAvgData = pd.DataFrame()
                        for i in range(0, 7306):
                            movAvgData.loc[i, 'AVG'] = np.mean(prevData.iloc[i+35:i+36]['OBS'])

                        movAvgData.dropna(inplace=True)

                        movAvgDiffData = np.nanmax(movAvgData['AVG']) - np.nanmin(movAvgData['AVG'])

                        # ==================================
                        # Fortran을 이용한 반복 수행
                        # ==================================
                        dimD = delMatData.transpose()
                        dimD.columns = ['val']

                        dimR = rMatData.transpose()
                        dimR.columns = ['val']

                        dimAvg = delMatData['Ensemble'].transpose()
                        dimAvg.columns = ['val']

                        dimMovAvg = pd.DataFrame( { 'val' : [movAvgDiffData] } )

                        inpDimFile = '{}/{}/{}.dat'.format(globalVar['outPath'], serviceName, 'input-dimD')
                        os.makedirs(os.path.dirname(inpDimFile), exist_ok=True)
                        dimD.to_csv(inpDimFile, index=False, header=False)

                        inpDimFile2 = '{}/{}/{}.dat'.format(globalVar['outPath'], serviceName, 'input-dimR')
                        os.makedirs(os.path.dirname(inpDimFile2), exist_ok=True)
                        dimR.to_csv(inpDimFile2, index=False, header=False)

                        inpDimFile3 = '{}/{}/{}.dat'.format(globalVar['outPath'], serviceName, 'input-dimAvg')
                        os.makedirs(os.path.dirname(inpDimFile3), exist_ok=True)
                        dimAvg.to_csv(inpDimFile3, index=False, header=False)

                        inpDimFile4 = '{}/{}/{}.dat'.format(globalVar['outPath'], serviceName, 'input-dimMovAvg')
                        os.makedirs(os.path.dirname(inpDimFile4), exist_ok=True)
                        dimMovAvg.to_csv(inpDimFile4, index=False, header=False)

                        # resDimFile = '{}/{}/{}.dat'.format(globalVar['outPath'], serviceName, 'res-dimD')
                        # resDimFile2 = '{}/{}/{}.dat'.format(globalVar['outPath'], serviceName, 'res-dimR')
                        # resDimFile3 = '{}/{}/{}.dat'.format(globalVar['outPath'], serviceName, 'res-dimAvg')

                        resDimFile = '{}/{}/{}/{}-{}-{}.dat'.format(globalVar['outPath'], serviceName, 'tmp/dimD', 'res-dimD', lonInfo, latInfo)
                        resDimFile2 = '{}/{}/{}/{}-{}-{}.dat'.format(globalVar['outPath'], serviceName, 'tmp/dimR', 'res-dimR', lonInfo, latInfo)
                        resDimFile3 = '{}/{}/{}/{}-{}-{}.dat'.format(globalVar['outPath'], serviceName, 'tmp/dimAvg', 'res-dimAvg', lonInfo, latInfo)

                        os.makedirs(os.path.dirname(resDimFile), exist_ok=True)
                        os.makedirs(os.path.dirname(resDimFile2), exist_ok=True)
                        os.makedirs(os.path.dirname(resDimFile3), exist_ok=True)

                        # inpFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'Historical')
                        # fileList = sorted(glob.glob(inpFile))

                        nmlFile = '{}/{}'.format(globalVar['fortranPath'], 'namelistInfo.nml')
                        os.makedirs(os.path.dirname(nmlFile), exist_ok=True)

                        nmlInfo = f90nml.read(nmlFile)

                        nmlInfo['input']['inpfile'] = inpDimFile
                        nmlInfo['input']['inpfile2'] = inpDimFile2
                        nmlInfo['input']['inpfile3'] = inpDimFile3
                        nmlInfo['input']['inpfile4'] = inpDimFile4

                        nmlInfo['output']['resfile'] = resDimFile
                        nmlInfo['output']['resfile2'] = resDimFile2
                        nmlInfo['output']['resfile3'] = resDimFile3

                        # saveNmlFile = '{}/{}'.format(globalVar['fortranPath'], 'namelistInfo.nml')

                        saveNmlFile = tempfile.NamedTemporaryFile(suffix='.nml').name
                        os.makedirs(os.path.dirname(saveNmlFile), exist_ok=True)
                        nmlInfo.write(saveNmlFile)

                        cmd = '{}/a.out {}'.format(globalVar['fortranPath'], saveNmlFile)

                        result = exeSubProc(cmd)

                        log.info("[CHECK] returncode : {}".format(result.returncode))
                        if (result.returncode == 0):
                            log.info("[CHECK] stdout : {}".format(result.stdout))
                        else:
                           log.info("[CHECK] stderr : {}".format(result.stderr))

                    except Exception as e:
                        log.error("Exception : {}".format(e))

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
