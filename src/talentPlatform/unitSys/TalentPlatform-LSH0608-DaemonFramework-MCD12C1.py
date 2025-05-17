# ================================================
# 요구사항
# ================================================
# Python 이용한 NetCDF 파일 처리 및 3종 증발산량 (Penman, Hargreaves, Thornthwaite) 계산

# 	ubyte Land_Cover_Type_1_Percent(YDim\:MOD12C1, XDim\:MOD12C1, Num_IGBP_Classes\:MOD12C1) ;
# 		Land_Cover_Type_1_Percent:long_name = "Land_Cover_Type_1_Percent" ;
# 		Land_Cover_Type_1_Percent:units = "percent in integers" ;
# 		Land_Cover_Type_1_Percent:valid_range = 0UB, 100UB ;
# 		Land_Cover_Type_1_Percent:_FillValue = 255UB ;
# 		Land_Cover_Type_1_Percent:Layer\ 0 = "water" ;
# 		Land_Cover_Type_1_Percent:Layer\ 1 = "evergreen needleleaf forest" ;
# 		Land_Cover_Type_1_Percent:Layer\ 2 = "evergreen broadleaf forest" ;
# 		Land_Cover_Type_1_Percent:Layer\ 3 = "deciduous needleleaf forest" ;
# 		Land_Cover_Type_1_Percent:Layer\ 4 = "deciduous broadleaf forest" ;
# 		Land_Cover_Type_1_Percent:Layer\ 5 = "mixed forests" ;
# 		Land_Cover_Type_1_Percent:Layer\ 6 = "closed shrubland" ;
# 		Land_Cover_Type_1_Percent:Layer\ 7 = "open shrublands" ;
# 		Land_Cover_Type_1_Percent:Layer\ 8 = "woody savannas" ;
# 		Land_Cover_Type_1_Percent:Layer\ 9 = "savannas" ;
# 		Land_Cover_Type_1_Percent:Layer\ 10 = "grasslands" ;
# 		Land_Cover_Type_1_Percent:Layer\ 11 = "permanent wetlands" ;
# 		Land_Cover_Type_1_Percent:Layer\ 12 = "croplands" ;
# 		Land_Cover_Type_1_Percent:Layer\ 13 = "urban and built-up" ;
# 		Land_Cover_Type_1_Percent:Layer\ 14 = "cropland/natural vegetation mosaic" ;
# 		Land_Cover_Type_1_Percent:Layer\ 15 = "snow and ice" ;
# 		Land_Cover_Type_1_Percent:Layer\ 16 = "barren or sparsely vegetated" ;

# gdalinfo to tif 변환, 0.1도 간격

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

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

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
                # 'srtDate': globalVar['srtDate']
                # , 'endDate': globalVar['endDate']
                'srtDate': '1990-01-01',
                'endDate': '2022-01-01',

                # 경도 최소/최대/간격
                'lonMin': -180,
                'lonMax': 180,
                'lonInv': 0.1,

                # 위도 최소/최대/간격
                'latMin': -90,
                'latMax': 90,
                'latInv': 0.1,

                # cmd 실행 정보
                'cmd': 'export PROJ_LIB=/SYSTEMS/LIB/anaconda3/envs/py38/share/proj && {exe} HDF4_EOS:EOS_GRID:"{inpFile}":MOD12C1:Land_Cover_Type_1_Percent "{outFile}"',
                'exe': '/SYSTEMS/LIB/anaconda3/envs/py38/bin/gdal_translate'
                # 'exe': '/data2/hzhenshao/EMI/py38/bin/gdal_translate'
            }

            # 도법 설정
            proj4326 = 'epsg:4326'
            mapProj4326 = Proj(proj4326)

            lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
            latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])

            log.info('[CHECK] len(lonList) : {}'.format(len(lonList)))
            log.info('[CHECK] len(latList) : {}'.format(len(latList)))

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1Y')
            # dtIncDateInfo = dtIncDateList[0]

            dataL5 = xr.Dataset()
            for i, dtIncDateInfo in enumerate(dtIncDateList):
                # log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))
                sYear = dtIncDateInfo.strftime('%Y')

                inpFile = '{}/{}/*/MCD12C1.A{}*.hdf'.format(globalVar['inpPath'], serviceName, sYear)
                # inpFile = '{}/{}/*/MCD12C1.A{}*.tif'.format(globalVar['inpPath'], serviceName, sYear)

                fileList = sorted(glob.glob(inpFile))
                if fileList is None or len(fileList) < 1:
                    log.error(f"inpFile : {inpFile} / 입력 자료를 확인해주세요")
                    continue

                fileInfo = fileList[0]
                log.info(f'[CHECK] fileInfo : {fileInfo}')

                # fileNameNoExt = os.path.basename(fileInfo).split('.hdf')[0]
                filePath = os.path.dirname(fileInfo)
                fileNameNoExt =  os.path.basename(fileInfo).split('.hdf')[0]
                cmd = sysOpt['cmd'].format(exe=sysOpt['exe'], inpFile=f"{filePath}/{fileNameNoExt}.hdf", outFile=f"{filePath}/{fileNameNoExt}.tif")
                log.info(f'[CHECK] cmd : {cmd}')

                try:
                    res = subprocess.run(cmd, shell=True, executable='/bin/bash')
                    log.info(f'returncode : {res.returncode} / args : {res.args}')
                    if res.returncode != 0: log.error(f'cmd 실패 : {cmd}')
                except Exception as e:
                    raise ValueError(f'Exception: {e}')

                inpFile = '{}/{}/*/MCD12C1.A{}*.tif'.format(globalVar['inpPath'], serviceName, sYear)

                fileList = sorted(glob.glob(inpFile))
                if fileList is None or len(fileList) < 1:
                    log.error(f"inpFile : {inpFile} / 입력 자료를 확인해주세요")
                    continue

                fileInfo = fileList[0]
                log.info(f'[CHECK] fileInfo : {fileInfo}')

                # data = xr.open_mfdataset(fileInfo)
                # data = xr.open_dataset(fileInfo)
                # data = xr.open_dataset(fileInfo, engine="h5netcdf")
                # data = xr.open_dataset(fileInfo, engine="rasterio")
                # data = xr.open_rasterio(fileInfo, chunks={"band": 1, "x": 100, "y": 100})
                data = xr.open_rasterio(fileInfo)

                dataL1 = data.rio.reproject(proj4326)
                dataL2 = dataL1.sel(band=13)
                dataL3 = dataL2.interp(x=lonList, y=latList, method='nearest')

                # Land_Cover_Type_1_Percent:Layer\ 13 = "urban and built-up" ;
                # SUBDATASET_3_NAME=HDF4_EOS:EOS_GRID:"MCD12C1.A2019001.061.2022170020638.hdf":MOD12C1:Land_Cover_Type_1_Percent
                # SUBDATASET_3_DESC=[3600x7200x17] Land_Cover_Type_1_Percent MOD12C1 (8-bit unsigned integer)
                # dataL1 = data.sel(band=13).interp(x=lonList, y=latList, method='linear')

                # 자료 변환
                lon1D = dataL1['x'].values
                lat1D = dataL1['y'].values
                time1D = pd.to_datetime(sYear, format='%Y')

                dataL4 = xr.Dataset(
                    {
                        'Land_Cover_Type_1_Percent': (('time', 'lat', 'lon'), (dataL1['band_data'].values).reshape(1, len(lat1D), len(lon1D)))
                    }
                    , coords={
                        'time': pd.date_range(time1D, periods=1)
                        , 'lat': lat1D
                        , 'lon': lon1D
                    }
                )

                # key = 'Land_Cover_Type_1_Percent'
                # saveImg = '{}/{}/{}/{}-{}.png'.format(globalVar['figPath'], serviceName, 'MCD12C1', fileNameNoExt, key)
                # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                # dataL2[key].plot()
                # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                # plt.tight_layout()
                # plt.show()
                # plt.close()
                # log.info(f'[CHECK] saveImg : {saveImg}')

                dataL5 = xr.merge([dataL5, dataL4])

            # 자료 저장
            timeList = dataL5['time'].values
            minYear = pd.to_datetime(timeList.min()).strftime('%Y')
            maxYear = pd.to_datetime(timeList.max()).strftime('%Y')

            saveFile = '{}/{}/{}_{}-{}.nc'.format(globalVar['outPath'], serviceName, 'Land_Cover_Type_1_Percent', minYear, maxYear)
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            dataL5.to_netcdf(saveFile)
            log.info('[CHECK] saveFile : {}'.format(saveFile))

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