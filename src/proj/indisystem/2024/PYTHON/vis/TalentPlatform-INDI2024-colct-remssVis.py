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

from remss.ssmis.bytemaps import sys
from remss.ssmis.bytemaps import Dataset
from remss.ssmis.bytemaps import Verify

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


class SSMISdaily(Dataset):
    """ Read daily SSMIS bytemaps. """
    """
    Public data:
        filename = name of data file
        missing = fill value used for missing data;
                  if None, then fill with byte codes (251-255)
        dimensions = dictionary of dimensions for each coordinate
        variables = dictionary of data for each variable
    """

    def __init__(self, filename, missing=None):
        """
        Required arguments:
            filename = name of data file to be read (string)

        Optional arguments:
            missing = fill value for missing data,
                      default is the value used in verify file
        """
        self.filename = filename
        self.missing = missing
        Dataset.__init__(self)

    # Dataset:

    def _attributes(self):
        return ['coordinates', 'long_name', 'units', 'valid_min', 'valid_max']

    def _coordinates(self):
        return ('orbit_segment', 'variable', 'latitude', 'longitude')

    def _shape(self):
        return (2, 5, 720, 1440)

    def _variables(self):
        return ['time', 'wspd_mf', 'vapor', 'cloud', 'rain',
                'longitude', 'latitude', 'land', 'ice', 'nodata']

        # _default_get():

    def _get_index(self, var):
        return {'time': 0,
                'wspd_mf': 1,
                'vapor': 2,
                'cloud': 3,
                'rain': 4,
                }[var]

    def _get_scale(self, var):
        return {'time': 0.1,
                'wspd_mf': 0.2,
                'vapor': 0.3,
                'cloud': 0.01,
                'rain': 0.1,
                }[var]

    def _get_offset(self, var):
        return {'cloud': -0.05,
                }[var]

    # _get_ attributes:

    def _get_long_name(self, var):
        return {'time': 'Fractional Hour GMT',
                'wspd_mf': '10m Surface Wind Speed',
                'vapor': 'Columnar Water Vapor',
                'cloud': 'Cloud Liquid Water',
                'rain': 'Surface Rain Rate',
                'longitude': 'Grid Cell Center Longitude',
                'latitude': 'Grid Cell Center Latitude',
                'land': 'Is this land?',
                'ice': 'Is this ice?',
                'nodata': 'Is there no data?',
                }[var]

    def _get_units(self, var):
        return {'time': 'Fractional Hour GMT',
                'wspd_mf': 'm/s',
                'vapor': 'mm',
                'cloud': 'mm',
                'rain': 'mm/hr',
                'longitude': 'degrees east',
                'latitude': 'degrees north',
                'land': 'True or False',
                'ice': 'True or False',
                'nodata': 'True or False',
                }[var]

    def _get_valid_min(self, var):
        return {'time': 0.0,
                'wspd_mf': 0.0,
                'vapor': 0.0,
                'cloud': -0.05,
                'rain': 0.0,
                'longitude': 0.0,
                'latitude': -90.0,
                'land': False,
                'ice': False,
                'nodata': False,
                }[var]

    def _get_valid_max(self, var):
        return {'time': 24.0,
                'wspd_mf': 50.0,
                'vapor': 75.0,
                'cloud': 2.45,
                'rain': 25.0,
                'longitude': 360.0,
                'latitude': 90.0,
                'land': True,
                'ice': True,
                'nodata': True,
                }[var]

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 위성측정기반 바람 산출물 수집

    # cd /home/hanul/SYSTEMS/KIER/PROG/PYTHON/colct
    # nohup /home/hanul/anaconda3/envs/py38/bin/python3 TalentPlatform-INDI2024-colct-remssAMSR2.py --modelList SAT-AMSR2 --cpuCoreNum 5 --srtDate 2024-08-01 --endDate 2024-08-15 &

    # ps -ef | grep "TalentPlatform-INDI2024-colct-remssSMAP.py" | awk '{print $2}' | xargs kill -9
    # ps -ef | grep "RunShell-get-gfsncep2.sh" | awk '{print $2}' | xargs kill -9
    # ps -ef | egrep "RunShell|Repro" | awk '{print $2}' | xargs kill -9

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
        # contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'
        contextPath = os.getcwd() if env in 'local' else '/vol01/SYSTEMS/KIER/PROG/PYTHON'

    prjName = 'test'
    serviceName = 'INDI2024'

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
                # 수행 목록
                'modelList': ['SAT-AMSR2', 'SAT-SMAP', 'SAT-SSMI']
                # 'modelList': [globalVar['modelList']]

                # 비동기 다중 프로세스 개수
                , 'cpuCoreNum': '5'
                # , 'cpuCoreNum': globalVar['cpuCoreNum']

                , 'SAT-SSMI': {
                    # 시작일, 종료일, 시간 간격 (연 1y, 월 1h, 일 1d, 시간 1h)
                    'srtDate': '2024-08-01'
                    , 'endDate': '2024-08-15'
                    # 'srtDate': globalVar['srtDate']
                    # , 'endDate': globalVar['endDate']
                    , 'invDate': '1d'
                    , 'request': {
                        'url': 'https://data.remss.com'
                        , 'filePath': '/ssmi/f18/bmaps_v08/y%Y/m%m'
                        , 'fileNamePattern': 'f18_(\d{4})(\d{2})(\d{2})v(\d+)\.gz'
                    }
                    , 'tmp': '/HDD/DATA/data1/SAT/SSMI/%Y/%m/.{}'
                    , 'target': '/HDD/DATA/data1/SAT/SSMI/%Y/%m/{}'
                }

                , 'SAT-AMSR2': {
                    # 시작일, 종료일, 시간 간격 (연 1y, 월 1h, 일 1d, 시간 1h)
                    'srtDate': '2024-08-01'
                    , 'endDate': '2024-08-15'
                    # 'srtDate': globalVar['srtDate']
                    # , 'endDate': globalVar['endDate']
                    , 'invDate': '1d'
                    , 'request': {
                        'url': 'https://data.remss.com'
                        , 'filePath': '/amsr2/ocean/L3/v08.2/daily/%Y'
                        , 'fileNamePattern': 'RSS_AMSR2_ocean_L3_daily_(\d{4})-(\d{2})-(\d{2})_v(\d+\.\d+)\.nc'
                    }
                    , 'tmp': '/HDD/DATA/data1/SAT/AMSR2/%Y/%m/.{}'
                    , 'target': '/HDD/DATA/data1/SAT/AMSR2/%Y/%m/{}'
                }

                , 'SAT-SMAP': {
                    # 시작일, 종료일, 시간 간격 (연 1y, 월 1h, 일 1d, 시간 1h)
                    'srtDate': '2024-08-01'
                    , 'endDate': '2024-08-15'
                    # 'srtDate': globalVar['srtDate']
                    # , 'endDate': globalVar['endDate']
                    , 'invDate': '1d'
                    , 'request': {
                        'url': 'https://data.remss.com'
                        , 'filePath': '/smap/wind/L3/v01.0/daily/NRT/%Y'
                        , 'fileNamePattern': 'RSS_smap_wind_daily_(\d{4})_(\d{2})_(\d{2})_NRT_v(\d+\.\d+)\.nc'
                    }
                    , 'tmp': '/HDD/DATA/data1/SAT/SMAP/%Y/%m/.{}'
                    , 'target': '/HDD/DATA/data1/SAT/SMAP/%Y/%m/{}'
                }
            }


            print('test')


            # /HDD/DATA/data1/SAT/AMSR2/2024/03/RSS_AMSR2_ocean_L3_daily_2024-03-05_v08.2.nc
            # /HDD/DATA/data1/SAT/SMAP/2024/08/RSS_smap_wind_daily_2024_08_11_NRT_v01.0.nc

            # import xarray as xr
            # data = xr.open_dataset('/HDD/DATA/data1/SAT/AMSR2/2024/03/RSS_AMSR2_ocean_L3_daily_2024-03-05_v08.2.nc')
            #
            # # data['wind_speed_AW'].sel({'pass': 2}).plot()
            # # plt.show()
            #
            # meanData = data['wind_speed_AW'].mean(dim=['pass'])
            # meanData.plot()
            # plt.show()
            #
            #
            # data = xr.open_dataset('/HDD/DATA/data1/SAT/SMAP/2024/08/RSS_smap_wind_daily_2024_08_11_NRT_v01.0.nc')
            #
            # # data['node'].values
            # #
            # # data['wind'].sel({'node': 0}).plot()
            # # data['wind'].sel({'node': 1}).plot()
            # # plt.show()
            #
            # meanData = data['wind'].mean(dim=['node'])
            # meanData.plot()
            # plt.show()







            ssmi = SSMISdaily('/HDD/DATA/TMP/f18_20240807v8.gz')

            dim = ssmi.dimensions
            for key, val in dim.items():
                print(key, val)

            var = ssmi.variables
            # for key, val in var.items():
            #     print(key, val.shape)

            dataL2 = xr.Dataset(
                coords={
                    'orbit': np.arange(dim['orbit_segment'])
                    , 'lat': var['latitude']
                    , 'lon': var['longitude']
                }
            )

            for key, val in var.items():
                if re.search('longitude|latitude', key, re.IGNORECASE): continue

                # Time:  7.10 = fractional hour GMT, NOT local time,  valid data range=0 to 24.0,  255 = land
                # Wind: 255=land, 253=bad data,  251=no wind calculated, other data <=50.0 is 10-meter wind speed
                # Water Vapor:  255=land, 253=bad data, other data <=75 is water vapor (mm)
                # Cloud Liquid Water:  255=land,  253=bad data, other data <=2.5 is cloud (mm)
                # Rain:  255=land, 253=bad data, other data <= 25 is rain (mm/hr)
                if re.search('time', key, re.IGNORECASE):
                    val2 = xr.where((0 <= val) & (val <= 24), val, np.nan)
                elif re.search('wspd_mf', key, re.IGNORECASE):
                    val2 = xr.where((val <= 50), val, np.nan)
                elif re.search('vapor', key, re.IGNORECASE):
                    val2 = xr.where((val <= 75), val, np.nan)
                elif re.search('cloud', key, re.IGNORECASE):
                    val2 = xr.where((val <= 2.5), val, np.nan)
                elif re.search('rain', key, re.IGNORECASE):
                    val2 = xr.where((val <= 25), val, np.nan)
                else:
                    val2 = val

                try:
                    dataL2[key] = (('orbit', 'lat', 'lon'), (val2))
                except Exception as e:
                    pass

            # import matplotlib.pyplot as plt
            # # dataL2['wspd_mf'].plot()
            # dataL2['wspd_mf'].sel(orbit = 1).plot()
            # plt.show()
            # dataL2['wspd_mf'].sel(orbit = 0).plot()
            # plt.show()

            meanData = dataL2['wspd_mf'].mean(dim=['orbit'])
            meanData.plot()
            plt.show()

            # d = xr.Dataset.from_dict(ssmi)
            # d = xr.DataArray.from_dict(ssmi)
            #
            # d = xr.DataArray.from_dict(ssmi._get_variables())



            # meanData = data['wind_speed_AW'].mean(dim=['pass'])
            # meanData.plot()
            # plt.show()


            # **************************************************************************************************************
            # 비동기 다중 프로세스 수행
            # **************************************************************************************************************
            # 비동기 다중 프로세스 개수
            # pool = Pool(int(sysOpt['cpuCoreNum']))
            #
            # for modelType in sysOpt['modelList']:
            #     log.info(f'[CHECK] modelType : {modelType}')
            #
            #     modelInfo = sysOpt.get(modelType)
            #     if modelInfo is None: continue
            #
            #     # 시작일/종료일 설정
            #     dtSrtDate = pd.to_datetime(modelInfo['srtDate'], format='%Y-%m-%d')
            #     dtEndDate = pd.to_datetime(modelInfo['endDate'], format='%Y-%m-%d')
            #     dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=modelInfo['invDate'])
            #
            #     for dtDateInfo in dtDateList:
            #         log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
            #         pool.apply_async(subColct, args=(modelInfo, dtDateInfo))
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
