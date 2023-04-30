# -*- coding: utf-8 -*-
import glob
import logging
import logging.handlers
import os
import platform
import sys
import argparse
import traceback
import warnings
# from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# import pyproj
import xarray as xr
# from mizani.transforms import trans
from scipy.stats import linregress
import pandas as pd
import math
from scipy import spatial
from pandas.tseries.offsets import Day, Hour, Minute, Second
from scipy.interpolate import Rbf

# os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share"
# from pygem import IDW
# import eccodes
# import pygrib
# import pykrige.kriging_tools as kt
# import haversine as hs
import pytz
import pvlib
import pandas as pd

# from auto_ts import auto_timeseries
# from plotnine import ggplot
# from pycaret.regression import setup
# from pycaret.regression import compare_models
from pycaret.regression import *
import numpy as np
import pandas as pd
import datetime
# from sklearn.model_selection import train_test_split
import time
# import h2o
# from h2o.automl import H2OAutoML
# from h2o.estimators.gbm import H2OGradientBoostingEstimator
# from pycaret.utils import check_metric

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
dtKst = datetime.timedelta(hours=9)

# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.node()
        , prjName
        , datetime.datetime.now().strftime("%Y%m%d")
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
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
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

        log.info("[CHECK] {} / val : {}".format(key, val))

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
    # Python을 이용한 한국 월별 수출액과 코스피 종간 간의 회귀분석

    # conda activate
    # python3 TalentPlatform-LSH0255-ASOS.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "2020-09"

    # python3 TalentPlatform-LSH0255-ASOS.py --inpPattzKsth "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "2020-09" &
    # nohup bash RunShell-Python.sh "2020-10" &

    #  /SYSTEMS/OUTPUT/OBS/202109/01/AWS_OBS_202109010000.txt
    # python3 ${contextPath}/TalentPlatform-LSH0255-GK2A.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "$1" --endDate "$2"
    # python3 TalentPlatform-LSH0255-GK2A.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "20210901" --endDate "20210902"
    # bash RunShell-Python.sh "20210901" "20210902"

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PV'

    prjName = 'test'
    serviceName = 'LSH0255'

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

        # import pandas as pd
        # import numpy as np

        try:
            if (platform.system() == 'Windows'):

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2019-01-01'
                    , 'endDate': '2021-12-31'
                }

                globalVar['inpPath'] = 'E:/DATA'
                globalVar['outPath'] = 'E:/DATA'

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': globalVar['srtDate']
                    , 'endDate': globalVar['endDate']
                }

            inpPosFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/GA_STN_INFO.xlsx')
            posData = pd.read_excel(inpPosFile, engine='openpyxl')
            posDataL1 = posData[['id', 'lat', 'lon']]

            # 예측 데이터
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))

            inpData = pd.DataFrame({
                'dtDate': dtIncDateList
                # , 'dtDateKst': dtIncDateList.tz_localize(tzKst).tz_convert(tzKst)
                , 'dtDateKst': dtIncDateList + dtKst
            })

            # 테스트 데이터
            # dtSrtDate = pd.to_datetime('2021-01-01', format='%Y-%m-%d')
            # dtEndDate = pd.to_datetime('2021-12-31', format='%Y-%m-%d')
            # dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))
            #
            # testData = pd.DataFrame({
            #     'dtDate': dtIncDateList
            #     , 'dtDateKst': dtIncDateList.tz_localize(tzKst).tz_convert(tzKst)
            # })

            # print(posId, posLat, posLon)

            # *******************************************************
            # 관측자료 읽기
            # *******************************************************
            inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST_20220123/PV/GA_Yield_2019_2022.xlsx')

            pvData = pd.DataFrame()
            for sheetInfo in range(2019, 2023):
                tmpData = pd.read_excel(inpFile, sheet_name=str(sheetInfo), engine='openpyxl')
                pvData = pvData.append(tmpData)

            # 이름 변경
            pvDataL1 = pvData.rename(
                columns={
                    'Vkey': 'id'
                    , 'localdate': 'time'
                    , 'totalYield': 'pv'
                }
            )

            # trainData min : 2020-09-01 00:00:00 / max : 2021-09-12 00:00:00
            pvDataL1['dtDateKst'] = pd.to_datetime(pvDataL1['time'], format='%Y-%m-%d %H')
            # pvDataL1['dtDateKst'] = pvDataL1['dtDate'].dt.tz_localize(tzKst).dt.tz_convert(tzKst)
            pvDataL1['dtDate'] = pvDataL1['dtDateKst'] - dtKst
            # log.info("[CHECK] trainData min : {} / max : {}".format(pvDataL2['dtDate'].min(), pvDataL2['dtDate'].max()))

            # GK2A 데이터
            # inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/SAT/GK2A_*.nc')
            inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST_20220123/SAT/GK2A_*.nc')
            fileList = sorted(glob.glob(inpFile))
            gk2aData = xr.open_mfdataset(fileList)

            # H8
            # inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/SAT/H8_*.nc')
            inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST_20220123/SAT/H8_*.nc')
            fileList = sorted(glob.glob(inpFile))
            h8Data = xr.open_mfdataset(fileList)

            for i, posInfo in posDataL1.iterrows():
                posId = int(posInfo['id'])
                posLat = posInfo['lat']
                posLon = posInfo['lon']

                log.info("[CHECK] posId (posLon, posLat) : {} ({}. {})".format(posId, posLon, posLat))

                # pv 데이터
                pvDataL2 = pvDataL1.loc[(pvDataL1['id'] == posId) & (pvDataL1['pv'] > 0)]

                # ASOS 데이터
                # inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/OBS/ASOS_OBS_*.nc')
                inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST_20220123/OBS/ASOS_OBS_*.nc')
                fileList = sorted(glob.glob(inpFile))

                asosDataL5 = pd.DataFrame()
                for fileInfo in fileList:
                    asosData = xr.open_mfdataset(fileInfo)
                    # asosDataL1 = asosData.where((asosData['CA_TOT'] >= 0) & (asosData['PA'] >= 940))
                    # asosDataL2 = asosData.interpolate_na()
                    asosDataL3 = asosData.sel(lat=posLat, lon=posLon)
                    asosDataL4 = asosDataL3.to_dataframe()
                    # asosDataL4['dtDateKst'] = asosDataL4.index.tz_localize(tzKst).tz_convert(tzKst)
                    asosDataL4['dtDateKst'] = asosDataL4.index
                    asosDataL5 = asosDataL5.append(asosDataL4)

                asosDataL6 = asosDataL5.drop_duplicates(['dtDateKst'])

                # PM10 데이터
                # inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/OBS/PM10_OBS_*.nc')
                inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST_20220123/OBS/PM10_OBS_*.nc')
                fileList = sorted(glob.glob(inpFile))

                pmDataL5 = pd.DataFrame()
                for fileInfo in fileList:
                    pmData = xr.open_mfdataset(fileInfo)
                    # pmDataL1 = pmData.where(pmData['PM10'] <= 500)
                    # pmDataL2 = pmDataL1.interpolate_na()
                    pmDataL3 = pmData.sel(lat=posLat, lon=posLon)
                    pmDataL4 = pmDataL3.to_dataframe()
                    # pmDataL4['dtDateKst'] = pmDataL4.index.tz_localize(tzKst).tz_convert(tzKst)
                    pmDataL4['dtDateKst'] = pmDataL4.index
                    pmDataL5 = pmDataL5.append(pmDataL4)

                pmDataL6 = pmDataL5.drop_duplicates(['dtDateKst'])

                # GK2A 데이터
                # gk2aDataL1 = gk2aData.where(gk2aData['DSR'] > 0)
                # gk2aDataL2 = gk2aData.interpolate_na(method='linear', fill_value="extrapolate")
                # gk2aDataL2 = gk2aDataL1
                gk2aDataL3 = gk2aData.sel(lat=posLat, lon=posLon)
                gk2aDataL4 = gk2aDataL3.to_dataframe()
                # gk2aDataL4['dtDateKst'] = gk2aDataL4.index.tz_localize(tzUtc).tz_convert(tzKst)
                gk2aDataL4['dtDateKst'] = gk2aDataL4.index + dtKst

                # H8
                h8DataL3 = h8Data.sel(lat=posLat, lon=posLon)
                h8DataL4 = h8DataL3.to_dataframe()
                # h8DataL4['dtDateKst'] = h8DataL4.index.tz_localize(tzUtc).tz_convert(tzKst)
                h8DataL4['dtDateKst'] = h8DataL4.index + dtKst

                inpDataL2 = inpData.merge(pvDataL2, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                    .merge(asosDataL6, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                    .merge(pmDataL6, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                    .merge(gk2aDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                    .merge(h8DataL4, how='left', left_on='dtDateKst', right_on='dtDateKst')

                trainData = inpDataL2.rename( {'dtDate_x': 'dtDate'}, axis='columns')
                trainDataL1 = trainData[['dtDate', 'dtDateKst', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'PM10', 'SWR', 'pv']]
                trainDataL2 = trainDataL1.reset_index(drop=True)

                trainDataL2['CA_TOT'] = np.where(trainDataL2['CA_TOT'] < 0, 0, trainDataL2['CA_TOT'])
                trainDataL2['CA_TOT'] = np.where(trainDataL2['CA_TOT'] > 1, 1, trainDataL2['CA_TOT'])

                # i = 0
                for i in trainDataL2.index:
                    lat = posLat
                    lon = posLon
                    pa = trainDataL2._get_value(i, 'PA') * 100.0
                    ta = trainDataL2._get_value(i, 'TA')
                    # dtDateTime = trainDataL2._get_value(i, 'dtDateKst')
                    dtDateTime = trainDataL2._get_value(i, 'dtDate')

                    solPosInfo = pvlib.solarposition.get_solarposition(dtDateTime, lat, lon, pressure=pa, temperature=ta, method='nrel_numpy')
                    trainDataL2._set_value(i, 'sza', solPosInfo['zenith'].values)
                    trainDataL2._set_value(i, 'aza', solPosInfo['azimuth'].values)
                    trainDataL2._set_value(i, 'et', solPosInfo['equation_of_time'].values)

                saveXlsxFile = '{}/{}/{}-SRV{:05d}-{}-{}-{}.xlsx'.format(globalVar['outPath'], 'ACT', serviceName, posId, 'final', 'proc', 'act')
                os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)
                log.info("[CHECK] saveXlsxFile : {}".format(saveXlsxFile))
                trainDataL2.to_excel(saveXlsxFile, index=False)


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
