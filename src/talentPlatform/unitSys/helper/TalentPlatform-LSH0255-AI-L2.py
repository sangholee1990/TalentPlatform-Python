# -*- coding: utf-8 -*-
import glob
import seaborn as sns
import logging
import logging.handlers
import os
import platform
import sys
import argparse
import traceback
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
from mizani.transforms import trans
from scipy.stats import linregress
import pandas as pd
import cartopy.crs as ccrs
import math
from scipy import spatial
from pandas.tseries.offsets import Day, Hour, Minute, Second
from scipy.interpolate import Rbf
from numpy import zeros, newaxis
import sklearn
from sklearn.preprocessing import *

# os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share"
# from pygem import IDW
import eccodes
import pygrib
# import pykrige.kriging_tools as kt
import haversine as hs
import pytz
import pvlib
import pandas as pd
import matplotlib.dates as mdates


# from auto_ts import auto_timeseries
from plotnine import ggplot
from pycaret.regression import setup
from pycaret.regression import compare_models
from pycaret.regression import *
from pycaret.utils import check_metric
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.gbm import H2OGradientBoostingEstimator
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
mpl.rcParams['timezone'] = 'Asia/Seoul'

# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')

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

# 산점도 시각화
def makeUserScatterPlot(prdVal, refVal, xlab, ylab, mainTitle, saveImg, minVal, maxVal, xIntVal, yIntVal, isSame):

    # 그리드 설정
    plt.grid(True)

    plt.scatter(prdVal, refVal)

    # 제목, x축, y축 설정
    plt.title(mainTitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
    Bias = np.mean(prdVal - refVal)
    rBias = (Bias / np.mean(refVal)) * 100.0
    RMSE = np.sqrt(np.mean((prdVal - refVal) ** 2))
    rRMSE = (RMSE / np.mean(refVal)) * 100.0
    MAPE = np.mean(np.abs((prdVal - refVal) / prdVal)) * 100.0

    # 선형회귀곡선에 대한 계산
    lmFit = linregress(prdVal, refVal)
    slope = lmFit[0]
    intercept = lmFit[1]
    R = lmFit[2]
    Pvalue = lmFit[3]
    N = len(prdVal)

    lmfit = (slope * prdVal) + intercept
    # plt.plot(prdVal, lmfit, color='red', linewidth=2,linestyle="-")
    plt.plot([minVal, maxVal], [minVal, maxVal], color="black", linestyle="--", linewidth=2)
    plt.plot(prdVal, lmfit, color='red', linewidth=2, linestyle="-")
    # 라벨 추가
    plt.annotate('%s = %.2f x (%s) + %.2f' % (ylab, slope, xlab, intercept),
                 xy=(minVal + xIntVal, maxVal - yIntVal),
                 color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('R = %.2f  (p-value < %.2f)' % (R, Pvalue), xy=(minVal + xIntVal, maxVal - yIntVal * 2),
                 color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')

    if (isSame == True):
        # plt.axes().set_aspect('equal')
        # plt.axes().set_aspect(1)
        # plt.gca().set_aspect('equal')
        plt.xlim(minVal, maxVal)
        plt.ylim(minVal, maxVal)

        plt.annotate('Bias = %.2f  (%%Bias = %.2f %%)' % (Bias, rBias), xy=(minVal + xIntVal, maxVal - yIntVal * 3),
                     color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
        plt.annotate('RMSE = %.2f  (%%RMSE = %.2f %%)' % (RMSE, rRMSE), xy=(minVal + xIntVal, maxVal - yIntVal * 4),
                     color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
        plt.annotate('MAPE = %.2f %%' % (MAPE), xy=(minVal + xIntVal, maxVal - yIntVal * 5),
                     color='black', xycoords='data', horizontalalignment='left', verticalalignment='center')
        plt.annotate('N = %d' % N, xy=(minVal + xIntVal, maxVal - yIntVal * 6), color='black',
                     xycoords='data', horizontalalignment='left', verticalalignment='center')
    else:
        plt.annotate('N = %d' % N, xy=(minVal + xIntVal, maxVal - yIntVal * 3), color='black',
                     xycoords='data', horizontalalignment='left', verticalalignment='center')

    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


# 시계열 시각화
def makeUserTimeSeriesPlot(dtDate, mlPrdVal, dlPrdVal, refVal, mlPrdValLabel, dlPrdValLabel, refValLabel, xlab, ylab, mainTitle, saveImg, isFore):

    # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
    mlRMSE = np.sqrt(np.mean((mlPrdVal - refVal) ** 2))
    mlReRMSE = (mlRMSE / np.mean(refVal)) * 100.0

    dlRMSE = np.sqrt(np.mean((dlPrdVal - refVal) ** 2))
    dlReRMSE = (dlRMSE / np.mean(refVal)) * 100.0

    # 선형회귀곡선에 대한 계산
    mlFit = linregress(mlPrdVal, refVal)
    mlR = mlFit[2]

    dlFit = linregress(dlPrdVal, refVal)
    dlR = dlFit[2]

    prdValLabel_ml = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(mlPrdValLabel, mlR, mlReRMSE)
    prdValLabel_dnn = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(dlPrdValLabel, dlR, dlReRMSE)

    plt.grid(True)

    plt.plot(dtDate, mlPrdVal, label=prdValLabel_ml, marker='o')
    plt.plot(dtDate, dlPrdVal, label=prdValLabel_dnn, marker='o')
    plt.plot(dtDate, refVal, label=refValLabel, marker='o')

    # 제목, x축, y축 설정
    plt.title(mainTitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.ylim(0, 1000)

    if (isFore == True):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.gcf().autofmt_xdate()
        plt.xticks(rotation=45, ha='right')

    else:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.gcf().autofmt_xdate()
        plt.xticks(rotation=0, ha='right')

    plt.legend(loc='upper left')

    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


def cartesian(latitude, longitude, elevation=0):
    # Convert to radians
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371  # 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)

    return (X, Y, Z)


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

    # python3 TalentPlatform-LSH0255-ASOS.py --inpPath "/SYSTEMS/OUTPUT" --outPath "/SYSTEMS/OUTPUT" --srtDate "2020-09" &
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

        try:

            import pandas as pd

            if (platform.system() == 'Windows'):

                globalVar['inpPath'] = 'E:/DATA/OUTPUT'
                globalVar['outPath'] = 'E:/DATA/OUTPUT'

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2021-10-01'
                    , 'endDate': '2021-11-01'
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
            posData = pd.read_excel(inpPosFile)
            posDataL1 = posData[['id', 'lat', 'lon']]

            lat1D = np.array(posDataL1['lat'])
            lon1D = np.array(posDataL1['lon'])


            dtSrtDate = pd.to_datetime('2020-09-01', format='%Y-%m-%d')
            dtEndDate = pd.to_datetime('2020-09-02', format='%Y-%m-%d')
            # dtSrtDate = pd.to_datetime('2021-09-12', format='%Y-%m-%d')
            # dtEndDate = pd.to_datetime('2021-10-31', format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))
            # dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))
            # dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))

            testData = pd.DataFrame({
                'dtDate': dtIncDateList
                , 'dtDateKst': dtIncDateList.tz_localize(tzKst).tz_convert(tzKst)
            })

            for i, posInfo in posDataL1.iterrows():
                posId = posInfo['id']
                posLat = posInfo['lat']
                posLon = posInfo['lon']


            # *******************************************************
            # 관측자료 읽기
            # *******************************************************
            # inpFilePattern = 'PV/{}/{}/{}/UMKR_l015_unis_*_{}.grb2'.format(dtDateYm, dtDateDay, dtDateHour,
            #                                                                   dtDateYmdHm)
            inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/PV/DW태양광발전소발전량(9월12_10월29일).xlsx')
            trainData = pd.read_excel(inpFile, sheet_name='Sheet1')


            # trainData min : 2020-09-01 00:00:00 / max : 2021-09-12 00:00:00
            trainData['dtDate'] = pd.to_datetime(trainData['time'], format='%Y-%m-%d %H', utc=False)
            trainData['dtDateKst'] = trainData['dtDate'].dt.tz_localize(tzKst).dt.tz_convert(tzKst)
            # trainData.index = trainData['dtDate']
            # trainData.dtDate = trainData.dtDate.tz_localize(tzKst).tz_convert(tzKst)
            # trainData['dtDate'] = pd.to_datetime(trainData['time'], format='%Y-%m-%d %H', utc=False).tz_convert('Asia/Seoul')

            trainDataL1 = trainData.loc[trainData['dtDate'].between('2020-09-01', '2021-09-12')]
            trainDataL2 = trainDataL1.loc[trainDataL1['pv'] > 0]

            # 2021-09-12 00:00:00 / max : 2021-10-29 23:00:00
            # testData['dtDate'] = pd.to_datetime(testData['time'], format='%Y-%m-%d %H', utc=False)
            # testData.index = testData['dtDate']
            # testData.index = testData.index.tz_localize(tzKst).tz_convert(tzKst)

            orgSheet2Data = pd.read_excel(inpFile, sheet_name='Sheet2')
            orgSheet2Data['dtDate'] = pd.to_datetime(orgSheet2Data['time'], format='%Y-%m-%d %H', utc=False)
            orgSheet2Data['dtDateKst'] = orgSheet2Data['dtDate'].dt.tz_localize(tzKst).dt.tz_convert(tzKst)
            orgSheet2DataL1 = orgSheet2Data.loc[orgSheet2Data['pv'] > 0]
            # testData = orgSheet2DataL1.loc[orgSheet2DataL1['dtDate'].between('2021-09-12', '2021-09-30')]
            # validData = orgSheet2DataL1.loc[orgSheet2DataL1['dtDate'].between('2021-10-01', '2021-10-29')]

            log.info("[CHECK] trainData min : {} / max : {}".format(trainData['dtDate'].min(), trainData['dtDate'].max()))
            log.info("[CHECK] testData min : {} / max : {}".format(orgSheet2DataL1['dtDate'].min(), orgSheet2DataL1['dtDate'].max()))

            # plt.plot(trainData['dtDate'], trainData['pv'])
            # plt.show()
            #
            # plt.plot(testData['dtDate'], testData['pv'])
            # plt.plot(testData['dtDate'], testData['prd'])
            # plt.plot(testData['dtDate'], testData['prd2'])
            # plt.show()
            # dtIncDateList

            # ASOS 데이터
            inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/OBS/ASOS_OBS_*.nc')
            fileList = sorted(glob.glob(inpFile))
            asosData = xr.open_mfdataset(fileList)
            # asosDataL1 = asosData.where((asosData['CA_TOT'] >= 0) & (asosData['PA'] >= 940) & (asosData['SS'] > 0))
            asosDataL1 = asosData.where((asosData['CA_TOT'] >= 0) & (asosData['PA'] >= 940))
            asosDataL2 = asosData.interpolate_na()
            # asosDataL2 = asosData
            asosDataL3 = asosDataL2.sel(lat=posLat, lon=posLon)
            asosDataL4 = asosDataL3.to_dataframe()
            asosDataL4['dtDateKst'] = asosDataL4.index.tz_localize(tzKst).tz_convert(tzKst)

            # plt.scatter(asosDataL1['time'], asosDataL1['CA_TOT'])
            # plt.scatter(asosDataL2['time'], asosDataL2['CA_TOT'])
            # plt.scatter(asosDataL1['time'], asosDataL1['HM'])
            # plt.scatter(asosDataL1['time'], asosDataL1['PA'])
            # plt.scatter(asosDataL2['time'], asosDataL2['SI'])
            # plt.scatter(asosDataL2['time'], asosDataL2['SS'])
            # plt.scatter(asosDataL1['time'], asosDataL1['TA'])
            # plt.scatter(asosDataL1['time'], asosDataL1['TD'])
            # plt.scatter(asosDataL1['time'], asosDataL1['WD'])
            # plt.scatter(asosDataL1['time'], asosDataL1['WS'])
            # plt.scatter(asosDataL2['time'], asosDataL2['WS'])
            # plt.scatter(asosDataL3['time'], asosDataL3['WS'])
            # plt.show()

            # PM10 데이터
            inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/OBS/PM10_OBS_*.nc')
            fileList = sorted(glob.glob(inpFile))
            pmData = xr.open_mfdataset(fileList)
            pmDataL1 = pmData.where(pmData['PM10'] <= 500)
            pmDataL2 = pmDataL1.interpolate_na()
            # pmDataL2 = pmDataL1
            pmDataL3 = pmDataL2.sel(lat=posLat, lon=posLon)
            pmDataL4 = pmDataL3.to_dataframe()
            pmDataL4['dtDateKst'] = pmDataL4.index.tz_localize(tzKst).tz_convert(tzKst)

            # plt.scatter(pmDataL2['time'], pmDataL2['PM10'])
            # plt.scatter(pmDataL2['time'], pmDataL2['PM10'])
            # plt.scatter(pmDataL3['dtDateKst'], pmDataL3['PM10'])
            # plt.show()

            # GK2A 데이터
            inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/SAT/GK2A_*.nc')
            fileList = sorted(glob.glob(inpFile))
            gk2aData = xr.open_mfdataset(fileList)
            gk2aDataL1 = gk2aData.where(gk2aData['DSR'] > 0)
            gk2aDataL2 = gk2aDataL1.interpolate_na(method = 'linear', fill_value="extrapolate")
            # gk2aDataL2 = gk2aDataL1
            gk2aDataL3 = gk2aDataL2.sel(lat=posLat, lon=posLon)
            gk2aDataL4 = gk2aDataL3.to_dataframe()
            gk2aDataL4['dtDateKst'] = gk2aDataL4.index.tz_localize(tzUtc).tz_convert(tzKst)


            # plt.scatter(gk2aDataL1['time'], gk2aDataL1['DSR'])
            # plt.scatter(gk2aDataL2['time'], gk2aDataL2['DSR'])
            # plt.scatter(gk2aDataL2['time'], gk2aDataL2['CA'])
            # plt.scatter(gk2aDataL2['time'], gk2aDataL2['CLD'])
            # plt.scatter(gk2aDataL2['time'], gk2aDataL2['CF'])
            # plt.show()

            # H8
            inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/SAT/H8_*.nc')
            fileList = sorted(glob.glob(inpFile))
            h8Data = xr.open_mfdataset(fileList)
            h8DataL1 = h8Data.where(h8Data['SWR'] > 0)
            h8DataL2 = h8DataL1.interpolate_na()
            # h8DataL2 = h8DataL1
            h8DataL3 = h8DataL2.sel(lat=posLat, lon=posLon)
            h8DataL4 = h8DataL3.to_dataframe()
            h8DataL4['dtDateKst'] = h8DataL4.index.tz_localize(tzUtc).tz_convert(tzKst)

            # plt.scatter(h8DataL1['time'], h8DataL1['SWR'])
            # plt.scatter(h8DataL2['time'], h8DataL2['SWR'])
            # plt.show()
            #
            # trainDataL3 = prdData.merge(trainDataL2, how='left', left_on='dtDateKst', right_on='dtDateKst') \
            #     .merge(asosDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
            #     .merge(pmDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
            #     .merge(gk2aDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
            #     .merge(h8DataL4, how='left', left_on='dtDateKst', right_on='dtDateKst')

            testDataL1 = testData.merge(orgSheet2DataL1, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                .merge(asosDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                .merge(pmDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                .merge(gk2aDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                .merge(h8DataL4, how='left', left_on='dtDateKst', right_on='dtDateKst')

            # 학습 모델 저장
            saveMlModel = '{0}/{1}-{2}-{3}-{4}-{5}.model.pkl'.format(globalVar['outPath'], serviceName, 'final', 'pycaret', 'train', '*')
            saveMlModelList = sorted(glob.glob(saveMlModel), reverse=True)

            # 학습 모델 불러오기
            mlModel = load_model(os.path.splitext(saveMlModelList[0])[0])

            # H2o 딥러닝
            h2o.init()
            saveDlModel = '{0}/{1}-{2}-{3}-{4}-{5}.model'.format(globalVar['outPath'], serviceName, 'final', 'h2o','train', '*')
            saveDlModelList = sorted(glob.glob(saveDlModel), reverse=True)
            dlModel = h2o.load_model(path=saveDlModelList[0])


            testDataL2 = testDataL1
            for i in testDataL2.index:
                lat = posLat
                lon = posLon
                pa = testDataL2._get_value(i, 'PA')
                ta = testDataL2._get_value(i, 'TA')
                dtDateTime = testDataL2._get_value(i, 'dtDateKst')

                solPosInfo = pvlib.solarposition.get_solarposition(dtDateTime, lat, lon, pressure=pa, temperature=ta,
                                                                   method='nrel_numpy')
                testDataL2._set_value(i, 'sza', solPosInfo['zenith'].values)
                testDataL2._set_value(i, 'aza', solPosInfo['azimuth'].values)
                testDataL2._set_value(i, 'et', solPosInfo['equation_of_time'].values)

            testDataL3 = testDataL2[['dtDateKst', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'sza', 'aza', 'et', 'pv']].dropna().reset_index(drop=True)
            testDataL4 = predict_model(mlModel, data=testDataL3).rename({'Label': 'ML'}, axis='columns')[['dtDateKst', 'ML']]
            testDataL4['DL'] = dlModel.predict(h2o.H2OFrame(testDataL3)).as_data_frame()

            testDataL5 = testDataL3.merge(testDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst')


            # dtDateList = testDataL5['dtDateKst'].dt.strftime('%Y%m%d').unique()
            # for j, dtDateInfo in enumerate(dtDateList):
            #
            #     testDataL6 = testDataL5.loc[
            #         testDataL5['dtDateKst'].dt.strftime('%Y%m%d') == dtDateInfo
            #         ]
            #
            #     mainTitle = '[{}] {}'.format(dtDateInfo, '기상 실황 정보 (지상 관측소, 위성)를 활용한 예측 시계열')
            #     saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            #     makeUserTimeSeriesPlot(testDataL6['dtDateKst'], testDataL6['ML'], testDataL6['DL'], testDataL6['pv'], '예측 (머신러닝)', '예측 (딥러닝)', '실측 (발전량)', '시간 (시)', '발전량', mainTitle, saveImg, False)
            #
            # mainTitle = '[{}-{}] {}'.format(min(dtDateList), max(dtDateList), '기상 실황 정보 (지상 관측소, 위성)를 활용한 머신러닝 (예측) 산점도')
            # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            # makeUserScatterPlot(testDataL5['ML'], testDataL5['pv'], '머신러닝', '실측', mainTitle, saveImg, 0, 1000, 20, 60, True)
            #
            # mainTitle = '[{}-{}] {}'.format(min(dtDateList), max(dtDateList), '기상 실황 정보 (지상 관측소, 위성)를 활용한 딥러닝 (예측) 산점도')
            # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            # makeUserScatterPlot(testDataL5['DL'], testDataL5['pv'], '딥러닝', '실측', mainTitle, saveImg, 0, 1000, 20, 60, True)



            # UM
            inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/MODEL/UMKR_l015_unis_*.nc')
            fileList = sorted(glob.glob(inpFile))
            umData = xr.open_mfdataset(fileList)
            umDataL1 = umData.interpolate_na(method='linear', fill_value="extrapolate")


            umDataL10 = pd.DataFrame()

            # saveCsvFile = '{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, 'TrainData')

            # CSV 파일 저장
            # umDataL10.to_csv(saveCsvFile, index=False)

            try:
                dtAnaTimeList = umData['anaTime'].values
                for j, dtAnaTimeInfo in enumerate(dtAnaTimeList):
                    print(dtAnaTimeInfo)

                    umDataL2 = umDataL1.sel(lat=posLat, lon=posLon, anaTime=dtAnaTimeInfo)
                    umDataL3 = umDataL2.to_dataframe()
                    umDataL3['dtDate'] = umDataL3.index
                    umDataL3['dtDateKst'] = umDataL3.index.tz_localize(tzUtc).tz_convert(tzKst)

                    umDataL4 = umDataL3.rename(
                        {'SS': 'SWR'}, axis='columns'
                    )[['dtDate', 'dtDateKst', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR']]

                    umDataL5 = umDataL4
                    for i in umDataL5.index:
                        lat = posLat
                        lon = posLon
                        pa = umDataL5._get_value(i, 'PA')
                        ta = umDataL5._get_value(i, 'TA')
                        dtDateTime = umDataL5._get_value(i, 'dtDateKst')

                        solPosInfo = pvlib.solarposition.get_solarposition(dtDateTime, lat, lon, pressure=pa,
                                                                           temperature=ta, method='nrel_numpy')
                        umDataL5._set_value(i, 'sza', solPosInfo['zenith'].values)
                        umDataL5._set_value(i, 'aza', solPosInfo['azimuth'].values)
                        umDataL5._set_value(i, 'et', solPosInfo['equation_of_time'].values)

                    umDataL6 = umDataL5.merge(orgSheet2DataL1, how='left', left_on='dtDateKst', right_on='dtDateKst')

                    umDataL7 = umDataL6[
                        ['dtDateKst', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'sza', 'aza', 'et',
                         'pv']].dropna().reset_index(drop=True)
                    umDataL8 = predict_model(mlModel, data=umDataL7).rename({'Label': 'ML'}, axis='columns')[
                        ['dtDateKst', 'ML']]
                    umDataL8['DL'] = dlModel.predict(h2o.H2OFrame(umDataL7)).as_data_frame()

                    umDataL9 = umDataL7.merge(umDataL8, how='left', left_on='dtDateKst', right_on='dtDateKst')

                    # mainTitle = '[{}] {}'.format(pd.to_datetime(dtAnaTimeInfo).strftime('%Y%m%d'), '기상 예보 정보 (수치모델)를 활용한 48시간 예측 시계열')
                    # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                    # makeUserTimeSeriesPlot(umDataL9['dtDateKst'], umDataL9['ML'], umDataL9['DL'], umDataL9['pv'], '예측 (머신러닝)', '예측 (딥러닝)', '실측 (발전량)', '시간 (시)', '발전량', mainTitle, saveImg, True)

                    umDataL9['anaTime'] = pd.to_datetime(dtAnaTimeInfo).strftime('%Y%m%d')
                    umDataL10 = umDataL10.append(umDataL9)

            except Exception as e:
                log.error("Exception : {}".format(e))

            # dtAnaTimeFmtList = pd.to_datetime(dtAnaTimeList).strftime('%Y%m%d')

            # mainTitle = '[{}-{}] {}'.format(min(dtAnaTimeFmtList), max(dtAnaTimeFmtList), '기상 예보 정보 (수치모델)를 활용한 머신러닝 (48시간 예측) 산점도')
            # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            # makeUserScatterPlot(umDataL10['ML'], umDataL10['pv'], '머신러닝', '실측', mainTitle, saveImg, 0, 1000, 20, 60, True)
            #
            # mainTitle = '[{}-{}] {}'.format(min(dtAnaTimeFmtList), max(dtAnaTimeFmtList), '기상 예보 정보 (수치모델)를 활용한 딥러닝 (48시간 예측) 산점도')
            # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            # makeUserScatterPlot(umDataL10['DL'], umDataL10['pv'], '딥러닝', '실측', mainTitle, saveImg, 0, 1000, 20, 60, True)



            # umDataL10
            saveCsvFile = '{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, 'TrainData')
            # CSV 파일 저장
            umDataL10.to_csv(saveCsvFile, index=False)



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
