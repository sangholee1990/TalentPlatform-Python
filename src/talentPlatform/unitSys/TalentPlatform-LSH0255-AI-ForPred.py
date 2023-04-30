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
# import pyproj
# import xarray as xr
# from mizani.transforms import trans
from scipy.stats import linregress
import pandas as pd
# import cartopy.crs as ccrs
import math
from scipy import spatial
from pandas.tseries.offsets import Day, Hour, Minute, Second
from scipy.interpolate import Rbf
from numpy import zeros, newaxis
import sklearn
from sklearn.preprocessing import *

# os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share"
# from pygem import IDW
# import eccodes
# import pygrib
# import pykrige.kriging_tools as kt
# import haversine as hs
import pytz
import pvlib
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.cm as cm


# from auto_ts import auto_timeseries
# from plotnine import ggplot
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
import shutil
import mariadb
import pymysql
import re
import configparser
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

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

    log.info('[START] {}'.format('makeUserScatterPlot'))

    result = None

    try:

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

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeUserScatterPlot'))


# 빈도분포 2D 시각화
def makeUserHist2DPlot(prdVal, refVal, xlab, ylab, mainTitle, saveImg, minVal, maxVal, xIntVal, yIntVal, nbinVal, isSame):

    log.info('[START] {}'.format('makeUserHist2DPlot'))

    result = None

    try:

        # 그리드 설정
        plt.grid(True)

        # plt.scatter(prdVal, refVal)
        # nbins = 250
        hist2D, xEdge, yEdge = np.histogram2d(prdVal, refVal, bins=nbinVal)
        # hist2D, xEdge, yEdge = np.histogram2d(prdVal, refVal)

        # hist2D 전처리
        hist2D = np.rot90(hist2D)
        hist2D = np.flipud(hist2D)

        # 마스킹
        hist2DVal = np.ma.masked_where(hist2D == 0, hist2D)

        plt.pcolormesh(xEdge, yEdge, hist2DVal, cmap=cm.get_cmap('jet'), vmin=0, vmax=100)


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

        # 컬러바
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('빈도수')

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

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeUserHist2DPlot'))


# 시계열 시각화
def makeUserTimeSeriesPlot(dtDate, mlPrdVal, dlPrdVal, refVal, mlPrdValLabel, dlPrdValLabel, refValLabel, xlab, ylab, mainTitle, saveImg, isFore):

    log.info('[START] {}'.format('makeUserTimeSeriesPlot'))

    result = None

    try:
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

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeUserTimeSeriesPlot'))

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

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    # contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PV'

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

            # h2o.init()

            if (platform.system() == 'Windows'):

                globalVar['inpPath'] = 'E:/DATA/OUTPUT'
                globalVar['outPath'] = 'E:/DATA/OUTPUT'

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2021-10-01'
                    , 'endDate': '2021-11-01'
                    
                    # 모델 버전 (날짜)
                    , 'modelVer': '*'
                    # , 'modelVer': '20220220'
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

            isDlModelInit = False

            # modelDirKeyList = ['AI']
            # figActDirKeyList = ['ACT']
            # figForDirKeyList = ['FOR']

            # modelDirKeyList = ['AI_2Y', 'AI_7D', 'AI_15D', 'AI_1M', 'AI_3M', 'AI_6M']
            # figActDirKeyList = ['ACT_2Y', 'ACT_7D', 'ACT_15D', 'ACT_1M', 'ACT_3M', 'ACT_6M']
            # figForDirKeyList = ['FOR_2Y', 'FOR_7D', 'FOR_15D', 'FOR_1M', 'FOR_3M', 'FOR_6M']
            modelDirKeyList = ['AI_2Y']
            figActDirKeyList = ['ACT_2Y']
            figForDirKeyList = ['FOR_2Y']
            modelVer = sysOpt['modelVer']

            # DB 연결 정보
            pymysql.install_as_MySQLdb()

            # 환경 변수 읽기
            config = configparser.ConfigParser()
            config.read(globalVar['sysPath'], encoding='utf-8')
            dbUser = config.get('mariadb', 'user')
            dbPwd = config.get('mariadb', 'pwd')
            dbHost = config.get('mariadb', 'host')
            dbPort = config.get('mariadb', 'port')
            dbName = config.get('mariadb', 'dbName')

            # dbCon = create_engine('mysql://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName))
            dbCon = create_engine('mariadb://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName))

            for k, modelDirKey in enumerate(modelDirKeyList):
                figActDirKey = figActDirKeyList[k]
                figForDirKey = figForDirKeyList[k]

                log.info("[CHECK] modelDirKey : {}".format(modelDirKey))
                log.info("[CHECK] figActDirKey : {}".format(figActDirKey))
                log.info("[CHECK] figForDirKey : {}".format(figForDirKey))

                for i, posInfo in posDataL1.iterrows():
                    posId = int(posInfo['id'])
                    posLat = posInfo['lat']
                    posLon = posInfo['lon']

                    if (not re.search('17', str(posId))): continue

                    # *******************************************************
                    # 관측자료 읽기
                    # *******************************************************
                    inpFile = '{}/{}/{}-SRV{:05d}-{}-{}-{}.xlsx'.format(globalVar['outPath'], 'FOR', serviceName, posId, 'final', 'proc', 'for')
                    fileList = sorted(glob.glob(inpFile))

                    # 파일 없을 경우 예외 처리
                    if fileList is None or len(fileList) < 1:
                        log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                        continue

                    fileInfo = fileList[0]
                    inpData = pd.read_excel(fileInfo, engine='openpyxl')

                    inpDataL1 = inpData.rename({'dtDate_x': 'dtDate'}, axis='columns')

                    # **********************************************************************************************************
                    # 머신러닝
                    # **********************************************************************************************************
                    # saveMlModel = '{}/{}/{}-SRV{:05d}-{}-{}-{}-{}.model.pkl'.format(globalVar['outPath'], modelDirKey, serviceName, posId, 'final', 'pycaret', 'for', modelVer)
                    # saveMlModelList = sorted(glob.glob(saveMlModel), reverse=True)
                    #
                    # if (len(saveMlModelList) > 0):
                    #     saveMlModelInfo = saveMlModelList[0]
                    #     log.info("[CHECK] saveMlModelInfo : {}".format(saveMlModelInfo))
                    #
                    #     mlModel = load_model(os.path.splitext(saveMlModelInfo)[0])
                    #
                    # mlModelPred = predict_model(mlModel, data=inpDataL1).rename({'Label': 'ML'}, axis='columns')[['dtDateKst', 'anaTime', 'ML']]
                    # mlModelPred = predict_model(mlModel, data=inpDataL1).rename({'Label': 'ML'}, axis='columns')[['dtDateKst', 'anaTime', 'ML']]

                    # **********************************************************************************************************
                    # 딥러닝
                    # **********************************************************************************************************
                    # saveDlModel = '{}/{}/{}-SRV{:05d}-{}-{}-{}-{}.model'.format(globalVar['outPath'], modelDirKey, serviceName, posId, 'final', 'h2o', 'for', '*')
                    saveDlModel = '{}/{}/{}-SRV{:05d}-{}-{}-{}-{}.model'.format(globalVar['outPath'], modelDirKey, serviceName, posId, 'final', 'h2o', 'for', modelVer)
                    saveDlModelList = sorted(glob.glob(saveDlModel), reverse=True)

                    # 학습 모델 불러오기
                    if (len(saveDlModelList) > 0):
                        saveDlModelInfo = saveDlModelList[0]
                        log.info("[CHECK] saveDlModelInfo : {}".format(saveDlModelInfo))

                        if (isDlModelInit == False):
                            h2o.init()
                            isDlModelInit = True

                        # dlModel = h2o.load_model(path=saveDlModelInfo)
                        dlModel = h2o.import_mojo(saveDlModelInfo)

                    inpDataL1['year'] = inpDataL1['dtDateKst'].dt.strftime('%Y').astype('int64')
                    inpDataL1['month'] = inpDataL1['dtDateKst'].dt.strftime('%m').astype('int64')
                    inpDataL1['day'] = inpDataL1['dtDateKst'].dt.strftime('%d').astype('int64')
                    inpDataL1['hour'] = inpDataL1['dtDateKst'].dt.strftime('%H').astype('int64')

                    # tmpData = inpDataL1[['dtDateKst', 'anaTime', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'pv', 'sza', 'aza', 'et']].dropna().reset_index(drop=True)
                    tmpData = inpDataL1[['year', 'month', 'day', 'hour', 'dtDateKst', 'anaTime', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'pv', 'sza', 'aza', 'et']].dropna().reset_index(drop=True)
                    # tmpData = inpDataL1[['hour', 'dtDateKst', 'anaTime', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'pv', 'sza', 'aza', 'et']].dropna().reset_index(drop=True)
                    dlModelPred = dlModel.predict(h2o.H2OFrame(tmpData)).as_data_frame().rename({'predict': 'DL'}, axis='columns')
                    dlModelPredL1 = pd.concat([tmpData[['dtDateKst', 'anaTime']], dlModelPred], axis=1)

                    # inpDataL2 = inpDataL1.merge(mlModelPred, how='left', left_on=['dtDateKst', 'anaTime'], right_on=['dtDateKst', 'anaTime']) \
                    #     .merge(dlModelPredL1, how='left', left_on=['dtDateKst', 'anaTime'], right_on=['dtDateKst', 'anaTime'])
                    inpDataL2 = inpDataL1.merge(dlModelPredL1, how='left', left_on=['dtDateKst', 'anaTime'], right_on=['dtDateKst', 'anaTime'])

                    # dtDateKst 및 anaTime을 기준으로 중복 제거
                    inpDataL2.drop_duplicates(subset=['dtDateKst', 'anaTime'], inplace=True)
                    inpDataL2 = inpDataL2.reset_index(drop=True)
                    # inpDataL2['anaTime'] = inpDataL2['anaTime'].astype(str)

                    # **********************************************************************************************************
                    # 엑셀 저장
                    # **********************************************************************************************************
                    # saveXlsxFile = '{}/{}/{}-SRV{:05d}-{}-{}-{}.xlsx'.format(globalVar['outPath'], 'FOR', serviceName, posId, 'final', 'pred', 'for')
                    # os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)
                    # log.info("[CHECK] saveXlsxFile : {}".format(saveXlsxFile))
                    # inpDataL2.to_excel(saveXlsxFile, index=False)

                    # **********************************************************************************************************
                    # DB 삽입
                    # **********************************************************************************************************
                    inpDataL2['anaYear'] = inpDataL2['anaTime'].dt.strftime("%Y").astype(str)
                    anaYearList = inpDataL2['anaYear'].unique()

                    # anaYearInfo = anaYearList[0]
                    # for j, anaYearInfo in enumerate(anaYearList):
                    #
                    #     inpDataL3 = inpDataL2.loc[
                    #         inpDataL2['anaYear'] == anaYearInfo
                    #         ].dropna().reset_index(drop=True)
                    #
                    #     if (len(inpDataL3) < 1): continue
                    #
                    #     inpDataL3['SRV'] = 'SRV{:05d}'.format(posId)
                    #     inpDataL3['REG_DATE'] = datetime.now()
                    #     iAnaYear = int(anaYearInfo)
                    #
                    #     dbData = inpDataL3.rename(
                    #         {
                    #             'anaTime': 'ANA_DATE'
                    #             , 'dtDateKst': 'DATE_TIME_KST'
                    #             , 'dtDate': 'DATE_TIME'
                    #             , 'sza': 'SZA'
                    #             , 'aza': 'AZA'
                    #             , 'et': 'ET'
                    #         }
                    #         , axis='columns'
                    #     )
                    #
                    #     dbData = dbData.drop(['id', 'time', 'pv', 'PlantCapacity', 'anaYear'], axis=1)
                    #
                    #     # 테이블 없을 시 생성
                    #     dbCon.execute(
                    #         """
                    #         create table IF NOT EXISTS TB_FOR_DATA_%s
                    #         (
                    #             SRV           varchar(10) not null comment '관측소 정보',
                    #             ANA_DATE      date        not null comment '예보일',
                    #             DATE_TIME     datetime    not null comment '예보날짜 UTC',
                    #             DATE_TIME_KST datetime    null comment '예보날짜 KST',
                    #             CA_TOT        float       null comment '전운량',
                    #             HM            float       null comment '상대습도',
                    #             PA            float       null comment '현지기압',
                    #             TA            float       null comment '기온',
                    #             TD            float       null comment '이슬점온도',
                    #             WD            float       null comment '풍향',
                    #             WS            float       null comment '풍속',
                    #             SZA           float       null comment '태양 천정각',
                    #             AZA           float       null comment '태양 방위각',
                    #             ET            float       null comment '태양 시간각',
                    #             SWR           float       null comment '일사량',
                    #             ML            float       null comment '머신러닝',
                    #             DL            float       null comment '딥러닝',
                    #             REG_DATE      datetime    null comment '등록일',
                    #             MOD_DATE      datetime    null comment '수정일',
                    #             primary key (SRV, DATE_TIME, ANA_DATE)
                    #         )
                    #             comment '기상 예보 테이블_%s';
                    #         """
                    #         , (iAnaYear, iAnaYear)
                    #     )
                    #
                    #     # 삽입
                    #     selDbTable = 'TB_FOR_DATA_{}'.format(iAnaYear)
                    #     dbData.to_sql(name=selDbTable, con=dbCon, if_exists='append', index=False)
                        # dbData.to_sql(name=selDbTable, con=dbCon, if_exists='replace', index=False)



                    # **********************************************************************************************************
                    # 시각화
                    # **********************************************************************************************************
                    # # 폴더 삭제
                    # delFile = '{}/{}/{}/SRV{:05d}'.format(globalVar['figPath'], serviceName, figForDirKey, posId)
                    # shutil.rmtree(delFile, ignore_errors=True)

                    # idxInfo = inpDataL2.loc[inpDataL2['dtDateKst'] >= pd.to_datetime('2021-01-01', format='%Y-%m-%d')].index.to_numpy()
                    # idxInfo = inpDataL2.loc[inpDataL2['dtDateKst'] >= pd.to_datetime('2021-01-01', format='%Y-%m-%d')].index.to_numpy()
                    # idxinfo = inpDataL2.loc[inpDataL2['dtDateKst'] >= pd.to_datetime('2021-11-01', format='%y-%m-%d')].index.to_numpy()
                    idxInfo = inpDataL2.loc[inpDataL2['dtDateKst'] >= pd.to_datetime('2021-06-01', format='%Y-%m-%d')].index.to_numpy()

                    if (len(idxInfo) < 1): continue
                    idx = idxInfo[0]
                    trainData, testData = inpDataL2[0:idx], inpDataL2[idx:len(inpDataL2)]

                    if (len(testData) < 1): continue
                    log.info('[CHECK] testData : {} - {}'.format(testData['dtDateKst'].min(), testData['dtDateKst'].max()))

                    trainDataL1 = trainData.dropna().reset_index(drop=True)
                    testDataL1 = testData.dropna().reset_index(drop=True)

                    anaTimeList = testDataL1['anaTime'].unique()
                    # min(anaTimeList).datetime.strftime("%Y%m%d")

                    minAnaTime = pd.to_datetime(anaTimeList).min().strftime("%Y%m%d")
                    maxAnaTime = pd.to_datetime(anaTimeList).max().strftime("%Y%m%d")

                    # anaTimeInfo = anaTimeList[0]
                    # for j, anaTimeInfo in enumerate(anaTimeList):
                    #
                    #     testDataL2 = testDataL1.loc[
                    #         testDataL1['anaTime'] == anaTimeInfo
                    #         ].dropna().reset_index(drop=True)

                        # mainTitle = '[{}] {}'.format(anaTimeInfo, '기상 예보 정보 (수치모델)를 활용한 48시간 예측 시계열')
                        # saveImg = '{}/{}/{}/SRV{:05d}/{}.png'.format(globalVar['figPath'], serviceName, figForDirKey, posId, mainTitle)
                        # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                        #
                        # if (os.path.exists(saveImg)): continue
                        # rtnInfo = makeUserTimeSeriesPlot(pd.to_datetime(testDataL2['dtDate']), testDataL2['ML'], testDataL2['DL'], testDataL2['pv'], '예측 (머신러닝)', '예측 (딥러닝)', '실측 (발전량)', '시간 (시)', '발전량', mainTitle, saveImg, True)
                        # log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

                    # mainTitle = '[{}-{}] {}'.format(min(anaTimeList), max(anaTimeList), '기상 예보 정보 (수치모델)를 활용한 머신러닝 (48시간 예측) 산점도')
                    # saveImg = '{}/{}/{}/SRV{:05d}/{}.png'.format(globalVar['figPath'], serviceName, figForDirKey, posId, mainTitle)
                    # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                    # rtnInfo = makeUserScatterPlot(testDataL1['ML'], testDataL1['pv'], '머신러닝', '실측', mainTitle, saveImg, 0, 1000, 20, 60, True)
                    # log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

                    # anaTimeList.strftime("%Y%m%d")
                    # mainTitle = '[{}-{}] {}'.format(minAnaTime, maxAnaTime, '기상 예보 정보 (수치모델)를 활용한 머신러닝 (48시간 예측) 2D 산점도')
                    # saveImg = '{}/{}/{}/SRV{:05d}/{}.png'.format(globalVar['figPath'], serviceName, figForDirKey, posId, mainTitle)
                    # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                    # rtnInfo = makeUserHist2DPlot(testDataL1['ML'], testDataL1['pv'], '머신러닝', '실측', mainTitle, saveImg, 0, 1000, 20, 60, 20, True)
                    # log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

                    # mainTitle = '[{}-{}] {}'.format(min(anaTimeList), max(anaTimeList), '기상 예보 정보 (수치모델)를 활용한 딥러닝 (48시간 예측) 산점도')
                    # saveImg = '{}/{}/{}/SRV{:05d}/{}.png'.format(globalVar['figPath'], serviceName, figForDirKey, posId, mainTitle)
                    # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                    # rtnInfo = makeUserScatterPlot(testDataL1['DL'], testDataL1['pv'], '딥러닝', '실측', mainTitle, saveImg, 0, 1000, 20, 60, True)
                    # log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

                    mainTitle = '[{}-{}] {}'.format(minAnaTime, maxAnaTime, '기상 예보 정보 (수치모델)를 활용한 딥러닝 (48시간 예측) 2D 산점도')
                    saveImg = '{}/{}/{}/SRV{:05d}/{}.png'.format(globalVar['figPath'], serviceName, figForDirKey, posId, mainTitle)
                    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                    rtnInfo = makeUserHist2DPlot(testDataL1['DL'], testDataL1['pv'], '딥러닝', '실측', mainTitle, saveImg, 0, 1000, 20, 60, 20, True)
                    log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

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
