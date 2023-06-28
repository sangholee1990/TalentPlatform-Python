import glob
# import seaborn as sns
import logging
import logging.handlers
import os
import platform
import sys
import argparse
import traceback
import warnings
# import datetime as dt
# from datetime import datetime
import pvlib
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


import pandas as pd
import math
from scipy import spatial
from pandas.tseries.offsets import Day, Hour, Minute, Second
from scipy.interpolate import Rbf
from numpy import zeros, newaxis

os.environ["PROJ_LIB"] = 'C:\ProgramData\Anaconda3\Library\share'
# from pygem import IDW
# import eccodes
# import pykrige.kriging_tools as kt
import pytz
import requests
import datetime
import h2o
# from pycaret.regression import *
from sqlalchemy import create_engine
import re
import configparser
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pymysql
import re
import configparser
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from scipy.stats import linregress
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
import xarray as xr
import pandas as pd
from pvlib import location
from pvlib import irradiance

# import cartopy.crs as ccrs
# import cartopy as crt
import pyproj

import h2o
from h2o.automl import H2OAutoML

# from pycaret.regression import *
from matplotlib import font_manager, rc
from metpy.units import units
from metpy.calc import wind_components, wind_direction, wind_speed

try:
    from pycaret.regression import *
except Exception as e:
    print("Exception : {}".format(e))

# try:
#     from pycaret.regression import *
# except Exception as e:
#     print("Exception : {}".format(e))


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
# 기상정보
# =================================================
# ASOS 데이터
#  1. TM     : 관측시각 (KST)
#  2. STN    : 국내 지점번호
#  3. WD     : 풍향 (16방위)
#  4. WS     : 풍속 (m/s)
#  5. GST_WD : 돌풍향 (16방위)
#  6. GST_WS : 돌풍속 (m/s)
#  7. GST_TM : 돌풍속이 관측된 시각 (시분)
#  8. PA     : 현지기압 (hPa)
#  9. PS     : 해면기압 (hPa)
# 10. PT     : 기압변화경향 (Code 0200)
# 11. PR     : 기압변화량 (hPa)
# 12. TA     : 기온 (C)
# 13. TD     : 이슬점온도 (C)
# 14. HM     : 상대습도 (%)
# 15. PV     : 수증기압 (hPa)
# 16. RN     : 강수량 (mm) : 여름철에는 1시간강수량, 겨울철에는 3시간강수량
# 17. RN_DAY : 위 관측시간까지의 일강수량 (mm)
# 18. RN_INT : 강수강도 (mm/h) : 관측하는 곳이 별로 없음
# 19. SD_HR3 : 3시간 신적설 (cm) : 3시간 동안 내린 신적설의 높이
# 20. SD_DAY : 일 신적설 (cm) : 00시00분부터 위 관측시간까지 내린 신적설의 높이
# 21. SD_TOT : 적설 (cm) : 치우지 않고 그냥 계속 쌓이도록 놔눈 경우의 적설의 높이
# 22. WC     : GTS 현재일기 (Code 4677)
# 23. WP     : GTS 과거일기 (Code 4561) .. 3(황사),4(안개),5(가랑비),6(비),7(눈),8(소나기),9(뇌전)
# 24. WW     : 국내식 일기코드 (문자열 22개) : 2자리씩 11개까지 기록 가능 (코드는 기상자원과 문의)
# 25. CA_TOT : 전운량 (1/10)
# 26. CA_MID : 중하층운량 (1/10)
# 27. CH_MIN : 최저운고 (100m)
# 28. CT     : 운형 (문자열 8개) : 2자리 코드로 4개까지 기록 가능
# 29. CT_TOP : GTS 상층운형 (Code 0509)
# 30. CT_MID : GTS 중층운형 (Code 0515)
# 31. CT_LOW : GTS 하층운형 (Code 0513)
# 32. VS     : 시정 (10m)
# 33. SS     : 일조 (hr)
# 34. SI     : 일사 (MJ/m2)
# 35. ST_GD  : 지면상태 코드 (코드는 기상자원과 문의)
# 36. TS     : 지면온도 (C)
# 37. TE_005 : 5cm 지중온도 (C)
# 38. TE_01  : 10cm 지중온도 (C)
# 39. TE_02  : 20cm 지중온도 (C)
# 40. TE_03  : 30cm 지중온도 (C)
# 41. ST_SEA : 해면상태 코드 (코드는 기상자원과 문의)
# 42. WH     : 파고 (m) : 해안관측소에서 목측한 값
# 43. BF     : Beaufart 최대풍력(GTS코드)
# 44. IR     : 강수자료 유무 (Code 1819) .. 1(Sec1에 포함), 2(Sec3에 포함), 3(무강수), 4(결측)
# 45. IX     : 유인관측/무인관측 및 일기 포함여부 (code 1860) .. 1,2,3(유인) 4,5,6(무인) / 1,4(포함), 2,5(생략), 3,6(결측)

# AWS 데이터
#--------------------------------------------------------------------------------------------------
#  WD1    : 1분 평균 풍향 (degree) : 0-N, 90-E, 180-S, 270-W, 360-무풍
#  WS1    : 1분 평균 풍속 (m/s)
#  WDS    : 최대 순간 풍향 (degree)
#  WSS    : 최대 순간 풍속 (m/s)
#  WD10   : 10분 평균 풍향 (degree)
#  WS10   : 10분 평균 풍속 (m/s)
#  TA     : 1분 평균 기온 (C)
#  RE     : 강수감지 (0-무강수, 0이 아니면-강수)
#  RN-15m : 15분 누적 강수량 (mm)
#  RN-60m : 60분 누적 강수량 (mm)
#  RN-12H : 12시간 누적 강수량 (mm)
#  RN-DAY : 일 누적 강수량 (mm)
#  HM     : 1분 평균 상대습도 (%)
#  PA     : 1분 평균 현지기압 (hPa)
#  PS     : 1분 평균 해면기압 (hPa)
#  TD     : 이슬점온도 (C)
#  *) -50 이하면 관측이 없거나, 에러처리된 것을 표시
#--------------------------------------------------------------------------------------------------

# =================================================
# 1. 초기 설정
# =================================================
warnings.filterwarnings("ignore")
# font_manager._rebuild()

#plt.rc('font', family='Malgun Gothic')
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

    saveLogFile = "{}/{}_{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
        , datetime.datetime.now().strftime("%Y%m%d")
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

        # 글꼴 설정
        plt.rc('font', family='Malgun Gothic')

    # 리눅스 환경
    if globalVar['sysOs'] in 'Linux':
        parser = argparse.ArgumentParser()

        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)

        inParInfo = vars(parser.parse_args())

        # 글꼴 설정
        fileList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        fontName = font_manager.FontProperties(fname=fileList[0]).get_name()
        plt.rc('font', family=fontName)

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


def cartesian(latitude, longitude, elevation=0):
    # Convert to radians
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371  # 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)

    return (X, Y, Z)


# 산점도 시각화
def makeUserScatterPlot(prdVal, refVal, xlab, ylab, mainTitle, saveImg, minVal, maxVal, xIntVal, yIntVal, isSame):

    log.info('[START] {}'.format('makeUserScatterPlot'))

    result = None

    try:

        # 그리드 설정
        plt.grid(True)

        # 결측값 마스킹
        mask = ~np.isnan(refVal)
        N = len(refVal[mask])

        plt.scatter(prdVal, refVal)

        # 제목, x축, y축 설정
        plt.title(mainTitle)
        plt.xlabel(xlab)
        plt.ylabel(ylab)


        # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
        Bias = np.mean(prdVal[mask] - refVal[mask])
        rBias = (Bias / np.mean(refVal[mask])) * 100.0
        RMSE = np.sqrt(np.mean((prdVal[mask] - refVal[mask]) ** 2))
        rRMSE = (RMSE / np.mean(refVal[mask])) * 100.0

        MAPE = np.mean(np.abs((prdVal[mask] - refVal[mask]) / prdVal[mask])) * 100.0

        # 선형회귀곡선에 대한 계산
        slope, intercept, rVal, pVal, stdErr = linregress(prdVal[mask], refVal[mask])

        lmfit = (slope * prdVal) + intercept
        # plt.plot(prdVal, lmfit, color='red', linewidth=2,linestyle="-")
        plt.plot([minVal, maxVal], [minVal, maxVal], color="black", linestyle="--", linewidth=2)
        plt.plot(prdVal, lmfit, color='red', linewidth=2, linestyle="-")
        # 라벨 추가
        plt.annotate('%s = %.2f x (%s) + %.2f' % (ylab, slope, xlab, intercept),
                     xy=(minVal + xIntVal, maxVal - yIntVal),
                     color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
        plt.annotate('R = %.2f  (p-value < %.2f)' % (rVal, pVal), xy=(minVal + xIntVal, maxVal - yIntVal * 2),
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
def makeUserHist2dPlot(prdVal, refVal, xlab, ylab, mainTitle, saveImg, minVal, maxVal, xIntVal, yIntVal, nbinVal, isSame):

    log.info('[START] {}'.format('makeUserHist2dPlot'))

    result = None

    try:

        # 그리드 설정
        plt.grid(True)

        # 결측값 마스킹
        mask = ~np.isnan(refVal)
        N = len(refVal[mask])

        # plt.scatter(prdVal, refVal)
        # nbins = 250
        hist2D, xEdge, yEdge = np.histogram2d(prdVal[mask], refVal[mask], bins=nbinVal)
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
        slope, intercept, rVal, pVal, stdErr = linregress(prdVal[mask], refVal[mask])

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
        plt.annotate('R = %.2f  (p-value < %.2f)' % (rVal, pVal), xy=(minVal + xIntVal, maxVal - yIntVal * 2),
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
        log.info('[END] {}'.format('makeUserHist2dPlot'))

#
# def makeValidTimeSeriesPlot(dtDate, mlPrdVal, dlPrdVal, refVal, mlPrdValLabel, dlPrdValLabel, refValLabel, xlab, ylab, mainTitle, saveImg, isFore):
#
#     log.info('[START] {}'.format('makeUserTimeSeriesPlot'))
#
#     result = None
#
#     try:
#         # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
#         mlRMSE = np.sqrt(np.mean((mlPrdVal - refVal) ** 2))
#         mlReRMSE = (mlRMSE / np.mean(refVal)) * 100.0
#
#         dlRMSE = np.sqrt(np.mean((dlPrdVal - refVal) ** 2))
#         dlReRMSE = (dlRMSE / np.mean(refVal)) * 100.0
#
#         # 선형회귀곡선에 대한 계산
#         mlFit = linregress(mlPrdVal, refVal)
#         mlR = mlFit[2]
#
#         dlFit = linregress(dlPrdVal, refVal)
#         dlR = dlFit[2]
#
#         prdValLabel_ml = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(mlPrdValLabel, mlR, mlReRMSE)
#         prdValLabel_dnn = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(dlPrdValLabel, dlR, dlReRMSE)
#
#         plt.grid(True)
#
#         plt.plot(dtDate, mlPrdVal, label=prdValLabel_ml, marker='o')
#         plt.plot(dtDate, dlPrdVal, label=prdValLabel_dnn, marker='o')
#         plt.plot(dtDate, refVal, label=refValLabel, marker='o')
#
#         # 제목, x축, y축 설정
#         plt.title(mainTitle)
#         plt.xlabel(xlab)
#         plt.ylabel(ylab)
#         plt.ylim(0, 1000)
#
#         if (isFore == True):
#             plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#             plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
#             plt.gcf().autofmt_xdate()
#             plt.xticks(rotation=45, ha='right')
#
#         else:
#             plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#             plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
#             plt.gcf().autofmt_xdate()
#             plt.xticks(rotation=0, ha='right')
#
#         plt.legend(loc='upper left')
#
#         plt.savefig(saveImg, dpi=600, bbox_inches='tight')
#         plt.show()
#         plt.close()
#
#         result = {
#             'msg': 'succ'
#             , 'saveImg': saveImg
#             , 'isExist': os.path.exists(saveImg)
#         }
#
#         return result
#
#     except Exception as e:
#         log.error("Exception : {}".format(e))
#         return result
#
#     finally:
#         # try, catch 구문이 종료되기 전에 무조건 실행
#         log.info('[END] {}'.format('makeUserTimeSeriesPlot'))


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

# 딥러닝 예측
def makeDlModel(subOpt=None, xCol=None, yCol=None, inpData=None):

    log.info('[START] {}'.format('makeDlModel'))
    log.info('[CHECK] subOpt : {}'.format(subOpt))

    result = None

    try:

        saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['modelPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'h2o', 'act', '*')
        saveModelList = sorted(glob.glob(saveModel), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        data = inpData[xyCol]
        dataL1 = data.dropna()

        # h2o.shutdown(prompt=False)

        if (not subOpt['isInit']):
            h2o.init()
            h2o.no_progress()
            subOpt['isInit'] = True

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            # trainData, validData = train_test_split(dataL1, test_size=0.3)

            # dlModel = H2OAutoML(max_models=30, max_runtime_secs=99999, balance_classes=True, seed=123)
            dlModel = H2OAutoML(max_models=20, max_runtime_secs=99999, balance_classes=True, seed=123)

            #dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(trainData), validation_frame=h2o.H2OFrame(validData))
            #dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(dataL1), validation_frame=h2o.H2OFrame(dataL1))
            dlModel.train(x=xCol, y=yCol, training_frame=h2o.H2OFrame(dataL1), validation_frame=h2o.H2OFrame(dataL1))

            fnlModel = dlModel.get_best_model()
            #fnlModel = dlModel.leader

            # 학습 모델 저장
            saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['modelPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'h2o', 'act', datetime.datetime.now().strftime('%Y%m%d'))
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)

            # h2o.save_model(model=fnlModel, path=os.path.dirname(saveModel), filename=os.path.basename(saveModel), force=True)
            fnlModel.save_mojo(path=os.path.dirname(saveModel), filename=os.path.basename(saveModel), force=True)
        else:
            saveModel = saveModelList[0]
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            fnlModel = h2o.import_mojo(saveModel)

        result = {
            'msg': 'succ'
            , 'dlModel': fnlModel
            , 'saveModel': saveModel
            , 'isExist': os.path.exists(saveModel)
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeDlModel'))


# 머신러닝 예측
def makeMlModel(subOpt=None, xCol=None, yCol=None, inpData=None):

    log.info('[START] {}'.format('makeMlModel'))
    log.info('[CHECK] subOpt : {}'.format(subOpt))

    result = None

    try:
        saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model.pkl'.format(globalVar['modelPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'pycaret', 'act', '*')
#        log.info('[CHECK] saveModel : {}'.format(saveModel))

        saveModelList = sorted(glob.glob(saveModel), reverse=True)
        xyCol = xCol.copy()
        xyCol.append(yCol)
        data = inpData[xyCol]
        dataL1 = data.dropna()

        # 학습 모델이 없을 경우
        if (subOpt['isOverWrite']) or (len(saveModelList) < 1):

            # 7:3에 대한 학습/테스트 분류
            trainData, validData = train_test_split(data, test_size=0.3)
            # trainData = inpData

            pyModel = setup(
                data=dataL1
                , session_id=123
                , silent=True
                , target=yCol
            )

            # pyModel = setup(
            #     data=dataL1
            #     , session_id=123
            #     , silent=True
            #     , target=yCol
            #     , remove_outliers= True
            #     , remove_multicollinearity = True
            #     , ignore_low_variance = True
            #     , normalize=True
            #     , transformation= True
            #     , transform_target = True
            #     , combine_rare_levels = True
            # )

            # 각 모형에 따른 자동 머신러닝
            modelList = compare_models(sort='RMSE', n_select=3)

            # 앙상블 모형
            blendModel = blend_models(estimator_list=modelList, fold=10)

            # 앙상블 파라미터 튜닝
            tuneModel = tune_model(blendModel, fold=10, choose_better=True)

            # 학습 모형
            fnlModel = finalize_model(tuneModel)

            # 학습 모형 저장
            saveModel = '{}/{}/{}-{}-{}-{}-{}-{}.model'.format(globalVar['modelPath'], subOpt['modelKey'], serviceName, subOpt['srvId'], 'final', 'pycaret', 'act', datetime.datetime.now().strftime('%Y%m%d'))
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            os.makedirs(os.path.dirname(saveModel), exist_ok=True)
            save_model(fnlModel, saveModel)

        else:
            saveModel = saveModelList[0]
            log.info('[CHECK] saveModel : {}'.format(saveModel))
            fnlModel = load_model(os.path.splitext(saveModel)[0])

        result = {
            'msg': 'succ'
            , 'mlModel': fnlModel
            , 'saveModel': saveModel
            , 'isExist': os.path.exists(saveModel)
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makeMlModel'))


def initCfgInfo(sysPath):
    log.info('[START] {}'.format('initCfgInfo'))
    # log.info('[CHECK] sysPath : {}'.format(sysPath))

    result = None

    try:
        # DB 연결 정보
        pymysql.install_as_MySQLdb()

        # DB 정보
        config = configparser.ConfigParser()
        config.read(sysPath, encoding='utf-8')
        dbUser = config.get('mariadb', 'user')
        dbPwd = config.get('mariadb', 'pwd')
        dbHost = config.get('mariadb', 'host')
        dbPort = config.get('mariadb', 'port')
        dbName = config.get('mariadb', 'dbName')

        dbEngine = create_engine('mariadb://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
        # dbEngine = create_engine('mariadb+mariadbconnector://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
        sessMake = sessionmaker(bind=dbEngine)
        session = sessMake()

        # API 정보
        apiUrl = config.get('pv', 'url')
        apiToken = config.get('pv', 'token')

        result = {
            'dbEngine': dbEngine
            , 'session': session
            , 'apiUrl': apiUrl
            , 'apiToken': apiToken
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('initCfgInfo'))


def subAtmosActData(cfgInfo, posDataL1, dtIncDateList):

    log.info('[START] {}'.format('subAtmosActData'))
    result = None

    try:
        lat1D = np.array(posDataL1['LAT'])
        lon1D = np.array(posDataL1['LON'])
        lon2D, lat2D = np.meshgrid(lon1D, lat1D)

        # ASOS 설정 정보
        # inpAsosStnFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/ASOS_STN_INFO.csv')
        # asosStnData = pd.read_csv(inpAsosStnFile)
        # asosStnDataL1 = asosStnData[['STN', 'LON', 'LAT']]

        # ALL 설정 정보
        inpAllStnFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/ALL_STN_INFO.csv')
        allStnData = pd.read_csv(inpAllStnFile)
        allStnDataL1 = allStnData[['STN', 'LON', 'LAT']]
        # allStnDataL1 = allStnDataL1.sort_values(by=['LON'], ascending=True).sort_values(by=['LAT'], ascending=True)

        # GK2A 설정 정보
        cfgFile = '{}/{}'.format(globalVar['cfgPath'], 'satInfo/gk2a_ami_le2_cld_ko020lc_202009010000.nc')
        cfgData = xr.open_dataset(cfgFile)

        # 위/경도 반환
        imgProjInfo = cfgData['gk2a_imager_projection'].attrs

        # 1) ccrs 사용
        # mapLccProj = ccrs.LambertConformal(
        #     central_longitude=imgProjInfo['central_meridian']
        #     , central_latitude=imgProjInfo['origin_latitude']
        #     , secant_latitudes=(imgProjInfo['standard_parallel1'], imgProjInfo['standard_parallel2'])
        #     , false_easting=imgProjInfo['false_easting']
        #     , false_northing=imgProjInfo['false_northing']
        # )
        #
        # try:
        #     mapLccProjInfo = mapLccProj.to_proj4()
        # except Exception as e:
        #     log.error("Exception : {}".format(e))

        # 2) Proj 사용
        mapLccProjInfo = '+proj=lcc +lat_0=38 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs'
        mapProj = pyproj.Proj(mapLccProjInfo)

        nx = imgProjInfo['image_width']
        ny = imgProjInfo['image_height']
        xOffset = imgProjInfo['lower_left_easting']
        yOffset = imgProjInfo['lower_left_northing']

        res = imgProjInfo['pixel_size']

        # 직교 좌표
        rowEle = (np.arange(0, nx, 1) * res) + xOffset
        colEle = (np.arange(0, ny, 1) * res) + yOffset
        colEle = colEle[::-1]

        posRow, posCol = mapProj(lon1D, lat1D, inverse=False)

        # dtIncDateInfo = dtIncDateList[0]
        for ii, dtIncDateInfo in enumerate(dtIncDateList):
            log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))

            # if (dtIncDateInfo < pd.to_datetime('2021-10-01 09', format='%Y-%m-%d %H')): continue
            # if (dtIncDateInfo < pd.to_datetime('2021-10-01 02:40', format='%Y-%m-%d %H:%M')): continue

            dtAsosDatePath = dtIncDateInfo.strftime('%Y%m/%d')
            dtGk2aDatePath = dtIncDateInfo.strftime('%Y%m/%d/%H')
            dtH8DatePath = dtIncDateInfo.strftime('%Y%m/%d/%H')

            dtAsosDateName = dtIncDateInfo.strftime('%Y%m%d')
            dtGk2aDateName = dtIncDateInfo.strftime('%Y%m%d%H%M')
            dtH8DateName = dtIncDateInfo.strftime('%Y%m%d_%H%M')

            # ************************************************************************************************
            # ASOS, AWS 융합 데이터
            # ************************************************************************************************
            inpAsosFilePattern = 'OBS/{}/ASOS_OBS_{}*.txt'.format(dtAsosDatePath, dtAsosDateName)
            inpAsosFile = '{}/{}'.format(globalVar['inpPath'], inpAsosFilePattern)
            fileList = sorted(glob.glob(inpAsosFile))

            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpAsosFile : {} / {}'.format(inpAsosFile, '입력 자료를 확인해주세요.'))
                continue

            # log.info("[CHECK] fileList : {}".format(fileList))

            dataL1 = pd.DataFrame()
            for fileInfo in fileList:
                data = pd.read_csv(fileInfo, header=None, delimiter='\s+')

                # 컬럼 설정
                data.columns = ['TM', 'STN', 'WD', 'WS', 'GST_WD', 'GST_WS', 'GST_TM', 'PA', 'PS', 'PT', 'PR', 'TA',
                                  'TD', 'HM', 'PV', 'RN', 'RN_DAY', 'TMP', 'RN_INT', 'SD_HR3', 'SD_DAY', 'SD_TOT', 'WC',
                                  'WP', 'WW', 'CA_TOT', 'CA_MID', 'CH_MIN', 'CT', 'CT_TOP', 'CT_MID', 'CT_LOW', 'VS',
                                  'SS', 'SI', 'ST_GD', 'TS', 'TE_005', 'TE_01', 'TE_02', 'TE_03', 'ST_SEA', 'WH', 'BF',
                                  'IR', 'IX']
                data = data[['TM', 'STN', 'WD', 'WS', 'PA', 'TA', 'TD', 'HM', 'CA_TOT']]
                dataL1 = pd.concat([dataL1, data], ignore_index=False)

            inpAwsFilePattern = 'OBS/{}/AWS_OBS_{}*.txt'.format(dtAsosDatePath, dtAsosDateName)
            inpAwsFile = '{}/{}'.format(globalVar['inpPath'], inpAwsFilePattern)
            fileList = sorted(glob.glob(inpAwsFile))

            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpAwsFile : {} / {}'.format(inpAwsFile, '입력 자료를 확인해주세요.'))

            # log.info("[CHECK] fileList : {}".format(fileList))

            for fileInfo in fileList:
                data = pd.read_csv(fileInfo, header=None, delimiter='\s+')

                # 컬럼 설정
                data.columns = ['TM', 'STN', 'WD', 'WS', 'WDS', 'WSS', 'WD10', 'WS10', 'TA', 'RE', 'RN-15m', 'RN-60m',
                                'RN-12H', 'RN-DAY', 'HM', 'PA', 'PS', 'TD']
                data = data[['TM', 'STN', 'WD', 'WS', 'PA', 'TA', 'TD', 'HM']]
                data['CA_TOT'] = np.nan

                dataL1 = pd.concat([dataL1, data], ignore_index=False)

            dataL2 = dataL1
            
            # TM 및 STN을 기준으로 중복 제거
            dataL2['TM'] = dataL2['TM'].astype(str)
            dataL2.drop_duplicates(subset=['TM', 'STN'], inplace=True)

            # 결측값 제거
            dataL3 = dataL2
            dataL3['dtDate'] = pd.to_datetime(dataL3['TM'].astype(str), format='%Y%m%d%H%M')

            dataL4 = dataL3.loc[
                dataL3['dtDate'] == dtIncDateInfo
            ]

            dataL4['WD'][dataL4['WD'] < 0] = np.nan
            dataL4['WS'][dataL4['WS'] < 0] = np.nan
            dataL4['PA'][dataL4['PA'] < 0] = np.nan
            dataL4['TA'][dataL4['TA'] < -50.0] = np.nan
            dataL4['TD'][dataL4['TD'] < -50.0] = np.nan
            dataL4['HM'][dataL4['HM'] < 0] = np.nan
            dataL4['CA_TOT'][dataL4['CA_TOT'] < 0] = np.nan

            # 풍향, 풍속을 이용해서 U,V 벡터 변환
            dataL4['uVec'], dataL4['vVec'] = wind_components(dataL4['WS'].values * units('m/s'), dataL4['WS'].values * units.deg)

            # statData = dataL4.describe()

            dataL5 = pd.merge(left=dataL4, right=allStnDataL1, how='left', left_on='STN', right_on='STN')

            # colInfo = 'WD'
            # colInfo = 'uVec'
            # colInfo = 'CA_TOT'

            varList = {}
            colList = dataL4.columns
            for colInfo in colList:
                if (re.match('TM|STN|dtDate|WS|WD', colInfo)): continue

                # varList[colInfo] = np.empty((len(lon1D), len(lat1D))) * np.nan
                # varList[colInfo] = np.full(shape=(len(lon1D), len(lat1D)), fill_value=None)
                varList[colInfo] = np.full(shape=(len(lon1D), len(lat1D)), fill_value=np.nan)

                dataL6 = dataL5[['dtDate', 'LON', 'LAT', colInfo]].dropna()

                if (len(dataL6) < 1): continue

                posLon = dataL6['LON'].values
                posLat = dataL6['LAT'].values
                posVar = dataL6[colInfo].values

                # Radial basis function (RBF) interpolation in N dimensions.
                try:
                    rbfModel = Rbf(posLon, posLat, posVar, function='linear')
                    rbfRes = rbfModel(lon2D, lat2D)
                    varList[colInfo] = rbfRes
                except Exception as e:
                    log.error("Exception : {}".format(e))

            #  U,V 벡터를 이용해서 풍향, 풍속 변환
            varList['WD'] = wind_direction(varList['uVec']* units('m/s'), varList['vVec'] * units('m/s'), convention='from')
            varList['WS'] = wind_speed(varList['uVec']* units('m/s'), varList['vVec'] * units('m/s'))

            # ************************************************************************************************
            # GK2A 데이터
            # ************************************************************************************************
            # inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/SAT/GK2A_*.nc')
            inpFilePattern = 'SAT/{}/gk2a_*_{}.nc'.format(dtGk2aDatePath, dtGk2aDateName)
            inpFile = '{}/{}'.format(globalVar['inpPath'], inpFilePattern)
            fileList = sorted(glob.glob(inpFile))

            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                continue

            # log.info("[CHECK] fileList : {}".format(fileList))

            gk2aData = xr.open_mfdataset(fileList)
            gk2aDataL1 = gk2aData.assign_coords(
                {"dim_x": ("dim_x", rowEle)
                    , "dim_y": ("dim_y", colEle)
                 }
            )

            selGk2aNearData = gk2aDataL1.sel(dim_x=posRow, dim_y=posCol, method='nearest')
            selGk2aIntpData = gk2aDataL1.interp(dim_x=posRow, dim_y=posCol)

            # ************************************************************************************************
            # Himawari-8 데이터
            # ************************************************************************************************
            # inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/SAT/GK2A_*.nc')
            inpFilePattern = 'SAT/{}/H08_{}_*.nc'.format(dtH8DatePath, dtH8DateName)
            inpFile = '{}/{}'.format(globalVar['inpPath'], inpFilePattern)
            fileList = sorted(glob.glob(inpFile))

            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                continue

            # log.info("[CHECK] fileList : {}".format(fileList))

            h8Data = xr.open_mfdataset(fileList)

            selH8NearData = h8Data.sel(latitude=lat1D, longitude=lon1D, method='nearest')
            selH8IntpData = h8Data.interp(latitude=lat1D, longitude=lon1D)

            # ************************************************************************************************
            # 융합 데이터
            # ************************************************************************************************
            try:
                actData = xr.Dataset(
                    {
                        # ASOS 및 AWS 융합
                        'WD': (('time', 'lat', 'lon'), (varList['WD']).reshape(1, len(lat1D), len(lon1D)))
                        , 'WS': (('time', 'lat', 'lon'), (varList['WS']).reshape(1, len(lat1D), len(lon1D)))
                        , 'PA': (('time', 'lat', 'lon'), (varList['PA']).reshape(1, len(lat1D), len(lon1D)))
                        , 'TA': (('time', 'lat', 'lon'), (varList['TA']).reshape(1, len(lat1D), len(lon1D)))
                        , 'TD': (('time', 'lat', 'lon'), (varList['TD']).reshape(1, len(lat1D), len(lon1D)))
                        , 'HM': (('time', 'lat', 'lon'), (varList['HM']).reshape(1, len(lat1D), len(lon1D)))
                        , 'CA_TOT': (('time', 'lat', 'lon'), (varList['CA_TOT']).reshape(1, len(lat1D), len(lon1D)))

                        # H8
                        , 'PAR': ( ('time', 'lat', 'lon'), (selH8NearData['PAR'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'SWR': ( ('time', 'lat', 'lon'), (selH8NearData['SWR'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'TAAE': (('time', 'lat', 'lon'), (selH8NearData['TAAE'].values).reshape(1, len(lat1D), len(lon1D)))
                        , 'TAOT_02': ( ('time', 'lat', 'lon'), (selH8NearData['TAOT_02'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'UVA': ( ('time', 'lat', 'lon'), (selH8NearData['UVA'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'UVB': ( ('time', 'lat', 'lon'), (selH8NearData['UVB'].values).reshape(1, len(lat1D), len(lon1D)) )

                        , 'PAR_intp': ( ('time', 'lat', 'lon'), (selH8IntpData['PAR'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'SWR_intp': ( ('time', 'lat', 'lon'), (selH8IntpData['SWR'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'TAAE_intp': (('time', 'lat', 'lon'), (selH8IntpData['TAAE'].values).reshape(1, len(lat1D), len(lon1D)))
                        , 'TAOT_02_intp': ( ('time', 'lat', 'lon'), (selH8IntpData['TAOT_02'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'UVA_intp': ( ('time', 'lat', 'lon'), (selH8IntpData['UVA'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'UVB_intp': ( ('time', 'lat', 'lon'), (selH8IntpData['UVB'].values).reshape(1, len(lat1D), len(lon1D)) )

                        # GK2A
                        , 'CA': ( ('time', 'lat', 'lon'), (selGk2aNearData['CA'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'CF': ( ('time', 'lat', 'lon'), (selGk2aNearData['CF'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'CLD': (('time', 'lat', 'lon'), (selGk2aNearData['CLD'].values).reshape(1, len(lat1D), len(lon1D)))
                        , 'DSR': ( ('time', 'lat', 'lon'), (selGk2aNearData['DSR'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'ASR': ( ('time', 'lat', 'lon'), (selGk2aNearData['ASR'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'RSR': ( ('time', 'lat', 'lon'), (selGk2aNearData['RSR'].values).reshape(1, len(lat1D), len(lon1D)) )

                        , 'CA_intp': ( ('time', 'lat', 'lon'), (selGk2aIntpData['CA'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'CF_intp': ( ('time', 'lat', 'lon'), (selGk2aIntpData['CF'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'CLD_intp': (('time', 'lat', 'lon'), (selGk2aIntpData['CLD'].values).reshape(1, len(lat1D), len(lon1D)))
                        , 'DSR_intp': ( ('time', 'lat', 'lon'), (selGk2aIntpData['DSR'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'ASR_intp': ( ('time', 'lat', 'lon'), (selGk2aIntpData['ASR'].values).reshape(1, len(lat1D), len(lon1D)) )
                        , 'RSR_intp': ( ('time', 'lat', 'lon'), (selGk2aIntpData['RSR'].values).reshape(1, len(lat1D), len(lon1D)) )
                    }
                    , coords={
                        'time': pd.date_range(dtIncDateInfo, periods=1)
                        , 'lat': lat1D
                        , 'lon': lon1D
                    }
                )

            except Exception as e:
                log.error("Exception : {}".format(e))

            # plt.scatter(posLon, posLat, c=posVar)
            # plt.colorbar()
            # plt.show()

            # actDataL1 = actData.isel(time=0)
            # plt.scatter(lon1D, lat1D, c=np.diag(actDataL1['WD']))
            # plt.colorbar()
            # plt.show()

            # actDataL1 = actData.isel(time=0)
            # plt.scatter(lon1D, lat1D, c=np.diag(actDataL1['WS']))
            # plt.colorbar()
            # plt.show()

            for kk, posInfo in posDataL1.iterrows():
                # log.info("[CHECK] posId (posLon, posLat) : {} ({}. {})".format(posId, posLon, posLat))
                if (len(actData) < 1): continue

                posId = int(posInfo['ID'])
                posLat = posInfo['LAT']
                posLon = posInfo['LON']
                posSza = posInfo['STN_SZA']
                posAza = posInfo['STN_AZA']

                try:
                    actDataL2 = actData.sel(lat=posLat, lon=posLon)
                    if (len(actDataL2) < 1): continue

                    # actDataL3 = actDataL2.to_dataframe().dropna().reset_index(drop=True)
                    actDataL3 = actDataL2.to_dataframe().reset_index(drop=True)
                    # actDataL3['dtDate'] = pd.to_datetime(dtanaDateInfo) + (actDataL3.index.values * datetime.timedelta(hours=1))
                    actDataL3['DATE_TIME'] = pd.to_datetime(dtIncDateInfo)
                    # actDataL3['dtDateKst'] = actDataL3.index.tz_localize(tzUtc).tz_convert(tzKst)
                    actDataL3['DATE_TIME_KST'] = actDataL3['DATE_TIME'] + dtKst
                    actDataL4 = actDataL3
                    actDataL5 = actDataL4[[
                        'DATE_TIME_KST', 'DATE_TIME', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS'
                        , 'SWR', 'TAAE', 'UVA', 'UVB', 'SWR_intp', 'TAAE_intp', 'UVA_intp', 'UVB_intp'
                        , 'CA', 'CF', 'CLD', 'DSR', 'ASR', 'RSR', 'CA_intp', 'CF_intp', 'CLD_intp', 'DSR_intp', 'ASR_intp', 'RSR_intp'
                    ]]
                    actDataL5['SRV'] = 'SRV{:05d}'.format(posId)
                    # actDataL5['TA'] = actDataL5['TA'] - 273.15
                    # actDataL5['TD'] = actDataL5['TD'] - 273.15
                    # actDataL5['PA'] = actDataL5['PA'] / 100.0
                    actDataL5['CA_TOT'] = actDataL5['CA_TOT'] / 10.0
                    actDataL5['CA_TOT'] = np.where(actDataL5['CA_TOT'] < 0, 0, actDataL5['CA_TOT'])
                    actDataL5['CA_TOT'] = np.where(actDataL5['CA_TOT'] > 1, 1, actDataL5['CA_TOT'])

                    solPosInfo = pvlib.solarposition.get_solarposition(pd.to_datetime(actDataL5['DATE_TIME'].values),
                                                                       posLat, posLon,
                                                                       pressure=actDataL5['PA'].values * 100.0,
                                                                       temperature=actDataL5['TA'].values,
                                                                       method='nrel_numpy')
                    actDataL5['SZA'] = solPosInfo['apparent_zenith'].values
                    actDataL5['AZA'] = solPosInfo['azimuth'].values
                    actDataL5['ET'] = solPosInfo['equation_of_time'].values

                    # pvlib.location.Location.get_clearsky()
                    site = location.Location(posLat, posLon, tz='Asia/Seoul')
                    clearInsInfo = site.get_clearsky(pd.to_datetime(actDataL5['DATE_TIME'].values))
                    actDataL5['GHI_CLR'] = clearInsInfo['ghi'].values
                    actDataL5['DNI_CLR'] = clearInsInfo['dni'].values
                    actDataL5['DHI_CLR'] = clearInsInfo['dhi'].values

                    poaInsInfo = irradiance.get_total_irradiance(
                        surface_tilt=posSza,
                        surface_azimuth=posAza,
                        dni=clearInsInfo['dni'],
                        ghi=clearInsInfo['ghi'],
                        dhi=clearInsInfo['dhi'],
                        solar_zenith=solPosInfo['apparent_zenith'].values,
                        solar_azimuth=solPosInfo['azimuth'].values
                    )

                    actDataL5['GHI_POA'] = poaInsInfo['poa_global'].values
                    actDataL5['DNI_POA'] = poaInsInfo['poa_direct'].values
                    actDataL5['DHI_POA'] = poaInsInfo['poa_diffuse'].values

                    # 혼탁도
                    turbidity = pvlib.clearsky.lookup_linke_turbidity(pd.to_datetime(actDataL5['DATE_TIME'].values), posLat, posLon, interp_turbidity=True)
                    actDataL5['TURB'] = turbidity.values
                    
                    setAtmosActDataDB(cfgInfo, actDataL5)

                except Exception as e:
                    log.error("Exception : {}".format(e))

        result = {
            'msg': 'succ'
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('subAtmosActData'))


def setAtmosActDataDB(cfgInfo, dbData):

    # log.info('[START] {}'.format('setAtmosActDataDB'))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] dbData : {}'.format(dbData))

    try:

        session = cfgInfo['session']
        dbEngine = cfgInfo['dbEngine']
        iYear = int(dbData['DATE_TIME'][0].strftime("%Y"))
        # iYear = 2022
        selDbTable = 'TB_ACT_DATA_{}'.format(iYear)

        # 테이블 생성
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS `{}`
            (
                SRV           varchar(10) not null comment '관측소 정보',
                DATE_TIME     datetime    not null comment '실황날짜 UTC',
                DATE_TIME_KST datetime    null comment '실황날짜 KST',
                CA_TOT        float       null comment '전운량',
                HM            float       null comment '상대습도',
                PA            float       null comment '현지기압',
                TA            float       null comment '기온',
                TD            float       null comment '이슬점온도',
                WD            float       null comment '풍향',
                WS            float       null comment '풍속',
                SZA           float       null comment '태양 천정각',
                AZA           float       null comment '태양 방위각',
                ET            float       null comment '태양 시간각',
                TURB          float       null comment '혼탁도',
                GHI_CLR       float       null comment '맑은날 전천 일사량',
                DNI_CLR       float       null comment '맑은날 직달 일사량',
                DHI_CLR       float       null comment '맑은날 산란 일사량',
                GHI_POA       float       null comment '보정 맑은날 전천 일사량',
                DNI_POA       float       null comment '보정 맑은날 직달 일사량',
                DHI_POA       float       null comment '보정 맑은날 산란 일사량',
                SWR           float       null comment '히마와리8호 일사량',
                TAAE          float       null comment '히마와리8호 옹스트롬지수',
                UVA           float       null comment '히마와리8호 자외선A',
                UVB           float       null comment '히마와리8호 자외선B',
                SWR_intp      float       null comment '히마와리8호 일사량 내삽',
                TAAE_intp     float       null comment '히마와리8호 옹스트롬지수 내삽',
                UVA_intp      float       null comment '히마와리8호 자외선A 내삽',
                UVB_intp      float       null comment '히마와리8호 자외선B 내삽',
                CA            float       null comment '천리안위성2A호 구름운량',
                CF            float       null comment '천리안위성2A호 구름비율',
                CLD           float       null comment '천리안위성2A호 구름탐지',
                DSR           float       null comment '천리안위성2A호 일사량',
                ASR           float       null comment '천리안위성2A호 흡수단파복사',
                RSR           float       null comment '천리안위성2A호 상향단파복사',
                CA_intp       float       null comment '천리안위성2A호 구름운량 내삽',
                CF_intp       float       null comment '천리안위성2A호 구름비율 내삽',
                CLD_intp      float       null comment '천리안위성2A호 구름탐지 내삽',
                DSR_intp      float       null comment '천리안위성2A호 일사량 내삽',
                ASR_intp      float       null comment '천리안위성2A호 흡수단파복사 내삽',
                RSR_intp      float       null comment '천리안위성2A호 상향단파복사 내삽',
                ML            float       null comment '머신러닝',
                DL            float       null comment '딥러닝',
                ML2           float       null comment '머신러닝',
                DL2           float       null comment '딥러닝',
                REG_DATE      datetime    null comment '등록일',
                MOD_DATE      datetime    null comment '수정일',
                primary key (SRV, DATE_TIME)
            )
            comment '기상 실황 테이블_{}';
            """.format(selDbTable, iYear)
        )
        session.commit()

        for k, dbInfo in dbData.iterrows():
            dbInfo = dbInfo.replace( { np.nan : -999 } )

            # 테이블 중복 검사
            resChk = pd.read_sql(
                """
                SELECT COUNT(*) AS CNT FROM `{}`
                WHERE SRV = '{}' AND DATE_TIME = '{}'
                """.format(selDbTable, dbInfo['SRV'], dbInfo['DATE_TIME'])
                , con=dbEngine
            )

            if (resChk.loc[0, 'CNT'] > 0):
                dbInfo['MOD_DATE'] = datetime.datetime.now()

                session.execute(
                    """
                    UPDATE `{}`
                    SET DATE_TIME_KST = '{}', CA_TOT = '{}', HM = '{}', PA = '{}', TA = '{}', TD = '{}', WD = '{}', WS = '{}', SZA = '{}', AZA = '{}', ET = '{}', TURB = '{}'
                    , GHI_CLR = '{}', DNI_CLR = '{}', DHI_CLR = '{}', GHI_POA = '{}', DNI_POA = '{}', DHI_POA = '{}', MOD_DATE = '{}'
                    , SWR = '{}', TAAE = '{}', UVA = '{}', UVB = '{}', SWR_intp = '{}', TAAE_intp = '{}', UVA_intp = '{}', UVB_intp = '{}'
                    , CA = '{}', CF = '{}', CLD = '{}', DSR = '{}', ASR = '{}', RSR = '{}'
                    , CA_intp = '{}', CF_intp = '{}', CLD_intp = '{}', DSR_intp = '{}', ASR_intp = '{}', RSR_intp = '{}'
                    WHERE SRV = '{}' AND DATE_TIME = '{}';
                    """.format(selDbTable
                               , dbInfo['DATE_TIME_KST'], dbInfo['CA_TOT'], dbInfo['HM'], dbInfo['PA'], dbInfo['TA']
                               , dbInfo['TD'], dbInfo['WD'], dbInfo['WS'], dbInfo['SZA'], dbInfo['AZA'], dbInfo['ET'], dbInfo['TURB'], dbInfo['GHI_CLR']
                               , dbInfo['DNI_CLR'], dbInfo['DHI_CLR'], dbInfo['GHI_POA'], dbInfo['DNI_POA'], dbInfo['DHI_POA'], dbInfo['MOD_DATE']
                               , dbInfo['SWR'], dbInfo['TAAE'], dbInfo['UVA'], dbInfo['UVB'], dbInfo['SWR_intp'], dbInfo['TAAE_intp'], dbInfo['UVA_intp'], dbInfo['UVB_intp']
                               , dbInfo['CA'], dbInfo['CF'], dbInfo['CLD'], dbInfo['DSR'], dbInfo['ASR'], dbInfo['RSR']
                               , dbInfo['CA_intp'], dbInfo['CF_intp'], dbInfo['CLD_intp'], dbInfo['DSR_intp'], dbInfo['ASR_intp'], dbInfo['RSR_intp']
                               , dbInfo['SRV'], dbInfo['DATE_TIME'])
                )

            else:
                dbInfo['REG_DATE'] = datetime.datetime.now()

                session.execute(
                    """
                    INSERT INTO `{}` (SRV, DATE_TIME, DATE_TIME_KST, CA_TOT, HM, PA, TA, TD, WD, WS, SZA, AZA, ET, TURB, GHI_CLR, DNI_CLR, DHI_CLR, GHI_POA, DNI_POA, DHI_POA, REG_DATE
                        , SWR, TAAE, UVA, UVB, SWR_intp, TAAE_intp, UVA_intp, UVB_intp, CA, CF, CLD, DSR, ASR, RSR, CA_intp, CF_intp, CLD_intp, DSR_intp, ASR_intp, RSR_intp
                    )
                    VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}'
                        , '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}'
                    )
                    """.format(selDbTable
                               , dbInfo['SRV'], dbInfo['DATE_TIME'], dbInfo['DATE_TIME_KST'], dbInfo['CA_TOT'], dbInfo['HM'], dbInfo['PA'], dbInfo['TA']
                               , dbInfo['TD'], dbInfo['WD'], dbInfo['WS'], dbInfo['SZA'], dbInfo['AZA'], dbInfo['ET'], dbInfo['TURB'], dbInfo['GHI_CLR'], dbInfo['DNI_CLR'], dbInfo['DHI_CLR'], dbInfo['GHI_POA'], dbInfo['DNI_POA'], dbInfo['DHI_POA'], dbInfo['REG_DATE']
                               , dbInfo['SWR'], dbInfo['TAAE'], dbInfo['UVA'], dbInfo['UVB'], dbInfo['SWR_intp'], dbInfo['TAAE_intp'], dbInfo['UVA_intp'], dbInfo['UVB_intp']
                               , dbInfo['CA'], dbInfo['CF'], dbInfo['CLD'], dbInfo['DSR'], dbInfo['ASR'], dbInfo['RSR']
                               , dbInfo['CA_intp'], dbInfo['CF_intp'], dbInfo['CLD_intp'], dbInfo['DSR_intp'], dbInfo['ASR_intp'], dbInfo['RSR_intp']
                               )
                )

            session.commit()

    except Exception as e:
        log.error('Exception : {}'.format(e))
        session.rollback()

    finally:
        session.close()
        # try, catch 구문이 종료되기 전에 무조건 실행
        # log.info('[END] {}'.format('setAtmosActDataDB'))



def subActModelInpData(sysOpt, cfgInfo, srvId, posVol, dtIncDateList):

    log.info('[START] {}'.format('subActModelInpData'))
    # log.info('[CHECK] sysOpt : {}'.format(sysOpt))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] srvId : {}'.format(srvId))
    # log.info('[CHECK] posVol : {}'.format(posVol))
    # log.info('[CHECK] dtIncDateList : {}'.format(dtIncDateList))

    result = None

    try:

        dtYearList = dtIncDateList.strftime('%Y').unique().tolist()
        session = cfgInfo['session']
        dbEngine = cfgInfo['dbEngine']

        # *****************************************************
        # 발전량 데이터
        # *****************************************************
        pvData = pd.DataFrame()
        # dtYearInfo = dtYearList[0]
        for idx, dtYearInfo in enumerate(dtYearList):
            log.info("[CHECK] dtYearInfo : {}".format(dtYearInfo))

            selDbTable = 'TB_PV_DATA_{}'.format(dtYearInfo)

            # 테이블 존재 여부
            resTableExist = pd.read_sql(
                """
                    SELECT COUNT(*) AS CNT FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME  = '{}'
                """.format('DMS02', selDbTable)
                , con=dbEngine
            )

            if (resTableExist.loc[0, 'CNT'] < 1): continue

            res = pd.read_sql(
                """
                SELECT SRV, DATE_TIME, DATE_TIME_KST, PV
                FROM `{}`
                WHERE SRV = '{}' AND DATE_TIME_KST BETWEEN '{}' AND '{}'
                """.format(selDbTable, srvId, sysOpt['srtDate'], sysOpt['endDate'])
                , con=dbEngine
            )
            if (len(res) < 0): continue
            pvData = pd.concat([pvData, res], ignore_index=False)

        # *****************************************************
        # 기상정보 데이터
        # *****************************************************
        forData = pd.DataFrame()
        # dtYearInfo = dtYearList[0]
        for idx, dtYearInfo in enumerate(dtYearList):
            log.info("[CHECK] dtYearInfo : {}".format(dtYearInfo))

            selDbTable = 'TB_ACT_DATA_{}'.format(dtYearInfo)

            # 테이블 존재 여부
            resTableExist = pd.read_sql(
                """
                    SELECT COUNT(*) AS CNT FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME  = '{}'
                """.format('DMS02', selDbTable)
                , con=dbEngine
            )

            if (resTableExist.loc[0, 'CNT'] < 1): continue

            res = pd.read_sql(
                """
                SELECT *
                FROM `{}`
                WHERE SRV = '{}' AND DATE_TIME BETWEEN '{}' AND '{}'
                """.format(selDbTable, srvId, sysOpt['srtDate'], sysOpt['endDate'])
                , con=dbEngine
            )
            if (len(res) < 0): continue
            forData = pd.concat([forData, res], ignore_index=False)

        # *****************************************************
        # 데이터 전처리 및 병합
        # *****************************************************
        pvDataL1 = pvData.loc[(pvData['PV'] > 0) & (pvData['PV'] <= posVol)]
        pvDataL1['PV'][pvDataL1['PV'] < 0] = np.nan

        forData['CA_TOT'][forData['CA_TOT'] < 0] = np.nan
        forData['WS'][forData['WS'] < 0] = np.nan
        forData['WD'][forData['WD'] < 0] = np.nan
        forData['SWR'][forData['SWR'] < 0] = np.nan
        forData['SWR_intp'][forData['SWR_intp'] < 0] = np.nan
        forData['DSR'][forData['DSR'] < 0] = np.nan
        forData['DSR_intp'][forData['DSR_intp'] < 0] = np.nan
        forData['WD'][forData['WD'] < 0] = np.nan
        forData['WS'][forData['WS'] < 0] = np.nan
        forData['PA'][forData['PA'] < 0] = np.nan
        forData['TA'][forData['TA'] < -50.0] = np.nan
        forData['TD'][forData['TD'] < -50.0] = np.nan

        inpData = forData.merge(pvDataL1, how='left', left_on=['SRV', 'DATE_TIME_KST', 'DATE_TIME'], right_on=['SRV', 'DATE_TIME_KST', 'DATE_TIME'])
        # inpDataL1 = inpData.sort_values(by=['ANA_DATE','DATE_TIME_KST'], axis=0)
        inpDataL1 = inpData.drop_duplicates(['SRV', 'DATE_TIME', 'DATE_TIME_KST']).sort_values(by=['SRV', 'DATE_TIME', 'DATE_TIME_KST'], axis=0).reset_index(drop=True)
        prdData = inpDataL1.copy()

        result = {
            'msg': 'succ'
            , 'inpDataL1': inpDataL1
            , 'prdData': prdData
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('subActModelInpData'))




def subForModelInpData(sysOpt, cfgInfo, srvId, posVol, dtIncDateList):

    log.info('[START] {}'.format('subForModelInpData'))
    # log.info('[CHECK] sysOpt : {}'.format(sysOpt))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] srvId : {}'.format(srvId))
    # log.info('[CHECK] posVol : {}'.format(posVol))
    # log.info('[CHECK] dtIncDateList : {}'.format(dtIncDateList))

    result = None

    try:

        dtYearList = dtIncDateList.strftime('%Y').unique().tolist()
        session = cfgInfo['session']
        dbEngine = cfgInfo['dbEngine']

        # *****************************************************
        # 발전량 데이터
        # *****************************************************
        pvData = pd.DataFrame()
        # dtYearInfo = dtYearList[0]
        for idx, dtYearInfo in enumerate(dtYearList):
            log.info("[CHECK] dtYearInfo : {}".format(dtYearInfo))

            selDbTable = 'TB_PV_DATA_{}'.format(dtYearInfo)

            # 테이블 존재 여부
            resTableExist = pd.read_sql(
                """
                    SELECT COUNT(*) AS CNT FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME  = '{}'
                """.format('DMS02', selDbTable)
                , con=dbEngine
            )

            if (resTableExist.loc[0, 'CNT'] < 1): continue

            res = pd.read_sql(
                """
                SELECT SRV, DATE_TIME, DATE_TIME_KST, PV
                FROM `{}`
                WHERE SRV = '{}' AND DATE_TIME_KST BETWEEN '{}' AND '{}'
                """.format(selDbTable, srvId, sysOpt['srtDate'], sysOpt['endDate'])
                , con=dbEngine
            )
            if (len(res) < 0): continue
            pvData = pd.concat([pvData, res], ignore_index=False)

        # *****************************************************
        # 기상정보 데이터
        # *****************************************************
        forData = pd.DataFrame()
        # dtYearInfo = dtYearList[0]
        for idx, dtYearInfo in enumerate(dtYearList):
            log.info("[CHECK] dtYearInfo : {}".format(dtYearInfo))

            # selDbTable = 'TB_FOR_DATA_{}'.format(dtYearInfo)
            selDbTable = 'TB_FOR_DATA_{}'.format(dtYearInfo)

            # 테이블 존재 여부
            resTableExist = pd.read_sql(
                """
                    SELECT COUNT(*) AS CNT FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME  = '{}'
                """.format('DMS02', selDbTable)
                , con=dbEngine
            )

            if (resTableExist.loc[0, 'CNT'] < 1): continue

            res = pd.read_sql(
                """
                SELECT *
                FROM `{}`
                WHERE SRV = '{}' AND ANA_DATE BETWEEN '{}' AND '{}'
                """.format(selDbTable, srvId, sysOpt['srtDate'], sysOpt['endDate'])
                , con=dbEngine
            )
            if (len(res) < 0): continue
            forData = pd.concat([forData, res], ignore_index=False)

        # *****************************************************
        # 데이터 전처리 및 병합
        # *****************************************************
        pvDataL1 = pvData.loc[(pvData['PV'] > 0) & (pvData['PV'] <= posVol)]
        pvDataL1['PV'][pvDataL1['PV'] < 0] = np.nan

        forData['CA_TOT'][forData['CA_TOT'] < 0] = np.nan
        forData['WS'][forData['WS'] < 0] = np.nan
        forData['WD'][forData['WD'] < 0] = np.nan
        forData['SWR'][forData['SWR'] < 0] = np.nan

        inpData = forData.merge(pvDataL1, how='left', left_on=['SRV', 'DATE_TIME_KST', 'DATE_TIME'], right_on=['SRV', 'DATE_TIME_KST', 'DATE_TIME'])
        #inpDataL1 = inpData.sort_values(by=['ANA_DATE','DATE_TIME_KST'], axis=0)
        inpDataL1 = inpData.drop_duplicates(['SRV', 'ANA_DATE', 'DATE_TIME_KST']).sort_values(by=['SRV', 'ANA_DATE', 'DATE_TIME_KST'], axis=0).reset_index(drop=True)
        prdData = inpDataL1.copy()

        result = {
            'msg': 'succ'
            , 'inpDataL1': inpDataL1
            , 'prdData': prdData
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('subForModelInpData'))


def subVisPrd(subOpt, prdData):

    log.info('[START] {}'.format('subVisPrd'))
    log.info('[CHECK] subOpt : {}'.format(subOpt))
    # log.info('[CHECK] prdData : {}'.format(prdData))

    result = None

    try:

        anaDataList = prdData['ANA_DATE'].unique()

        minAnaData = pd.to_datetime(anaDataList).min().strftime("%Y%m%d")
        maxAnaData = pd.to_datetime(anaDataList).max().strftime("%Y%m%d")
        # anaDataInfo = anaDataList[0]
        # for idx, anaDataInfo in enumerate(anaDataList):
        #
        #     prdDataL1 = prdData.loc[
        #         prdData['ANA_DATE'] == anaDataInfo
        #         ].dropna().reset_index(drop=True)
        #
        #     mainTitle = '[{}] {}'.format(pd.to_datetime(anaDataInfo).strftime("%Y%m%d"), '기상 예보 정보 (수치모델)를 활용한 48시간 예측 시계열')
        #     saveImg = '{}/{}/{}/{}.png'.format(globalVar['figPath'], subOpt['mlModel']['modelKey'], subOpt['mlModel']['srvId'], mainTitle)
        #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        #
        #     if (os.path.exists(saveImg)): continue
        #     rtnInfo = makeUserTimeSeriesPlot(pd.to_datetime(prdDataL1['DATE_TIME']), prdDataL1['ML2'], prdDataL1['DL2'], prdDataL1['PV'], '예측 (머신러닝)', '예측 (딥러닝)', '실측 (발전량)', '시간 (시)', '발전량', mainTitle, saveImg, True)
        #     log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

        mainTitle = '[{}-{}] {}'.format(minAnaData, maxAnaData, '기상 예보 정보 (수치모델)를 활용한 머신러닝 (48시간 예측) 2D 산점도')
        saveImg = '{}/{}/{}/{}.png'.format(globalVar['figPath'], subOpt['mlModel']['modelKey'], subOpt['mlModel']['srvId'], mainTitle)
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        rtnInfo = makeUserHist2dPlot(prdData['ML2'], prdData['PV'], '머신러닝', '실측', mainTitle, saveImg, 0, 1000, 20, 60, 20, True)
        log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

        mainTitle = '[{}-{}] {}'.format(minAnaData, maxAnaData, '기상 예보 정보 (수치모델)를 활용한 딥러닝 (48시간 예측) 2D 산점도')
        saveImg = '{}/{}/{}/{}.png'.format(globalVar['figPath'], subOpt['mlModel']['modelKey'], subOpt['mlModel']['srvId'], mainTitle)
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        rtnInfo = makeUserHist2dPlot(prdData['DL2'], prdData['PV'], '딥러닝', '실측', mainTitle, saveImg, 0, 1000, 20, 60, 20, True)
        log.info('[CHECK] rtnInfo : {}'.format(rtnInfo))

        result = {
            'msg': 'succ'
            , 'prdData': prdData
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('subVisPrd'))


def reqPvApi(cfgInfo, dtYmd, id):

    # log.info('[START] {}'.format('subVisPrd'))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] dtYmd : {}'.format(dtYmd))
    # log.info('[CHECK] id : {}'.format(id))

    result = None

    try:

        apiUrl = cfgInfo['apiUrl']
        apiToken = cfgInfo['apiToken']
        stnId = id
        srvId = 'SRV{:05d}'.format(id)

        reqUrl = '{}/{}/{}'.format(apiUrl, dtYmd, id)
        reqHeader = {'Authorization': 'Bearer {}'.format(apiToken)}
        res = requests.get(reqUrl, headers=reqHeader)

        if not (res.status_code == 200): return result
        resJson = res.json()

        if not (resJson['success'] == True): return result
        resInfo = resJson['pvs']

        if (len(resInfo) < 1): return result
        resData = pd.DataFrame(resInfo)

        resData = resData.rename(
            {
                'pv': 'PV'
            }
            , axis='columns'
        )

        resData['SRV'] = srvId
        resData['DATE_TIME_KST'] = pd.to_datetime(resData['date'], format='%Y-%m-%d %H')
        resData['DATE_TIME'] = resData['DATE_TIME_KST'] - dtKst
        resData = resData.drop(['date'], axis='columns')

        result = {
            'msg': 'succ'
            , 'resData': resData
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    # finally:
    #     # try, catch 구문이 종료되기 전에 무조건 실행
    #     log.info('[END] {}'.format('reqPvApi'))


def setPvDataDB(cfgInfo, dbData, dtYear):

    # log.info('[START] {}'.format('setPvDataDB'))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] dbData : {}'.format(dbData))
    # log.info('[CHECK] dtYear : {}'.format(dtYear))

    # result = None

    try:

        session = cfgInfo['session']
        dbEngine = cfgInfo['dbEngine']
        selDbTable = 'TB_PV_DATA_{}'.format(dtYear)

        session.execute(
            """
                CREATE TABLE IF NOT EXISTS `{}`
                (
                    SRV           varchar(10) not null comment '관측소 정보',
                    DATE_TIME     datetime    not null comment '날짜 UTC',
                    DATE_TIME_KST datetime    null comment '날짜 KST',
                    PV            float       null comment '발전량',
                    REG_DATE      datetime    null comment '등록일',
                    MOD_DATE      datetime    null comment '수정일',
                    primary key (SRV, DATE_TIME)
                )    
                    comment '발전량 테이블_{}';
            """.format(selDbTable, dtYear)
        )

        for idx, dbInfo in dbData.iterrows():

            # 중복 검사
            resChk = pd.read_sql(
                """
                SELECT COUNT(*) AS CNT FROM `{}`
                WHERE SRV = '{}' AND DATE_TIME = '{}'
                """.format(selDbTable, dbInfo['SRV'], dbInfo['DATE_TIME'])
                , dbEngine
            )

            if (resChk.loc[0, 'CNT'] > 0):
                dbInfo['MOD_DATE'] = datetime.datetime.now()
                session.execute(
                    """
                    UPDATE `{}` SET PV = '{}', MOD_DATE = '{}' WHERE SRV = '{}' AND DATE_TIME = '{}'; 
                    """.format(selDbTable, dbInfo['PV'], dbInfo['MOD_DATE'], dbInfo['SRV'], dbInfo['DATE_TIME'])
                )

            else:
                dbInfo['REG_DATE'] = datetime.datetime.now()
                session.execute(
                    """
                    INSERT INTO `{}` (SRV, DATE_TIME, DATE_TIME_KST, PV, REG_DATE, MOD_DATE) VALUES ('{}', '{}', '{}', '{}', '{}', '{}') 
                    """.format(selDbTable, dbInfo['SRV'], dbInfo['DATE_TIME'], dbInfo['DATE_TIME_KST'], dbInfo['PV'], dbInfo['REG_DATE'], dbInfo['REG_DATE'])
                )

            session.commit()

    except Exception as e:
        log.error('Exception : {}'.format(e))
        session.rollback()

    finally:
        session.close()

        # try, catch 구문이 종료되기 전에 무조건 실행
        # log.info('[END] {}'.format('setPvDataDB'))


def subPvData(cfgInfo, sysOpt, posDataL1, dtIncDateList):

    log.info('[START] {}'.format('subPvData'))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] sysOpt : {}'.format(sysOpt))
    # log.info('[CHECK] posDataL1 : {}'.format(posDataL1))
    # log.info('[CHECK] dtIncDateList : {}'.format(dtIncDateList))
    result = None

    try:
        # dtIncDateInfo = dtIncDateList[0]
        stnId = sysOpt.get('stnId')

        for i, dtIncDateInfo in enumerate(dtIncDateList):
            log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))

            dtYmd = dtIncDateInfo.strftime('%Y/%m/%d')
            dtYear = dtIncDateInfo.strftime('%Y')

            isSearch = True if ((stnId == None) or (len(stnId) < 1)) else False
            if (isSearch):
                for j, posInfo in posDataL1.iterrows():
                    id = int(posInfo['ID'])

                    result = reqPvApi(cfgInfo, dtYmd, id)
                    # log.info("[CHECK] result : {}".format(result))

                    resData = result['resData']
                    if (len(resData) < 1): continue

                    setPvDataDB(cfgInfo, resData, dtYear)
            else:
                id = int(stnId)

                result = reqPvApi(cfgInfo, dtYmd, id)
                # log.info("[CHECK] result : {}".format(result))

                resData = result['resData']
                if (len(resData) < 1): continue

                setPvDataDB(cfgInfo, resData, dtYear)

        result = {
            'msg': 'succ'
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('subPvData'))


def setUpdActDataDB(cfgInfo, dbData):

    # log.info('[START] {}'.format('setPrdActDataDB'))
    # log.info('[CHECK] cfgInfo : {}'.format(cfgInfo))
    # log.info('[CHECK] dbData : {}'.format(dbData))

    try:

        session = cfgInfo['session']
        dbEngine = cfgInfo['dbEngine']

        for k, dbInfo in dbData.iterrows():

            #iYear = int(dbInfo['ANA_DATE'].strftime("%Y"))
            iYear = int(dbInfo['DATE_TIME'].strftime("%Y"))
            selDbTable = 'TB_ACT_DATA_{}'.format(iYear)

            # 테이블 존재 여부
            resTableExist = pd.read_sql(
                """
                    SELECT COUNT(*) AS CNT FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME  = '{}'
                """.format('DMS02', selDbTable)
                , con=dbEngine
            )

            if (resTableExist.loc[0, 'CNT'] < 1): continue

            # 테이블 중복 검사
            resChk = pd.read_sql(
                """
                    SELECT COUNT(*) AS CNT FROM `{}`
                    WHERE SRV = '{}' AND DATE_TIME = '{}'
                """.format(selDbTable, dbInfo['SRV'], dbInfo['DATE_TIME'])
                , con=dbEngine
            )

            if (resChk.loc[0, 'CNT'] < 1): continue

            dbInfo['MOD_DATE'] = datetime.datetime.now()

            session.execute(
                """
                UPDATE `{}` SET CA_TOT = '{}', HM = '{}', PA = '{}', TA = '{}', TD = '{}', WD = '{}', WS = '{}'
                WHERE SRV = '{}' AND DATE_TIME = '{}';
                """.format(selDbTable
                           , dbInfo['CA_TOT'], dbInfo['HM'], dbInfo['PA'], dbInfo['TA'], dbInfo['TD'], dbInfo['WD'], dbInfo['WS']
                           , dbInfo['SRV'], dbInfo['DATE_TIME'])
            )

            session.commit()

    except Exception as e:
        log.error('Exception : {}'.format(e))
        session.rollback()

    finally:
        session.close()
        # try, catch 구문이 종료되기 전에 무조건 실행
        # log.info('[END] {}'.format('setPrdActDataDB'))

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
    # python3 /SYSTEMS/PROG/PYTHON/PV/TalentPlatform-LSH0255-RealTime-For.py --inpPath "/DATA" --outPath "/SYSTEMS/OUTPUT" --modelPath "/DATA" --srtDate "20220101" --endDate "20220102"

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PV_20220523'

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

            if (platform.system() == 'Windows'):

                # 'srtDate': '2022-08-01'
                # , 'endDate': '2022-09-21'

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': '2021-10-01'
                    # , 'endDate': '2021-10-10'
                    'srtDate': '2022-08-01'
                    , 'endDate': '2022-09-21'
                    # 'srtDate': '2019-01-01'
                    # , 'endDate': '2022-05-22'
                    # , 'endDate': '2021-11-01'

                    #  딥러닝
                    , 'dlModel': {
                        # 초기화
                        'isInit': False

                        # 모형 업데이트 여부
                        # , 'isOverWrite': True
                        , 'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }

                    #  머신러닝
                    , 'mlModel': {
                        # 모델 업데이트 여부
                        # 'isOverWrite': True
                        'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }
                }

                globalVar['inpPath'] = 'E:/DATA/OUTPUT'
                globalVar['outPath'] = 'E:/DATA/OUTPUT'
                globalVar['figPath'] = 'E:/DATA/OUTPUT'
                globalVar['modelPath'] = 'E:/DATA'

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': globalVar['srtDate']
                    # , 'endDate': globalVar['endDate']
                    'srtDate': '2018-01-01'
                    , 'endDate': '2022-07-31'

                    #  딥러닝
                    , 'dlModel': {
                        # 초기화
                        'isInit': False

                        # 모형 업데이트 여부
                        # , 'isOverWrite': True
                        , 'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }

                    #  머신러닝
                    , 'mlModel': {
                        # 모델 업데이트 여부
                        # 'isOverWrite': True
                        'isOverWrite': False

                        # 모형 버전 (날짜)
                        , 'modelVer': '*'
                    }
                }

                globalVar['inpPath'] = '/DATA'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['modelPath'] = '/DATA'

            # DB 정보
            cfgInfo = initCfgInfo(globalVar['sysPath'])
            dbEngine = cfgInfo['dbEngine']

            # =======================================================================
            # 관측소 정보
            # =======================================================================
            posDataL1 = pd.read_sql(
                """
               SELECT *
               FROM TB_STN_INFO
               WHERE OPER_YN = 'Y';
               """
            , con = dbEngine)

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Minute(10))
            dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))

            # =======================================================================
            # 기상정보 자료 수집 및 DB 삽입
            # =======================================================================
            #subAtmosActData(cfgInfo, posDataL1, dtIncDateList)

            # =======================================================================
            # 발전량 자료 수집 및 DB 삽입
            # =======================================================================
            #subPvData(cfgInfo, sysOpt, posDataL1, dtPvDateList)

            # =======================================================================
            # 발전량 관측소에 따른 머신러닝/딥러닝 예측
            # =======================================================================
            for idx, posInfo in posDataL1.iterrows():
                posId = int(posInfo['ID'])
                posVol = posInfo['VOLUME']

                srvId = 'SRV{:05d}'.format(posId)
                log.info("[CHECK] srvId : {}".format(srvId))

                actRes = subActModelInpData(sysOpt, cfgInfo, srvId, posVol, dtIncDateList)
                # actInpDataL1 = actRes['inpDataL1']
                actData = actRes['prdData']

                # 시간 변환
                actDataL1 = actData[['SRV', 'DATE_TIME', 'DATE_TIME_KST', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS']]
                actDataL1['DATE_TIME_KST'] = actDataL1['DATE_TIME']
                actDataL1['DATE_TIME'] = actDataL1['DATE_TIME'] - dtKst

                # actData['sDate'] = pd.to_datetime(actData['DATE_TIME_KST']).dt.strftime('%Y%m%d')
                # for i, dtDayInfo in enumerate(dtDayList):
                #     sDate = pd.to_datetime(dtDayInfo).strftime('%Y%m%d')
                #
                #     actDataL1 = actData.loc[
                #         actData['sDate'] == sDate
                #     ]
                #
                #     if (len(actDataL1) < 1): continue
                #
                #     break
                #
                # colInfo = 'TA'
                #
                # plt.plot(actDataL1['DATE_TIME_KST'], actDataL1[colInfo], label='실황', marker='o')
                # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
                # plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
                # plt.gcf().autofmt_xdate()
                # plt.xticks(rotation=45, ha='right')
                # plt.legend(loc='upper left')
                # plt.show()

                # # DB 삽입
                setUpdActDataDB(cfgInfo, actDataL1)


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
