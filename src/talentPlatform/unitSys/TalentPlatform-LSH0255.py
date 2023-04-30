# -*- coding: utf-8 -*-

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
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
from scipy.stats import linregress
import pandas as pd
import cartopy.crs as ccrs
import math
from scipy import spatial
from pandas.tseries.offsets import Day, Hour, Minute, Second
from scipy.interpolate import Rbf
from numpy import zeros, newaxis

from src.talentPlatform.tmp.Test import serviceName

os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share"
# from pygem import IDW

# import pykrige.kriging_tools as kt

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


# 시계열 시각화
def makeUserTimeSeriesPlot(dtDate, prdVal, refVal, prdValLabel, refValLabel, xlab, ylab, mainTitle, saveImg):

    # 그리드 설정
    plt.grid(True)

    plt.plot(dtDate, prdVal, label=prdValLabel)
    plt.plot(dtDate, refVal, label=refValLabel)

    # 제목, x축, y축 설정
    plt.title(mainTitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left')

    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

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
    plt.plot(prdVal, lmfit, color='red', linewidth=2)

    # 라벨 추가
    plt.annotate('%s = %.2f x (%s) + %.2f' % (ylab, slope, xlab, intercept), xy=(minVal + xIntVal, maxVal - yIntVal),
                 color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')
    plt.annotate('R = %.2f  (p-value < %.2f)' % (R, Pvalue), xy=(minVal + xIntVal, maxVal - yIntVal * 2),
                 color='red', xycoords='data', horizontalalignment='left', verticalalignment='center')

    if (isSame == True):
        # plt.axes().set_aspect('equal')

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


def cartesian(latitude, longitude, elevation=0):
    # Convert to radians
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371  # 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)

    return (X, Y, Z)

# 외삽 과정에서 결측값 처리
def extrapolate_nans(x, y, v):
    if np.ma.is_masked(v):
        nans = v.mask
    else:
        nans = np.isnan(v)
    notnans = np.logical_not(nans)
    v[nans] = griddata((x[notnans], y[notnans]), v[notnans],
        (x[nans], y[nans]), method='nearest').ravel()
    return v

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 한국 월별 수출액과 코스피 종간 간의 회귀분석

    # conda activate
    #  python3 TalentPlatform-LSH0255.py --inpPath "/SYSTEMS/OUTPUT" --srtDate "2019-01"
    #  /SYSTEMS/OUTPUT/OBS/202109/01/AWS_OBS_202109010000.txt

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
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

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': globalVar['srtDate']
                , 'endDate': globalVar['endDate']
                # , 'endDate': '2021-04-09'
            }

            # sysOpt = {
            #     # 시작/종료 시간
            #     'srtDate': '2020-09-01'
            #     , 'endDate': '2020-10-01'
            # }

            # 주소 전라북도 임실군 삼계면 오지리   산 82-1
            # 발전설비용량 : 996.45
            # Latitude:  35.545380
            # Longitude:  127.283937

            posInfo = {
                'lat' : 35.545380
                , 'lon' : 127.283937
                , 'size' : 996.45
                , 'addr' : '전라북도 임실군 삼계면 오지리 산 82-1'
            }

            globalVar['inpPath'] = 'E:/DATA/OUTPUT'
            globalVar['outPath'] = 'E:/DATA/OUTPUT'

            # *******************************************************
            # 지상 관측소
            # *******************************************************
            inpAsosStnFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/ASOS_STN_INFO.csv')
            asosStnData = pd.read_csv(inpAsosStnFile)
            asosStnDataL1 = asosStnData[['STN', 'LON', 'LAT']]

            # 1 km 해상도 생산
            gridLon = [124, 131]
            gridLat = [33, 39]
            gridSize = 0.01

            lon1D = np.arange(min(gridLon), max(gridLon) + gridSize, gridSize)
            lat1D = np.arange(min(gridLat), max(gridLat) + gridSize, gridSize)
            lon2D, lat2D = np.meshgrid(lon1D, lat1D)

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            # dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))

            # dtIncDateInfo = dtIncDateList[0]
            # for i, dtIncDateInfo in enumerate(dtIncDateList):
                # print(i, dtIncDateInfo)

                # python3 TalentPlatform-LSH0255.py --inpPath "/SYSTEMS/OUTPUT" --srtDate "2020-09"
            dtDateYm = dtSrtDate.strftime('%Y%m')

            # globalVar['srtDate']

            # /SYSTEMS/OUTPUT/OBS/202109/01/AWS_OBS_202109010000.txt
            # inpAsosFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ASOS_OBS_*.txt')
            # sysOpt
            inpAsosFilePattern = 'OBS/{0}/*/ASOS_OBS_{0}*.txt'.format(dtDateYm)
            inpAsosFile = '{}/{}'.format(globalVar['inpPath'], inpAsosFilePattern)
            fileList = sorted(glob.glob(inpAsosFile))

            log.info("[CHECK] inpAsosFile : {}".format(inpAsosFile))
            log.info("[CHECK] fileList : {}".format(fileList))

            # sys.exit()

            #--------------------------------------------------------------------------------------------------
            #  기상청 지상관측 시간자료 [입력인수형태][예] ?tm=201007151200&stn=0&help=1
            #--------------------------------------------------------------------------------------------------
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

            dataL1 = pd.DataFrame()
            for fileInfo in fileList:
                data = pd.read_csv(fileInfo, header=None, delimiter='\s+')
                dataL1 = dataL1.append(data)

            # dataL1 = dataL1.reset_index(drop=True)
            dataL1.columns = ['TM', 'STN', 'WD', 'WS', 'GST_WD', 'GST_WS', 'GST_TM', 'PA', 'PS', 'PT', 'PR', 'TA', 'TD', 'HM', 'PV', 'RN', 'RN_DAY', 'TMP'
                               , 'RN_INT', 'SD_HR3', 'SD_DAY', 'SD_TOT', 'WC', 'WP', 'WW', 'CA_TOT', 'CA_MID', 'CH_MIN', 'CT', 'CT_TOP', 'CT_MID'
                               , 'CT_LOW', 'VS', 'SS', 'SI', 'ST_GD', 'TS', 'TE_005', 'TE_01', 'TE_02', 'TE_03', 'ST_SEA', 'WH', 'BF', 'IR', 'IX']

            dataL2 = dataL1[['TM', 'STN', 'WD', 'WS', 'PA', 'TA', 'TD', 'HM', 'CA_TOT', 'SS', 'SI']]

            # TM 및 STN을 기준으로 중복 제거
            dataL2['TM'] = dataL2['TM'].astype(str)
            # dataL2['STN'] = dataL2['STN'].astype(str)
            dataL2.drop_duplicates(subset=['TM', 'STN'], inplace=True)

            # 결측값 제거
            # dataL3 = dataL2.replace([-9.0, -99.0, -999.0], np.nan)
            dataL3 = dataL2
            dataL3['dtDate'] = pd.to_datetime(dataL3['TM'], format='%Y%m%d%H%M')

            dtDateList = dataL3['dtDate'].unique()

            dsData = xr.Dataset()
            for i, dtDateInfo in enumerate(dtDateList):
                log.info('[CHECK] dtDateInfo : {}'.format(dtDateInfo))

                dataL4 = dataL3.loc[
                    dataL3['dtDate'] == dtDateInfo
                ]

                dataL5 = pd.merge(left=dataL4, right=asosStnDataL1, how='left', left_on = 'STN', right_on = 'STN')

                colList = dataL3.columns
                # colInfo = colList[2]
                varList = {}

                for colInfo in colList:
                    if (colInfo == 'TM') or (colInfo == 'STN') or (colInfo == 'dtDate'): continue

                    dataL6 = dataL5[['dtDate', 'LON', 'LAT', colInfo]].dropna()

                    if (len(dataL6) < 1): continue

                    posLon = dataL6['LON'].values
                    posLat = dataL6['LAT'].values
                    posVar = dataL6[colInfo].values

                    # Radial basis function (RBF) interpolation in N dimensions.
                    # 성공
                    rbfModel = Rbf(posLon, posLat, posVar, function="linear")
                    rbfRes = rbfModel(lon2D, lat2D)
                    # varList[colInfo] = rbfRes
                    varList[colInfo] = rbfRes

                ds = xr.Dataset(
                    {
                        'WD' : ( ('time', 'lat', 'lon'),  varList['WD'][newaxis, ...] )
                        , 'WS' : ( ('time', 'lat', 'lon'), varList['WS'][newaxis, ...] )
                        , 'PA' : ( ('time', 'lat', 'lon'), varList['PA'][newaxis, ...] )
                        , 'TA' : ( ('time', 'lat', 'lon'), varList['TA'][newaxis, ...] )
                        , 'TD' : ( ('time', 'lat', 'lon'), varList['TD'][newaxis, ...] )
                        , 'HM' : ( ('time', 'lat', 'lon'), varList['HM'][newaxis, ...] )
                        , 'CA_TOT' : ( ('time', 'lat', 'lon'), varList['CA_TOT'][newaxis, ...] )
                        , 'SS' : ( ('time', 'lat', 'lon'), varList['SS'][newaxis, ...] )
                        , 'SI' : ( ('time', 'lat', 'lon'), varList['SI'][newaxis, ...] )
                     }
                    , coords={
                        'time': pd.date_range(dtDateInfo, periods=1)
                        , 'lat': lat1D
                        , 'lon': lon1D
                    }
                )

                dsData = dsData.merge(ds)


            # saveFile = '{}/{}/ASOS/ASOS_OBS_{}.nc'.format(globalVar['inpPath'], serviceName, pd.to_datetime(dtDateInfo).strftime('%Y%m'))
            saveFile = '{}/TEST/ASOS_OBS_{}.nc'.format(globalVar['outPath'], pd.to_datetime(dtDateInfo).strftime('%Y%m'))
            dsData.to_netcdf(saveFile)

            log.info('[CHECK] saveFile : {}'.format(saveFile))

            # #
            # # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ASOS/*.nc')
            # inpFile = '{}/TEST/{}'.format(globalVar['inpPath'], 'ASOS_OBS_*.nc')
            # #
            # fileList = glob.glob(inpFile)
            # if fileList is None or len(fileList) < 1:
            #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #     raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            # # 자료 읽기
            # dsData = xr.open_mfdataset(fileList)

            # dtDateInfo
            # val = dsData['temp']

            # dsData.sel(
            #     time =
            # )

            # date = var['time'].data
            #
            # for info in date:
            #     print(info)
            #     varL1 = var.sel(
            #         time = info
            #     )

            # dtSrtDate = (pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')).strftime("%Y-%m-%d")
            # dtEndDate = (pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')).strftime("%Y-%m-%d")


            # dtIncDate = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))
            dtIncDate = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Minute(1))
            # dtIncDate = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))


            selNearVal = dsData.sel(lon=posInfo['lon'], time=dtIncDate, lat=posInfo['lat'], method='nearest')
            selIntpVal = dsData.interp(lon=posInfo['lon'], lat=posInfo['lat'])


            saveFile = '{}/TEST/ASOS_OBS_POS_{}.nc'.format(globalVar['outPath'], pd.to_datetime(dtDateInfo).strftime('%Y%m'))
            selIntpVal.to_netcdf(saveFile)

            selIntpVal['SS']
            selIntpVal['time']

            plt.plot(selIntpVal['TA']['time'], selIntpVal['TA'].values, 'o')

            plt.plot(selIntpVal['SS']['time'], selIntpVal['SS'].values, 'o')
            plt.ylim(0, 10)
            plt.show()

            # posInfo['lat']
            # posInfo['lon']


            # g = pd.date_range("2000-01-01", periods=1)
            # ds = xr.Dataset(
            #     {"val": (('time', "lon", "lat"), np.random.rand(2, 4, 4))},
            #     coords={
            #         "time": pd.date_range("2000-01-01", periods=2)
            #         , "lon": [10, 20, 30, 40]
            #         , 'lat': [30, 40, 50, 60]
            #     }
            # )

            # ds = xr.Dataset(
            #     {"val": (('time', "lon", "lat"), np.random.rand(1, 4, 4))},
            #     coords={
            #         "time": pd.date_range("2000-01-01", periods=2)
            #         , "lon": [10, 20, 30, 40]
            #         , 'lat': [30, 40, 50, 60]
            #     }
            # )




            # val = ds['val']
            val = ds['temp']

            selNearVal = np.nan_to_num(val.sel(lon=25, lat=24, method='nearest'))
            selIntpVal = np.nan_to_num(val.interp(lon=25, lat=24))

            ds = ds.sel(
                time = pd.to_datetime('2000-01-01', format='%Y-%m-%d')
            )

            ds['val'].mean(['lon', 'lat']).plot()
            plt.show()


            lon = ds['lon'].data
            lat = ds['lat'].data
            val = ds['val'].data

            plt.contourf(lon, lat, val[:,:])
            plt.show()


            ds.to_netcdf("saved_on_disk.nc")



            # rbf3 = Rbf(lons, lats, data, function="cubic", smooth=5)
            # nval2D = rbf3(gridLon, gridLat)

            posLon = dataL6['LON'][0]
            posLat = dataL6['LAT'][0]


            # from mba import *
            #
            # interp = mba2(lo=[-0.1, -0.1], hi=[1.1, 1.1], grid=[3, 3],
            #               coo=[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0],
            #                    [1.0, 1.0], [0.4, 0.4], [0.6, 0.6]],
            #               val=[0.2, 0.0, 0.0, -0.2, -1.0, 1.0]
            #               )
            #
            # w = interp([[0.3, 0.7]])



            eleData = pd.DataFrame()
            #
            # 501*627
            #
            # gridLon.size

            posList = []

            # kdTree를 위한 초기 데이터
            for i in range(0, lon2D.shape[0]):
                for j in range(0, lon2D.shape[1]):

                    coord = [lat2D[i, j], lon2D[i, j]]
                    posList.append(cartesian(*coord))

            # 250 314
            # 157314

            # row = 157314 / gridLon.shape[0]
            # row2 = 157314 % gridLon.shape[0]

            # row = int(157314 / gridLon.shape[1])
            # col = 157314 % gridLon.shape[1]



            # saveCsvFile = '{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, sysOpt['id'], '그리드 설정')
            # eleData.to_csv(saveCsvFile, index=False)
            # log.info('[CHECK] saveCsvFile : {}'.format(saveCsvFile))

            # kdTree 학습
            tree = spatial.KDTree(posList)

            dataL7 = dataL6

            # kdTree를 통해 최근접 위/경도 인덱스 설정
            for i, item in dataL7.iterrows():
                coord = cartesian(item['LAT'], item['LON'])
                closest = tree.query([coord], k=1)

                cloIdx = closest[1][0]
                # print(item['LAT'], item['LON'], cloIdx)

                row = int(cloIdx / lon2D.shape[1])
                col = cloIdx % lon2D.shape[1]

                dataL7._set_value(i, 'idx', closest[1][0])
                dataL7._set_value(i, 'dist', closest[0][0])
                dataL7._set_value(i, 'row', int(cloIdx / lon2D.shape[1]))
                dataL7._set_value(i, 'col', cloIdx % lon2D.shape[1])
                dataL7._set_value(i, 'rbf', rbfRes[row, col])

                selNearVal = np.nan_to_num(val.sel(lon=item['LON'], lat=item['LAT'], method='nearest'))
                selIntpVal = np.nan_to_num(val.interp(lon=item['LON'], lat=item['LAT']))

                dataL7._set_value(i, 'near', selNearVal)
                dataL7._set_value(i, 'intp', selIntpVal)




            # s = interpolate.InterpolatedUnivariateSpline(lons, lats)

            # plt.scatter(gridLon, gridLat, c=nval2D)
            # plt.scatter(gridLon, gridLat, c=nval2D)
            # plt.scatter(dataL6['LON'], dataL6['LAT'], c=dataL6['temp'])
            # plt.scatter(dataL6['LON'], dataL6['LAT'], c=dataL6['lenar'])
            # plt.scatter(dataL7['LON'], dataL7['LAT'], c=dataL7['temp'])

            # plt.scatter(dataL7['LON'], dataL7['LAT'], c=dataL7['temp'])
            # plt.scatter(dataL7['LON'], dataL7['LAT'], c=dataL7['lear'])
            plt.scatter(dataL7['LON'], dataL7['LAT'], c=dataL7['rbf'])
            plt.colorbar()
            plt.show()

            # plt.contourf(gridLon, gridLat, lear)
            # plt.contourf(gridLon2, gridLat2, lear2)
            # plt.scatter(gridLon, gridLat, c=lear)
            plt.scatter(lon2D, lat2D, c=rbfRes)
            # plt.contourf(lon1D, lat1D, rbfRes)
            plt.colorbar()
            plt.show()



            # ************************************************************************
            # 산점도 시각화
            # ************************************************************************
            mainTitle = 'ASOS 자료를 이용한 지표면온도 산점도 결과'
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)

            makeUserScatterPlot(dataL7['rbf'], dataL7['temp'], '예측', '실측', mainTitle, saveImg,
                                0, 6, 0.2, 0.4, True)

            mainTitle = 'ASOS 자료를 이용한 지표면온도 산점도 결과2'
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)

            makeUserScatterPlot(dataL7['near'], dataL7['temp'], '예측', '실측', mainTitle, saveImg,
                                0, 6, 0.2, 0.4, True)

            mainTitle = 'ASOS 자료를 이용한 지표면온도 산점도 결과3'
            saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, mainTitle)

            makeUserScatterPlot(dataL7['intp'], dataL7['temp'], '예측', '실측', mainTitle, saveImg,
                                0, 6, 0.2, 0.4, True)




            # ax.coastlines()
            # plt.contourf(lons, lats, nval2D)
            # plt.contourf(lonGrid, latGrid, nval2D)
            # plt.contourf(grid_lon, grid_lat, df_grid['Krig_gaussian'])
            # plt.contourf(ds['DSR'])
            # ax.gridlines(draw_labels=True)
            plt.colorbar()
            plt.show()

            from scipy.interpolate import Rbf
            rng = np.random.default_rng()
            x, y, z, d = rng.random((4, 50))
            rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance
            xi = yi = zi = np.linspace(0, 1, 20)
            di = rbfi(xi, yi, zi)  # interpolated values
            # di.shape



            nx, ny, nz = (20, 20, 20)
            mesh = np.zeros((nx * ny * nz, 3))
            xv = np.linspace(0, 1, nx)
            yv = np.linspace(0, 1, ny)
            zv = np.linspace(0, 1, nz)
            z, y, x = np.meshgrid(zv, yv, xv)
            mesh_points = np.array([x.ravel(), y.ravel(), z.ravel()])
            idw = IDW()
            idw.read_parameters('tests/test_datasets/parameters_idw_cube.prm')
            new_mesh_points = idw(mesh_points.T)



            data = np.array([[0.3, 1.2, 0.47],
                             [1.9, 0.6, 0.56],
                             [1.1, 3.2, 0.74],
                             [3.3, 4.4, 1.47],
                             [4.7, 3.8, 1.74]])

            gridx = np.arange(0.0, 5.5, 0.5)
            gridy = np.arange(0.0, 5.5, 0.5)

            from pykrige.rk import Krige
            from sklearn.model_selection import GridSearchCV

            # 2D Kring param opt

            param_dict = {
                "method": ["ordinary", "universal"],
                "variogram_model": ["linear", "power", "gaussian", "spherical"],
                # "nlags": [4, 6, 8],
                # "weight": [True, False]
            }

            estimator = GridSearchCV(Krige(), param_dict, verbose=True, return_train_score=True)

            # dummy data
            X = np.random.randint(0, 400, size=(100, 2)).astype(float)
            y = 5 * np.random.rand(100)

            # run the gridsearch
            estimator.fit(X=X, y=y)

            if hasattr(estimator, "best_score_"):
                print("best_score R² = {:.3f}".format(estimator.best_score_))
                print("best_params = ", estimator.best_params_)

            print("\nCV results::")
            if hasattr(estimator, "cv_results_"):
                for key in [
                    "mean_test_score",
                    "mean_train_score",
                    "param_method",
                    "param_variogram_model",
                ]:
                    print(" - {} : {}".format(key, estimator.cv_results_[key]))

            # 3D Kring param opt

            param_dict3d = {
                "method": ["ordinary3d", "universal3d"],
                "variogram_model": ["linear", "power", "gaussian", "spherical"],
                # "nlags": [4, 6, 8],
                # "weight": [True, False]
            }

            estimator = GridSearchCV(Krige(), param_dict3d, verbose=True, return_train_score=True)

            # dummy data
            X3 = np.random.randint(0, 400, size=(100, 3)).astype(float)
            y = 5 * np.random.rand(100)

            # run the gridsearch
            estimator.fit(X=X3, y=y)

            if hasattr(estimator, "best_score_"):
                print("best_score R² = {:.3f}".format(estimator.best_score_))
                print("best_params = ", estimator.best_params_)

            print("\nCV results::")
            if hasattr(estimator, "cv_results_"):
                for key in [
                    "mean_test_score",
                    "mean_train_score",
                    "param_method",
                    "param_variogram_model",
                ]:
                    print(" - {} : {}".format(key, estimator.cv_results_[key]))


            # *******************************************************
            # GK2A OK
            # *******************************************************
            # inpGk2aFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'gk2a_ami_le2_swrad_ko020lc_202111230500.nc')
            inpGk2aFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'gk2a_ami_le2_swrad_ea020lc_202111230500.nc')

            # inpFilePattern = 'swe/{:04d}/{:02d}/*swe*.nc'.format(searchYear, searchMonth)
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, inpFilePattern)
            fileList = glob.glob(inpGk2aFile)

            # if fileList is None or len(fileList) < 1:
            #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #     raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

            # ds = xr.open_mfdataset(fileList)
            ds = xr.open_mfdataset(fileList)

            # 위/경도 반환
            imgProjInfo = ds['gk2a_imager_projection'].attrs

            # ccrs.LambertConformal()
            mapLccProj = ccrs.LambertConformal(
                central_longitude=imgProjInfo['central_meridian']
                , central_latitude=imgProjInfo['origin_latitude']
                , secant_latitudes=(imgProjInfo['standard_parallel1'], imgProjInfo['standard_parallel2'])
                , false_easting=imgProjInfo['false_easting']
                , false_northing=imgProjInfo['false_northing']
            )

            mapProj = pyproj.Proj(mapLccProj.to_proj4())


            nx = imgProjInfo['image_width']
            ny = imgProjInfo['image_height']
            xOffset = imgProjInfo['lower_left_easting']
            yOffset = imgProjInfo['lower_left_northing']

            res = imgProjInfo['pixel_size']

            # lon1D = np.arange(min(gridLon), max(gridLon) + gridSize, gridSize)
            # aa = np.arange(0, nx, 1)
            # aa *res + xOffset
            # 직교 좌표
            rowEle = (np.arange(0, nx, 1) * res) + xOffset
            colEle = (np.arange(0, ny, 1) * res) + yOffset
            colEle = colEle[::-1]

            # rowEle2D, colEle2D = np.meshgrid(rowEle, colEle, sparse=False, indexing='xy')
            # rowEle1D = rowEle2D.ravel()
            # colEle1D = colEle2D.ravel()
            #
            # lonGeo, latGeo = mapProj(rowEle1D, colEle1D, inverse=True)
            #
            # lonGeo2D = lonGeo.reshape(ny, nx)
            # latGeo2D = latGeo.reshape(ny, nx)



            ds = ds.assign_coords(
                {"dim_x": ("dim_x", rowEle)
                    , "dim_y": ("dim_y", colEle)
                 }
            )


            val = ds['DSR']


            # 성공
            ax = plt.axes(projection=mapLccProj)
            ax.coastlines()
            plt.contourf(ds['dim_x'], ds['dim_y'], ds['DSR'])
            # plt.contourf(ds['DSR'])
            ax.gridlines(draw_labels=True)
            plt.colorbar()
            plt.show()


            posLon, posLat = mapProj(imgProjInfo['upper_left_easting'], imgProjInfo['upper_left_northing'], inverse=True)

            # posLon = 128.2
            posLon = 160
            posLat = 30

            posRow, posCol = mapProj(posLon, posLat, inverse=False)

            selNearVal = np.nan_to_num(val.sel(dim_x=posRow, dim_y=posCol, method='nearest'))
            selIntpVal = np.nan_to_num(val.interp(dim_x=posRow, dim_y=posCol))

            mapProj(posRow, posCol, inverse=True)

            # *******************************************************
            # 히마와리 # *******************************************************
            # *******************************************************
            # inpHimaFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'H08_20211123_0000_rFL010_FLDK.02701_02601.nc')
            inpHimaFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'H08_*_*_rFL010_FLDK.02701_02601.nc')
            fileList = glob.glob(inpHimaFile)
            ds = xr.open_mfdataset(fileList)
            val = ds['SWR']

            selNearVal = np.nan_to_num(val.sel(latitude = posInfo['lat'], longitude = posInfo['lon'], method='nearest'))
            selIntpVal =  np.nan_to_num(val.interp(latitude = posInfo['lat'], longitude = posInfo['lon']))




            # other = xr.DataArray(
            #     np.sin(0.4 * np.arange(9).reshape(3, 3)),
            #     [("time", [0.9, 1.9, 2.9]), ("space", [0.15, 0.25, 0.35])],
            # )

            # dtDateTimeFore = pd.to_datetime((ds.time + i.step).values).strftime('%Y%m%d%H%M')
            # nearVal = np.nan_to_num(i.sel(latitude = inLat, longitude = inLon, method='nearest'))
            # interpVal = np.nan_to_num(i.interp(latitude = inLat, longitude = inLon))

            # plt.scatter(val['longitude'], val['latitude'], c=val.data)
            # val.plot(val)
            plt.contourf(val['longitude'], val['latitude'], val.data)
            plt.colorbar()
            plt.show()
            plt.close()


            # # 2021-11-23 05:00
            # # *******************************************************
            # # UM LDAPS
            # # *******************************************************
            # inpUmFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'l015_v070_erlo_unis_h000.2016101200.gb2')
            # # outNcFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'l015_v070_erlo_unis_h000.2016101200.nc')
            #
            # fileList = glob.glob(inpUmFile)
            #
            # # import cfgrib
            # # ds = cfgrib.open_dataset(inpUmFile)
            #
            # # ds = xr.open_dataset(fileList[0], engine="pynio")
            #
            # # d = xr.open_dataset(inpUmFile, engine='cfgrib')
            # # print(d)
            #
            # import eccodes
            # import pygrib
            # grbData = pygrib.open(inpUmFile)
            # varList = grbData.read()
            #
            # grbInfo = grbData.select(name="Temperature")[1]
            #
            # lat, lon = grbInfo.latlons()
            # var = grbInfo.values
            #
            # # 최근접 좌표
            # lat2D = lat
            # lon2D = lon
            #
            # # eleData = pd.DataFrame()
            # posList = []
            #
            # # kdTree를 위한 초기 데이터
            # for i in range(0, lon2D.shape[0]):
            #     for j in range(0, lon2D.shape[1]):
            #         coord = [lat2D[i, j], lon2D[i, j]]
            #         posList.append(cartesian(*coord))
            #
            # tree = spatial.KDTree(posList)
            #
            # coord = cartesian(posInfo['lat'], posInfo['lon'])
            # closest = tree.query([coord], k=1)
            # cloIdx = closest[1][0]
            # row = int(cloIdx / lon2D.shape[1])
            # col = cloIdx % lon2D.shape[1]
            # idx = closest[1][0]
            # dist = closest[0][0]
            # val  = var[row, col]
            #
            # import haversine as hs
            # loc1 = (28.426846, 77.088834)
            # loc2 = (28.394231, 77.050308)
            # hs.haversine(loc1, loc2)
            #
            # posData = pd.DataFrame()
            # idx = 0
            # for i in range(0, lon2D.shape[0]):
            #     for j in range(0, lon2D.shape[1]):
            #         loc1 = (lat2D[i, j], lon2D[i, j])
            #         loc2 = (posInfo['lat'], posInfo['lon'])
            #
            #         dist = hs.haversine(loc1, loc2)
            #
            #         posData._set_value(idx, 'dist', dist)
            #         posData._set_value(idx, 'i', i)
            #         posData._set_value(idx, 'j', j)
            #
            #         idx = idx + 1
            #
            # # grbInfo['forecastTime']
            # # grbInfo['time']
            # # grbInfo['dataDate']
            #
            # # attrInfo = grbData.select(name='LambertConformal_Projection')
            #
            # # grbData['LambertConformal_Projection']
            #
            # imgProjInfo = {
            #     'central_meridian': 126.0
            #     , 'origin_latitude': 30.0
            #     , 'standard_parallel1': 30.0
            #     , 'standard_parallel2': 60.0
            #     , 'false_easting': 0.0
            #     , 'false_northing': 0.0
            #     , 'image_width': 781
            #     , 'image_height': 602
            # }
            #
            #
            # # ccrs.LambertConformal()
            # mapLccProj = ccrs.LambertConformal(
            #     central_longitude=imgProjInfo['central_meridian']
            #     , central_latitude=imgProjInfo['origin_latitude']
            #     , secant_latitudes=(imgProjInfo['standard_parallel1'], imgProjInfo['standard_parallel2'])
            #     , false_easting=imgProjInfo['false_easting']
            #     , false_northing=imgProjInfo['false_northing']
            # )
            #
            # mapProj = pyproj.Proj(mapLccProj.to_proj4())
            # mapProj4326 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
            #
            # mapProj(389.478893, 618.363966, inverse=True)
            # mapProj(126.00403694776627, 30.005578238312996, inverse=False)
            # # mapProj(126.00403694776627, 30.005578238312996, inverse=False)
            #
            # # lon
            #
            #
            # # mapProj(132.53187417045402, 43.12964515366301, inverse=True)
            # xOffset, yOffset = mapProj(lon.min(), lat.min(), inverse=False)
            # xOffset, yOffset = mapProj(126.00403694776627, 30.005578238312996, inverse=False)
            # xOffset, yOffset = mapProj(121.834429, 32.256875, inverse=False)
            # mapProj(xOffset, xOffset, inverse=True)
            # mapProj4326(xOffset, yOffset, inverse=True)
            #
            # # mapProj(132.53187417045402, 43.12964515366301, inverse=True)
            # # maxRow, maxCol = mapProj(132.53187417045402, 43.12964515366301, inverse=False)
            # maxRow, maxCol = mapProj(lon.max(), lat.min(), inverse=False)
            #
            # b = abs(xOffset) + abs(maxRow)
            # b/2
            #
            # # minRow
            # lon.min()
            # lat.min()
            #
            # from pyproj import Proj, transform
            #
            # x1, y1 = xOffset, yOffset
            #
            # lon1, lat1 = transform(mapProj, mapProj4326, xOffset, yOffset)
            # # lat1 = pyproj.transform(mapProj, mapProj4326, xOffset, yOffset)
            # pyproj.transform(mapProj, mapProj4326, lon1, lat1)
            # pyproj.transform(mapProj, mapProj4326, maxRow, maxCol)
            #
            # from pyproj import Transformer
            # transformer = Transformer.from_crs(mapProj.crs, mapProj4326.crs)
            # x3, y3 = transformer.transform(x1, y1)
            #
            # ny, nx = lon.shape
            # res = 2000
            # res = 1500
            #
            # rowEle = (np.arange(0, nx, 1) * res) + xOffset
            # colEle = (np.arange(0, ny, 1) * res) + yOffset
            #
            # rowEle[301]
            #
            # rowEle.max()
            # colEle.max()
            #
            # # 직교 좌표
            # rowEle = (range(nx) * res) + xOffset
            # colEle = (range(ny) * res) + yOffset
            #
            # # lat.min()
            # # Out[131]: 32.188086607613876
            # # lat.max()
            # # Out[132]: 43.12964515366301
            # # lon.min()
            # # Out[133]: 121.0602948180667
            # # lon.max()
            # # Out[134]: 132.53187417045402
            #
            #
            # dtDateInfo = pd.to_datetime('{0}{1:02d}'.format(grbInfo['dataDate'], grbInfo['forecastTime']), format='%Y%m%d%H')
            # # grbInfo['validDate']
            #
            # # g1 = grbs[1]
            # # lats, lons = g1.latlons()
            # # latin1 = g1['Latin1InDegrees']
            # # lov = g1['LoVInDegrees']
            # # grbs.close()
            # # nlats, nlons = np.shape(lats)
            #
            #
            # # grb = t_messages[0]
            #
            # # maxt = grb.values
            # # maxt.shape
            # # maxt.min()
            # # maxt.max()
            # #
            # # grb.values
            #
            #
            # # lats, lons = grb.latlons()
            #
            # # UM 그림 그리기
            # plt.scatter(lons, lats, c=maxt)
            # plt.colorbar()
            # plt.show()
            #
            # # plt.contourf(lons, lats, maxt)
            # # plt.colorbar()
            # # plt.show()
            #
            # # dd = np.random.rand(1, len(lon1D), len(lat1D))
            #
            # dsData = np.empty((1, len(lon), 602))
            # dsData[0, :, :] = var
            #
            # lat.min()
            # lat.max()
            #
            # lon.min()
            # lon.max()
            #
            # # lat.min()
            # # Out[131]: 32.188086607613876
            # # lat.max()
            # # Out[132]: 43.12964515366301
            # # lon.min()
            # # Out[133]: 121.0602948180667
            # # lon.max()
            # # Out[134]: 132.53187417045402
            #
            # ds = xr.Dataset(
            #     {
            #         'temp': (('time', 'lat', 'lon'), dsData)
            #         , 'lat2D': (('lat', 'lon'), lat)
            #         , 'lon2D': (('lat', 'lon'), lon)
            #     }
            #     , coords={
            #         'time': pd.date_range(dtDateInfo, periods=1)
            #         # , 'lat': lat
            #         # , 'lon': lon
            #     }
            # )
            #
            # # ASOS_OBS_ *.txt
            #
            # saveFile = '{}/{}/ASOS/ASOS_OBS_{}.nc'.format(globalVar['inpPath'], serviceName,
            #                                               pd.to_datetime(dtDateInfo).strftime('%Y%m%d%H%M'))
            # ds.to_netcdf(saveFile)
            #
            # log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ASOS/*.nc')
            #
            # fileList = glob.glob(inpFile)
            # if fileList is None or len(fileList) < 1:
            #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #     raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            # # 자료 읽기
            # dsData = xr.open_mfdataset(fileList)
            #
            # val = dsData['temp']
            #
            # date = var['time'].data
            #
            # for info in date:
            #     print(info)
            #     varL1 = var.sel(
            #         time=info
            #     )
            #
            # dtSrtDate = (pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')).strftime("%Y-%m-%d")
            # dtEndDate = (pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')).strftime("%Y-%m-%d")
            #
            # # dtIncDate = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))
            # dtIncDate = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Minute(1))
            #
            # selNearVal = val.sel(lon=posInfo['lon'], time=dtIncDate, lat=posInfo['lat'], method='nearest')
            # selIntpVal = val.interp(lon=posInfo['lon'], time=dtIncDate, lat=posInfo['lat'])
            #
            # plt.plot(selIntpVal['time'], selIntpVal.values, 'o')
            # plt.plot(selNearVal['time'], selNearVal.values, 'o')
            # plt.show()
            #
            #
            #
            # # time_diff = timedelta(hours=9)
            #
            # # df = pd.DataFrame({
            # #     "validDate": [msg.validDate + time_diff for msg in t_messages],
            # #     "temperature": [
            # #         msg.data(
            # #             lat1=35.6745 - 0.025,
            # #             lat2=35.6745 + 0.025,
            # #             lon1=139.7169 - 0.03125,
            # #             lon2=139.7169 + 0.03125,
            # #         )[0][0][0] - 273.15 for msg in t_messages
            # #     ]
            # # })
            #
            #
            #
            #
            #
            # # GDAL
            # from osgeo import gdal
            # arq1 = gdal.Open(inpUmFile)
            #
            # GT_entrada = arq1.GetGeoTransform()
            # print(GT_entrada)
            #
            # from osgeo import gdal
            # # Change the following variables to the file you want to convert (inputfile) and
            # # what you want to name your input file (outputfile).
            # # inputfile = 'Greenland_vel_mosaic_250_vy_v1.tif'
            # # outputfile = 'Greenland_vel_mosaic_250_vy_v1.nc'
            # # Do not change this line, the following command will convert the geoTIFF to a netCDF
            # ds = gdal.Translate(outNcFile, inpUmFile, format='NetCDF')
            #
            ds2 = xr.open_dataset(outNcFile, engine='netcdf4')
            # # ds2['Band132']
            #
            # # conda install -c conda-forge pygrib
            #
            # val = ds['SWR']
            #
            # selNearVal = np.nan_to_num(val.sel(latitude=posInfo['lat'], longitude=posInfo['lon'], method='nearest'))
            # selIntpVal = np.nan_to_num(val.interp(latitude=posInfo['lat'], longitude=posInfo['lon']))


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
