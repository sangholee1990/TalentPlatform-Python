# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import math
import os
import platform
import sys
import traceback
from datetime import datetime

import h2o
import matplotlib as mpl
import matplotlib.pyplot as plt
import pvlib
# os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share"
# from pygem import IDW
# import pykrige.kriging_tools as kt
import pytz
import xarray as xr
from h2o.automl import H2OAutoML
from pycaret.regression import *
from scipy.stats import linregress

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
def makeUserTimeSeriesPlot(dtDate, prdVal1, prdVal2, refVal, prdValLabel1, prdValLabel2, refValLabel, xlab, ylab,
                           mainTitle, saveImg):

    #####################################################################################
    # 검증스코어 계산 : Bias (Relative Bias), RMSE (Relative RMSE)
    RMSE_ml = np.sqrt(np.mean((prdVal1 - refVal) ** 2))
    rRMSE_ml = (RMSE_ml / np.mean(refVal)) * 100.0

    RMSE_dnn = np.sqrt(np.mean((prdVal2 - refVal) ** 2))
    rRMSE_dnn = (RMSE_dnn / np.mean(refVal)) * 100.0

    # 선형회귀곡선에 대한 계산
    lmFit1 = linregress(prdVal1, refVal)
    R_ml = lmFit1[2]

    lmFit2 = linregress(prdVal2, refVal)
    R_dnn = lmFit2[2]

    prdValLabel_ml = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(prdValLabel1, R_ml, rRMSE_ml)
    prdValLabel_dnn = '{0:s} : R = {1:.3f}, %RMSE = {2:.3f} %'.format(prdValLabel2, R_dnn, rRMSE_dnn)
    #####################################################################################

    plt.grid(True)

    plt.plot(dtDate, prdVal1, label=prdValLabel_ml, marker='o')
    plt.plot(dtDate, prdVal2, label=prdValLabel_dnn, marker='o')
    plt.plot(dtDate, refVal, label=refValLabel, marker='o')

    # 제목, x축, y축 설정
    plt.title(mainTitle)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.xticks(rotation=45, ha='right')
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

            import pandas as pd

            # 옵션 설정
            # sysOpt = {
            #     # 시작/종료 시간
            #     'srtDate': globalVar['srtDate']
            #     , 'endDate': globalVar['endDate']
            # }

            sysOpt = {
                # 시작/종료 시간
                'srtDate': '2021-10-01'
                , 'endDate': '2021-11-01'
            }

            # 주소 전라북도 임실군 삼계면 오지리   산 82-1
            # 발전설비용량 : 996.45
            # Latitude:  35.545380
            # Longitude:  127.283937

            posInfo = {
                'lat': 35.545380
                , 'lon': 127.283937
                , 'size': 996.45
                , 'addr': '전라북도 임실군 삼계면 오지리 산 82-1'
            }

            posLon = posInfo['lon']
            posLat = posInfo['lat']

            globalVar['inpPath'] = 'E:/DATA/OUTPUT'
            globalVar['outPath'] = 'E:/DATA/OUTPUT'

            dtSrtDate = pd.to_datetime('2020-09-01', format='%Y-%m-%d')
            dtEndDate = pd.to_datetime('2021-09-12', format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))

            prdData = pd.DataFrame({
                'dtDate': dtIncDateList
                , 'dtDateKst': dtIncDateList.tz_localize(tzKst).tz_convert(tzKst)
            })

            dtSrtDate = pd.to_datetime('2021-09-12', format='%Y-%m-%d')
            dtEndDate = pd.to_datetime('2021-10-31', format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))
            # dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))

            testData = pd.DataFrame({
                'dtDate': dtIncDateList
                , 'dtDateKst': dtIncDateList.tz_localize(tzKst).tz_convert(tzKst)
            })


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
            asosDataL1 = asosData.where((asosData['CA_TOT'] >= 0) & (asosData['PA'] >= 940) & (asosData['SS'] > 0))
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

            trainDataL3 = prdData.merge(trainDataL2, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                .merge(asosDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                .merge(pmDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                .merge(gk2aDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                .merge(h8DataL4, how='left', left_on='dtDateKst', right_on='dtDateKst')

            testDataL1 = testData.merge(orgSheet2DataL1, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                .merge(asosDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                .merge(pmDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                .merge(gk2aDataL4, how='left', left_on='dtDateKst', right_on='dtDateKst') \
                .merge(h8DataL4, how='left', left_on='dtDateKst', right_on='dtDateKst')



            # covid_data_daily.rename({'Korea, South': 'val'}, axis='columns')
            trainDataL4 = trainDataL3.rename(
                {'dtDate_x': 'dtDate'}, axis='columns'
            # )[['dtDate', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'PM10', 'DSR', 'CLD', 'CF', 'SWR', 'pv']]
            # )[['dtDate', 'dtDateKst', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'PM10', 'DSR', 'CF', 'CLD', 'SWR', 'pv']].dropna()
            )[['dtDate', 'dtDateKst', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'PM10', 'CF', 'CLD', 'SWR', 'pv']]

            # sDateTime = "20190701000000"
            # dtDateTime = pd.to_datetime(sDateTime, format='%Y%m%d%H%M%S')
            #
            # dataL1 = data
            #

            trainDataL6 = trainDataL4.reset_index(drop=True)
            # trainDataL6 = trainDataL4

            for i in trainDataL6.index:
                lat = posLat
                lon = posLon
                pa = trainDataL6._get_value(i, 'PA')
                ta = trainDataL6._get_value(i, 'TA')
                dtDateTime = trainDataL6._get_value(i, 'dtDateKst')

                solPosInfo = pvlib.solarposition.get_solarposition(dtDateTime, lat, lon, pressure=pa, temperature=ta, method='nrel_numpy')
                trainDataL6._set_value(i, 'sza', solPosInfo['zenith'].values)
                # trainDataL6._set_value(i, 'vza', solPosInfo['elevation'].values)
                trainDataL6._set_value(i, 'aza', solPosInfo['azimuth'].values)
                trainDataL6._set_value(i, 'et', solPosInfo['equation_of_time'].values)


            # plt.scatter(trainDataL6['SWR'], trainDataL6['pv'])
            # plt.scatter(trainDataL6['sza'], trainDataL6['pv'])
            # plt.scatter(trainDataL6['aza'], trainDataL6['pv'])
            # plt.scatter(trainDataL6['dtDateKst'], trainDataL6['pv'])

            # import seaborn as sns
            #
            # corr = trainDataL6.corr(method='pearson')
            # cmap = sns.diverging_palette(220, 10, as_cmap=True)
            # sns.heatmap(corr, square=True, annot=False, cmap=cmap, vmin=-1.0, vmax=1.0, linewidths=0.5)
            # plt.show()


            # plt.show()




            # MSE: 5544.3155559281695
            # RMSE: 74.46016086423779
            # MAE: 54.11916878773443
            # RMSLE: NaN
            # R ^ 2: 0.8680844614602129

            # trainDataL7 = trainDataL6.loc[trainDataL6['CLD'] == 0 or trainDataL6['CLD'] == 1].reset_index(drop=True)
            # trainDataL7 = trainDataL6.loc[trainDataL6['CLD'] == 1].reset_index(drop=True)
            # trainDataL7 = trainDataL6.loc[trainDataL6['CLD'] == 2].reset_index(drop=True)

            # plt.scatter(trainDataL3['dtDate'], trainDataL3['CA_TOT'])
            # plt.scatter(trainDataL4['dtDate'], trainDataL4['CA_TOT'])
            # plt.scatter(asosDataL1['time'], asosDataL1['HM'])
            # plt.scatter(asosDataL1['time'], asosDataL1['PA'])
            # # plt.scatter(asosDataL1['time'], asosDataL1['SI'])
            # plt.scatter(asosDataL1['time'], asosDataL1['SS'])
            # plt.scatter(asosDataL1['time'], asosDataL1['TA'])
            # plt.scatter(asosDataL1['time'], asosDataL1['TD'])
            # plt.scatter(asosDataL1['time'], asosDataL1['WD'])
            # plt.scatter(asosDataL1['time'], asosDataL1['WS'])
            # plt.show()

            # from auto_ts import auto_timeseries
            # from plotnine import ggplot
            # from pycaret.regression import setup
            # from pycaret.regression import compare_models
            # from pycaret.regression import *


            # model = auto_timeseries(score_type='rmse', time_interval='Month', non_seasonal_pdq=None, seasonality=False,
            #                         seasonal_period=12, model_type=['Prophet'], verbose=2)
            # model = auto_timeseries(score_type='rmse', time_interval='H', model_type='best')
            # model.fit(traindata=trainDataL4, ts_column="dtDate", target="pv")
            # model.get_best_model()
            # prd22 = model.predict(testdata=trainDataL4)

            # pyModel = setup(data = dataL2, target = 'Product Amount')
            # trainDataL5 = trainDataL4[['CA_TOT', 'DSR', 'CLD', 'CF', 'SWR', 'pv']]
            # trainDataL5 = trainDataL4[['SWR', 'pv']].reset_index(drop=True)
            # trainDataL5 = trainDataL4.reset_index(drop=True)
            # trainDataL6 = trainDataL5[['CA_TOT', 'DSR', 'CLD', 'CF', 'SWR', 'pv']]
            # trainDataL5.dtypes

            # trainDataL5['pv'] = trainDataL5['pv'].astype(float)

            # trainDataL4
            #
            # trainDataL5.describe()

            #### 1. h2o 분석 준비하기 ####

            h2o.init()
            # # h2o.no_progress()
            #
            # h2o.show_progress()
            #
            #
            # # h2o_train = h2o.H2OFrame(train)
            # # h2o_valid = h2o.H2OFrame(valid)
            # #
            # # # For binary classification, response should be a factor
            # # h2o_train[y] = h2o_train[y]
            # # h2o_valid[y] = h2o_valid[y]
            # # b = ['HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'CA_TOT', 'DSR', 'CLD', 'CF', 'SWR']
            #
            # # ################################################################
            # # # Run AutoML for 1000 seconds                                  #
            # # ################################################################
            aml = H2OAutoML(max_models=20, max_runtime_secs=10000, balance_classes=True, seed=1)
            # # aml = H2OAutoML(max_models=20, max_runtime_secs=10000, seed=1)
            # # # model_id                                                   mean_residual_deviance      rmse       mse       mae       rmsle
            # # # -------------------------------------------------------  ------------------------  --------  --------  --------  ----------
            # # # StackedEnsemble_AllModels_4_AutoML_1_20211230_211812                      8651.61   93.014    8651.61   68.0968  nan
            # #
            # # # None: Do not perform any transformations on the data.
            # # # Standardize: Standardizing subtracts the mean and then divides each variable by its standard deviation.
            # # # Normalize: Scales all numeric variables in the range [0,1].
            # # # Demean: The mean for each variable is subtracting from each observation resulting in mean zero. Note that it is not always advisable to demean the data if the Moving Average parameter is of primary interest to estimate.
            # # # Descale: Divides by the standard deviation of each column.
            # #
            # # # from sklearn.preprocessing import MinMaxScaler
            # #
            # # # col = ['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'PM10', 'CF', 'SWR']
            # # # trainDataL4Std = trainDataL4
            # # # # trainDataL4Std[col] = StandardScaler().fit_transform(trainDataL4Std[col])
            # # # trainDataL4Std[col] = MinMaxScaler().fit_transform(trainDataL4Std[col])
            # # # print(ss_data)
            # #
            # #
            # # # train, test = train_test_split(trainDataL4, test_size=0.3)
            # # # train, test = train_test_split(trainDataL6, test_size=0.3)
            # # # train, test = train_test_split(trainDataL7, test_size=0.3)
            # # # train, test = train_test_split(trainDataL4Std, test_size=0.3)
            # train = train.reset_index(drop=True)
            # test = test.reset_index(drop=True)
            # #
            # # # test = trainDataL6.drop_na()
            train = trainDataL6.dropna().reset_index(drop=True)
            test = trainDataL6.dropna().reset_index(drop=True)
            # # #
            # # # train = trainDataL7.dropna()
            # # # test = trainDataL7.dropna()
            # #
            # # # boston_glm = H2OGeneralizedLinearEstimator(H2OAutoML = True)
            # # # aml.train(x=x, y=y, training_frame=h2o.H2OFrame(), validation_frame=h2o_valid)
            # # # aml.train(x=['SWR'], y='pv', training_frame=h2o.H2OFrame(trainDataL5))
            # # # aml.train(x=['HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'CA_TOT', 'DSR', 'CLD', 'CF', 'SWR'], y='pv', training_frame=h2o.H2OFrame(trainDataL4))
            # # # aml.train(x=['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'PM10', 'CF', 'SWR'], y='pv', training_frame=h2o.H2OFrame(train), validation_frame=h2o.H2OFrame(test))
            # # # aml.train(x=['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'PM10', 'CF', 'SWR', 'sza', 'vza', 'aza', 'et'], y='pv', training_frame=h2o.H2OFrame(train), validation_frame=h2o.H2OFrame(test))
            aml.train(x=['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'sza', 'aza', 'et'], y='pv', training_frame=h2o.H2OFrame(train), validation_frame=h2o.H2OFrame(test))
            # # # aml.train(x=['SWR'], y='pv', training_frame=h2o.H2OFrame(train), validation_frame=h2o.H2OFrame(test))
            # # # aml.train(x=['HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'CA_TOT', 'CF', 'SWR'], y='pv', training_frame=h2o.H2OFrame(trainDataL4Std))
            # #
            # # # m = aml.get_best_model()
            # #
            # # # plt.scatter(asosDataL1['time'], asosDataL1['CA_TOT'])
            # # # plt.scatter(asosDataL2['time'], asosDataL2['CA_TOT'])
            # # # plt.scatter(asosDataL1['time'], asosDataL1['HM'])
            # # # plt.scatter(asosDataL1['time'], asosDataL1['PA'])
            # # # plt.scatter(asosDataL2['time'], asosDataL2['SI'])
            # # # plt.scatter(asosDataL2['time'], asosDataL2['SS'])
            # # # plt.scatter(asosDataL1['time'], asosDataL1['TA'])
            # # # plt.scatter(asosDataL1['time'], asosDataL1['TD'])
            # # # plt.scatter(asosDataL1['time'], asosDataL1['WD'])
            # # # plt.scatter(asosDataL1['time'], asosDataL1['WS'])
            # # # plt.show()
            # #
            leaderboard = aml.leaderboard
            # # # performance = aml.leader.model_performance(testData)  # (Optional) Evaluate performance on a test set
            # # #
            # # # model_id = aml.leader.model_id  # 최고 모델 명
            # # # accuracy = performance.accuracy()  # 정확도
            # # # precision = performance.precision()  # precision
            # # # recall = performance.recall()  # recall
            # # # F1 = performance.F1()  # f1
            # # # auc = performance.auc()  # auc
            # # # variable_importance = aml.leader.varimp()  # 중요한 입력 변수
            # # #
            # # # # print(model_id, accuracy, precision, recall, F1, auc, variable_importance)
            # # # print(performance)
            # #
            # # # h2o.estimators.xgboost.H2OXGBoostEstimator.available()
            # #


            # f = os.path.basename(saveModel)
            saveModel = '{0}/{1}-{2}-{3}-{4}-{5}.model'.format(globalVar['outPath'], serviceName, 'final', 'h2o','test', datetime.now().strftime("%Y%m%d"))
            h2o.save_model(model=aml.get_best_model(), path=os.path.dirname(saveModel), filename=os.path.basename(saveModel), force=True)

            aa = h2o.load_model(path=saveModel)

            # # # View the AutoML Leaderboard
            # # lb = aml.leaderboard
            # # lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
            # #
            # # # valid_origin = valid_result.as_data_frame()
            # # test_result = aml.predict(h2o.H2OFrame(test)).as_data_frame()
            # # test['prd'] = test_result

            # test_result = aml.predict(h2o.H2OFrame(test)).as_data_frame()
            # test_result['prd2'] = test_result
            #
            # wrwr = aa.predict(h2o.H2OFrame(test)).as_data_frame()
            # test_result['prd3'] = wrwr


            # test_result = aml.predict(h2o.H2OFrame(trainDataL6)).as_data_frame()
            # test['prd'] = test_result

            # plt.scatter(test['pv'], test['prd'])
            # plt.plot(test['dtDate'], test['pv'])
            # plt.plot(test['dtDate'], test['prd'])
            #
            # plt.plot(test['dtDate'], test['pv'] - test['prd'])
            # plt.plot(test['dtDate'], test['pv'] - test['Label'])
            # plt.show()
            #
            # for j, dtDateInfo in enumerate(dtDateList):
            #     print(dtDateInfo)
            #
            #     testL1 = test.loc[test['dtDate'] == dtDateInfo]


            # valid_result
            # result_x = np.array(valid_result)
            # result_y = np.array(valid[['gap_real']])
            #
            # print(len(result_x))
            # print(len(result_y))

            # test = trainDataL6.drop_na()
            train = trainDataL6.dropna().reset_index(drop=True)
            test = trainDataL6.dropna().reset_index(drop=True)

            # test = trainDataL7.dropna().reset_index(drop=True)

            #
            # testL1 = test[['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'PM10', 'SWR', 'sza', 'aza', 'et', 'pv']]
            testL1 = test[['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'sza', 'aza', 'et', 'pv']]

            pyModel = setup(
                data=testL1
                , target='pv'
                , session_id=123
            )

            try:

                # 각 모형에 따른 자동 머신러닝
                modelList = compare_models(sort='RMSE', n_select=3)

                # 앙상블 모형
                blendModel = blend_models(estimator_list=modelList, fold=2)

                # 앙상블 튜닝
                tuneModel = tune_model(blendModel, fold=2, choose_better=True)

                # 학습 모델
                fnlModel = finalize_model(tuneModel)

            except Exception as e:
                log.error("Exception : {}".format(e))

            # evaluate_model(tuneModel)

            # pred_holdout = predict_model(fnlModel)

            # print(fnlModel)

            # 회귀 시각화
            plot_model(fnlModel, plot='error')
            plt.show()



            #
            # plot_model(fnlModel, plot='error', save=True)
            # saveOrgImg = '{}/{}.png'.format(globalVar['contextPath'], 'Prediction Error')
            # saveImg = '{}/{}_{}-{}-{}-{}-{}.png'.format(globalVar['figPath'], serviceName, 'final', 'blend', 'model', 'train', datetime.now().strftime("%Y%m%d"))
            # shutil.move(saveOrgImg, saveImg)

            # 분류 시각화
            # plot_model(fnlModel, plot='auc')
            # plot_model(fnlModel, plot='pr')
            # plot_model(fnlModel, plot='feature')
            # plot_model(fnlModel, plot = 'confusion_matrix')

            # 학습 모델 저장
            saveModel = '{0}/{1}-{2}-{3}-{4}-{5}.model'.format(globalVar['outPath'], serviceName, 'final', 'pycaret', 'train', datetime.now().strftime("%Y%m%d"))
            # save_model(fnlModel, saveModel)
            #
            # # 학습 모델 불러오기
            fnlModel = load_model(saveModel)
            #
            # 예측
            predData = predict_model(fnlModel, data=test)

            # 24.4427

            check_metric(predData['pv'], predData['Label'], metric= 'RMSE')


            # from plotnine import *
            # plotData = (
            #         predData >>
            #         dfply.gather('key', 'val', ['cnt', 'Label'])
            # )

            # plot = (ggplot(plotData, aes(x='dtDec', y='val', color='key'))
            #         + geom_point()
            #         + stat_smooth(span=0.3)
            #         + labs(x='Date', y='Value')
            #         )
            # plot.save(saveImg, bbox_inches='tight', width=10, height=6, dpi=600)
            #
            # saveImg = '{}/{}/{}-{}-{}-{}-{}-{}.png'.format(globalVar['figPath'], serviceName, 'final', 'blend', 'model', 'test', 'timeSeries', datetime.now().strftime("%Y%m%d"))
            # plot = (ggplot(predData, aes(x='Label', y='cnt'))
            #         + geom_point()
            #         + stat_smooth(method='lm')
            #         + labs(x='Prd Value', y='Obs Value')
            #         )
            # plot.save(saveImg, bbox_inches='tight', width=10, height=6, dpi=600)

            # for i in trainDataL4.index:
            #     lat = data._get_value(i, 'lat')
            #     lon = data._get_value(i, 'lon')
            #     sp = data._get_value(i, 'sp')
            #     t2m = data._get_value(i, 't2m')

            # solPosInfo = pvlib.solarposition.get_solarposition(trainDataL3['dtDate'], trainDataL3['lat'], lon, pressure=sp, temperature=t2m, method='nrel_numpy')
            # dataL1._set_value(i, 'spaSza', solPosInfo['zenith'].values)

            testDataL2 = testDataL1
            for i in testDataL2.index:
                lat = posLat
                lon = posLon
                pa = testDataL2._get_value(i, 'PA')
                ta = testDataL2._get_value(i, 'TA')
                dtDateTime = testDataL2._get_value(i, 'dtDateKst')

                solPosInfo = pvlib.solarposition.get_solarposition(dtDateTime, lat, lon, pressure=pa, temperature=ta, method='nrel_numpy')
                testDataL2._set_value(i, 'sza', solPosInfo['zenith'].values)
                testDataL2._set_value(i, 'aza', solPosInfo['azimuth'].values)
                testDataL2._set_value(i, 'et', solPosInfo['equation_of_time'].values)

            testDataL3 = predict_model(fnlModel, data=testDataL2)
            testDataL4 = testDataL3[['Label', 'pv', 'dtDateKst', 'SWR']].dropna()
            # check_metric(testDataL4['pv'], testDataL4['Label'], metric='RMSE')

            plt.scatter(testDataL4['Label'], testDataL4['pv'])
            plt.show()

            # testDataL4['dtDateKst'].strftime('%Y%m')

            dtDateList = testDataL4['dtDateKst'].dt.strftime('%Y%m%d').unique()


            for j, dtDateInfo in enumerate(dtDateList):
                print(dtDateInfo)

                testDataL5 = testDataL4.loc[testDataL4['dtDateKst'].dt.strftime('%Y%m%d') == dtDateInfo]

                saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, 'sdsd')

                # plt.grid(True)
                #
                # plt.plot(dtDate, prdVal1, label=prdValLabel_ml, marker='o')
                # plt.plot(dtDate, prdVal2, label=prdValLabel_dnn, marker='o')
                # plt.plot(dtDate, refVal, label=refValLabel, marker='o')
                #
                # # 제목, x축, y축 설정
                # plt.title(mainTitle)
                # plt.xlabel(xlab)
                # plt.ylabel(ylab)
                #
                # plt.xticks(rotation=45, ha='right')
                # plt.legend(loc='upper left')
                #
                # plt.savefig(saveImg, dpi=600, bbox_inches='tight')
                # plt.show()
                # plt.close()


                plt.plot(testDataL5['dtDateKst'], testDataL5['pv'], label='실측')
                plt.plot(testDataL5['dtDateKst'], testDataL5['Label'], label='예측 (머신러닝)')
                plt.plot(testDataL5['dtDateKst'], testDataL5['SWR'], label='예측 (딥러닝)')
                plt.xticks(rotation=45, ha='right')
                plt.legend(loc='upper left')
                plt.savefig(saveImg, dpi=600, bbox_inches='tight')
                plt.show()

                #
                # plt.scatter(testDataL5['dtDateKst'], testDataL5['pv'])
                # plt.scatter(testDataL5['dtDateKst'], testDataL5['Label'])
                # plt.scatter(testDataL5['dtDateKst'], testDataL5['SWR'])
                # # seaborn.set()
                # plt.plot(testDataL5['dtDateKst'], testDataL5['pv'])
                # plt.plot(testDataL5['dtDateKst'], testDataL5['pv'])
                # plt.legend(['input', 'resample', 'asfreq'],
                #            loc='upper left');
                # plt.show()

            # UM
            inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/MODEL/UMKR_l015_unis_*.nc')
            fileList = sorted(glob.glob(inpFile))
            umData = xr.open_mfdataset(fileList)
            umDataL1 = umData.interpolate_na(method='linear', fill_value="extrapolate")
            umDataL2 = umDataL1.sel(lat=posLat, lon=posLon, anaTime=umData['anaTime'].values[0])
            umDataL3 = umDataL2.to_dataframe()
            umDataL3['dtDate'] = umDataL3.index
            umDataL3['dtDateKst'] = umDataL3.index.tz_localize(tzUtc).tz_convert(tzKst)

            umDataL3.columns

            umDataL4 = umDataL3.rename( {'SS': 'SWR'}, axis='columns'
            )[['dtDate', 'dtDateKst', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR']]

            umDataL5 = umDataL4
            for i in umDataL5.index:
                lat = posLat
                lon = posLon
                pa = umDataL5._get_value(i, 'PA')
                ta = umDataL5._get_value(i, 'TA')
                dtDateTime = umDataL5._get_value(i, 'dtDateKst')

                solPosInfo = pvlib.solarposition.get_solarposition(dtDateTime, lat, lon, pressure=pa, temperature=ta,
                                                                   method='nrel_numpy')
                umDataL5._set_value(i, 'sza', solPosInfo['zenith'].values)
                umDataL5._set_value(i, 'aza', solPosInfo['azimuth'].values)
                umDataL5._set_value(i, 'et', solPosInfo['equation_of_time'].values)


            predData = predict_model(fnlModel, data=umDataL5)

            # 24.4427
            from pycaret.utils import check_metric
            check_metric(predData['pv'], predData['Label'], metric='RMSE')



            # testL1 = test[['CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'PM10', 'SWR', 'sza', 'aza', 'et', 'pv']]
            # predData = predict_model(fnlModel, data=test)

            # plt.scatter(umDataL2['time'], umDataL2['CA_TOT'])
            # plt.scatter(umDataL2['time'], umDataL2['HM'])
            # plt.scatter(umDataL2['time'], umDataL2['PA'])
            # plt.scatter(umDataL2['time'], umDataL2['SS'])
            # plt.scatter(umDataL2['time'], umDataL2['TA'])
            # plt.scatter(umDataL2['time'], umDataL2['TD'])
            # plt.scatter(umDataL2['time'], umDataL2['WD'])
            # plt.scatter(umDataL2['time'], umDataL2['WS'])
            # plt.show()
            #
            # dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            # dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            # dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))
            #
            # posLon = posInfo['lon']
            # posLat = posInfo['lat']
            # lon1D = np.array(posLon).reshape(1)
            # lat1D = np.array(posLat).reshape(1)
            #
            # dsDataL2 = xr.Dataset()
            # # dtIncDateInfo = dtIncDateList[0]
            # for i, dtIncDateInfo in enumerate(dtIncDateList):
            #     log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))
            #
            #     # UMKR_l015_unis_H001_202110010000.grb2
            #     saveFile = '{}/TEST/MODEL/UMKR_l015_unis_{}_{}.nc'.format(globalVar['outPath'],
            #                                                               pd.to_datetime(dtSrtDate).strftime('%Y%m%d'),
            #                                                               pd.to_datetime(dtEndDate).strftime('%Y%m%d'))
            #     if (os.path.exists(saveFile)):
            #         continue
            #
            #     dtDateYm = dtIncDateInfo.strftime('%Y%m')
            #     dtDateDay = dtIncDateInfo.strftime('%d')
            #     dtDateHour = dtIncDateInfo.strftime('%H')
            #     dtDateYmd = dtIncDateInfo.strftime('%Y%m%d')
            #     dtDateHm = dtIncDateInfo.strftime('%H%M')
            #     dtDateYmdHm = dtIncDateInfo.strftime('%Y%m%d%H%M')
            #
            #     # /SYSTEMS/OUTPUT/OBS/202109/01/AWS_OBS_202109010000.txt
            #     # inpAsosFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ASOS_OBS_*.txt')
            #     # UMKR_l015_unis_H001_202110010000.grb2
            #     inpFilePattern = 'MODEL/{}/{}/{}/UMKR_l015_unis_*_{}.grb2'.format(dtDateYm, dtDateDay, dtDateHour,
            #                                                                       dtDateYmdHm)
            #     inpFile = '{}/{}'.format(globalVar['inpPath'], inpFilePattern)
            #     fileList = sorted(glob.glob(inpFile))
            #
            #     if (len(fileList) < 1):
            #         continue
            #         # raise Exception("[ERROR] fileInfo : {} : {}".format("입력 자료를 확인해주세요.", inpFile))
            #
            #     grbInfo = pygrib.open(fileList[0]).select(name='Temperature')[1]
            #     lat2D, lon2D = grbInfo.latlons()
            #
            #     # 최근접 좌표
            #     posList = []
            #
            #     # kdTree를 위한 초기 데이터
            #     for i in range(0, lon2D.shape[0]):
            #         for j in range(0, lon2D.shape[1]):
            #             coord = [lat2D[i, j], lon2D[i, j]]
            #             posList.append(cartesian(*coord))
            #
            #     tree = spatial.KDTree(posList)
            #
            #     coord = cartesian(posInfo['lat'], posInfo['lon'])
            #     closest = tree.query([coord], k=1)
            #     cloIdx = closest[1][0]
            #     row = int(cloIdx / lon2D.shape[1])
            #     col = cloIdx % lon2D.shape[1]
            #
            #     dtAnalDate = grbInfo.analDate
            #
            #     # fileInfo = fileList[2]
            #     for j, fileInfo in enumerate(fileList):
            #         log.info("[CHECK] fileInfo : {}".format(fileInfo))
            #
            #         try:
            #             grb = pygrib.open(fileInfo)
            #             grbInfo = grb.select(name='Temperature')[1]
            #
            #             dtValidDate = grbInfo.validDate
            #
            #             uVec = grb.select(name='10 metre U wind component')[0].values[row, col]
            #             vVec = grb.select(name='10 metre V wind component')[0].values[row, col]
            #             WD = (270 - np.rad2deg(np.arctan2(vVec, uVec))) % 360
            #             WS = np.sqrt(np.square(uVec) + np.square(vVec))
            #             PA = grb.select(name='Surface pressure')[0].values[row, col]
            #             TA = grbInfo.values[row, col]
            #             TD = grb.select(name='Dew point temperature')[0].values[row, col]
            #             HM = grb.select(name='Relative humidity')[0].values[row, col]
            #             lowCA = grb.select(name='Low cloud cover')[0].values[row, col]
            #             medCA = grb.select(name='Medium cloud cover')[0].values[row, col]
            #             higCA = grb.select(name='High cloud cover')[0].values[row, col]
            #             CA_TOT = np.mean([lowCA, medCA, higCA])
            #             SS = grb.select(name='unknown')[0].values[row, col]
            #
            #             dsDataL1 = xr.Dataset(
            #                 {
            #                     'uVec': (
            #                     ('anaTime', 'time', 'lat', 'lon'), (uVec).reshape(1, 1, len(lat1D), len(lon1D)))
            #                     , 'vVec': (
            #                 ('anaTime', 'time', 'lat', 'lon'), (vVec).reshape(1, 1, len(lat1D), len(lon1D)))
            #                     , 'WD': (('anaTime', 'time', 'lat', 'lon'), (WD).reshape(1, 1, len(lat1D), len(lon1D)))
            #                     , 'WS': (('anaTime', 'time', 'lat', 'lon'), (WS).reshape(1, 1, len(lat1D), len(lon1D)))
            #                     , 'PA': (('anaTime', 'time', 'lat', 'lon'), (PA).reshape(1, 1, len(lat1D), len(lon1D)))
            #                     , 'TA': (('anaTime', 'time', 'lat', 'lon'), (TA).reshape(1, 1, len(lat1D), len(lon1D)))
            #                     , 'TD': (('anaTime', 'time', 'lat', 'lon'), (TD).reshape(1, 1, len(lat1D), len(lon1D)))
            #                     , 'HM': (('anaTime', 'time', 'lat', 'lon'), (HM).reshape(1, 1, len(lat1D), len(lon1D)))
            #                     , 'lowCA': (
            #                 ('anaTime', 'time', 'lat', 'lon'), (lowCA).reshape(1, 1, len(lat1D), len(lon1D)))
            #                     , 'medCA': (
            #                 ('anaTime', 'time', 'lat', 'lon'), (medCA).reshape(1, 1, len(lat1D), len(lon1D)))
            #                     , 'higCA': (
            #                 ('anaTime', 'time', 'lat', 'lon'), (higCA).reshape(1, 1, len(lat1D), len(lon1D)))
            #                     , 'CA_TOT': (
            #                 ('anaTime', 'time', 'lat', 'lon'), (CA_TOT).reshape(1, 1, len(lat1D), len(lon1D)))
            #                     , 'SS': (('anaTime', 'time', 'lat', 'lon'), (SS).reshape(1, 1, len(lat1D), len(lon1D)))
            #                 }
            #                 , coords={
            #                     'anaTime': pd.date_range(dtAnalDate, periods=1)
            #                     , 'time': pd.date_range(dtValidDate, periods=1)
            #                     , 'lat': lat1D
            #                     , 'lon': lon1D
            #                 }
            #             )
            #
            #             dsDataL2 = dsDataL2.merge(dsDataL1)
            #
            #         except Exception as e:
            #             log.error("Exception : {}".format(e))
            #
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #
            # dsDataL2.to_netcdf(saveFile)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))
            #
            # ds = xr.open_dataset(saveFile)
            #
            # tt = ds['anaTime'].values
            # ds2 = ds.sel(anaTime=tt[0])
            #
            # ds2['time'].values
            #
            # dtSrtDate = pd.to_datetime('2021-10-01', format='%Y-%m-%d')
            # dtEndDate = pd.to_datetime('2021-10-03', format='%Y-%m-%d')
            # dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Minute(1))
            #
            # # ds3 = ds2.interpolate_na(dim="time", method="linear", fill_value="extrapolate")
            # ds3 = ds2.interpolate_na(dim="time", method="linear", fill_value="extrapolate")
            # selIntpVal = ds2.interp(time=dtIncDateList)
            # selIntpVal = ds3.interp(time=dtIncDateList)
            #
            # # plt.scatter(ds2['time'].values, ds2['TA'].values)
            # plt.scatter(ds3['time'].values, ds3['TA'].values)
            # plt.scatter(selIntpVal['time'].values, selIntpVal['TA'].values)
            # plt.show()






            # dsDataL2 = xr.open_dataset(saveFile)
            #
            # dtSrtPrdDate = pd.to_datetime('2020-09-01', format='%Y-%m-%d')
            # dtEndPrdDate = pd.to_datetime('2020-09-02', format='%Y-%m-%d')
            # dtIncPrdDateList = pd.date_range(start=dtSrtPrdDate, end=dtEndPrdDate, freq=Minute(1))
            #
            # dsDataL3 = dsDataL2.sel(lon = posLon, lat = posLat)
            #
            # dsDataL4 = dsDataL3.interp(time=dtIncPrdDateList)
            # dsDataL5 = dsDataL4.interpolate_na(dim="time", method="linear", fill_value="extrapolate")
            #
            # plt.plot(dsDataL4['PM10'][:, 0, 0])
            # plt.show()
            #
            # dsDataL3['PM10'].plot()
            # plt.show()
            #
            # bb = dsDataL3['time']
            # bb1 = dsDataL3['PM10']
            # plt.scatter(dsDataL4['time'], dsDataL4['PM10'])
            # plt.plot(dsDataL3['PM10'].values)
            #
            # plt.plot(bb1)
            # # plt.scatter(bb1)
            # # plt.scatter(selIntpVal['time'].values, selIntpVal['TA'].values)
            # plt.show()
            #
            # dd = dsDataL2['PM10']
            #
            # plt.plot(dd['time'], dd[ :, 0,0])
            # plt.scatter(dsDataL3['time'].values, dsDataL3['PM10'][ :, 0,0].values, c = dsDataL3['PM10'][ :, 0,0].values)
            # # plt.contourf(dsDataL2['time'], dsDataL2['lat'], dsDataL2['PM10'][:, :, :])
            # # plt.contourf(ds['lon'], ds['lat'], ds['PM10'][0, :, :], levels=100)
            # # plt.colorbar()
            # plt.show()

            # import sqlite3
            # import pandas.io.sql as psql
            #
            # con = sqlite3.connect(':memory:')
            # cur = con.cursor()
            #
            # trainData.to_sql(name='trainData', con=con)
            # df = psql.read_sql("SELECT * FROM trainData;", con)
            #
            # df.to_sql(name='df', con=con)
            # psql.read_sql("SELECT * FROM df;", con)
            # df2 = pd.DataFrame([['sample3', 'CCC', '2017-07-16 00:00:00']], columns=['title', 'body', 'created'],
            #                    index=[2])
            #
            # cur.execute(
            #     """
            #     SELECT * FROM trainData
            #     """
            # )
            # df = pd.DataFrame(data=[[0, '10/11/12'], [1, '12/11/10']],
            #                   columns=['int_column', 'date_column'])
            # df.to_sql('test_data', conn)

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
