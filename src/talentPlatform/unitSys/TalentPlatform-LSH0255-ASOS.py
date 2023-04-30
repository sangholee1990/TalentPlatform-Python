# -*- coding: utf-8 -*-
import glob
# import seaborn as sns
import logging
import logging.handlers
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

# os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share"
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

# --------------------------------------------------------------------------------------------------
#  기상청 지상관측 시간자료
# --------------------------------------------------------------------------------------------------
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

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
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

            if (platform.system() == 'Windows'):

                globalVar['inpPath'] = 'E:/DATA/OUTPUT'
                globalVar['outPath'] = 'E:/DATA/OUTPUT'

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2020-09-01'
                    , 'endDate': '2020-09-03'
                }
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
            lon2D, lat2D = np.meshgrid(lon1D, lat1D)

            inpAsosStnFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/ASOS_STN_INFO.csv')
            asosStnData = pd.read_csv(inpAsosStnFile)
            asosStnDataL1 = asosStnData[['STN', 'LON', 'LAT']]

            # *******************************************************
            # 지상 관측소
            # *******************************************************
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))

            # dtIncDateInfo = dtIncDateList[0]
            # dtIncDateInfo = dtIncDateList[1]
            dsDataL1 = xr.Dataset()
            for i, dtIncDateInfo in enumerate(dtIncDateList):

                # log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))
                # saveFile = '{}/TEST/OBS-L2/ASOS_OBS_{}_{}_2.nc'.format(globalVar['outPath'], pd.to_datetime(dtSrtDate).strftime('%Y%m%d'), pd.to_datetime(dtEndDate).strftime('%Y%m%d'))
                saveFile = '{}/TEST/OBS/ASOS_OBS_{}_{}.nc'.format(globalVar['outPath'], pd.to_datetime(dtSrtDate).strftime('%Y%m%d'), pd.to_datetime(dtEndDate).strftime('%Y%m%d'))

                if (os.path.exists(saveFile)):
                    continue

                dtDateYm = dtIncDateInfo.strftime('%Y%m')
                dtDateDay = dtIncDateInfo.strftime('%d')
                dtDateHour = dtIncDateInfo.strftime('%H')
                dtDateYmd = dtIncDateInfo.strftime('%Y%m%d')
                dtDateHm = dtIncDateInfo.strftime('%H%M')
                dtDateYmdH = dtIncDateInfo.strftime('%Y%m%d%H')
                dtDateYmdHm = dtIncDateInfo.strftime('%Y%m%d%H%M')


                # /SYSTEMS/OUTPUT/OBS/202109/01/AWS_OBS_202109010000.txt
                inpAsosFilePattern = 'OBS/{}/{}/ASOS_OBS_{}.txt'.format(dtDateYm, dtDateDay, dtDateYmdHm)
                inpAsosFile = '{}/{}'.format(globalVar['inpPath'], inpAsosFilePattern)
                fileList = sorted(glob.glob(inpAsosFile))

                if (len(fileList) < 1):
                    continue

                dataL1 = pd.DataFrame()
                for fileInfo in fileList:
                    data = pd.read_csv(fileInfo, header=None, delimiter='\s+')
                    dataL1 = dataL1.append(data)

                # dataL1 = dataL1.reset_index(drop=True)
                dataL1.columns = ['TM', 'STN', 'WD', 'WS', 'GST_WD', 'GST_WS', 'GST_TM', 'PA', 'PS', 'PT', 'PR', 'TA', 'TD', 'HM', 'PV', 'RN', 'RN_DAY', 'TMP', 'RN_INT', 'SD_HR3', 'SD_DAY', 'SD_TOT', 'WC', 'WP', 'WW', 'CA_TOT', 'CA_MID', 'CH_MIN', 'CT', 'CT_TOP', 'CT_MID', 'CT_LOW', 'VS', 'SS', 'SI', 'ST_GD', 'TS', 'TE_005', 'TE_01', 'TE_02', 'TE_03', 'ST_SEA', 'WH', 'BF', 'IR', 'IX']

                # dataL2 = dataL1[['TM', 'STN', 'WD', 'WS', 'PA', 'TA', 'TD', 'HM', 'CA_TOT', 'SS', 'SI']]
                dataL2 = dataL1[['TM', 'STN', 'WD', 'WS', 'PA', 'TA', 'TD', 'HM', 'CA_TOT']]

                # TM 및 STN을 기준으로 중복 제거
                dataL2['TM'] = dataL2['TM'].astype(str)
                dataL2.drop_duplicates(subset=['TM', 'STN'], inplace=True)

                # 결측값 제거
                dataL3 = dataL2
                dataL3['dtDate'] = pd.to_datetime(dataL3['TM'], format='%Y%m%d%H%M')

                colList = dataL3.columns
                dtDateList = dataL3['dtDate'].unique()

                # dtDateInfo = dtDateList[0]
                dsDataL1 = xr.Dataset()
                for i, dtDateInfo in enumerate(dtDateList):

                    log.info('[CHECK] dtDateInfo : {}'.format(dtDateInfo))

                    dataL4 = dataL3.loc[
                        dataL3['dtDate'] == dtDateInfo
                    ]

                    dataL4['WD'][dataL4['WD'] < 0] = np.nan
                    dataL4['WS'][dataL4['WS'] < 0] = np.nan
                    dataL4['PA'][dataL4['PA'] < 0] = np.nan
                    dataL4['HM'][dataL4['HM'] < 0] = np.nan
                    dataL4['CA_TOT'][dataL4['CA_TOT'] < 0] = np.nan

                    dataL5 = pd.merge(left=dataL4, right=asosStnDataL1, how='left', left_on='STN', right_on='STN')

                    # log.info('[CHECK] colList : {}'.format(colList))

                    # colInfo = 'WD'
                    varList = {}
                    for colInfo in colList:
                        if (colInfo == 'TM') or (colInfo == 'STN') or (colInfo == 'dtDate'): continue


                        dataL6 = dataL5[['dtDate', 'LON', 'LAT', colInfo]].dropna()

                        if (len(dataL6) < 1):
                            continue

                        posLon = dataL6['LON'].values
                        posLat = dataL6['LAT'].values
                        posVar = dataL6[colInfo].values

                        # log.info('[CHECK] posVar : {}'.format(posVar))

                        # Radial basis function (RBF) interpolation in N dimensions.
                        try:
                            rbfModel = Rbf(posLon, posLat, posVar, function="linear")
                            rbfRes = rbfModel(lon2D, lat2D)
                            varList[colInfo] = rbfRes
                        except Exception as e:
                            log.error("Exception : {}".format(e))

                    try:
                        dsData = xr.Dataset(
                            {
                                'WD': (('time', 'lat', 'lon'), (varList['WS']).reshape(1, len(lat1D), len(lon1D)))
                                , 'WS': (('time', 'lat', 'lon'), (varList['WS']).reshape(1, len(lat1D), len(lon1D)))
                                , 'PA': (('time', 'lat', 'lon'), (varList['PA']).reshape(1, len(lat1D), len(lon1D)))
                                , 'TA': (('time', 'lat', 'lon'), (varList['TA']).reshape(1, len(lat1D), len(lon1D)))
                                , 'TD': (('time', 'lat', 'lon'), (varList['TD']).reshape(1, len(lat1D), len(lon1D)))
                                , 'HM': (('time', 'lat', 'lon'), (varList['HM']).reshape(1, len(lat1D), len(lon1D)))
                                , 'CA_TOT': ( ('time', 'lat', 'lon'), (varList['CA_TOT']).reshape(1, len(lat1D), len(lon1D)))
                                # , 'SS': ( ('time', 'lat', 'lon'), (varList['SS']).reshape(1, len(lat1D), len(lon1D)) )
                                # , 'SI': ( ('time', 'lat', 'lon'), (varList['SI']).reshape(1, len(lat1D), len(lon1D)) )
                            }
                            , coords={
                                'time': pd.date_range(dtDateInfo, periods=1)
                                , 'lat': lat1D
                                , 'lon': lon1D
                            }
                        )

                        dsDataL1 = dsDataL1.merge(dsData)
                    except Exception as e:
                        log.error("Exception : {}".format(e))

            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            dsDataL1.to_netcdf(saveFile)
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
