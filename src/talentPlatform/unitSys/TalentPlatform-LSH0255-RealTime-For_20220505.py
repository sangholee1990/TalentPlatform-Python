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
# import datetime as dt
# from datetime import datetime
import pvlib

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
# import eccodes
import pygrib
# import pykrige.kriging_tools as kt
import haversine as hs
import pytz
import datetime
import h2o
# from pycaret.regression import *
from sqlalchemy import create_engine
import pymysql
import re
import configparser

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
                globalVar['modelPath'] = 'E:/DATA'

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2020-09-01'
                    , 'endDate': '2021-11-01'

                    # 모델 버전 (날짜)
                    , 'modelVer': '*'
                    # , 'modelVer': '20220220'
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': globalVar['srtDate']
                    , 'endDate': globalVar['endDate']

                    # 모델 버전 (날짜)
                    , 'modelVer': '*'
                    # , 'modelVer': '20220220'
                }

            # modelDirKeyList = ['AI_2Y']
            # figActDirKeyList = ['ACT_2Y']
            # figForDirKeyList = ['FOR_2Y']
            #
            # for k, modelDirKey in enumerate(modelDirKeyList):
            #     figActDirKey = figActDirKeyList[k]
            #     figForDirKey = figForDirKeyList[k]

            modelDirKey = 'AI_2Y'
            figActDirKey = 'ACT_2Y'
            figForDirKey = 'FOR_2Y'
            modelVer = sysOpt['modelVer']

            isDlModelInit = False

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

            import sqlalchemy
            from sqlalchemy.ext.declarative import declarative_base

            # dbCon = create_engine('mysql://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName))
            dbCon = create_engine('mariadb://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName))

            # 관측소 정보
            # inpPosFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/GA_STN_INFO.xlsx')
            # posData = pd.read_excel(inpPosFile)
            # posDataL1 = posData[['id', 'lat', 'lon']]

            res = dbCon.execute(
                """
                SELECT *
                FROM TB_STN_INFO
                """
            ).fetchall()

            posDataL1 = pd.DataFrame(res).rename(
                            {
                                'ID': 'id'
                                , 'dtDateKst': 'DATE_TIME_KST'
                                , 'LAT': 'lat'
                                , 'LON': 'lon'
                            }
                            , axis='columns'
                        )

            lat1D = np.array(posDataL1['lat'])
            lon1D = np.array(posDataL1['lon'])

            # *******************************************************
            # UM 자료 읽기
            # *******************************************************
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(1))

            # posLon = posInfo['lon']
            # posLat = posInfo['lat']
            # lon1D = np.array(posLon).reshape(1)
            # lat1D = np.array(posLat).reshape(1)

            cfgFile = '{}/{}'.format(globalVar['cfgPath'], 'modelInfo/UMKR_l015_unis_H000_202110010000.grb2')
            # log.info("[CHECK] cfgFile : {}".format(cfgFile))

            cfgInfo = pygrib.open(cfgFile).select(name='Temperature')[1]
            lat2D, lon2D = cfgInfo.latlons()

            # =======================================================================
            # 최근접 좌표
            # =======================================================================
            posList = []

            # kdTree를 위한 초기 데이터
            for i in range(0, lon2D.shape[0]):
                for j in range(0, lon2D.shape[1]):
                    coord = [lat2D[i, j], lon2D[i, j]]
                    posList.append(cartesian(*coord))

            tree = spatial.KDTree(posList)

            # coord = cartesian(posInfo['lat'], posInfo['lon'])
            row1D = []
            col1D = []
            for ii, posInfo in posDataL1.iterrows():
                coord = cartesian(posInfo['lat'], posInfo['lon'])
                closest = tree.query([coord], k=1)
                cloIdx = closest[1][0]
                row = int(cloIdx / lon2D.shape[1])
                col = cloIdx % lon2D.shape[1]

                row1D.append(row)
                col1D.append(col)

            row2D, col2D = np.meshgrid(row1D, col1D)


            # dtIncDateInfo = dtIncDateList[0]
            dsDataL2 = xr.Dataset()
            for ii, dtIncDateInfo in enumerate(dtIncDateList):
                log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))

                # UMKR_l015_unis_H001_202110010000.grb2
                # saveFile = '{}/TEST/MODEL/UMKR_l015_unis_{}_{}.nc'.format(globalVar['outPath'], pd.to_datetime(dtSrtDate).strftime('%Y%m%d'), pd.to_datetime(dtEndDate).strftime('%Y%m%d'))

                # if (os.path.exists(saveFile)):
                #     continue

                dtDateYm = dtIncDateInfo.strftime('%Y%m')
                dtDateDay = dtIncDateInfo.strftime('%d')
                dtDateHour = dtIncDateInfo.strftime('%H')
                dtDateYmd = dtIncDateInfo.strftime('%Y%m%d')
                dtDateHm = dtIncDateInfo.strftime('%H%M')
                dtDateYmdHm = dtIncDateInfo.strftime('%Y%m%d%H%M')

                # UMKR_l015_unis_H001_202110010000.grb2
                inpFilePattern = 'MODEL/{}/{}/{}/UMKR_l015_unis_*_{}.grb2'.format(dtDateYm, dtDateDay, dtDateHour,
                                                                                  dtDateYmdHm)
                inpFile = '{}/{}'.format(globalVar['inpPath'], inpFilePattern)
                fileList = sorted(glob.glob(inpFile))

                if (len(fileList) < 1): continue
                    # raise Exception("[ERROR] fileInfo : {} : {}".format("입력 자료를 확인해주세요.", inpFile))

                # fileInfo = fileList[2]
                for jj, fileInfo in enumerate(fileList):
                    log.info("[CHECK] fileInfo : {}".format(fileInfo))

                    try:
                        grb = pygrib.open(fileInfo)
                        grbInfo = grb.select(name='Temperature')[1]

                        validIdx = int(re.findall('H\d{3}', fileInfo)[0].replace('H', ''))
                        dtValidDate = grbInfo.validDate
                        dtAnalDate = grbInfo.analDate

                        uVec = grb.select(name='10 metre U wind component')[0].values[row2D, col2D]
                        vVec = grb.select(name='10 metre V wind component')[0].values[row2D, col2D]
                        WD = (270 - np.rad2deg(np.arctan2(vVec, uVec))) % 360
                        WS = np.sqrt(np.square(uVec) + np.square(vVec))
                        PA = grb.select(name='Surface pressure')[0].values[row2D, col2D]
                        TA = grbInfo.values[row2D, col2D]
                        TD = grb.select(name='Dew point temperature')[0].values[row2D, col2D]
                        HM = grb.select(name='Relative humidity')[0].values[row2D, col2D]
                        lowCA = grb.select(name='Low cloud cover')[0].values[row2D, col2D]
                        medCA = grb.select(name='Medium cloud cover')[0].values[row2D, col2D]
                        higCA = grb.select(name='High cloud cover')[0].values[row2D, col2D]
                        CA_TOT = np.mean([lowCA, medCA, higCA], axis=0)
                        SS = grb.select(name='unknown')[0].values[row2D, col2D]

                        dsDataL1 = xr.Dataset(
                            {
                                'uVec': (('anaTime', 'time', 'lat', 'lon'), (uVec).reshape(1, 1, len(lat1D), len(lon1D)))
                                , 'vVec': (('anaTime', 'time', 'lat', 'lon'), (vVec).reshape(1, 1, len(lat1D), len(lon1D)))
                                , 'WD': (('anaTime', 'time', 'lat', 'lon'), (WD).reshape(1, 1, len(lat1D), len(lon1D)))
                                , 'WS': (('anaTime', 'time', 'lat', 'lon'), (WS).reshape(1, 1, len(lat1D), len(lon1D)))
                                , 'PA': (('anaTime', 'time', 'lat', 'lon'), (PA).reshape(1, 1, len(lat1D), len(lon1D)))
                                , 'TA': (('anaTime', 'time', 'lat', 'lon'), (TA).reshape(1, 1, len(lat1D), len(lon1D)))
                                , 'TD': (('anaTime', 'time', 'lat', 'lon'), (TD).reshape(1, 1, len(lat1D), len(lon1D)))
                                , 'HM': (('anaTime', 'time', 'lat', 'lon'), (HM).reshape(1, 1, len(lat1D), len(lon1D)))
                                , 'lowCA': (('anaTime', 'time', 'lat', 'lon'), (lowCA).reshape(1, 1, len(lat1D), len(lon1D)))
                                , 'medCA': (('anaTime', 'time', 'lat', 'lon'), (medCA).reshape(1, 1, len(lat1D), len(lon1D)))
                                , 'higCA': (('anaTime', 'time', 'lat', 'lon'), (higCA).reshape(1, 1, len(lat1D), len(lon1D)))
                                , 'CA_TOT': (('anaTime', 'time', 'lat', 'lon'), (CA_TOT).reshape(1, 1, len(lat1D), len(lon1D)))
                                , 'SS': (('anaTime', 'time', 'lat', 'lon'), (SS).reshape(1, 1, len(lat1D), len(lon1D)))
                            }
                            , coords={
                                'anaTime': pd.date_range(dtAnalDate, periods=1)
                                , 'time': pd.date_range(dtValidDate, periods=1)
                                , 'lat': lat1D
                                , 'lon': lon1D
                            }
                        )

                    except Exception as e:
                        log.error("Exception : {}".format(e))


                    for kk, posInfo in posDataL1.iterrows():
                        posId = int(posInfo['id'])
                        posLat = posInfo['lat']
                        posLon = posInfo['lon']

                        log.info("[CHECK] posId (posLon, posLat) : {} ({}. {})".format(posId, posLon, posLat))

                        # umData = dsDataL2
                        umData = dsDataL1
                        dtAnaTimeList = umData['anaTime'].values
                        # umDataL8 = pd.DataFrame()
                        for ll, dtAnaTimeInfo in enumerate(dtAnaTimeList):
                            log.info("[CHECK] dtAnaTimeInfo : {}".format(dtAnaTimeInfo))

                            try:
                                umDataL2 = umData.sel(lat=posLat, lon=posLon, anaTime=dtAnaTimeInfo)
                                umDataL3 = umDataL2.to_dataframe().dropna().reset_index(drop=True)
                                # umDataL3['dtDate'] = pd.to_datetime(dtAnaTimeInfo) + (umDataL3.index.values * datetime.timedelta(hours=1))
                                umDataL3['dtDate'] = pd.to_datetime(dtAnaTimeInfo) + (validIdx * datetime.timedelta(hours=1))
                                # umDataL3['dtDateKst'] = umDataL3.index.tz_localize(tzUtc).tz_convert(tzKst)
                                umDataL3['dtDateKst'] = umDataL3['dtDate'] + dtKst
                                umDataL4 = umDataL3.rename({'SS': 'SWR'}, axis='columns')
                                umDataL5 = umDataL4[['dtDateKst', 'dtDate', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR']]
                                umDataL5['SRV'] = 'SRV{:05d}'.format(posId)
                                umDataL5['TA'] = umDataL5['TA'] - 273.15
                                umDataL5['TD'] = umDataL5['TD'] - 273.15
                                umDataL5['PA'] = umDataL5['PA'] / 100.0
                                umDataL5['CA_TOT'] = np.where(umDataL5['CA_TOT'] < 0, 0, umDataL5['CA_TOT'])
                                umDataL5['CA_TOT'] = np.where(umDataL5['CA_TOT'] > 1, 1, umDataL5['CA_TOT'])

                                umDataL6 = umDataL5
                                for i in umDataL6.index:
                                    lat = posLat
                                    lon = posLon
                                    pa = umDataL6._get_value(i, 'PA') * 100.0
                                    ta = umDataL6._get_value(i, 'TA')
                                    # dtDateTime = umDataL6._get_value(i, 'dtDateKst')
                                    dtDateTime = umDataL6._get_value(i, 'dtDate')

                                    solPosInfo = pvlib.solarposition.get_solarposition(dtDateTime, lat, lon, pressure=pa, temperature=ta, method='nrel_numpy')
                                    umDataL6._set_value(i, 'sza', solPosInfo['zenith'].values)
                                    umDataL6._set_value(i, 'aza', solPosInfo['azimuth'].values)
                                    umDataL6._set_value(i, 'et', solPosInfo['equation_of_time'].values)

                                # umDataL7 = umDataL6.merge(pvDataL2, how='left', left_on=['dtDateKst'], right_on=['dtDateKst'])
                                umDataL7 = umDataL6
                                umDataL7['anaTime'] = pd.to_datetime(dtAnaTimeInfo)

                                # umDataL8 = umDataL8.append(umDataL7)

                            except Exception as e:
                                log.error("Exception : {}".format(e))

                        # log.info("[CHECK] modelDirKey : {}".format(modelDirKey))
                        # log.info("[CHECK] figActDirKey : {}".format(figActDirKey))

                        # *******************************************************
                        # 관측자료 읽기
                        # *******************************************************
                        # inpData = pd.read_excel(fileInfo, engine='openpyxl')
                        # inpData = umDataL7
                        inpData = umDataL7
                        inpDataL1 = inpData.rename({'dtDate_x': 'dtDate'}, axis='columns')
                        # log.info("[CHECK] inpDataL1 : {}".format(inpDataL1))

                        # log.info("[CHECK] inpDataL1['SRV'] : {}".format(inpDataL1['SRV'][0]))
                        # log.info("[CHECK] inpDataL1['anaTime'] : {}".format(inpDataL1['anaTime'][0]))
                        # log.info("[CHECK] inpDataL1['dtDate'] : {}".format(inpDataL1['dtDate'][0]))

                        iAnaYear = int(inpDataL1['anaTime'][0].strftime("%Y"))

                        # 테이블 없을 시 생성
                        dbCon.execute(
                            """
                            create table IF NOT EXISTS TB_FOR_DATA_%s
                            (
                                SRV           varchar(10) not null comment '관측소 정보',
                                ANA_DATE      date        not null comment '예보일',
                                DATE_TIME     datetime    not null comment '예보날짜 UTC',
                                DATE_TIME_KST datetime    null comment '예보날짜 KST',
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
                                SWR           float       null comment '일사량',
                                ML            float       null comment '머신러닝',
                                DL            float       null comment '딥러닝',
                                REG_DATE      datetime    null comment '등록일',
                                MOD_DATE      datetime    null comment '수정일',
                                primary key (SRV, DATE_TIME, ANA_DATE)
                            )    
                                comment '기상 예보 테이블_%s';
                            """
                            , (iAnaYear, iAnaYear)
                        )

                        keyChk = dbCon.execute(
                            """
                            SELECT COUNT(*) AS CNT
                            FROM TB_FOR_DATA_%s
                            WHERE  SRV = %s AND ANA_DATE = %s AND DATE_TIME = %s
                            """
                            , (iAnaYear, inpDataL1['SRV'][0], inpDataL1['anaTime'][0], inpDataL1['dtDate'][0])
                        ).fetchone()

                        # log.info("[CHECK] keyChk['CNT'] : {}".format(keyChk['CNT']))

                        if (keyChk['CNT'] > 0): continue

                        # **********************************************************************************************************
                        # 머신러닝
                        # **********************************************************************************************************
                        # saveMlModel = '{}/{}/{}-SRV{:05d}-{}-{}-{}-{}.model.pkl'.format(globalVar['modelPath'], modelDirKey, serviceName, posId, 'final', 'pycaret', 'for', '*')
                        # saveMlModel = '{}/{}/{}-SRV{:05d}-{}-{}-{}-{}.model.pkl'.format(globalVar['modelPath'], modelDirKey, serviceName, posId, 'final', 'pycaret', 'for', '20220220')
                        # saveMlModelList = sorted(glob.glob(saveMlModel), reverse=True)
                        #
                        # # from pycaret.regression import *
                        #
                        # if (len(saveMlModelList) > 0):
                        #     saveMlModelInfo = saveMlModelList[0]
                        #     log.info("[CHECK] saveMlModelInfo : {}".format(saveMlModelInfo))
                        #
                        #     mlModel = load_model(os.path.splitext(saveMlModelInfo)[0])
                        #
                        # mlModelPred = predict_model(mlModel, data=inpDataL1).rename({'Label': 'ML'}, axis='columns')[['dtDateKst', 'anaTime', 'ML']]

                        # **********************************************************************************************************
                        # 딥러닝
                        # **********************************************************************************************************
                        # saveDlModel = '{}/{}/{}-SRV{:05d}-{}-{}-{}-{}.model'.format(globalVar['modelPath'], modelDirKey, serviceName, posId, 'final', 'h2o', 'for', '*')
                        saveDlModel = '{}/{}/{}-SRV{:05d}-{}-{}-{}-{}.model'.format(globalVar['modelPath'], modelDirKey, serviceName, posId, 'final', 'h2o', 'for', modelVer)
                        saveDlModelList = sorted(glob.glob(saveDlModel), reverse=True)

                        if (isDlModelInit == False):
                            h2o.init()
                            isDlModelInit = True

                        # 학습 모델 불러오기
                        if (len(saveDlModelList) > 0):
                            saveDlModelInfo = saveDlModelList[0]
                            log.info("[CHECK] saveDlModelInfo : {}".format(saveDlModelInfo))

                            dlModel = h2o.load_model(path=saveDlModelInfo)

                        tmpData = inpDataL1[['dtDateKst', 'anaTime', 'CA_TOT', 'HM', 'PA', 'TA', 'TD', 'WD', 'WS', 'SWR', 'sza', 'aza',
                                             'et']].dropna().reset_index(drop=True)
                        dlModelPred = dlModel.predict(h2o.H2OFrame(tmpData)).as_data_frame().rename({'predict': 'DL'}, axis='columns')
                        dlModelPredL1 = pd.concat([tmpData[['dtDateKst', 'anaTime']], dlModelPred], axis=1)

                        # 머신러닝 또는 딥러닝
                        # inpDataL2 = inpDataL1.merge(mlModelPred, how='left', left_on=['dtDateKst', 'anaTime'],right_on=['dtDateKst', 'anaTime'])\
                        #     .merge(dlModelPredL1, how='left', left_on=['dtDateKst', 'anaTime'], right_on=['dtDateKst', 'anaTime'])

                        # 딥러닝
                        inpDataL2 = inpDataL1.merge(dlModelPredL1, how='left', left_on=['dtDateKst', 'anaTime'], right_on=['dtDateKst', 'anaTime'])

                        # dtDateKst 및 anaTime을 기준으로 중복 제거
                        inpDataL2.drop_duplicates(subset=['dtDateKst', 'anaTime'], inplace=True)
                        inpDataL2 = inpDataL2.reset_index(drop=True)

                        dbData = inpDataL2.rename(
                            {
                                'anaTime': 'ANA_DATE'
                                , 'dtDateKst': 'DATE_TIME_KST'
                                , 'dtDate': 'DATE_TIME'
                                , 'sza': 'SZA'
                                , 'aza': 'AZA'
                                , 'et': 'ET'
                            }
                            , axis='columns'
                        )

                        res = dbCon.execute(
                            """
                            SELECT COUNT(*) AS CNT
                            FROM TB_FOR_DATA_%s
                            WHERE  SRV = %s AND ANA_DATE = %s AND DATE_TIME = %s
                            """
                            , (iAnaYear, dbData['SRV'][0], dbData['ANA_DATE'][0], dbData['DATE_TIME'][0])
                        ).fetchone()

                        log.info("[CHECK] res['CNT'] : {}".format(res['CNT']))

                        # 삽입 및 수정
                        if (res['CNT'] == 0):
                            dbData['REG_DATE'] = datetime.datetime.now()
                        else:
                            dbData['MOD_DATE'] = datetime.datetime.now()

                        # 삽입
                        selDbTable = 'TB_FOR_DATA_{}'.format(iAnaYear)
                        dbData.to_sql(name=selDbTable, con=dbCon, if_exists='append', index=False)

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