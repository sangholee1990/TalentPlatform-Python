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
from scipy.stats import linregress
import pandas as pd
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
import pytz
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
import pvlib
import xarray as xr
from pvlib import location
from pvlib import irradiance


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

                globalVar['inpPath'] = 'E:/DATA'
                globalVar['outPath'] = 'E:/DATA/OUTPUT'
                globalVar['modelPath'] = 'E:/DATA'

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2019-01-01'
                    , 'endDate': '2022-06-08'
                    # , 'endDate': '2021-11-01'

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

            dbEngine = create_engine('mariadb://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
            sessMake = sessionmaker(bind=dbEngine)
            session = sessMake()


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

            # *******************************************************
            # 관측자료 읽기
            # *******************************************************
            saveFile = '{}/{}_{}.xlsx'.format(globalVar['outPath'], serviceName, 'FOR-ACT_SWR')
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            writer = pd.ExcelWriter(saveFile, engine='openpyxl')

            for idx, posInfo in posDataL1.iterrows():
                posId = int(posInfo['ID'])
                posVol = posInfo['VOLUME']
                posLat = posInfo['LAT']
                posLon = posInfo['LON']
                posSza = posInfo['STN_SZA']
                posAza = posInfo['STN_AZA']

                srvId = 'SRV{:05d}'.format(posId)
                log.info("[CHECK] srvId : {}".format(srvId))

                # *****************************************************
                # 기상정보 데이터
                # *****************************************************
                dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
                dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
                dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(6))

                dtYearList = dtIncDateList.strftime('%Y').unique().tolist()

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


                actData = pd.DataFrame()
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
                    actData = pd.concat([actData, res], ignore_index=False)


                # *****************************************************
                # 데이터 전처리 및 병합
                # *****************************************************
                forDataL1 = forData[['SRV', 'ANA_DATE', 'DATE_TIME', 'DATE_TIME_KST', 'SWR']].rename(
                    {
                        'SWR': 'UM_SWR'
                    }
                    , axis='columns'
                )

                actDataL1 = actData[['SRV', 'DATE_TIME', 'DATE_TIME_KST', 'SWR']].rename(
                    {
                        'SWR': 'H8_SWR'
                    }
                    , axis='columns'
                )

                data = forDataL1.merge(actDataL1, how='left', left_on=['SRV', 'DATE_TIME', 'DATE_TIME_KST'], right_on=['SRV', 'DATE_TIME', 'DATE_TIME_KST'])
                dataL1 = data.drop_duplicates(['SRV', 'DATE_TIME', 'DATE_TIME_KST']).sort_values(by=['SRV', 'ANA_DATE', 'DATE_TIME', 'DATE_TIME_KST'], axis=0).reset_index(drop=True)

                dataL1.to_excel(writer, sheet_name=srvId, index=False)

            writer.save()
            log.info('[CHECK] saveFile : {}'.format(saveFile))

                # dataL1['dtYmd'] = pd.to_datetime(dataL1['DATE_TIME_KST']).dt.strftime("%Y%m%d")
                # dtYmdList = sorted(dataL1['dtYmd'].unique())
                #
                # # dtYmdInfo = dtYmdList[0]
                # for i, dtYmdInfo in enumerate(dtYmdList):
                #     log.info("[CHECK] dtYmdInfo : {}".format(dtYmdInfo))
                #
                #     dataL2 = dataL1.loc[
                #         (dataL1['dtYmd'] == dtYmdInfo)
                #         ]
                #
                #     plt.plot(dataL2['DATE_TIME_KST'], dataL2['UM_SWR'], 'o')
                #     plt.plot(dataL2['DATE_TIME_KST'], dataL2['H8_SWR'], 'o')
                #     plt.show()


                # *****************************************************
                # 데이터 전처리 및 병합
                # *****************************************************
                # pvDataL1 = pvData.loc[(pvData['PV'] > 0) & (pvData['PV'] <= posVol)]
                # pvDataL1['PV'][pvDataL1['PV'] < 0] = np.nan
                #
                # forData['CA_TOT'][forData['CA_TOT'] < 0] = np.nan
                # forData['WS'][forData['WS'] < 0] = np.nan
                # forData['WD'][forData['WD'] < 0] = np.nan
                # forData['SWR'][forData['SWR'] < 0] = np.nan
                #
                # inpData = forData.merge(pvDataL1, how='left', left_on=['SRV', 'DATE_TIME_KST', 'DATE_TIME'],
                #                         right_on=['SRV', 'DATE_TIME_KST', 'DATE_TIME'])
                # # inpDataL1 = inpData.sort_values(by=['ANA_DATE','DATE_TIME_KST'], axis=0)
                # inpDataL1 = inpData.drop_duplicates(['SRV', 'ANA_DATE', 'DATE_TIME_KST']).sort_values(
                #     by=['SRV', 'ANA_DATE', 'DATE_TIME_KST'], axis=0).reset_index(drop=True)
                # prdData = inpDataL1.copy()


                # inpFilePattern = '/ACT/LSH0255-SRV{:03d}-final-proc-act.xlsx'.format(posId)
                # inpFile = '{}/{}'.format(globalVar['inpPath'], inpFilePattern)
                # fileList = sorted(glob.glob(inpFile))
                #
                # if (len(fileList) < 1): continue
                #
                # data = pd.read_excel(fileList[0])
                #
                # # 이름 변경
                # dataL1 = data.rename(
                #     columns={
                #         'dtDate': 'DATE_TIME'
                #         , 'dtDateKst': 'DATE_TIME_KST'
                #         , 'sza': 'SZA'
                #         , 'aza': 'AZA'
                #         , 'et': 'ET'
                #     }
                # )
                # dataL1['SRV'] = srvId
                #
                # site = location.Location(posLat, posLon, tz='Asia/Seoul')
                # clearInsInfo = site.get_clearsky(pd.to_datetime(dataL1['DATE_TIME'].values))
                # dataL1['GHI_CLR'] = clearInsInfo['ghi'].values
                # dataL1['DNI_CLR'] = clearInsInfo['dni'].values
                # dataL1['DHI_CLR'] = clearInsInfo['dhi'].values
                #
                # poaInsInfo = irradiance.get_total_irradiance(
                #     surface_tilt=posSza,
                #     surface_azimuth=posAza,
                #     dni=clearInsInfo['dni'],
                #     ghi=clearInsInfo['ghi'],
                #     dhi=clearInsInfo['dhi'],
                #     solar_zenith=dataL1['SZA'].values,
                #     solar_azimuth=dataL1['AZA'].values
                # )
                #
                # dataL1['GHI_POA'] = poaInsInfo['poa_global'].values
                # dataL1['DNI_POA'] = poaInsInfo['poa_direct'].values
                # dataL1['DHI_POA'] = poaInsInfo['poa_diffuse'].values
                #
                # # 혼탁도
                # turbidity = pvlib.clearsky.lookup_linke_turbidity(pd.to_datetime(dataL1['DATE_TIME'].values), posLat, posLon, interp_turbidity=True)
                # dataL1['TURB'] = turbidity.values
                #
                #
                # dbData = dataL1
                # for i, dbInfo in dbData.iterrows():
                #     # dbInfo['SRV'] = 'SRV{:05d}'.format(dbInfo['id'])
                #
                #     log.info("[CHECK] DATE_TIME : {}".format(dbInfo['DATE_TIME']))
                #
                #     iYear = int(dbInfo['DATE_TIME_KST'].strftime("%Y"))
                #     selDbTable = 'TB_ACT_DATA_{}'.format(iYear)
                #
                #     # 테이블 생성
                #     session.execute(
                #         """
                #         CREATE TABLE IF NOT EXISTS `{}`
                #         (
                #             SRV           varchar(10) not null comment '관측소 정보',
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
                #             TURB          float       null comment '혼탁도',
                #             GHI_CLR       float       null comment '맑은날 전천 일사량',
                #             DNI_CLR       float       null comment '맑은날 직달 일사량',
                #             DHI_CLR       float       null comment '맑은날 산란 일사량',
                #             GHI_POA       float       null comment '보정 맑은날 전천 일사량',
                #             DNI_POA       float       null comment '보정 맑은날 직달 일사량',
                #             DHI_POA       float       null comment '보정 맑은날 산란 일사량',
                #             SWR           float       null comment '일사량',
                #             ML            float       null comment '머신러닝',
                #             DL            float       null comment '딥러닝',
                #             ML2            float       null comment '머신러닝',
                #             DL2            float       null comment '딥러닝',
                #             REG_DATE      datetime    null comment '등록일',
                #             MOD_DATE      datetime    null comment '수정일',
                #             primary key (SRV, DATE_TIME)
                #         )
                #         comment '기상 실황 테이블_{}';
                #         """.format(selDbTable, iYear)
                #     )
                #     session.commit()
                #
                #     # 테이블 중복 검사
                #     resChk = pd.read_sql(
                #         """
                #         SELECT COUNT(*) AS CNT FROM `{}`
                #         WHERE SRV = '{}' AND DATE_TIME = '{}'
                #         """.format(selDbTable, dbInfo['SRV'], dbInfo['DATE_TIME'])
                #         , con=dbEngine
                #     )
                #
                #     if (resChk.loc[0, 'CNT'] > 0):
                #         dbInfo['MOD_DATE'] = datetime.datetime.now()
                #
                #         session.execute(
                #             """
                #             UPDATE `{}`
                #             SET DATE_TIME_KST = '{}', CA_TOT = '{}', HM = '{}', PA = '{}', TA = '{}', TD = '{}', WD = '{}', WS = '{}', SZA = '{}', AZA = '{}', ET = '{}', TURB = '{}'
                #             , GHI_CLR = '{}', DNI_CLR = '{}', DHI_CLR = '{}', GHI_POA = '{}', DNI_POA = '{}', DHI_POA = '{}', SWR = '{}', MOD_DATE = '{}'
                #             WHERE SRV = '{}' AND DATE_TIME = '{}';
                #             """.format(selDbTable
                #                        , dbInfo['DATE_TIME_KST'], dbInfo['CA_TOT'], dbInfo['HM'], dbInfo['PA'], dbInfo['TA']
                #                        , dbInfo['TD'], dbInfo['WD'], dbInfo['WS'], dbInfo['SZA'], dbInfo['AZA'], dbInfo['ET'], dbInfo['TURB'], dbInfo['GHI_CLR']
                #                        , dbInfo['DNI_CLR'], dbInfo['DHI_CLR'], dbInfo['GHI_POA'], dbInfo['DNI_POA'], dbInfo['DHI_POA'], dbInfo['SWR'], dbInfo['MOD_DATE']
                #                        , dbInfo['SRV'], dbInfo['DATE_TIME']
                #                        )
                #         )
                #
                #     else:
                #         dbInfo['REG_DATE'] = datetime.datetime.now()
                #
                #         session.execute(
                #             """
                #             INSERT INTO `{}` (SRV, DATE_TIME, DATE_TIME_KST, CA_TOT, HM, PA, TA, TD, WD, WS, SZA, AZA, ET, TURB, GHI_CLR, DNI_CLR, DHI_CLR, GHI_POA, DNI_POA, DHI_POA, SWR, REG_DATE)
                #             VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')
                #             """.format(selDbTable
                #                        , dbInfo['SRV'], dbInfo['DATE_TIME'], dbInfo['DATE_TIME_KST'], dbInfo['CA_TOT'], dbInfo['HM'], dbInfo['PA'], dbInfo['TA']
                #                        , dbInfo['TD'], dbInfo['WD'], dbInfo['WS'], dbInfo['SZA'], dbInfo['AZA'], dbInfo['ET'], dbInfo['TURB'], dbInfo['GHI_CLR'], dbInfo['DNI_CLR']
                #                        , dbInfo['DHI_CLR'], dbInfo['GHI_POA'], dbInfo['DNI_POA'], dbInfo['DHI_POA'], dbInfo['SWR'], dbInfo['REG_DATE']
                #                        )
                #         )
                #     session.commit()

        except Exception as e:
            log.error("Exception : {}".format(e))
            session.rollback()
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