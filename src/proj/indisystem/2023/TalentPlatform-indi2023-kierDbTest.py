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
from datetime import datetime
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
from urllib.parse import quote_plus
# os.environ["PROJ_LIB"] = "C:\ProgramData\Anaconda3\Library\share"
# from pygem import IDW
# import eccodes
import pygrib
# import pykrige.kriging_tools as kt
import pytz
import requests
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
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
import psycopg2


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
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
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

        # 글꼴 설정
        # plt.rc('font', family='Malgun Gothic')

    # 리눅스 환경
    if globalVar['sysOs'] in 'Linux':
        parser = argparse.ArgumentParser()

        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)

        inParInfo = vars(parser.parse_args())

        # 글꼴 설정
        # fileList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        # fontName = font_manager.FontProperties(fname=fileList[0]).get_name()
        # plt.rc('font', family=fontName)

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


# def initCfgInfo(sysPath, sysOpt):
#     log.info('[START] {}'.format('initCfgInfo'))
#
#     result = None
#
#     try:
#
#         # DB 정보
#         config = configparser.ConfigParser()
#         config.read(sysPath, encoding='utf-8')
#
#         configKey = 'postgresql-clova-kier'
#         dbUser = config.get(configKey, 'user')
#         dbPwd = quote_plus(config.get(configKey, 'pwd'))
#         dbHost = config.get(configKey, 'host')
#         dbHost = 'localhost' if dbHost == sysOpt['updIp'] else dbHost
#         dbPort = config.get(configKey, 'port')
#         dbName = config.get(configKey, 'dbName')
#
#         sqlDbUrl = f'mysql+pymysql://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}'
#         engine = create_engine(sqlDbUrl)
#         session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#         # session().execute('SELECT * FROM TB_FILE_INFO_DTL').fetchall()
#
#         result = {
#             'dbEngine': engine
#             , 'session': session
#         }
#
#         return result
#
#     except Exception as e:
#         log.error('Exception : {}'.format(e))
#         return result
#
#     finally:
#         # try, catch 구문이 종료되기 전에 무조건 실행
#         log.info(f'[END] {}'.format('initCfgInfo'))

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 포스트SQL 연동 테스트

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'INDI2023'

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

            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'


            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': '2019-01-01'
                , 'endDate': '2022-05-22'

                # DB 정보
                , 'dbUser' : 'kier'
                , 'dbPwd' : 'kier20230707!@#'
                , 'dbHost': '223.130.134.136'
                , 'dbPort': '5432'
                , 'dbName': 'kier'

                # 서버 정보
                , 'serverHost': '223.130.134.136'
            }

            # DB 정보
            # cfgInfo = initCfgInfo(globalVar['sysPath'])
            # dbEngine = cfgInfo['dbEngine']
            # kier@223.130.134.136-5432-kier20230707!@#

            dbUser = sysOpt['dbUser']
            dbPwd = quote_plus(sysOpt['dbPwd'])
            dbHost = sysOpt['dbHost']
            dbHost = 'localhost' if dbHost == sysOpt['serverHost'] else dbHost
            dbPort = sysOpt['dbPort']
            dbName = sysOpt['dbName']

            sqlDbUrl = f'postgresql://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}'
            engine = create_engine(sqlDbUrl)
            session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            # session().execute('SELECT * FROM dms01.model').fetchall()

            # dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            # dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            # dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(6))

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'MODEL/KIER-LDAPS/wrfsolar_d02.2023-06-30_03:00:00.nc')
            fileList = sorted(glob.glob(inpFile))

            if fileList is None or len(fileList) < 1:
                log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                # continue

            fileInfo = fileList[0]
            log.info('[CHECK] fileInfo : {}'.format(fileInfo))

            fileNameNoExt = os.path.basename(fileInfo).split('.')[0]
            data = xr.open_dataset(fileInfo)

            # 분석장
            anaDate = pd.to_datetime(data.START_DATE, format='%Y-%m-%d_%H:%M:%S')
            forDate = pd.to_datetime(data['Times'].values[0].decode(), format='%Y-%m-%d_%H:%M:%S')
            modelType = 'KIER-LDAPS'
            swD = data['SWDOWN'].isel(Time=0).values

            from sqlalchemy import create_engine, text
            from sqlalchemy.dialects.postgresql import ARRAY

            data = [[31.148539, 30.739117, 33.336533], [31.985273, 256.6858, 286.0836], [33.42487, 267.13397, 273.16577]]
            with engine.connect() as connection:
                sql = text('INSERT INTO dms01.table (data) VALUES (:data)')
                connection.execute(sql, data=data)

            sql = text('INSERT INTO dms01.table (data) VALUES (:data)')
            engine.execute(sql, data=data)
            # session().query(sql, data=data)

            # query = """
            #     INSERT INTO dms01.model (
            #         ana_date, for_date, model_type, sw_d
            #     ) VALUES (
            #         ana_date, for_date,:model_type :sw_d
            #     )
            # """
            #
            # with engine.begin() as connection:
            #     connection.execute(query, {
            #         "ana_date": anaDate
            #         , "for_date": forData
            #         , "model_type": modelType
            #         , "sw_d": swD.tolist()
            #     })

            selDbTable = 'dms01.model'

            # session().execute(
            #     """
            #     INSERT INTO dms01.model (ana_date, for_date, model_type, sw_d)
            #     VALUES (:anaDate, :for_date, :model_type, :sw_d)
            #     ON CONFLICT (ana_date, for_date, model_type)
            #     DO UPDATE
            #     SET ana_date = :anaDate, for_date = :forDate, model_type = :modelType, sw_d = :swD.tolist();
            #     """,
            #     {
            #         "anaDate": anaDate,
            #         "for_date": forDate,
            #         "model_type": modelType,
            #         "sw_d": swD.tolist()
            #     }
            # )

            # sw_d = swD.tolist()
            #
            # session().execute(
            #     """
            #     INSERT INTO dms01.model (ana_date, for_date, model_type, sw_d)
            #     VALUES (:anaDate, :for_date, :model_type, :sw_d)
            #     ON CONFLICT (ana_date, for_date, model_type)
            #     DO UPDATE
            #     SET ana_date = :anaDate, for_date = :forDate, model_type = :modelType, sw_d = :sw_d;
            #     """,
            #     {
            #         "anaDate": anaDate,
            #         "forDate": forDate,
            #         "modelType": modelType,
            #         "sw_d": ARRAY(sw_d)
            #     }
            # )

            # sql = text('INSERT INTO dms01.model (ana_date, for_date, model_type, sw_d) VALUES (:anaDate, :for_date, :model_type, :sw_d)')
            sql = text('INSERT INTO dms01.model (ana_date, for_date, model_type, sw_d) VALUES (:ana_date, :for_date, :model_type, :sw_d)')
            engine.execute(sql,  {
                    "ana_date": anaDate,
                    "for_date": forDate,
                    "model_type": modelType,
                    "sw_d": swD.tolist()
                }
            )


            # session().execute(
            #    f"""
            #     INSERT INTO `{selDbTable}` ('ana_date', 'for_date', 'model_type', 'sw_d')
            #       VALUES ({anaDate}, {forDate}, {modelType}, {swD.tolist()})
            #       ON CONFLICT (anaDate, for_date, model_type)
            #       DO UPDATE SET sw_d = swD.tolist()
            #       WHERE ana_date = {anaDate}, for_date = {forDate}, model_type = {modelType};
            #     """
            # )

            # session().execute(
            #     """
            #     INSERT INTO model (ana_date, for_date, model_type, sw_d)
            #     VALUES (:ana_date, :for_date, :model_type, :sw_d)
            #     ON CONFLICT (ana_date, for_date, model_type)
            #     DO UPDATE SET sw_d = excluded.sw_d
            #     WHERE model.ana_date = :ana_date AND model.for_date = :for_date AND model.model_type = :model_type;
            #     """,
            #     {
            #         "ana_date": anaDate,
            #         "for_date": forDate,
            #         "model_type": modelType,
            #         "sw_d": swD.tolist()
            #     }
            # )


            #   # 테이블 PK키를 통해 삽입/수정
            #   session.execute(
            #       """
            #     INSERT INTO `{}` (PRODUCT_SERIAL_NUMBER, DATE_TIME, TEMP, HMDTY, PM25, PM10, MVMNT, TVOC, HCHO, CO2, CO, BENZO, RADON, REG_DATE, MOD_DATE)
            #     VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')
            #     ON DUPLICATE KEY UPDATE
            #         PRODUCT_SERIAL_NUMBER = VALUES(PRODUCT_SERIAL_NUMBER)
            #         , TEMP = VALUES(TEMP)
            #         , HMDTY = VALUES(HMDTY)
            #         , PM25 = VALUES(PM25)
            #         , PM10 = VALUES(PM10)
            #         , MVMNT = VALUES(MVMNT)
            #         , TVOC = VALUES(TVOC)
            #         , HCHO = VALUES(HCHO)
            #         , CO2 = VALUES(CO2)
            #         , CO = VALUES(CO)
            #         , BENZO = VALUES(BENZO)
            #         , RADON = VALUES(RADON)
            #         , MOD_DATE = VALUES(MOD_DATE)
            #         ;
            #         """.format(selDbTable, dbInfo['PRODUCT_SERIAL_NUMBER'], dbInfo['DATE_TIME'], dbInfo['TEMP'], dbInfo['HMDTY'], dbInfo['PM25'], dbInfo['PM10']
            #                    , dbInfo['MVMNT'], dbInfo['TVOC'], dbInfo['HCHO'], dbInfo['CO2'], dbInfo['CO'], dbInfo['BENZO'], dbInfo['RADON'], dbInfo['REG_DATE'], dbInfo['MOD_DATE'])
            # )
            #
            # session.commit()





            #    dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            #             dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            #             dtMonthList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1M')

            # data['']
            # :START_DATE = "2023-06-28_12:00:00";

            time1D = data['Times'].values
            # lon2D = data['XLAT'].values
            # data['XLAT'].isel(Time = 5).plot()
            #
            # plt.show()
            # data['XLONG'].values

            data['SWDOWN'].isel(Time=5).plot()
            plt.show()

            dataL1 = data['SWDOWN'].isel(Time=0).to_dataframe()
            dataL2 = dataL1.reset_index(drop=False)

            data['SWDOWN'].values

        #   dataL2.to_dataframe().reset_index(drop=False).to_csv(saveFile, index=False)

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
