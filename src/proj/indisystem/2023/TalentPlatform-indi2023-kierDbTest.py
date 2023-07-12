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
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pymysql
import re
import configparser
from sqlalchemy.ext.declarative import declarative_base
from scipy.stats import linregress
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
import psycopg2
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import insert

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


def dbMergeData(session, table, dataList):
    try:
        stmt = insert(table)
        onConflictStmt = stmt.on_conflict_do_update(
            index_elements=['ANA_DT', 'FOR_DT', 'MODEL_TYPE']
            , set_=stmt.excluded
        )
        session.execute(onConflictStmt, dataList)
        session.commit()

    except Exception as e:
        session.rollback()
        log.error(f'Exception : {e}')

    finally:
        session.close()

# def add(a, b):
#     print(a + b)
#     return a + b
#
# def multiply(a, b):
#     return a * b

def dynamicFun(funName, *args, **kwargs):
    function = globals()[funName]
    function(*args, **kwargs)

# dynamicFun("add", 5, 3)

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 포스트SQL 연동 테스트

    # create table "TB_MODEL"
    # (
    #     "ANA_DT"     timestamp   not null,
    #     "FOR_DT"     timestamp   not null,
    #     "MODEL_TYPE" varchar(20) not null,
    #     "U"          real[],
    #     "V"          real[],
    #     "SW_D"       real[],
    #     "SW_DC"      real[],
    #     "U1"         real[],
    #     "V1"         real[],
    #     "SW_DDNI"    real[],
    #     "SW_DDIF"    real[],
    #     "SW_NET"     real[],
    #     "SW_UC"      real[],
    #     "SW_U"       real[],
    #     "U850"       real[],
    #     "U875"       real[],
    #     "U900"       real[],
    #     "U925"       real[],
    #     "U975"       real[],
    #     "U1000"      real[],
    #     "V850"       real[],
    #     "V875"       real[],
    #     "V900"       real[],
    #     "V925"       real[],
    #     "V975"       real[],
    #     "V1000"      real[],
    #     constraint "TB_MODEL_pk"
    #         primary key ("ANA_DT", "FOR_DT", "MODEL_TYPE")
    # );
    #
    # comment on column "TB_MODEL"."ANA_DT" is '분석시간';
    #
    # comment on column "TB_MODEL"."FOR_DT" is '예보시간';
    #
    # comment on column "TB_MODEL"."MODEL_TYPE" is '모델 종류';
    #
    # comment on column "TB_MODEL"."U" is '지표 U벡터';
    #
    # comment on column "TB_MODEL"."V" is '지표 V벡터';
    #
    # comment on column "TB_MODEL"."SW_D" is '지표 하향 전천일사량';
    #
    # comment on column "TB_MODEL"."SW_DC" is '지표 하향 청천일사량';
    #
    # comment on column "TB_MODEL"."U1" is '상층 U벡터 1 hPa';
    #
    # comment on column "TB_MODEL"."V1" is '상층 V벡터 1 hPa';
    #
    # comment on column "TB_MODEL"."SW_DDNI" is '지표 하향 직달일사량';
    #
    # comment on column "TB_MODEL"."SW_DDIF" is '지표 하향 산란일사량';
    #
    # comment on column "TB_MODEL"."SW_NET" is '지표 순 일사량';
    #
    # comment on column "TB_MODEL"."SW_UC" is '지표 상향 청천일사량';
    #
    # comment on column "TB_MODEL"."SW_U" is '지표 상향 전천일사량';
    #
    # comment on column "TB_MODEL"."U850" is '상층 U벡터 850 hPa';
    #
    # comment on column "TB_MODEL"."U875" is '상층 U벡터 875 hPa';
    #
    # comment on column "TB_MODEL"."U900" is '상층 U벡터 900 hPa';
    #
    # comment on column "TB_MODEL"."U925" is '상층 U벡터 925 hPa';
    #
    # comment on column "TB_MODEL"."U975" is '상층 U벡터 975 hPa';
    #
    # comment on column "TB_MODEL"."U1000" is '상층 U벡터 1000 hPa';
    #
    # comment on column "TB_MODEL"."V850" is '상층 V벡터 850 hPa';
    #
    # comment on column "TB_MODEL"."V875" is '상층 V벡터 875 hPa';
    #
    # comment on column "TB_MODEL"."V900" is '상층 V벡터 900 hPa';
    #
    # comment on column "TB_MODEL"."V925" is '상층 V벡터 925 hPa';
    #
    # comment on column "TB_MODEL"."V975" is '상층 V벡터 975 hPa';
    #
    # comment on column "TB_MODEL"."V1000" is '상층 V벡터 1000 hPa';
    #
    # alter table "TB_MODEL"
    #     owner to kier;

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
                # 시작일, 종료일, 시간 간격
                'srtDate': '2023-06-29'
                , 'endDate': '2023-07-01'
                , 'invHour': 1

                # DB 정보
                # 사용자, 비밀번호, 호스트, 포트, 스키마
                , 'dbInfo': {
                    'dbType': 'postgresql'
                    , 'dbUser': 'kier'
                    , 'dbPwd': 'kier20230707!@#'
                    , 'dbHost': '223.130.134.136'
                    , 'dbPort': '5432'
                    , 'dbName': 'kier'
                    , 'dbTable': 'TB_MODEL'
                    , 'dbSchema': 'dms01'

                    # 서버 정보
                    , 'serverHost': '223.130.134.136'
                }

                # 모델 종류에 따른 함수 정의
                , 'procList' : {
                    'KIER-LDAPS' : 'add'
                }

                # 모델 정보
                # 파일 경로, 파일명, 데이터 컬럼, DB 컬럼, 시간 간격
                , 'modelList' : {
                    'KIER-LDAPS' : {
                        'pres' : {
                            'filePath': '/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS'
                            , 'fileName' : 'wrfout_d02_%Y-%m-%d_%H:%M:*.nc'
                            , 'selCol': ['U10', 'V10']
                            , 'dbCol': ['U1', 'V1']
                        }
                        , 'sfc' : {
                            'filePath': '/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS'
                            , 'fileName' : 'wrfsolar_d02.%Y-%m-%d_%H:%M:*.nc'
                            , 'selCol': ['SWDOWN', 'SWDOWNC', 'GSW', 'SWDDNI', 'SWDDIF']
                            , 'dbCol': ['SW_D', 'SW_DC', 'SW_NET', 'SW_DDNI', 'SW_DDIF']
                        }
                    }
                }
            }

            # DB 정보
            # cfgInfo = initCfgInfo(globalVar['sysPath'])
            # dbEngine = cfgInfo['dbEngine']
            # kier@223.130.134.136-5432-kier20230707!@#

            dbInfo = sysOpt['dbInfo']
            dbType = dbInfo['dbType']
            dbUser = dbInfo['dbUser']
            dbPwd = quote_plus(dbInfo['dbPwd'])
            dbHost = 'localhost' if dbInfo['dbHost'] == dbInfo['serverHost'] else dbInfo['dbHost']
            dbPort = dbInfo['dbPort']
            dbName = dbInfo['dbName']
            dbTable = dbInfo['dbTable']
            dbSchema = dbInfo['dbSchema']

            sqlDbUrl = f'{dbType}://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}'
            engine = create_engine(sqlDbUrl)
            sessionMake = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            session = sessionMake()

            # 테이블 정보
            metaData = MetaData()
            table = Table(dbTable, metaData, autoload_with=engine, schema=dbSchema)

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(sysOpt['invHour']))

            for modelIdx, modelType in enumerate(sysOpt['modelList']):
                log.info(f'[CHECK] modelType : {modelType}')

                # procFun = sysOpt['procList'][modelType]

                # procDynamiCall
                # dynamicFun(procFun, 5, 3)

                for dtDateIdx, dtDateInfo in enumerate(dtDateList):
                    log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                    dbDataList = []
                    dbData = {}
                    for i, modelKey in enumerate(sysOpt['modelList'][modelType]):
                        modelInfo = sysOpt['modelList'][modelType][modelKey]
                        log.info(f'[CHECK] modelInfo : {modelInfo}')

                        inpFile = '{}/{}'.format(modelInfo['filePath'], modelInfo['fileName'])
                        inpFileDate = dtDateInfo.strftime(inpFile)
                        fileList = sorted(glob.glob(inpFileDate))

                        if fileList is None or len(fileList) < 1:
                            # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                            continue

                        # NetCDF 파일 읽기
                        orgData = xr.open_mfdataset(fileList)

                        # sfc에서 일사량 관련 인자의 경우 1시간 평균 생산
                        if modelKey == 'sfc':
                            data = orgData.mean(dim=['Time'], skipna = True)
                            timeByteList = [orgData['Times'].values[0]]
                        else:
                            data = orgData
                            timeByteList = orgData['Times'].values

                        # 분석시간
                        anaDate = pd.to_datetime(orgData.START_DATE, format='%Y-%m-%d_%H:%M:%S')
                        timeList = [item.decode('utf-8') for item in timeByteList]

                        for timeIdx, timeInfo in enumerate(timeList):
                            log.info(f'[CHECK] timeInfo : {timeInfo}')

                            # 예보시간
                            forDate = pd.to_datetime(timeInfo, format='%Y-%m-%d_%H:%M:%S')

                            # 필수 컬럼
                            dbData['ANA_DT'] = anaDate
                            dbData['FOR_DT'] = forDate
                            dbData['MODEL_TYPE'] = modelType

                            # 선택 컬럼
                            for selCol, dbCol in zip(modelInfo['selCol'], modelInfo['dbCol']):
                                try:
                                    # sfc에서 일사량 관련 인자의 경우 1시간 평균 생산
                                    if modelKey == 'sfc':
                                        dbData[dbCol] = data[selCol].values.tolist() if len(data[selCol].values) > 0 else None
                                    else:
                                        dbData[dbCol] = data[selCol].isel(Time=timeIdx).values.tolist() if len(data[selCol].isel(Time=timeIdx).values) > 0 else None

                                    log.info(f'[CHECK] selCol / dbCol : {selCol} / {dbCol}')
                                except Exception as e:
                                    # log.error('Exception : {}'.format(e))
                                    continue

                    # (삽입) 1건
                    if len(dbData) < 1: continue
                    dbMergeData(session, table, dbData)


                #     # 테스트
                #     dtSrtDate = pd.to_datetime('2021-01-01', format='%Y-%m-%d')
                #     dtEndDate = pd.to_datetime('2023-06-29', format='%Y-%m-%d')
                #     dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(sysOpt['invHour']))
                #
                #     for dtDateIdx, dtDateInfo in enumerate(dtDateList):
                #         log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
                #         dbData['ANA_DT'] = dtDateInfo
                #         dbMergeData(session, table, dbData)
                #
                # # (삽입) N건
                    # dbMergeData(session, table, dbDataList)



            # dataList = [dbData, dbData]
            # dbMergeBatchData(session, table, dataList)



            # from sqlalchemy.dialects.mysql import insert
            #
            # def upsert_data(engine, table_name, orgData):
            #     stmt = insert(table_name).values(orgData)
            #     on_duplicate_key_stmt = stmt.on_duplicate_key_update(
            #         {key: orgData[key] for key in orgData}
            #     )
            #     engine.execute(on_duplicate_key_stmt)
            #
            # upsert_data(engine, 'dms01.model', dbData)


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

            # orgData['']
            # :START_DATE = "2023-06-28_12:00:00";

            # time1D = orgData['Times'].values
            # lon2D = orgData['XLAT'].values
            # orgData['XLAT'].isel(Time = 5).plot()
            #
            # plt.show()
            # orgData['XLONG'].values

            # orgData['SWDOWN'].isel(Time=5).plot()
            # plt.show()
            #
            # dataL1 = orgData['SWDOWN'].isel(Time=0).to_dataframe()
            # dataL2 = dataL1.reset_index(drop=False)
            #
            # orgData['SWDOWN'].values

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
