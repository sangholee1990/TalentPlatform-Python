import argparse
import glob
import logging
import logging.handlers
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime
from datetime import timedelta
from urllib.parse import quote_plus

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import xarray as xr
from pandas.tseries.offsets import Hour
from sqlalchemy import MetaData, Table
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker

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
        # , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        # , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        # , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        # , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        # , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        # , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        # , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        # , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        # , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
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


def initCfgInfo(sysOpt):
    log.info(f'[START] initCfgInfo')

    result = None

    try:
        dbInfo = sysOpt['dbInfo']
        dbType = dbInfo['dbType']
        dbUser = dbInfo['dbUser']
        dbPwd = quote_plus(dbInfo['dbPwd'])
        dbHost = 'localhost' if dbInfo['dbHost'] == dbInfo['serverHost'] else dbInfo['dbHost']
        dbPort = dbInfo['dbPort']
        dbName = dbInfo['dbName']
        dbSchema = dbInfo['dbSchema']

        sqlDbUrl = f'{dbType}://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}'

        # 커넥션 풀 관리
        # engine = create_engine(sqlDbUrl)
        engine = create_engine(sqlDbUrl, pool_size=20, max_overflow=0)
        sessionMake = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = sessionMake()

        # DB 연결 시 타임아웃 1시간 설정 : 60 * 60 * 1000
        session.execute(text("SET statement_timeout = 3600000;"))

        # 트랜잭션이 idle 상태 5분 설정 : 5 * 60 * 1000
        session.execute(text("SET idle_in_transaction_session_timeout = 300000;"))

        # 격리 수준 설정
        session.execute(text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;"))

        # 세션 커밋
        session.commit()

        # 테이블 정보
        metaData = MetaData()

        # 예보 모델 테이블
        tbModel = Table('TB_MODEL', metaData, autoload_with=engine, schema=dbSchema)
        tbByteModel = Table('TB_BYTE_MODEL', metaData, autoload_with=engine, schema=dbSchema)
        tbIntModel = Table('TB_INT_MODEL', metaData, autoload_with=engine, schema=dbSchema)

        # 기본 위경도 테이블
        tbGeo = Table('TB_GEO', metaData, autoload_with=engine, schema=dbSchema)

        # 상세 위경도 테이블
        tbGeoDtl = Table('TB_GEO_DTL', metaData, autoload_with=engine, schema=dbSchema)

        result = {
            'engine': engine
            , 'session': session
            , 'sessionMake': sessionMake
            , 'tbModel': tbModel
            , 'tbByteModel': tbByteModel
            , 'tbIntModel': tbIntModel
            , 'tbGeo': tbGeo
            , 'tbGeoDtl': tbGeoDtl
        }

        return result

    except Exception as e:
        log.error(f'Exception : {e}')
        return result

    finally:
        log.info(f'[END] initCfgInfo')


def dbMergeData(session, table, dataList, pkList=['ANA_DT', 'FOR_DT', 'MODEL_TYPE']):

    # log.info(f'[START] dbMergeData')

    try:
        stmt = insert(table)
        onConflictStmt = stmt.on_conflict_do_update(
            index_elements=pkList
            , set_=stmt.excluded
        )
        session.execute(onConflictStmt, dataList)
        session.commit()

    except Exception as e:
        session.rollback()
        log.error(f'Exception : {e}')

    finally:
        session.close()
        # log.info(f'[END] dbMergeData')

def dynamicFun(funName, *args, **kwargs):
    log.info(f'[START] dynamicFun')

    try:
        function = globals()[funName]

        function(*args, **kwargs)

    except Exception as e:
        log.error(f'Exception : {e}')

    finally:
        log.info(f'[END] dynamicFun')


def convFloatToIntList(val):

    scaleFactor = 10000
    addOffset = 0

    return ((np.around(val, 4) * scaleFactor) - addOffset).astype(int).tolist()

def readKierData(modelType, dtDateList, sysOpt, cfgOpt):

    log.info(f'[START] readKierData')

    result = None

    for dtDateIdx, dtDateInfo in enumerate(dtDateList):
        log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

        # dbDataList = []
        dbData = {}

        try:
            for i, modelKey in enumerate(sysOpt[modelType]):
                modelInfo = sysOpt[modelType][modelKey]
                # log.info(f'[CHECK] modelInfo : {modelInfo}')

                inpFile = '{}/{}'.format(modelInfo['filePath'], modelInfo['fileName'])
                inpFileDate = dtDateInfo.strftime(inpFile)
                fileList = sorted(glob.glob(inpFileDate))

                if fileList is None or len(fileList) < 1:
                    # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                    continue

                # NetCDF 파일 읽기
                fileInfo = fileList[0]
                orgData = xr.open_mfdataset(fileList)
                log.info(f'[CHECK] fileInfo : {fileInfo}')

                # wrfsolar에서 일사량 관련 인자의 경우 1시간 누적 생산 -> 평균 생산
                if modelKey == 'wrfsolar':
                    data = orgData.mean(dim=['Time'], skipna=True)
                    timeByteList = [orgData['Times'].values[0]]
                else:
                    data = orgData
                    timeByteList = orgData['Times'].values

                # 분석시간
                anaDate = pd.to_datetime(orgData.START_DATE, format='%Y-%m-%d_%H:%M:%S')
                # log.info(f'[CHECK] anaDate : {anaDate}')

                # 예보시간
                timeIdx = 0
                timeInfo = timeByteList[timeIdx].decode('utf-8')
                forDate = pd.to_datetime(timeInfo, format='%Y-%m-%d_%H:%M:%S')
                log.info(f'[CHECK] anaDate : {anaDate} / forDate : {forDate}')

                # 필수 컬럼
                dbData['ANA_DT'] = anaDate
                dbData['FOR_DT'] = forDate
                dbData['MODEL_TYPE'] = modelType

                # 선택 컬럼
                for selCol, dbCol in zip(modelInfo['selCol'], modelInfo['dbCol']):
                    try:
                        # wrfsolar에서 일사량 관련 인자의 경우 1시간 누적 생산 -> 평균 생산
                        if modelKey == 'wrfsolar':
                            # dbData[dbCol] = data[selCol].values.tolist() if len(data[selCol].values) > 0 else None
                            # dbData[dbCol] = np.around(data[selCol].values, 4).tobytes() if len(data[selCol].values) > 0 else None
                            dbData[dbCol] = convFloatToIntList(data[selCol].values) if len(data[selCol].values) > 0 else None

                        else:
                            key, levIdx = selCol.split('-')
                            # dbData[dbCol] = data[key].isel(Time=timeIdx, bottom_top=int(levIdx)).values.tolist() if len(data[key].isel(Time=timeIdx, bottom_top=int(levIdx)).values) > 0 else None
                            # dbData[dbCol] = np.around(data[key].isel(Time=timeIdx, bottom_top=int(levIdx)).values, 4).tobytes() if len(data[key].isel(Time=timeIdx, bottom_top=int(levIdx)).values) > 0 else None
                            dbData[dbCol] = convFloatToIntList(data[key].isel(Time=timeIdx, bottom_top=int(levIdx)).values) if len(data[key].isel(Time=timeIdx, bottom_top=int(levIdx)).values) > 0 else None

                        # log.info(f'[CHECK] selCol / dbCol : {selCol} / {dbCol}')
                    except Exception as e:
                        log.error(f'Exception : {e}')
                        # pass

            if len(dbData) < 1: continue
            log.info(f'[CHECK] dbData : {dbData.keys()}')
            # dbMergeData(cfgOpt['sessionMake'], cfgOpt['tbModel'], dbData)
            # dbMergeData(cfgOpt['sessionMake'], cfgOpt['tbIntModel'], dbData)
            dbMergeData(cfgOpt['session'], cfgOpt['tbIntModel'], dbData)

        except Exception as e:
            log.error(f'Exception : {e}')
            return result

        finally:
            log.info(f'[END] readKierData')

def readTmpData(modelType, dtAndDateList, sysOpt, cfgOpt):

    log.info(f'[START] readTmpData')

    result = None

    for dtAndDateIdx, dtAndDateInfo in enumerate(dtAndDateList):
        dtForDateList = pd.date_range(start=dtAndDateInfo, end=dtAndDateInfo + Hour(48), freq=Hour(1))
        for dtForDateIdx, dtForDateInfo in enumerate(dtForDateList):
            log.info(f'[CHECK] dtAndDateInfo : {dtAndDateInfo} / dtForDateInfo : {dtForDateInfo}')

            try:
                dbData = {}

                # 필수 컬럼
                dbData['ANA_DT'] = dtAndDateInfo
                dbData['FOR_DT'] = dtForDateInfo
                dbData['MODEL_TYPE'] = modelType

                # 선택 컬럼
                for dbidx, dbCol in enumerate(sysOpt['TMP']['dbCol']):
                    try:
                        # dbData[dbCol] = np.random.randn(1000, 1000).tolist()
                        # dbData[dbCol] = np.around(np.random.randn(1000, 1000), 4).tolist()
                        # dbData[dbCol] = np.around(np.random.randn(1000, 1000), 4).tobytes()
                        dbData[dbCol] = convFloatToIntList(np.random.randn(1000, 1000) * 10)
                    except Exception as e:
                        log.error(f'Exception : {e}')

                if len(dbData) < 1: continue
                log.info(f'[CHECK] dbData : {dbData.keys()}')
                # dbMergeData(cfgOpt['sessionMake'], cfgOpt['tbModel'], dbData)
                # dbMergeData(cfgOpt['sessionMake'], cfgOpt['tbByteModel'], dbData)
                # dbMergeData(cfgOpt['sessionMake'], cfgOpt['tbIntModel'], dbData)
                dbMergeData(cfgOpt['session'], cfgOpt['tbIntModel'], dbData)

                # dbDataList.append(dbData)

            # if len(dbDataList) < 1: continue
            # log.info(f'[CHECK] dbDataList : {dbDataList.keys()}')
            # dbMergeData(cfgOpt['session'], cfgOpt['tbModel'], dbDataList)

            except Exception as e:
                log.error(f'Exception : {e}')
                return result

            finally:
                log.info(f'[END] readTmpData')


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

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        # contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'
        contextPath = os.getcwd() if env in 'local' else '/vol01/SYSTEMS/KIER/PROG/PYTHON'

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

        log.info(f"[START] __init__ : init")

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info(f"[END] __init__ : init")

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info(f"[START] exec")

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
                # 'srtDate': '2023-06-28'
                # , 'endDate': '2023-06-28'
                # 'srtDate': globalVar['srtDate']
                # , 'endDate': globalVar['endDate']
                , 'invHour': 1

                # DB 정보 : 사용자, 비밀번호, 호스트, 포트, 스키마
                , 'dbInfo': {
                    'dbType': 'postgresql'
                    , 'dbUser': 'kier'
                    , 'dbPwd': 'kier20230707!@#'
                    # , 'dbHost': '223.130.134.136'
                    # , 'dbHost': '192.168.0.244'
                    , 'dbHost': 'dev3.indisystem.co.kr'
                    # , 'dbPort': '5432'
                    , 'dbPort': '55432'
                    , 'dbName': 'kier'
                    , 'dbSchema': 'DMS01'

                    # 서버 정보
                    , 'serverHost': '223.130.134.136'
                    # , 'serverHost': '192.168.0.244'
                    # , 'serverHost': 'dev3.indisystem.co.kr'
                }

                # , 'modelList': ['KIER-LDAPS']
                # , 'modelList': ['KIER-RDAPS']
                , 'modelList': ['KIER-LDAPS', 'KIER-RDAPS']
                # , 'modelList': ['TMP3']
                # , 'modelList': [globalVar['modelList']]

                # 모델 종류에 따른 함수 정의
                # , 'procList': {
                #     'KIER-LDAPS': 'readKierData'
                #     , 'KIER-RDAPS': 'readKierData'
                #     , 'TMP': 'readTmpData'
                #     , 'TMP2': 'readTmpData'
                #     , 'TMP3': 'readTmpData'
                #     , 'TMP4': 'readTmpData'
                # }

                # 모델 정보 : 파일 경로, 파일명, 데이터/DB 컬럼 (지표면 wrfsolar 동적 설정, 상층면 wrfout 정적 설정), 시간 간격
                , 'KIER-LDAPS': {
                    'wrfout': {
                        'filePath': '/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS'
                        # 'filePath': '/vol01/DATA/MODEL/KIER-LDAPS'
                        , 'fileName': 'wrfout*d02*%Y-%m-%d_%H:%M:*.nc'
                        , 'selCol': ['U-0', 'U-1', 'U-2', 'U-3', 'U-4', 'U-5', 'V-0', 'V-1', 'V-2', 'V-3', 'V-4', 'V-5']
                        , 'dbCol': ['U1000', 'U975', 'U925', 'U900', 'U875', 'U850', 'V1000', 'V975', 'V925', 'V900','V875', 'V850']
                    }
                    , 'wrfsolar': {
                        'filePath': '/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS'
                        # 'filePath': '/vol01/DATA/MODEL/KIER-LDAPS'
                        , 'fileName': 'wrfsolar*d02*%Y-%m-%d_%H:%M:*.nc'
                        , 'selCol': ['SWDOWN', 'SWDOWNC', 'GSW', 'SWDDNI', 'SWDDIF', 'U10', 'V10']
                        , 'dbCol': ['SW_D', 'SW_DC', 'SW_NET', 'SW_DDNI', 'SW_DDIF', 'U', 'V']
                    }
                }

                , 'KIER-RDAPS': {
                    'wrfout': {
                        'filePath': '/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS'
                        # 'filePath': '/vol01/DATA/MODEL/KIER-RDAPS'
                        , 'fileName': 'wrfout*d02*%Y-%m-%d_%H:%M:*.nc'
                        , 'selCol': ['U-0', 'U-1', 'U-2', 'U-3', 'U-4', 'U-5', 'V-0', 'V-1', 'V-2', 'V-3', 'V-4', 'V-5']
                        , 'dbCol': ['U1000', 'U975', 'U925', 'U900', 'U875', 'U850', 'V1000', 'V975', 'V925', 'V900', 'V875', 'V850']
                    }
                    , 'wrfsolar': {
                        'filePath': '/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS'
                        # 'filePath': '/vol01/DATA/MODEL/KIER-RDAPS'
                        , 'fileName': 'wrfsolar*d02*%Y-%m-%d_%H:%M:*.nc'
                        , 'selCol': ['SWDOWN', 'SWDOWNC', 'GSW', 'SWDDNI', 'SWDDIF', 'U10', 'V10']
                        , 'dbCol': ['SW_D', 'SW_DC', 'SW_NET', 'SW_DDNI', 'SW_DDIF', 'U', 'V']
                    }
                }
                , 'TMP': {
                    'dbCol': ['U1000', 'U975', 'U925', 'U900', 'U875', 'U850', 'V1000', 'V975', 'V925', 'V900', 'V875',
                              'V850', 'SW_D', 'SW_DC', 'SW_NET', 'SW_DDNI', 'SW_DDIF', 'U', 'V']
                }
                , 'TMP2': {
                    'dbCol': ['U1000', 'U975', 'U925', 'U900', 'U875', 'U850', 'V1000', 'V975', 'V925', 'V900', 'V875',
                              'V850', 'SW_D', 'SW_DC', 'SW_NET', 'SW_DDNI', 'SW_DDIF', 'U', 'V']
                }
                , 'TMP3': {
                    'dbCol': ['U1000', 'U975', 'U925', 'U900', 'U875', 'U850', 'V1000', 'V975', 'V925', 'V900', 'V875',
                              'V850', 'SW_D', 'SW_DC', 'SW_NET', 'SW_DDNI', 'SW_DDIF', 'U', 'V']
                }
                , 'TMP4': {
                    'dbCol': ['U1000', 'U975', 'U925', 'U900', 'U875', 'U850', 'V1000', 'V975', 'V925', 'V900', 'V875',
                              'V850', 'SW_D', 'SW_DC', 'SW_NET', 'SW_DDNI', 'SW_DDIF', 'U', 'V']
                }
            }

            # *********************************************
            # DB 세선 정보 및 테이블 메타정보 가져오기
            # *********************************************
            cfgOpt = initCfgInfo(sysOpt)
            if cfgOpt is None or len(cfgOpt) < 1:
                log.error(f"cfgOpt : {cfgOpt} / DB 접속 정보를 확인해주세요.")
                exit(1)

            # *********************************************
            # [템플릿] 기본 위경도 정보를 DB 삽입
            # *********************************************
            # dbData = {}
            # # modelType = 'KIER-LDAPS'
            # modelType = 'KIER-RDPAS'
            # dbData['MODEL_TYPE'] = modelType
            #
            # # 지표
            # # orgData = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS/wrfsolar_d02.2023-06-30_03:00:00.nc')
            # # orgData = xr.open_mfdataset('/vol01/DATA/MODEL/KIER-LDAPS/wrfsolar_d02.2023-06-30_03:00:00.nc')
            # # orgData = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfsolar_d02.2023-06-30_04:00:00.nc')
            # orgData = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfsolar_d02.2023-06-30_04:00:00.nc')
            # data = orgData['SWDOWN'].isel(Time=0)
            #
            # dbData['LON_SFC'] = data['XLONG'].values.tolist() if len(data['XLONG'].values) > 0 else None
            # dbData['LAT_SFC'] = data['XLAT'].values.tolist() if len(data['XLAT'].values) > 0 else None
            #
            # # 상층
            # # orgData2 = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS/wrfout_d02_2023-06-30_03:00:00.nc')
            # # orgData2 = xr.open_mfdataset('/vol01/DATA/MODEL/KIER-LDAPS/wrfout_d02_2023-06-30_03:00:00.nc')
            # # orgData2 = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfout_d02_2023-06-30_04:00:00.nc')
            # orgData2 = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfout_d02_2023-06-30_04:00:00.nc')
            # data2 = orgData2['U'].isel(Time = 0, bottom_top = 0)
            # dbData['LON_PRE'] = data2['XLONG_U'].values.tolist() if len(data2['XLONG_U'].values) > 0 else None
            # dbData['LAT_PRE'] = data2['XLAT_U'].values.tolist() if len(data2['XLAT_U'].values) > 0 else None
            #
            # dbMergeData(cfgOpt['session'], cfgOpt['tbGeo'], dbData, pkList=['MODEL_TYPE'])
            #
            # # *********************************************
            # # [템플릿] 상세 위경도 정보를 DB 삽입
            # # *********************************************
            # sfcData = orgData['SWDOWN'].isel(Time=0).to_dataframe().reset_index(drop=False).rename(
            #     columns={
            #         'south_north': 'ROW'
            #         , 'west_east': 'COL'
            #         , 'XLAT': 'LAT_SFC'
            #         , 'XLONG': 'LON_SFC'
            #     }
            # ).drop(['SWDOWN'], axis='columns')
            #
            # preData = orgData2['U'].isel(Time = 0, bottom_top = 0).to_dataframe().reset_index(drop=False).rename(
            #     columns={
            #         'south_north': 'ROW'
            #         , 'west_east_stag': 'COL'
            #         , 'XLAT_U': 'LAT_PRE'
            #         , 'XLONG_U': 'LON_PRE'
            #     }
            # ).drop(['U', 'XTIME'], axis='columns')
            #
            # dataL2 = pd.merge(left=sfcData, right=preData, how='inner', left_on=['ROW', 'COL'], right_on=['ROW', 'COL'])
            # dataL2['MODEL_TYPE'] = modelType
            #
            # dataList = dataL2.to_dict(orient='records')
            # dbMergeData(cfgOpt['session'], cfgOpt['tbGeoDtl'], dataList, pkList=['MODEL_TYPE', 'ROW', 'COL'])

            # sys.exit(1)

            # *********************************************
            # [템플릿] KIER-LDPAS 테스트
            # *********************************************
            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(sysOpt['invHour']))

            # 분석 시간
            dtAnaDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(6))

            for modelIdx, modelType in enumerate(sysOpt['modelList']):
                log.info(f'[CHECK] modelType : {modelType}')

                # 동적 함수 호출
                # procFun = sysOpt['procList'][modelType]
                # log.info(f'[CHECK] procFun : {procFun}')
                # dynamicFun(procFun, modelType, dtDateList, sysOpt, cfgOpt)

                # 정적 함수 호출 : 모델 종류, 날짜 목록, 모델 정보, DB 정보
                readKierData(modelType, dtDateList, sysOpt, cfgOpt)
                # readTmpData(modelType, dtAnaDateList, sysOpt, cfgOpt)

        except Exception as e:
            log.error(f"Exception : {e}")
            raise e

        finally:
            log.info(f"[END] exec")


# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print(f'[START] main')

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
        print('[END] main')
