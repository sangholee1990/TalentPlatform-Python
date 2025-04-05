import argparse
from datetime import datetime
from datetime import timedelta
import glob
import pandas as pd
# import seaborn as sns
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from builtins import enumerate

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pytz
from matplotlib import font_manager
import pymysql
import configparser
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pymysql
import random
from urllib.parse import quote_plus
from urllib.parse import unquote_plus
import urllib.parse
import sqlalchemy
from sqlalchemy import create_engine, text
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
# font_manager._rebuild()

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
def initArgument(globalVar):
    parser = argparse.ArgumentParser()

    for i, argv in enumerate(sys.argv[1:]):
        if not argv.__contains__('--'): continue
        parser.add_argument(argv)

    inParInfo = vars(parser.parse_args())
    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        if env not in 'local' and key.__contains__('Path'):
            os.makedirs(val, exist_ok=True)
        globalVar[key] = val

    return globalVar


def initCfgInfo(sysPath, key):

    # log.info('[START] {}'.format('initCfgInfo'))
    result = None

    try:

        # DB 연결 정보
        pymysql.install_as_MySQLdb()

        log.info(f'[CHECK] sysPath : {sysPath}')
        log.info(f'[CHECK] key : {key}')

        # DB 정보
        config = configparser.ConfigParser()
        config.read(sysPath, encoding='utf-8')
        dbUser = config.get(key, 'user')
        dbPwd = urllib.parse.quote(config.get(key, 'pwd'))
        dbHost = config.get(key, 'host')
        # dbHost = 'localhost'
        dbPort = config.get(key, 'port')
        dbName = config.get(key, 'dbName')

        # dbEngine = create_engine('mysql://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
        # dbEngine = sqlalchemy.create_engine(f'mysql://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}')
        # dbEngine = sqlalchemy.create_engine(f'mysql://{dbUser}:{str_db_password}@{dbHost}:{dbPort}/{dbName}')
        # dbEngine = create_engine(f'mysql://{dbUser}:{aa}@{dbHost}:{dbPort}/{dbName}')
        dbEngine = sqlalchemy.create_engine('mysql://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)

        # testData = pd.read_sql('select * from TB_COMM_CODE', con=dbEngine)

        sessMake = sessionmaker(bind=dbEngine)
        session = sessMake()

        # API 정보
        # apiUrl = config.get('pv', 'url')
        # apiToken = config.get('pv', 'token')

        result = {
            'dbEngine': dbEngine
            , 'session': session
            # , 'apiUrl': apiUrl
            # , 'apiToken': apiToken
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    # finally:
    #     # try, catch 구문이 종료되기 전에 무조건 실행
    #     log.info('[END] {}'.format('initCfgInfo'))


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 한국 월별 수출액과 코스피 종간 간의 회귀분석

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
    serviceName = 'PRJ2022'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info("[START] __init__ : {}".format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

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
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                # 'srtDate': '2023-01-01 00:00',
                # 'endDate': '2023-06-15 00:00',
                'srtDate': '2025-01-01 00:00',
                'endDate': '2026-01-01 00:00',

                # 설정 파일
                'sysPath': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
            }

            # ********************************************************************************
            # 제품 시리얼 생성
            # ********************************************************************************
            # data = pd.DataFrame()
            # for i in range(0, 1000):
            #     log.info('[CHECK] i : {}'.format(i))
            #
            #     dtDateTime = datetime.now()
            #     serialNumber = 'BDWIDE-{}'.format(generate(seed=dtDateTime).get_key())
            #
            #     dict = {
            #         'PRODUCT_SERIAL_NUMBER' : [serialNumber]
            #         , 'AUTH_YN' : ['Y']
            #         , 'USE_YN' : ['Y']
            #         , 'MOD_DATE' : [dtDateTime.strftime('%Y-%m-%d %H:%M:%S')]
            #         , 'REG_DATE' : [dtDateTime.strftime('%Y-%m-%d %H:%M:%S')]
            #     }
            #
            #     data = pd.concat([data, pd.DataFrame.from_dict(dict)], ignore_index=True)
            #
            # saveFile = '{}/{}/SerialNumber_{}.xlsx'.format(globalVar['outPath'], serviceName, datetime.now().strftime('%Y%m%d'))
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # data.to_excel(saveFile, index=False)
            # log.info('[CHECK] saveFile : {}'.format(saveFile))

            # DB 접근
            cfgInfo = initCfgInfo(sysOpt['sysPath'], 'mysql-iwin-dms01user01')

            dbEngine = cfgInfo['dbEngine']
            session = cfgInfo['session']

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d %H:%M')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d %H:%M')
            # dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1D')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1H')
            dtIncMonthList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1M')

            for i, dtIncDateInfo in enumerate(dtIncDateList):
                log.info("[CHECK] dtIncDateInfo : {}".format(dtIncDateInfo))

                # ********************************************************************************
                # TB_INPUT_DATA 생성
                # ********************************************************************************
                # proSerNumList = ['BDWIDE-0033f-05a3776796-89ff44-b7b3ec0-d30403e426', 'BDWIDE-820b897da-c8384be-d793ded9-1bc8d8-0f1260de']
                proSerNumList = ['BDWIDE-820b897da-c8384be-d793ded9-1bc8d8-0f1260de']
                for j, proSerNumInfo in enumerate(proSerNumList):
                    log.info(f'[CHECK] proSerNumInfo : {proSerNumInfo}')

                    dict = {
                        'PRODUCT_SERIAL_NUMBER': [proSerNumInfo]
                        , 'DATE_TIME': [dtIncDateInfo.strftime('%Y-%m-%d %H:%M:%S')]
                        , 'TEMP': [random.uniform(-10, 30)]
                        , 'HMDTY': [random.uniform(0, 100)]
                        , 'PM25': [random.uniform(0, 500)]
                        , 'PM10': [random.uniform(0, 500)]
                        , 'MVMNT': [round(random.uniform(0, 1))]
                        , 'TVOC': [random.uniform(0.02, 2000)]
                        , 'HCHO': [random.uniform(0.01, 50)]
                        , 'CO2': [random.uniform(350, 5000)]
                        , 'CO': [random.uniform(0, 5000)]
                        , 'BENZO': [random.uniform(0, 5)]
                        , 'RADON': [random.uniform(20, 9990)]
                        , 'MOD_DATE': [datetime.now()]
                        , 'REG_DATE': [datetime.now()]
                    }

                    dbData = pd.DataFrame.from_dict(dict)

                    selDbTable = 'TB_INPUT_DATA_{}'.format(dtIncDateInfo.strftime('%Y'))

                    # 테이블 생성
                    session.execute(text(
                        """
                        CREATE TABLE IF NOT EXISTS `{}`
                        (
                            PRODUCT_SERIAL_NUMBER varchar(63) not null comment '제품 시리얼 번호',
                            DATE_TIME             datetime    not null comment '날짜 시간 UTC 기준' primary key,
                            TEMP                  float       null comment '온도',
                            HMDTY                 float       null comment '습도',
                            PM25                  float       null comment '미세먼지2.5',
                            PM10                  float       null comment '미세먼지10',
                            MVMNT                 varchar(20) null comment '움직임 센서',
                            TVOC                  float       null comment '총 휘발성 유기화합물',
                            HCHO                  float       null comment '포름알데하이드',
                            CO2                   float       null comment '이산화탄소',
                            CO                    float       null comment '일산화탄소',
                            BENZO                 float       null comment '벤조피렌',
                            RADON                 float       null comment '라돈',
                            MOD_DATE              datetime    null comment '수정일',
                            REG_DATE              datetime    null comment '등록일',
                            TMP                   float       null comment '실수형 임시 변수',
                            TMP2                  float       null comment '실수형 임시 변수2',
                            TMP3                  float       null comment '실수형 임시 변수3',
                            TMP4                  float       null comment '실수형 임시 변수4',
                            TMP5                  float       null comment '실수형 임시 변수5',
                            TMP6                  varchar(20) null comment '문자형 임시 변수',
                            TMP7                  varchar(20) null comment '문자형 임시 변수2',
                            TMP8                  varchar(20) null comment '문자형 임시 변수3',
                            TMP9                  varchar(20) null comment '문자형 임시 변수4',
                            TMP10                 varchar(20) null comment '문자형 임시 변수5'
                        )
                            comment 'IOT 테이블 {}';
                        """.format(selDbTable, dtIncDateInfo.strftime('%Y'))
                    ))
                    session.commit()

                    # for k, dbInfo in dbData.iterrows():
                    #
                    #     # 테이블 PK키를 통해 삽입/수정
                    #     session.execute(text(
                    #         """
                    #         INSERT INTO `{}` (PRODUCT_SERIAL_NUMBER, DATE_TIME, TEMP, HMDTY, PM25, PM10, MVMNT, TVOC, HCHO, CO2, CO, BENZO, RADON, REG_DATE, MOD_DATE)
                    #         VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')
                    #         ON DUPLICATE KEY UPDATE
                    #             PRODUCT_SERIAL_NUMBER = VALUES(PRODUCT_SERIAL_NUMBER)
                    #             , TEMP = VALUES(TEMP)
                    #             , HMDTY = VALUES(HMDTY)
                    #             , PM25 = VALUES(PM25)
                    #             , PM10 = VALUES(PM10)
                    #             , MVMNT = VALUES(MVMNT)
                    #             , TVOC = VALUES(TVOC)
                    #             , HCHO = VALUES(HCHO)
                    #             , CO2 = VALUES(CO2)
                    #             , CO = VALUES(CO)
                    #             , BENZO = VALUES(BENZO)
                    #             , RADON = VALUES(RADON)
                    #             , MOD_DATE = VALUES(MOD_DATE)
                    #             ;
                    #             """.format(selDbTable, dbInfo['PRODUCT_SERIAL_NUMBER'], dbInfo['DATE_TIME'], dbInfo['TEMP'], dbInfo['HMDTY'], dbInfo['PM25'], dbInfo['PM10']
                    #                   , dbInfo['MVMNT'], dbInfo['TVOC'], dbInfo['HCHO'], dbInfo['CO2'], dbInfo['CO'], dbInfo['BENZO'], dbInfo['RADON'], dbInfo['REG_DATE'], dbInfo['MOD_DATE'])
                    #     ))
                    #
                    #     session.commit()

                # ********************************************************************************
                # TB_OUTPUT_DATA 생성
                # ********************************************************************************
                # customerLinkNumList = [1, 66]
                customerLinkNumList = [4]
                for j, customerLinkNumInfo in enumerate(customerLinkNumList):
                    log.info(f'[CHECK] customerLinkNumInfo : {customerLinkNumInfo}')

                    dict = {
                        'CUSTOMER_LINK_NUMBER': [customerLinkNumInfo]
                        , 'DATE_TIME': [dtIncDateInfo.strftime('%Y-%m-%d %H:%M:%S')]
                        , 'TEMP': [random.uniform(-10, 30)]
                        , 'HMDTY': [random.uniform(0, 100)]
                        , 'PM25': [random.uniform(0, 500)]
                        , 'PM10': [random.uniform(0, 500)]
                        , 'DUST': [random.uniform(0, 500)]
                        , 'CO2': [random.uniform(350, 5000)]
                        , 'PWR': [random.uniform(0, 5)]
                        , 'GAS': [random.uniform(0, 5)]
                        , 'WATER': [random.uniform(0, 5)]
                        , 'PRD_PWR': [random.uniform(0, 5)]
                        , 'PRD_GAS': [random.uniform(0, 5)]
                        , 'PRD_WATER': [random.uniform(0, 5)]
                        , 'MOD_DATE': [datetime.now()]
                        , 'REG_DATE': [datetime.now()]
                    }

                    dbData = pd.DataFrame.from_dict(dict)

                    selDbTable = 'TB_OUTPUT_DATA_{}'.format(dtIncDateInfo.strftime('%Y'))

                    # 테이블 생성
                    session.execute(text(
                        """
                        CREATE TABLE IF NOT EXISTS `{}`
                        (
                            CUSTOMER_LINK_NUMBER int(20)  not null comment '고객 연동 번호',
                            DATE_TIME            datetime not null comment '날짜 시간 UTC 기준' primary key,
                            TEMP                 float    null comment '온도',
                            HMDTY                float    null comment '습도',
                            PM25                 float    null comment '미세먼지2.5',
                            PM10                 float    null comment '미세먼지10',
                            DUST                 float    null comment '황사',
                            CO2                  float    null comment '이산화탄소',
                            PWR                  float    null comment '전력량',
                            GAS                  float    null comment '가스량',
                            WATER                float    null comment '수도량',
                            PRD_PWR              float    null comment '예측 전력량',
                            PRD_GAS              float    null comment '예측 가스량',
                            PRD_WATER            float    null comment '예측 수도량',
                            MOD_DATE             datetime null comment '수정일',
                            REG_DATE             datetime null comment '등록일',
                            constraint TB_OUTPUT_DATA_{}_TB_MEMBER_null_fk
                                foreign key (CUSTOMER_LINK_NUMBER) references DMS02.TB_MEMBER (CUSTOMER_LINK_NUMBER)
                                    on update cascade on delete cascade
                        )
                            comment 'API 테이블 {}';
                        """.format(selDbTable, dtIncDateInfo.strftime('%Y'), dtIncDateInfo.strftime('%Y'))
                    ))
                    session.commit()

                    for k, dbInfo in dbData.iterrows():

                        # 테이블 PK키를 통해 삽입/수정
                        session.execute(text(
                            """
                            INSERT INTO `{}` (CUSTOMER_LINK_NUMBER, DATE_TIME, TEMP, HMDTY, PM25, PM10, DUST, CO2, PWR, GAS, WATER, PRD_PWR, PRD_GAS, PRD_WATER, REG_DATE, MOD_DATE)
                            VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')
                            ON DUPLICATE KEY UPDATE
                                CUSTOMER_LINK_NUMBER = VALUES(CUSTOMER_LINK_NUMBER)
                                , TEMP = VALUES(TEMP)
                                , HMDTY = VALUES(HMDTY)
                                , PM25 = VALUES(PM25)
                                , PM10 = VALUES(PM10)
                                , DUST = VALUES(DUST)
                                , CO2 = VALUES(CO2)
                                , PWR = VALUES(PWR)
                                , GAS = VALUES(GAS)
                                , WATER = VALUES(WATER)
                                , PRD_PWR = VALUES(PRD_PWR)
                                , PRD_GAS = VALUES(PRD_GAS)
                                , PRD_WATER = VALUES(PRD_WATER)
                                , MOD_DATE = VALUES(MOD_DATE)
                                ;
                                """.format(selDbTable, dbInfo['CUSTOMER_LINK_NUMBER'], dbInfo['DATE_TIME'], dbInfo['TEMP'], dbInfo['HMDTY'], dbInfo['PM25'], dbInfo['PM10']
                                      , dbInfo['DUST'], dbInfo['CO2'], dbInfo['PWR'], dbInfo['GAS'], dbInfo['WATER'], dbInfo['PRD_PWR'], dbInfo['PRD_GAS'], dbInfo['PRD_WATER'], dbInfo['REG_DATE'], dbInfo['MOD_DATE'])
                        ))
                        session.commit()

            # ********************************************************************************
            # TB_OUTPUT_STAT_DATA 생성
            # ********************************************************************************
            for i, dtIncMonthInfo in enumerate(dtIncMonthList):
                log.info("[CHECK] dtIncMonthInfo : {}".format(dtIncMonthInfo))

                # customerLinkNumList = [1, 66]
                customerLinkNumList = [4]
                for j, customerLinkNumInfo in enumerate(customerLinkNumList):
                    log.info(f'[CHECK] customerLinkNumInfo : {customerLinkNumInfo}')

                    dict = {
                        'CUSTOMER_LINK_NUMBER': [customerLinkNumInfo]
                        , 'DATE_TIME': [dtIncMonthInfo.strftime('%Y-%m-01 00:00:00')]
                        , 'PRV_PWR': [random.uniform(0, 5) * 30]
                        , 'PRV_WATER': [random.uniform(0, 5) * 30]
                        , 'PRV_GAS': [random.uniform(0, 5) * 30]
                        , 'PRD_PRV_PWR': [random.uniform(0, 5) * 30]
                        , 'PRD_PRV_GAS': [random.uniform(0, 5) * 30]
                        , 'PRD_PRV_WATER': [random.uniform(0, 5) * 30]
                        , 'PRE_PWR': [random.uniform(0, 5) * 30]
                        , 'PRE_GAS': [random.uniform(0, 5) * 30]
                        , 'PRE_WATER': [random.uniform(0, 5) * 30]
                        , 'PRD_PRE_PWR': [random.uniform(0, 5) * 30]
                        , 'PRD_PRE_GAS': [random.uniform(0, 5) * 30]
                        , 'PRD_PRE_WATER': [random.uniform(0, 5) * 30]
                        , 'MOD_DATE': [datetime.now()]
                        , 'REG_DATE': [datetime.now()]
                    }

                    dbData = pd.DataFrame.from_dict(dict)

                    selDbTable = 'TB_OUTPUT_STAT_DATA_{}'.format(dtIncMonthInfo.strftime('%Y'))

                    # 테이블 생성
                    session.execute(text(
                        """
                        CREATE TABLE IF NOT EXISTS `{}`
                        (
                            CUSTOMER_LINK_NUMBER int(20)  not null comment '고객 연동 번호',
                            DATE_TIME            datetime not null comment '날짜 시간 UTC 기준' primary key,
                            PRV_PWR              float    null comment '전월 전력량',
                            PRV_GAS              float    null comment '전월 가스량',
                            PRV_WATER            float    null comment '전월 수도량',
                            PRD_PRV_PWR          float    null comment '예측 전월 전력량',
                            PRD_PRV_GAS          float    null comment '예측 전월 가스량',
                            PRD_PRV_WATER        float    null comment '예측 전월 수도량',
                            PRE_PWR              float    null comment '당월 전력량',
                            PRE_GAS              float    null comment '당월 가스량',
                            PRE_WATER            float    null comment '당월 수도량',
                            PRD_PRE_PWR          float    null comment '예측 당월 전력량',
                            PRD_PRE_GAS          float    null comment '예측 당월 가스량',
                            PRD_PRE_WATER        float    null comment '예측 당월 수도량',
                            MOD_DATE             datetime null comment '수정일',
                            REG_DATE             datetime null comment '등록일',
                            constraint TB_OUTPUT_STAT_DATA_{}_TB_MEMBER_null_fk
                                foreign key (CUSTOMER_LINK_NUMBER) references DMS02.TB_MEMBER (CUSTOMER_LINK_NUMBER)
                                    on update cascade on delete cascade
                        )
                            comment 'API 통계 테이블 {}';
                        """.format(selDbTable, dtIncMonthInfo.strftime('%Y'), dtIncMonthInfo.strftime('%Y'))
                    ))
                    session.commit()

                    for k, dbInfo in dbData.iterrows():
                        # 테이블 PK키를 통해 삽입/수정
                        session.execute(text(
                            """
                            INSERT INTO `{}` (CUSTOMER_LINK_NUMBER, DATE_TIME, PRV_PWR, PRV_GAS, PRV_WATER, PRD_PRV_PWR, PRD_PRV_GAS, PRD_PRV_WATER, PRE_PWR, PRE_GAS, PRE_WATER, PRD_PRE_PWR, PRD_PRE_GAS, PRD_PRE_WATER, REG_DATE, MOD_DATE)
                            VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')
                            ON DUPLICATE KEY UPDATE
                                CUSTOMER_LINK_NUMBER = VALUES(CUSTOMER_LINK_NUMBER)
                                , PRV_PWR = VALUES(PRV_PWR)
                                , PRV_GAS = VALUES(PRV_GAS)
                                , PRV_WATER = VALUES(PRV_WATER)
                                , PRD_PRV_PWR = VALUES(PRD_PRV_PWR)
                                , PRD_PRV_GAS = VALUES(PRD_PRV_GAS)
                                , PRD_PRV_WATER = VALUES(PRD_PRV_WATER)
                                , PRE_PWR = VALUES(PRE_PWR)
                                , PRE_GAS = VALUES(PRE_GAS)
                                , PRE_WATER = VALUES(PRE_WATER)
                                , PRD_PRE_PWR = VALUES(PRD_PRE_PWR)
                                , PRD_PRE_GAS = VALUES(PRD_PRE_GAS)
                                , PRD_PRE_WATER = VALUES(PRD_PRE_WATER)
                                , MOD_DATE = VALUES(MOD_DATE)
                                ;
                                """.format(selDbTable, dbInfo['CUSTOMER_LINK_NUMBER'], dbInfo['DATE_TIME'], dbInfo['PRV_PWR'], dbInfo['PRV_GAS'], dbInfo['PRV_WATER'], dbInfo['PRD_PRV_PWR']
                                           , dbInfo['PRD_PRV_GAS'], dbInfo['PRD_PRV_WATER'], dbInfo['PRE_PWR'], dbInfo['PRE_GAS'], dbInfo['PRE_WATER'], dbInfo['PRD_PRE_PWR'], dbInfo['PRD_PRE_GAS'], dbInfo['PRD_PRE_WATER'], dbInfo['REG_DATE'], dbInfo['MOD_DATE'])
                        ))
                        session.commit()

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
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
