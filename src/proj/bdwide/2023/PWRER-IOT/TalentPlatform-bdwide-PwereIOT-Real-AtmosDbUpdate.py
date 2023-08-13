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
from key_generator.key_generator import generate
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

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import requests
import xmltodict
from pandas.tseries.offsets import Hour
from pandas.tseries.offsets import Day, Hour, Minute, Second
from sqlalchemy.dialects.mysql import insert
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Float
from sqlalchemy.dialects.mysql import DOUBLE
from sqlalchemy import Table, Column, Integer, String, MetaData
from sqlalchemy import Column, Numeric
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.dialects.mysql import insert as mysql_insert
import googlemaps
import json

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
        #fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        #fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        #plt.rcParams['font.family'] = fontName

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


def initCfgInfo(sysOpt, sysPath):

    log.info('[START] {}'.format('initCfgInfo'))

    result = None

    try:

        config = configparser.ConfigParser()
        config.read(sysPath, encoding='UTF-8')

        # 구글 API 정보
        gmap = googlemaps.Client(key=config.get('googleApi', 'key'))

        # 공공데이터포털 API
        apiKey = config.get('dataApi-obs', 'key')

        # DB 정보
        configKey = 'mysql-clova-dms02'
        dbUser = config.get(configKey, 'user')
        dbPwd = quote_plus(config.get(configKey, 'pwd'))
        dbHost = 'localhost' if config.get(configKey, 'host') == sysOpt['updIp'] else config.get(configKey, 'host')
        dbPort = config.get(configKey, 'port')
        dbName = config.get(configKey, 'dbName')

        sqlDbUrl = f'mysql+pymysql://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}'

        engine = create_engine(sqlDbUrl, echo=False, pool_size=20, max_overflow=0)
        sessMake = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = sessMake()
        # session.execute("""SELECT * FROM TB_VIDEO_INFO""").fetchall()

        # 테이블 생성
        selYear = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d').strftime('%Y')
        selDbTable = f'TB_OUTPUT_DATA_{selYear}'

        # 테이블 생성
        session.execute(
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
            """.format(selDbTable, selYear, selYear)
        )

        session.commit()

        # 테이블 정보
        metaData = MetaData()

        # 테이블 가져오기
        tbMember = Table('TB_MEMBER', metaData, autoload_with=engine, schema=dbName)
        tbOutputData = Table(selDbTable, metaData, autoload_with=engine, schema=dbName)

        result = {
            'engine': engine
            , 'session': session
            , 'gmap': gmap
            , 'apiKey': apiKey
            , 'tbMember': tbMember
            , 'tbOutputData': tbOutputData
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('initCfgInfo'))

def dbMergeData(session, table, dataList, pkList=['ANA_DT', 'FOR_DT', 'MODEL_TYPE']):

    log.info(f'[START] dbMergeData')

    try:
        stmt = mysql_insert(table).values(dataList)
        updDict = {c.name: stmt.inserted[c.name] for c in stmt.inserted if c.name not in pkList}
        onConflictStmt = stmt.on_duplicate_key_update(**updDict)

        session.execute(onConflictStmt)
        session.commit()

    except Exception as e:
        session.rollback()
        log.error(f'Exception : {e}')

    finally:
        session.close()
        log.info(f'[END] dbMergeData')


def convAddrToGeo(gmap, data, column):

    log.info(f'[START] convAddrToGeo')
    result = None

    try:
        # 구글 위경도 변환
        addrList = set(data[column])

        matData = pd.DataFrame()
        for j, addrInfo in enumerate(addrList):

            # 초기값 설정
            matData.loc[j, column] = addrInfo
            matData.loc[j, '위도'] = None
            matData.loc[j, '경도'] = None

            try:
                rtnGeo = gmap.geocode(addrInfo, language='ko')
                if (len(rtnGeo) < 1): continue

                # 위/경도 반환
                matData.loc[j, '위도'] = rtnGeo[0]['geometry']['location']['lat']
                matData.loc[j, '경도'] = rtnGeo[0]['geometry']['location']['lng']

            except Exception as e:
                log.error(f"Exception : {e}")

        # addr를 기준으로 병합
        data = data.merge(matData, left_on=[column], right_on=[column], how='inner')

        result = {
            'msg': 'succ'
            , 'data': data
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        log.info(f'[END] convAddrToGeo')

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
    serviceName = 'BDWIDE2023'

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
                'srtDate': globalVar['srtDate']
                , 'endDate': globalVar['endDate']
                # 'srtDate': '2023-08-14 00:00'
                # , 'endDate': '2023-08-14 01:00'

                # 기상청_단기예보 ((구)_동네예보) 조회서비스 -
                , 'apiUrl': 'https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst'

                # 업로드 아이피
                , 'updIp': '223.130.134.136'
            }

            # 날짜 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            # dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Minute(10))
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(1))

            # DB 정보
            cfgInfo = initCfgInfo(sysOpt, f"{globalVar['cfgPath']}/system.cfg")
            if cfgInfo is None or len(cfgInfo) < 1:
                log.error(f"cfgInfo : {cfgInfo} / DB 접속 정보를 확인해주세요.")
                exit(1)

            session = cfgInfo['session']
            gmap = cfgInfo['gmap']
            apiKey = cfgInfo['apiKey']
            tbMember = cfgInfo['tbMember']
            tbOutputData = cfgInfo['tbOutputData']


            # ********************************************************************************
            # 주소 컬럼을 통해 위경도 변환
            # ********************************************************************************
            # sqlRes = session.query(tbMember)
            # memberList = pd.read_sql(sqlRes.statement, sqlRes.session.bind)
            #
            # for i, memberInfo in memberList.iterrows():
            #     log.info(memberInfo)

            memberData = pd.read_sql(session.query(tbMember).statement, session.bind)
            memberDataL1 = memberData[memberData['LAT'].isna() | memberData['LON'].isna()]

            if len(memberDataL1) > 0:
                convRes = convAddrToGeo(gmap, memberDataL1, 'ADDR')  # Replace 'address_column_name' with the actual column name for the address in your tbMember table

                for idx, row in convRes['data'].iterrows():
                    if pd.isna(row['위도']) or pd.isna(row['경도']): continue

                    session.query(tbMember).filter(tbMember.c.CUSTOMER_LINK_NUMBER == row['CUSTOMER_LINK_NUMBER']).update({
                        'LAT': row['위도']
                        , 'LON': row['경도']
                    })

                session.commit()

            # ********************************************************************************
            # 지점 정보
            # ********************************************************************************
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '기상청41_단기예보 조회서비스_오픈API활용가이드_격자_위경도(20230611).xlsx')
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1:
                log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요.')

            posData = pd.read_excel(fileList[0], sheet_name='최종 업데이트 파일_20230611')
            posDataL1 = posData.rename(
                columns={
                    '경도(초/100)': 'LON'
                    , '위도(초/100)': 'LAT'
                    , '격자 X': 'NX'
                    , '격자 Y': 'NY'
                }
            )

            # ********************************************************************************
            # 시공간 일치
            # ********************************************************************************
            memberData = pd.read_sql(session.query(tbMember).statement, session.bind)
            for i, memberInfo in memberData.iterrows():
                if pd.isna(memberInfo['LAT']) or pd.isna(memberInfo['LON']): continue

                posDataL1['dist'] = (posDataL1['LAT'] - memberInfo['LAT']) ** 2 + (posDataL1['LON'] - memberInfo['LON']) ** 2
                cloData = posDataL1[posDataL1['dist'] == posDataL1['dist'].min()]

                nx = cloData['NX'].values[0]
                ny = cloData['NY'].values[0]

                # ********************************************************************************
                # 자료 수집
                # ********************************************************************************
                # dtIncDateInfo = dtIncDateList[23]
                for j, dtIncDateInfo in enumerate(dtIncDateList):
                    log.info(f'[CHECK] {memberInfo["CUSTOMER_LINK_NUMBER"]} : {dtIncDateInfo}')

                    # dtYmdHm = dtIncDateInfo.strftime('%Y-%m-%d %H:%M')
                    dtYmd = dtIncDateInfo.strftime('%Y%m%d')
                    dtHm = dtIncDateInfo.strftime('%H%M')

                    try:
                        apiUrl = sysOpt['apiUrl']

                        inParams = {'serviceKey': apiKey, 'pageNo': '1', 'numOfRows': '1000', 'dataType': 'JSON', 'base_date': dtYmd, 'base_time': dtHm, 'nx': nx, 'ny': ny}
                        apiParams = urllib.parse.urlencode(inParams).encode('UTF-8')

                        res = requests.get(apiUrl, params=apiParams)

                        resCode = res.status_code
                        if resCode != 200: continue

                        # json 읽기
                        # resData = json.loads(res.read().decode('utf-8'))
                        resData = json.loads(res.content.decode('utf-8'))

                        # xml to json 읽기
                        # resData = xmltodict.parse(res.content.decode('utf-8'))

                        if resData.get('response') is None: continue

                        resultCode = resData['response']['header']['resultCode']
                        if resultCode != '00': continue

                        resBody = resData['response']['body']
                        totalCnt = int(resBody['totalCount'])
                        if (totalCnt < 1): break

                        itemList = resBody['items']['item']
                        # itemList = resBody['items']
                        if (len(itemList) < 1): break

                        data = pd.DataFrame.from_dict(itemList).pivot(index=['baseDate', 'baseTime', 'nx', 'ny'], columns='category', values='obsrValue').reset_index(drop=False)
                        dtYmdHm = pd.to_datetime(data['baseDate'].values[0] + '-' + data['baseTime'].values[0], format='%Y%m%d-%H%M')

                        selData = session.query(tbOutputData).filter(tbOutputData.c.CUSTOMER_LINK_NUMBER == memberInfo['CUSTOMER_LINK_NUMBER'], tbOutputData.c.DATE_TIME == dtYmdHm).all()

                        if len(selData) < 1:
                            stmt = mysql_insert(tbOutputData).values({
                                'CUSTOMER_LINK_NUMBER': memberInfo['CUSTOMER_LINK_NUMBER']
                                , 'DATE_TIME': dtYmdHm
                                , 'TEMP': float(data['T1H'].values[0])
                                , 'HMDTY': float(data['REH'].values[0])
                                , 'REG_DATE': datetime.now()
                            })

                            session.execute(stmt)

                        else:
                            session.query(tbOutputData).filter(tbOutputData.c.CUSTOMER_LINK_NUMBER == memberInfo['CUSTOMER_LINK_NUMBER'], tbOutputData.c.DATE_TIME == dtYmdHm).update({
                                'TEMP': float(data['T1H'].values[0])
                                , 'HMDTY': float(data['REH'].values[0])
                                , 'MOD_DATE': datetime.now()
                            })

                        log.info(f'[CHECK] data : {data}')
                        session.commit()

                    except Exception as e:
                        log.error("Exception : {}".format(e))

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