# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from urllib.parse import quote_plus
import configparser
import pymysql
import zipfile
import requests
import pytz
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
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'output', prjName)
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

        log.info("[CHECK] {} : {}".format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar

# def getPublicIp():
#     response = requests.get('https://api.ipify.org')
#     return response.text

def initCfgInfo(sysOpt, sysPath):
    log.info('[START] {}'.format('initCfgInfo'))

    result = None

    try:
        # DB 연결 정보
        # pymysql.install_as_MySQLdb()

        # DB 정보
        config = configparser.ConfigParser()
        config.read(sysPath, encoding='UTF-8')

        configKey = 'mysql-clova-dms02user01'
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

        # 테이블 정보
        metaData = MetaData()

        tbRsdDown = Table('TB_RSD_DOWN', metaData, autoload_with=engine, schema=dbName)
        tbRsdInfo = Table('TB_RSD_INFO', metaData, autoload_with=engine, schema=dbName)

        result = {
            'engine': engine
            , 'session': session
            , 'tbRsdDown': tbRsdDown
            , 'tbRsdInfo': tbRsdInfo
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
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

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 대한민국 성별, 연령별, 거주인구 전처리 및 구글 스튜디오 시각화

    # MySQL 테이블 내 데이터 삭제
    # TRUNCATE TABLE TB_RSD_DOWN;
    # TRUNCATE TABLE TB_RSD_INFO;

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
    serviceName = 'LSH0452'

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
                # 업로드 아이피 및 포트
                'updIp': '223.130.134.136'
                , 'updPort': '9000'
                ,  'colList' : ['GID', 'SIDO', 'SIGUNGU', 'TOWN', 'YEAR', 'CNT', 'AGE', 'SEX', 'LAT', 'LON']
            }

            # DB 정보
            cfgInfo = initCfgInfo(sysOpt, f"{globalVar['cfgPath']}/system.cfg")
            if cfgInfo is None or len(cfgInfo) < 1:
                log.error(f"cfgInfo : {cfgInfo} / DB 접속 정보를 확인해주세요.")
                exit(1)

            # *********************************************************************************
            # 파일 검색
            # *********************************************************************************
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '2023년 4월 대한민국 거주인구.csv')
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*.csv')
            fileList = sorted(glob.glob(inpFile))

            # fileInfo = fileList[1]
            data = pd.DataFrame()
            for i, fileInfo in enumerate(fileList):
                log.info(f'[CHECK] fileInfo: {fileInfo}')

                fileNameNoExt = os.path.basename(fileInfo).split('.')[0]
                tmpData = pd.read_csv(fileInfo, encoding='EUC-KR')
                tmpData['YEAR'] = int(fileNameNoExt.split('년')[0])
                tmpData['시도'] = tmpData['시도'].replace('강원도', '강원특별자치도')

                try:
                    tmpData.drop(['layer', 'path'], inplace=True, axis=1)
                    # tmpData['CNT'] = tmpData[sysOpt['colList']].sum(skipna=True, axis=1)
                except Exception as e:
                    log.error("Exception : {}".format(e))


                tmpDataL1 = pd.melt(tmpData, id_vars=['gid', '성별', '연도', '시도', '시군구', '읍면동', 'x', 'y', 'YEAR'], var_name='AGE', value_name='CNT')
                tmpDataL2 = tmpDataL1.dropna().reset_index(drop=True)

                data = pd.concat([data, tmpDataL2], ignore_index=True)

            dataL1 = data.rename(
                {
                    'gid': 'GID'
                    , '시도': 'SIDO'
                    , '시군구': 'SIGUNGU'
                    , '읍면동': 'TOWN'
                    , '성별': 'SEX'
                    , 'y': 'LAT'
                    , 'x': 'LON'
                }
                , axis=1
            )

            # 자료 필터
            dataL1 = dataL1.loc[dataL1['SEX'] != '꼈?']

            # *******************************************************************
            # 상세정보 가공
            # *******************************************************************
            # dataL2 = dataL1.loc[dataL1['SIDO'] == '서울시']
            dataL2 = dataL1

            colList = sysOpt['colList']
            dataL3 = dataL2[colList].reset_index(drop=True)

            # 중복 검사
            dataL4 = dataL3.groupby(['GID', 'SIDO', 'SIGUNGU', 'TOWN', 'YEAR', 'AGE', 'SEX', 'LAT', 'LON'])['CNT'].sum().reset_index()

            dbData = dataL4[colList]
            dbData['REG_DATE'] = datetime.now(pytz.timezone('Asia/Seoul'))

            # 신규 컬럼
            ordList = ['유아', '유소년', '초등학생', '중학생', '고등학생', '20대', '30대', '40대', '50대', '60대', '70대', '80대', '90대', '100세이상']
            dbData['ORD'] = dbData['AGE'].apply(lambda x: ordList.index(x) if x in ordList else None)
            dbData['GEO'] = dbData['LAT'].astype(str) + ", " + dbData['LON'].astype(str)
            dbData['YEAR_DATE'] = pd.to_datetime(dbData['YEAR'], format='%Y')

            saveFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'TB_RSD_INFO')
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            dbData.to_csv(saveFile, index=False)
            print(f'[CHECK] saveFile : {saveFile}')

            # 1건으로 처리
            # dataList = dbData.to_dict(orient='records')
            # dbMergeData(cfgInfo['session'], cfgInfo['tbRsdInfo'], dataList, pkList=[''])

            # 10만건으로 분할 처리
            # chunkSize = 100000
            # for i in range(0, len(dbData), chunkSize):
            #     log.info(f'[CHECK] i : {i}')
            #     dataList = dbData[i:i + chunkSize].to_dict(orient='records')
            #     dbMergeData(cfgInfo['session'], cfgInfo['tbRsdInfo'], dataList, pkList=[''])

            # *******************************************************************
            # 기본정보 가공
            # *******************************************************************
            dbDataL1 = pd.DataFrame()
            typeList = sorted(set(dataL1['SIDO']))
            for i, typeInfo in enumerate(typeList):
                log.info(f'[CHECK] typeInfo : {typeInfo}')

                selData = dataL1.loc[dataL1['SIDO'] == typeInfo].reset_index(drop=True)

                saveFile = '{}/{}/{}.csv'.format(globalVar['updPath'], '거주인구', typeInfo)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                selData.to_csv(saveFile, index=False)
                log.info(f'[CHECK] saveFile : {saveFile}')

                dbData = pd.DataFrame(
                    {
                        'TYPE': [typeInfo]
                        , 'CSV_INFO': [f"http://{sysOpt['updIp']}:{sysOpt['updPort']}/CSV{saveFile.replace(globalVar['updPath'], '')}"]
                    }
                )
                dbData['REG_DATE'] = datetime.now(pytz.timezone('Asia/Seoul'))
                dbDataL1 = pd.concat([dbDataL1, dbData], ignore_index=True)

            # 1건으로 처리
            dataList = dbDataL1.to_dict(orient='records')
            dbMergeData(cfgInfo['session'], cfgInfo['tbRsdDown'], dataList, pkList=[''])

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
