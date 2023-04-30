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

        log.info("[CHECK] {} : {}".format(key, val))

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
                    'srtDate': '2021-10-01'
                    , 'endDate': '2021-10-02'
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

            dbEngine = create_engine('mariadb://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName),
                                     echo=False)
            sessMake = sessionmaker(bind=dbEngine)
            session = sessMake()

            # *******************************************************
            # 관측자료 읽기
            # *******************************************************
            inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST_20220123/PV/GA_Yield_2019_2022.xlsx')

            pvData = pd.DataFrame()
            for sheetInfo in range(2019, 2023):
                tmpData = pd.read_excel(inpFile, sheet_name=str(sheetInfo), engine='openpyxl')
                pvData = pd.concat([pvData, tmpData], ignore_index=False)

            # 이름 변경
            pvDataL1 = pvData.rename(
                columns={
                    'Vkey': 'id'
                    , 'localdate': 'time'
                    , 'totalYield': 'PV'
                }
            )

            dbData = pvDataL1
            for i, dbInfo in dbData.iterrows():
                dbInfo['SRV'] = 'SRV{:05d}'.format(dbInfo['id'])
                dbInfo['DATE_TIME_KST'] = pd.to_datetime(dbInfo['time'], format='%Y-%m-%d %H')
                dbInfo['DATE_TIME'] = dbInfo['DATE_TIME_KST'] - dtKst

                log.info("[CHECK] DATE_TIME : {}".format(dbInfo['DATE_TIME']))

                iYear = int(dbInfo['DATE_TIME_KST'].strftime("%Y"))
                selDbTable = 'TB_PV_DATA_{}'.format(iYear)

                resChk = session.execute(
                    """
                    SELECT COUNT(*) AS CNT FROM `{}`
                    WHERE SRV = '{}' AND DATE_TIME = '{}'
                    """.format(selDbTable, dbInfo['SRV'], dbInfo['DATE_TIME'])
                ).fetchone()

                if (resChk['CNT'] > 0):
                    dbInfo['MOD_DATE'] = datetime.datetime.now()

                    session.execute(
                        """
                        UPDATE `{}`
                        SET DATE_TIME_KST = '{}', PV = '{}', MOD_DATE = '{}'
                        WHERE SRV = '{}' AND DATE_TIME = '{}';
                        """.format(selDbTable
                                   , dbInfo['DATE_TIME_KST'], dbInfo['PV'], dbInfo['MOD_DATE']
                                   , dbInfo['SRV'], dbInfo['DATE_TIME'])
                    )
                else:
                    dbInfo['REG_DATE'] = datetime.datetime.now()

                    session.execute(
                        """
                        INSERT INTO `{}` (SRV, DATE_TIME, DATE_TIME_KST, PV, REG_DATE)
                        VALUES ('{}', '{}', '{}', '{}', '{}')
                        """.format(selDbTable, dbInfo['SRV'], dbInfo['DATE_TIME'], dbInfo['DATE_TIME_KST'], dbInfo['PV'], dbInfo['REG_DATE'])
                    )

                session.commit()

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