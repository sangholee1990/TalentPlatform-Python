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
from _ast import expr
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc
import random
import time
import seaborn as sns
from paho.mqtt import client as mqtt_client
from paho.mqtt.enums import CallbackAPIVersion
from time import sleep

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
import configparser
from urllib.parse import quote_plus
import pytz
import pymysql
from sqlalchemy.dialects.mysql import insert as mysql_insert
import asyncio
import websockets
import os

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
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'output', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , 'sysCfg': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.json')
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

        # 글꼴 설정
        # fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        # fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        # plt.rcParams['font.family'] = fontName

    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        globalVar[key] = val

    # 전역 변수
    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)

        globalVar[key] = val.replace('\\', '/')

        log.info(f"[CHECK] {key} : {val}")

    return globalVar

def initCfgInfo(sysOpt, sysPath):
    log.info('[START] {}'.format('initCfgInfo'))

    result = None

    try:
        # DB 연결 정보
        pymysql.install_as_MySQLdb()

        # DB 정보
        config = configparser.ConfigParser()
        config.read(sysPath, encoding='utf-8')

        configKey = 'mysql-iwinv-dms01user01'
        dbUser = config.get(configKey, 'user')
        dbPwd = quote_plus(config.get(configKey, 'pwd'))
        dbHost = config.get(configKey, 'host')
        dbHost = 'localhost' if (sysOpt.get('updIp') is None or dbHost == sysOpt['updIp']) else dbHost
        dbPort = config.get(configKey, 'port')
        dbName = config.get(configKey, 'dbName')

        engine = create_engine(f"mysql+pymysql://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}", echo=False)
        # engine = create_engine('mariadb://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
        sessMake = sessionmaker(bind=engine)
        session = sessMake()
        # session.execute("""SELECT * FROM TB_VIDEO_INFO""").fetchall()

        # 메타정보
        metadata = MetaData()

        # 테이블 생성
        tbBeeIot = Table(
            'TB_BEE_IOT',
            metadata,
            Column('MES_DT', String(14), primary_key=True, comment="측정 시간")
            , Column('TMP', Float, comment="온도 (섭씨)")
            , Column('HUM', Float, comment="습도 (%)")
            , Column('CO2', Float, comment="CO2 (ppm)")
            , Column('WEG', Float, comment="무게 (g)")
            , Column('BAT', Float, comment="배터리 (%)")
            , Column('REG_DATE', DateTime, default=datetime.now(pytz.timezone('Asia/Seoul')), nullable=False, comment="등록일")
            , Column('MOD_DATE', DateTime, default=datetime.now(pytz.timezone('Asia/Seoul')), onupdate=datetime.now(pytz.timezone('Asia/Seoul')), nullable=True, comment="수정일")
            , extend_existing=True
        )

        metadata.create_all(engine)

        result = {
            'engine': engine
            , 'session': session
            , 'tbBeeIot': tbBeeIot
        }

        return result

    except Exception as e:
        log.error('Exception : {}'.format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('initCfgInfo'))

def dbMergeData(session, table, dataList, pkList):
    try:
        stmt = mysql_insert(table).values(dataList)
        updDict = {c.name: stmt.inserted[c.name] for c in stmt.inserted if c.name not in pkList}
        onConflictStmt = stmt.on_duplicate_key_update(**updDict)

        session.execute(onConflictStmt)
        session.commit()

    except Exception as e:
        session.rollback()
        log.error(f"Exception : {str(e)}")

    finally:
        session.close()

def onCon(client, userdata, flags, rc, properties=None):
    if rc == 0:
        log.info(f"Successfully connected to MQTT Broker")
    else:
        log.error(f"Failed to connect to MQTT Broker, return code : {rc}")


def onMsg(client, sysOpt, msg):

    try:
        # 센서 데이터
        if msg.topic == 'topic/mqtt':
            # 연월일시분초,온도,습도,co2,무게,배터리
            # msgStr = '241203004305,24.43,38.10,-2,1979,0.00'
            msgStr = msg.payload.decode("UTF-8")
            if msgStr is None or len(msgStr) < 1: return
            log.info(f"[CHECK] msgStr : {msgStr}")

            msgSplit = msgStr.split(',')
            if len(msgSplit) != 6: return

            colList = ['MES_DT', 'TMP', 'HUM', 'CO2', 'WEG', 'BAT']
            msgData = pd.DataFrame([msgSplit], columns=colList)

            msgData['MES_DT'] = pd.to_datetime(msgData['MES_DT'], format='%y%m%d%H%M%S').dt.strftime('%Y%m%d%H%M%S')
            msgData['TMP'] = msgData['TMP'].astype(float)
            msgData['HUM'] = msgData['HUM'].astype(float)
            msgData['CO2'] = msgData['CO2'].astype(float)
            msgData['WEG'] = msgData['WEG'].astype(float)
            msgData['BAT'] = msgData['BAT'].astype(float)

            dbMergeData(sysOpt['db']['session'], sysOpt['db']['table']['tbBeeIot'], msgData.to_dict(orient='records'), pkList=['MES_DT'])

        # 오디오 데이터
        # if msg.topic == 'topic/audio':

        # 비디오 데이터
        # if msg.topic == 'topic/video':

    except Exception as e:
        log.error(f"Exception : {str(e)}")

def mqttCon(sysOpt) -> mqtt_client.Client:
    try:
        # client = mqtt_client.Client(client_id=sysOpt['client_id'], protocol=sysOpt['callback_api_version'])
        client = mqtt_client.Client(client_id=sysOpt['client_id'], protocol=sysOpt['protocol'], callback_api_version=sysOpt['callback_api_version'])
        client.on_connect = onCon
        client.connect(sysOpt['broker'], sysOpt['port'])
    except Exception as e:
        log.error(f"Exception : {str(e)}")
        raise
    return client

def subscribe(client: mqtt_client.Client, sysOpt):
    try:
        for topic in sysOpt['topicList']:
            client.subscribe(topic)
            log.info(f"[CHECK] topic : {topic}")

        client.user_data_set(sysOpt)
        client.on_message = lambda client, userdata, msg: onMsg(client, userdata, msg)

    except Exception as e:
        log.error(f"Exception : {str(e)}")
        raise

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 메시지 mqtt 메시지 구독

    # conda activate py38
    # cd /SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2024/BEE-IOT
    # nohup python TalentPlatform-BDWIDE2024-beeIotSub.py &
    # tail -f nohup.out

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'BDWIDE2024'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info('[START] {}'.format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("init"))

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
                globalVar['cfgPath'] = '/SYSTEMS/PROG/PYTHON/IDE/resources/config'

            # 옵션 설정
            sysOpt = {
                # 'broker': "localhost"
                'broker': "49.247.41.71"
                , 'port': 1883
                , 'topicList': [
                    "topic/mqtt"
                ]
                , 'client_id': f"pub-{random.randint(0, 1000)}"
                , 'callback_api_version': CallbackAPIVersion.VERSION2
                , 'protocol': mqtt_client.MQTTv311
                , 'db': {
                    'engine': None
                    , 'session': None
                    , 'table': {
                        'tbBeeIot': None
                    }
                }
            }

            # DB
            cfgInfo = initCfgInfo(sysOpt, f"{globalVar['cfgPath']}/system.cfg")
            sysOpt['db']['engine'] = cfgInfo['engine']
            sysOpt['db']['session'] = cfgInfo['session']
            sysOpt['db']['table']['tbBeeIot'] = cfgInfo['tbBeeIot']

            # 메시지 구독
            try:
                client = mqttCon(sysOpt)
                subscribe(client, sysOpt)
                client.loop_forever()
            except Exception as e:
                log.error(f"Exception : {str(e)}")

        except Exception as e:
            log.error(f"Exception : {str(e)}")
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
