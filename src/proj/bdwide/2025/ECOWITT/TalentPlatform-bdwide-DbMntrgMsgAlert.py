﻿# ================================================
# 요구사항
# ================================================
# Python을 이용한 데이터베이스 내부온도/외부온도/데이터 감시 및 텔레그램 알림 발송

# 프로그램 종료
# ps -ef | grep python | grep TalentPlatform-bdwide-DbMntrgMsgAlert.py | awk '{print $2}' | xargs kill -9
# pkill -f TalentPlatform-bdwide-DbMntrgMsgAlert.py

# 프로그램 시작
# conda activate py38
# cd /SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT
# nohup /SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-bdwide-DbMntrgMsgAlert.py &
# nohup /SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-bdwide-DbMntrgMsgAlert.py > /dev/null 2>&1 &
# tail -f nohup.out

import argparse
import glob
import json
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import urllib.parse
import warnings
from builtins import enumerate
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import pytz
from datetime import timedelta
import configparser
import time

from dask.bag.text import delayed
from sqlalchemy.util import await_only
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import fnmatch
import re
import tempfile
import subprocess
import shutil
import asyncio

import argparse
import base64
import json
import os
import os.path as osp

import imgviz
import PIL.Image

# from labelme.logger import logger
from labelme import utils
from retrying import retry
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor
import threading
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from concurrent.futures import ProcessPoolExecutor, as_completed
import pymysql
import random
from urllib.parse import quote_plus
from urllib.parse import unquote_plus
import urllib.parse
import sqlalchemy
from sqlalchemy import create_engine, text
import requests
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import text

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

# plt.rc('font', family='Malgun Gothic')
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

    saveLogFile = "{}/{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
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
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1, backupCount=30, encoding='utf-8')

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

def objToDict(obj):
    result = None

    try:
        result = {
            column.key: getattr(obj, column.key)
            for column in obj.__table__.columns
        }

        return result
    except Exception as e:
        log.error(f'Exception : {e}')
        return result

#  초기 전달인자 설정
def initArgument(globalVar):
    parser = argparse.ArgumentParser()

    for i, argv in enumerate(sys.argv[1:]):
        if not argv.__contains__('--'): continue
        parser.add_argument(argv)

    inParInfo = vars(parser.parse_args())
    log.info(f"inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        if env not in 'local' and key.__contains__('Path'):
            os.makedirs(val, exist_ok=True)
        globalVar[key] = val

    return globalVar

# 벌통 내부 검사
def dbMntrgIndoor(sysOpt):
    try:
        query = text("""
                     WITH PRE_ECO_DATA AS (SELECT ECO.*,
                                                  dev.device_name,
                                                  ROW_NUMBER() OVER (PARTITION BY ECO.device_id ORDER BY ECO.tm DESC) AS rn
                                           FROM TB_ECOWITT_DATA AS ECO
                                                    LEFT OUTER JOIN
                                                TB_DEVICE_MASTER AS dev ON ECO.device_id = dev.device_id
                                           WHERE ECO.tm BETWEEN :srtDate AND :endDate
                                             AND ECO.indoor_temp <> -999)
                     SELECT tm          AS tm,
                            device_id   AS device_id,
                            device_name AS device_name,
                            indoor_temp AS indoor_temp,
                            CASE
                                WHEN indoor_temp >= 38.0 THEN '내부 고온 경보'
                                WHEN indoor_temp >= 35.0 THEN '내부 고온 주의보'
                                WHEN indoor_temp <= 6.0 THEN '내부 저온 주의보'
                                WHEN indoor_temp <= 0.0 THEN '내부 저온 경보'
                                END     AS state
                     FROM PRE_ECO_DATA
                     WHERE rn = 1;
                     """)

        endDate = datetime.now()
        srtDate = endDate - timedelta(minutes=sysOpt['mntrgMinInv'])

        with sysOpt['cfgDb']['sessionMake']() as session:
            dataList = session.execute(query, {"srtDate": srtDate, "endDate": endDate}).all()
            for dataInfo in dataList:
                if dataInfo.state is None: continue
                log.info(f"[CHECK] dataInfo : {dataInfo}")

                key = f"{dataInfo.state}-{dataInfo.device_id}-{dataInfo.device_name}"
                msgAlertDate = sysOpt['msgAlertHist'].get(key)
                if (msgAlertDate is None) or (endDate - msgAlertDate) >= timedelta(minutes=sysOpt['msgAlertMinInv']):
                    sendTgApi(sysOpt['cfgTg'], sysOpt['msgAlertTemplate'][dataInfo.state].format(device_id=dataInfo.device_id, device_name=dataInfo.device_name, indoor_temp=dataInfo.indoor_temp))
                    sysOpt['msgAlertHist'][key] = endDate
                    log.info(f"[CHECK] msgAlertHist : {sysOpt['msgAlertHist'].keys()}")
    except Exception as e:
        log.error(f'Exception : {e}')

# 벌통 외부 검사
def dbMntrgOutdoor(sysOpt):
    try:
        query = text("""
                     WITH PRE_ECO_DATA AS (SELECT ECO.*,
                                                  dev.device_name,
                                                  ROW_NUMBER() OVER (PARTITION BY ECO.device_id ORDER BY ECO.tm DESC) AS rn
                                           FROM TB_ECOWITT_DATA AS ECO
                                                    LEFT OUTER JOIN
                                                TB_DEVICE_MASTER AS dev ON ECO.device_id = dev.device_id
                                           WHERE ECO.tm BETWEEN :srtDate AND :endDate
                                             AND ECO.outdoor_temp <> -999
                                             AND ECO.outdoor_hmdty <> -999
                                             AND ECO.wind_speed <> -999),
                          CALC_TEMP_DATA AS (SELECT tm,
                         MONTH (tm) AS measurement_month, device_id, outdoor_temp, outdoor_hmdty, wind_speed,
                          -- 여름철(하절기) 체감온도 계산
                         (-0.2442
                         + (0.55399 * (
                         outdoor_temp * ATAN(0.151977 * POW(outdoor_hmdty + 8.313659, 0.5))
                         + ATAN(outdoor_temp + outdoor_hmdty)
                         - ATAN(outdoor_hmdty - 1.676331)
                         + (0.00391838 * POW(outdoor_hmdty, 1.5)) * ATAN(0.023101 * outdoor_hmdty)
                         - 4.686035
                         ))
                         + (0.45535 * outdoor_temp)
                         - (0.0022 * POW((
                         outdoor_temp * ATAN(0.151977 * POW(outdoor_hmdty + 8.313659, 0.5))
                         + ATAN(outdoor_temp + outdoor_hmdty)
                         - ATAN(outdoor_hmdty - 1.676331)
                         + (0.00391838 * POW(outdoor_hmdty, 1.5)) * ATAN(0.023101 * outdoor_hmdty)
                         - 4.686035
                         ), 2))
                         + (0.00278 * (
                         outdoor_temp * ATAN(0.151977 * POW(outdoor_hmdty + 8.313659, 0.5))
                         + ATAN(outdoor_temp + outdoor_hmdty)
                         - ATAN(outdoor_hmdty - 1.676331)
                         + (0.00391838 * POW(outdoor_hmdty, 1.5)) * ATAN(0.023101 * outdoor_hmdty)
                         - 4.686035
                         ) * outdoor_temp)
                         + 3.0) AS summer_feels_like,
                          -- 겨울철(동절기) 체감온도 계산
                         (13.12 + (0.6215 * outdoor_temp) - (11.37 * POW(wind_speed * 3.6, 0.16)) + (0.3965 * outdoor_temp * POW(wind_speed * 3.6, 0.16))) AS winter_feels_like
                     FROM
                         PRE_ECO_DATA
                     WHERE
                         rn = 1
                         )
                     SELECT calcs.tm           AS tm,
                            dev.device_id      AS device_id,
                            dev.device_name    AS device_name,
                            calcs.outdoor_temp AS outdoor_temp,
                            CASE
                                WHEN calcs.measurement_month BETWEEN 5 AND 9 THEN calcs.summer_feels_like
                                ELSE calcs.winter_feels_like
                                END            AS fill_temp,
                            CASE
                                WHEN calcs.measurement_month BETWEEN 5 AND 9 THEN
                                    CASE
                                        WHEN calcs.summer_feels_like >= 35 THEN '외부 폭염 경보'
                                        WHEN calcs.summer_feels_like >= 33 THEN '외부 폭염 주의보'
                                        END
                                ELSE
                                    CASE
                                        WHEN calcs.winter_feels_like <= -12 THEN '외부 한파 주의보'
                                        WHEN calcs.winter_feels_like <= -15 THEN '외부 한파 경보'
                                        END
                                END            AS state
                     FROM CALC_TEMP_DATA AS calcs
                              LEFT OUTER JOIN
                          TB_DEVICE_MASTER AS dev ON calcs.device_id = dev.device_id;
                     """)

        endDate = datetime.now()
        srtDate = endDate - timedelta(minutes=sysOpt['mntrgMinInv'])
        with sysOpt['cfgDb']['sessionMake']() as session:
            dataList = session.execute(query, {"srtDate": srtDate, "endDate": endDate}).all()

            for dataInfo in dataList:
                if dataInfo.state is None: continue
                log.info(f"[CHECK] dataInfo : {dataInfo}")

                key = f"{dataInfo.state}-{dataInfo.device_id}-{dataInfo.device_name}"
                msgAlertDate = sysOpt['msgAlertHist'].get(key)
                if (msgAlertDate is None) or (endDate - msgAlertDate) >= timedelta(minutes=sysOpt['msgAlertMinInv']):
                    sendTgApi(sysOpt['cfgTg'], sysOpt['msgAlertTemplate'][dataInfo.state].format(device_id=dataInfo.device_id,
                                                                                             device_name=dataInfo.device_name,
                                                                                             outdoor_temp=dataInfo.outdoor_temp,
                                                                                             fill_temp=dataInfo.fill_temp))
                    sysOpt['msgAlertHist'][key] = endDate
                    log.info(f"[CHECK] msgAlertHist : {sysOpt['msgAlertHist'].keys()}")
    except Exception as e:
        log.error(f'Exception : {e}')

# 데이터 적재 검사
def dbMntrgData(sysOpt):
    try:
        query = text("""
                     WITH RANK_DATA AS (SELECT ECO.*,
                                               dev.device_name,
                                               dev.location,
                                               ROW_NUMBER() OVER (PARTITION BY ECO.device_id ORDER BY ECO.tm DESC) AS rn
                                        FROM TB_ECOWITT_DATA AS ECO
                                                 LEFT OUTER JOIN
                                             TB_DEVICE_MASTER AS dev ON ECO.device_id = dev.device_id)
                     SELECT *
                     FROM RANK_DATA
                     WHERE rn = 1;
                     """)

        with sysOpt['cfgDb']['sessionMake']() as session:
            dataList = session.execute(query, {}).all()
            endDate = datetime.now()

            for dataInfo in dataList:
                delayMin = (datetime.now() - dataInfo.tm).total_seconds() / 60
                log.info(f"[CHECK] id : {dataInfo.device_id} / tm : {dataInfo.tm} / preMin : {delayMin:.1f}")

                for thrMinInfo, thrMsgInfo in zip(sysOpt['thrMinList'], sysOpt['thrMsgList']):
                    if delayMin <= thrMinInfo: continue

                    key = f"데이터 적재 실패-{dataInfo.device_id}-{dataInfo.device_name}"
                    msgAlertDate = sysOpt['msgAlertHist'].get(key)
                    if (msgAlertDate is None) or (endDate - msgAlertDate) >= timedelta(minutes=sysOpt['msgAlertMinInv']):
                        sendTgApi(sysOpt['cfgTg'],
                                  sysOpt['msgAlertTemplate']['데이터 적재 실패'].format(device_id=dataInfo.device_id,
                                                                                 device_name=dataInfo.device_name,
                                                                                 tm=dataInfo.tm.strftime(
                                                                                     "%Y-%m-%d %H:%M"),
                                                                                 thrMsgInfo=thrMsgInfo))
                        sysOpt['msgAlertHist'][key] = endDate
                        log.info(f"[CHECK] msgAlertHist : {sysOpt['msgAlertHist'].keys()}")
                        break
    except Exception as e:
        log.error(f'Exception : {e}')

async def asyncSchdl(sysOpt):
    scheduler = AsyncIOScheduler()

    jobList = [
        (dbMntrgIndoor, 'cron', {'minute': '*/1', 'second': '0'}, {'args': [sysOpt]}),
        (dbMntrgOutdoor, 'cron', {'minute': '*/1', 'second': '0'}, {'args': [sysOpt]}),
        (dbMntrgData, 'cron', {'minute': '*/1', 'second': '0'}, {'args': [sysOpt]}),
    ]

    for fun, trigger, triggerArgs, kwargs in jobList:
        try:
            scheduler.add_job(fun, trigger, **triggerArgs, **kwargs)
        except Exception as e:
            log.error(f"Exception : {e}")

    scheduler.start()
    asyncEvent = asyncio.Event()

    try:
        await asyncEvent.wait()
    except Exception as e:
        log.error(f"Exception : {e}")
    finally:
        if scheduler.running:
            scheduler.shutdown()

def initCfgInfo(config, key):

    result = None

    try:
        # DB 연결 정보
        pymysql.install_as_MySQLdb()

        log.info(f'[CHECK] key : {key}')

        # DB 정보
        dbUser = config.get(key, 'user')
        dbPwd = urllib.parse.quote(config.get(key, 'pwd'))
        dbHost = config.get(key, 'host')
        # dbHost = 'localhost'
        dbPort = config.get(key, 'port')
        dbName = config.get(key, 'dbName')

        engine = sqlalchemy.create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}?charset=utf8'.format(dbUser, dbPwd, dbHost, dbPort, dbName), echo=False)
        sessionMake = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        # session = sessionMake()

        base = automap_base()
        base.prepare(autoload_with=engine)
        tableList = base.classes.keys()

        result = {
            'engine': engine
            , 'sessionMake': sessionMake
            # , 'session': session
            , 'tableList': tableList
            , 'tableCls': base.classes
        }

        return result

    except Exception as e:
        log.error(f'Exception : {e}')
        return result

def sendTgApi(cfgTg, msg):
    try:
        url = f"https://api.telegram.org/bot{cfgTg['bot_token']}/sendMessage"
        payload = {
            "chat_id": cfgTg['chat_id'],
            "text": msg,
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        log.error(f"Exception : {e}")

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'BDWIDE2025'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info('[START] {}'.format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

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
                globalVar['inpPath'] = '/HDD/DATA/INPUT'
                globalVar['outPath'] = '/HDD/DATA/OUTPUT'
                globalVar['figPath'] = '/HDD/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 설정 파일
                'cfgDbKey': 'mysql-iwin-dms01user01-DMS03',
                'cfgTgKey': 'telegram-smartHiveMntrg',
                'cfgDb': None,
                'cfgTg': None,
                'cfgFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',

                # 모니터링 주기 1분
                'mntrgMinInv': 1,

                # 데이터 적재 검사
                'thrMinList': [60 * 24 * 7, 60 * 24, 60 * 12, 60, 30],
                'thrMsgList': ['7일', '1일', '12시간', '1시간', '30분'],

                # 메시지 알림 이력
                'msgAlertHist': {},

                # 모니터링 주기 60분
                'msgAlertMinInv': 60,

                # 메시지 알림 템플릿
                'msgAlertTemplate': {
                    # 내부 온도
                    '내부 고온 경보': '[🚨내부 고온 경보] {device_name} 농장 / {device_id}번 벌통\n- 현재 상태: 내부 온도 {indoor_temp:.1f}℃ (경보 기준: 38℃)\n- 권장 조치:\n  ▶ 즉시 벌통 주변 환기 강화\n  ▶ 차광막 설치 또는 보강\n  ▶ 급수시설 확인 및 보충',
                    '내부 고온 주의보': '[🚨내부 고온 주의보] {device_name} 농장 / {device_id}번 벌통\n- 현재 상태: 내부 온도 {indoor_temp:.1f}℃ (주의보 기준: 35℃)\n- 권장 조치:\n  ▶ 즉시 벌통 주변 환기 강화\n  ▶ 차광막 설치 또는 보강\n  ▶ 급수시설 확인 및 보충',
                    '내부 저온 주의보': '[❄️내부 저온 주의보] {device_name} 농장 / {device_id}번 벌통\n- 현재 상태: 내부 온도 {indoor_temp:.1f}℃ (주의보 기준: 6℃)\n- 권장 조치:\n  ▶ 봉군 보온재 상태 점검\n  ▶ 비상 먹이(사양액) 잔량 확인 및 보충 준비',
                    '내부 저온 경보': '[❄️내부 저온 경보] {device_name} 농장 / {device_id}번 벌통\n- 현재 상태: 내부 온도 {indoor_temp:.1f}℃ (경보 기준: 0℃)\n- 권장 조치:\n  ▶ 봉군 보온재 상태 점검\n  ▶ 비상 먹이(사양액) 잔량 확인 및 보충 준비',
                    # 외부 폭염/한파
                    '외부 폭염 경보': '[☀️외부 폭염 경보] {device_name} 농장 / {device_id}번 벌통\n- 현재 상태: 외부 온도 {outdoor_temp:.1f}℃ / 체감 온도 {fill_temp:.1f}℃ (경보 기준: 35℃)\n- 권장 조치:\n  ▶ 전체 농가 차광막 설치\n  ▶ 차광막 설치 또는 보강\n  ▶ 급수 시설 점검 및 물 보충',
                    '외부 폭염 주의보': '[☀️외부 폭염 주의보] {device_name} 농장 / {device_id}번 벌통\n- 현재 상태: 외부 온도 {outdoor_temp:.1f}℃ / 체감 온도 {fill_temp:.1f}℃ (주의보 기준: 33℃)\n- 권장 조치:\n  ▶ 전체 농가 차광막 설치\n  ▶ 차광막 설치 또는 보강\n  ▶ 급수 시설 점검 및 물 보충',
                    '외부 한파 주의보': '[🧊외부 한파 주의보] {device_name} 농장 / {device_id}번 벌통\n- 현재 상태: 외부 온도 {outdoor_temp:.1f}℃ / 체감 온도 {fill_temp:.1f}℃ (주의보 기준: -12℃)\n- 권장 조치:\n  ▶ 벌통 덮개 및 고정장치 결박 상태 점검\n  ▶ 눈 가림막 및 방풍 시설 확인',
                    '외부 한파 경보': '[🧊외부 한파 경보] {device_name} 농장 / {device_id}번 벌통\n- 현재 상태: 외부 온도 {outdoor_temp:.1f}℃ / 체감 온도 {fill_temp:.1f}℃ (경보 기준: -15℃)\n- 권장 조치:\n  ▶ 벌통 덮개 및 고정장치 결박 상태 점검\n  ▶ 눈 가림막 및 방풍 시설 확인',
                    # 데이터 적재
                    '데이터 적재 실패': '[⚠️데이터 적재 실패] {device_name} 농장 / {device_id}번 벌통\n- 현재 상태: 최근 일시 {tm} ({thrMsgInfo} 이상 경과)',
                }
            }

            config = configparser.ConfigParser()
            config.read(sysOpt['cfgFile'], encoding='utf-8')
            
            sysOpt['cfgDb'] = initCfgInfo(config, sysOpt['cfgDbKey'])
            sysOpt['cfgTg'] = {
                'bot_token': config.get(sysOpt['cfgTgKey'], 'bot_token'),
                'chat_id': config.get(sysOpt['cfgTgKey'], 'chat_id'),
            }

            # 파일 스케줄러
            asyncio.run(asyncSchdl(sysOpt))

        except Exception as e:
            log.error(f"Exception : {e}")
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