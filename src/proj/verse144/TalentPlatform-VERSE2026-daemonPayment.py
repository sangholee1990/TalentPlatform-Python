# ================================================
# 요구사항
# ================================================
# Python을 이용한 정기결제 데몬 스케줄

# ps -ef | grep "TalentPlatform-VERSE2026-daemonPayment" | awk '{print $2}' | xargs kill -9
# pkill -f "TalentPlatform-VERSE2026-daemonPayment"

# 프로그램 시작
# conda activate py39
# cd /SYSTEMS/PROG/PYTHON
# /SYSTEMS/LIB/anaconda3/envs/py39/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-VERSE2026-daemonPayment.py
# nohup /SYSTEMS/LIB/anaconda3/envs/py39/bin/python/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-VERSE2026-daemonPayment.py > /dev/null 2>&1 &

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
from datetime import timedelta
from urllib.parse import quote_plus

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from pandas.tseries.offsets import Hour
import yaml
from multiprocessing import Pool
import multiprocessing as mp
from retrying import retry
# import cdsapi
import shutil
import io
import uuid
import requests
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime
import subprocess
from pandas.tseries.offsets import DateOffset
import configparser
from urllib.parse import urlparse, parse_qs
from lxml import etree
import urllib.parse
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import os
import firebase_admin
from firebase_admin import firestore
import google.auth
import requests
import json
import requests
import base64
import time
from datetime import datetime

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
        # , 'resPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        # , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        # , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        # , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        # , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        # , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        # , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        # , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        # , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        # , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        # , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    return globalVar


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

def payProc(sysOpt):
    try:
        preDt = datetime.now(pytz.UTC)
        preDtKst = preDt.now(pytz.timezone('Asia/Seoul'))

        docList = sysOpt['cfgDb'].collection('users').get()
        for docInfo in docList:
            try:
                docItem = docInfo.to_dict()
                subStatus = docItem.get('subscriptionStatus') or {}
                subTier = docItem.get('subscriptionTier') or {}
                nextPaymentDate = docItem.get('nextPaymentDate').astimezone(pytz.timezone('Asia/Seoul')) if docItem.get('nextPaymentDate') else None

                billingConfig = docItem.get('billingConfig') or {}
                billingKey = billingConfig.get('billingKey')

                # 구독 검사
                if not subStatus == "active": continue
                if subTier not in ["standard", "premium"]: continue

                # 빌링키 검사
                if not billingKey: continue

                # 결재일 검사
                if not nextPaymentDate: continue
                if preDtKst < nextPaymentDate: continue
                log.info(f"uid : {docItem.get('uid')}, docItem : {docItem}")

                # 토스 API 연계
                # https://docs.tosspayments.com/guides/v2/billing/integration-api#1-%EB%B9%8C%EB%A7%81%ED%82%A4-%EB%B0%9C%EA%B8%89%ED%95%98%EA%B8%B0
                url = f"{sysOpt['cfgToss']['BASE_TOSS_URL']}/v1/billing/{billingKey}"
                payload = {
                    "customerKey": docItem.get('uid'),
                    "amount": 4800 if subTier == 'standard' else 9800,
                    "orderId": preDtKst.strftime('%Y%m%d%H%M%S'),
                    "orderName": f"{subTier} 구독",
                    "customerEmail": docItem.get('email'),
                    "customerName": docItem.get('name'),
                    "taxFreeAmount": 0
                }

                response = requests.post(url, headers=sysOpt['headers'], json=payload)
                if not response.status_code == 200: continue

                result = response.json()
                if result.get('status') == 'DONE':
                    log.info(f"uid : {docItem.get('uid')}, docItem : {docItem}, result : {result}")
                    docInfo.reference.update({
                        'subscriptionStatus': 'active',
                        'nextPaymentDate': (pd.Timestamp(preDtKst) + pd.DateOffset(months=1)).to_pydatetime()
                    })

            except Exception as e:
                log.error(f"Exception : {e}")

    except Exception as e:
        log.error(f'Exception : {e}')

async def asyncSchdl(sysOpt):
    scheduler = AsyncIOScheduler()

    jobList = [
        (payProc, 'cron', {'minute': '*'}, {'args': [sysOpt], 'max_instances': 1, 'misfire_grace_time': None, 'coalesce': False})
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
    serviceName = 'VERSE2026'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info(f"[START] __init__ : init")

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

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
                # 설정 정보
                'cfgFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/application_default_credentials.json',
                'cfgDb': None,

                'cfgFile2': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
                'cfgTossKey': 'toss-pay-test',
                # 'cfgTossKey': 'toss-pay-oper',
                'cfgToss': None,

                'headers': None,
            }

            # **********************************************************************************************************
            # 설정 정보
            # **********************************************************************************************************
            creds, _ = google.auth.load_credentials_from_file(sysOpt['cfgFile'])
            firebase_admin.initialize_app(credential=creds, options={'projectId': 'project-p-32424'})
            sysOpt['cfgDb'] = firestore.client()

            config = configparser.ConfigParser()
            config.read(sysOpt['cfgFile2'], encoding='utf-8')
            sysOpt['cfgToss'] = {
                'BASE_TOSS_URL': config.get(sysOpt['cfgTossKey'], 'BASE_TOSS_URL'),
                'TOSS_PAY_GCK': config.get(sysOpt['cfgTossKey'], 'TOSS_PAY_GCK'),
                'TOSS_PAY_GSK': config.get(sysOpt['cfgTossKey'], 'TOSS_PAY_GSK'),
                'TOSS_API_CK': config.get(sysOpt['cfgTossKey'], 'TOSS_API_CK'),
                'TOSS_API_SK': config.get(sysOpt['cfgTossKey'], 'TOSS_API_SK'),
                'TOSS_API_SE': config.get(sysOpt['cfgTossKey'], 'TOSS_API_SE'),
            }

            auth = f"{sysOpt['cfgToss']['TOSS_API_SK']}:"
            authEnc = base64.b64encode(auth.encode('utf-8')).decode('utf-8')

            sysOpt['headers'] = {
                "Authorization": f"Basic {authEnc}",
                "Content-Type": "application/json"
            }

            # 파일 스케줄러
            asyncio.run(asyncSchdl(sysOpt))

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
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] main')