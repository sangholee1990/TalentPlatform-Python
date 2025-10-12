# ================================================
# 요구사항
# ================================================
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
# /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-colct-naverSearchApi.py
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-colct-naverSearchApi.py &
# tail -f nohup.out

# pkill -f TalentPlatform-LSH0627-DaemonFramework-colct-naverSearchApi.py


# 0 0 * * * cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys && /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-LSH0627-DaemonFramework-colct-naverSearchApi.py

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
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import pandas as pd
import re
import json
from datetime import datetime, timedelta
# from konlpy.tag import Okt
from collections import Counter
import pytz
import os
import sys
import urllib.request
import os
import sys
import requests
import json
from konlpy.tag import Okt
from newspaper import Article

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
    serviceName = 'LSH0612'

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

            if (platform.system() == 'Windows'):
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 수행 목록
                'modelList': ['SHOP'],

                # 세부 정보
                'SHOP': {
                    'client_id': 'BNDqTQqb0NECaQN56flk',
                    'client_secret': '5t2GECIt1m',
                    'display': '100',
                    'sort': 'date',
                    # 중고:렌탈:해외직구,구매대행
                    'exclude': 'used:rental:cbshop',
                    'preDt': datetime.now(),
                    'typeList': ['알톤 자전거', '삼천리 자전거', '스마트 자전거'],
                    'cateList': ['전기자전거', '하이브리드', 'MTB', '사이클', '일반자전거', '미니벨로'],
                    'url': 'https://openapi.naver.com/v1/search/shop.json',
                    'saveCsvFile': '/HDD/DATA/OUTPUT/LSH0627/%Y%m/%d/naverShop_{queryInfo}_%Y%m%d.csv',
                },
            }

            # =================================================================
            # 네이버 API 수집
            # =================================================================
            for modelType in sysOpt['modelList']:
                log.info(f'[CHECK] modelType : {modelType}')

                modelInfo = sysOpt.get(modelType)
                if modelInfo is None: continue

                typeList = modelInfo['typeList']
                cateList = modelInfo['cateList']
                for typeInfo in typeList:
                    for cateInfo in cateList:
                        queryInfo = f"{typeInfo} {cateInfo}"

                        dataL1 = pd.DataFrame()
                        pageList = np.arange(1, 1001, 1)
                        for pageInfo in pageList:
                            try:
                                headers = {
                                    'X-Naver-Client-Id': modelInfo['client_id'],
                                    'X-Naver-Client-Secret': modelInfo['client_secret'],
                                }

                                params = {
                                    'query': queryInfo,
                                    'display': modelInfo['display'],
                                    'sort': modelInfo['sort'],
                                    'exclude': modelInfo['exclude'],
                                    'start': pageInfo,
                                }

                                res = requests.get(modelInfo['url'], headers=headers, params=params)
                                if res.status_code != 200: continue

                                resJson = res.json().get('items')
                                if resJson is None or len(resJson) < 1: continue

                                resData = pd.DataFrame(resJson)

                                dataL1 = pd.concat([dataL1, resData], ignore_index=True)
                                dataL1['type'] = typeInfo
                                dataL1['cate'] = cateInfo
                                dataL1['date'] = modelInfo['preDt'].strftime('%Y%m%d')
                                log.info(f'[CHECK] modelType : {modelType} / queryInfo : {queryInfo} / per : {round((pageInfo / 1001) * 100, 1)}  / cnt : {len(dataL1)}')

                            except Exception as e:
                                log.error(f"Exception : {e}")

                        if len(dataL1) > 0:
                            saveCsvFile = modelInfo['preDt'].strftime(modelInfo['saveCsvFile']).format(queryInfo=queryInfo)
                            os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)
                            dataL1.to_csv(saveCsvFile, index=False)
                            log.info(f"[CHECK] saveCsvFile : {saveCsvFile}")

            # =================================================================
            # 전처리
            # =================================================================
            # fileList = sorted(glob.glob('/HDD/DATA/OUTPUT/LSH0627/naverShop_알톤 자전거_*.csv'), reverse=True)
            # fileList = sorted(glob.glob('/HDD/DATA/OUTPUT/LSH0627/naverShop_알톤*_20251004.csv'), reverse=True)
            fileList = sorted(glob.glob('/HDD/DATA/OUTPUT/LSH0627/*/*/naverShop_*자전거*_*.csv'), reverse=True)
            data = pd.DataFrame()
            for fileInfo in fileList:
                orgData = pd.read_csv(fileInfo)
                orgDataL1 = orgData[(orgData['category1'] == '스포츠/레저') & (orgData['category2'] == '자전거') & (orgData['category3'] == '자전거/MTB')]
                data = pd.concat([data, orgDataL1], ignore_index=False)
            dataL1 = data.drop_duplicates(subset=['title', 'link', 'image', 'lprice', 'hprice', 'mallName', 'productId', 'productType', 'brand', 'maker', 'category1', 'category2', 'category3', 'category4', 'type', 'cate', 'date']).sort_values(['title', 'date'], ascending=False)
            dataL1.to_csv('/HDD/DATA/OUTPUT/LSH0627/naverShop_자전거.csv', index=False)

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
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))