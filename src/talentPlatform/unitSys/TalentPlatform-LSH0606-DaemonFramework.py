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
from pyparsing import col
from selenium.webdriver.support.ui import Select

from urllib.request import urlopen
from urllib import parse
from urllib.request import Request
from urllib.error import HTTPError
import json
import math
from scipy import spatial
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from lxml import etree
import xml.etree.ElementTree as et
import requests
from lxml import html
import urllib
import unicodedata2
from urllib import parse
import time
from urllib.parse import quote_plus, urlencode

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

import time
from selenium import webdriver
import chardet
from selenium.common.exceptions import NoSuchWindowException
import requests
from bs4 import BeautifulSoup
import pytz
from pytrends.request import TrendReq

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

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 중국 빅데이터 사이트 조사 및 셀레늄 기반 로그인, 기본 및 부가정보 수집 및 추출

    # 원도우 X11 (X Window System) 프로토콜 지원
    # xming

    # 리눅스 CLI 실행
    # google-chrome --no-sandbo

    # 프로그램 실행
    # cd /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
    # /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0605-DaemonFramework.py
    # nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0605-DaemonFramework.py &
    # tail -f nohup.out

    # 프로그램 종료
    # ps -ef | grep "TalentPlatform-LSH0605-DaemonFramework" | awk '{print $2}' | xargs kill -9
    # ps -ef | grep "chrome" | awk '{print $2}' | xargs kill -9

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'LSH0606'

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

            # 옵션 설정
            sysOpt = {
                'loginUrl': "https://www.pkulaw.com/login",
                'listUrl': "https://www.pkulaw.com",

                'chromeInfo': "/DATA/INPUT/LSH0602/chrome-linux64/chrome",
                'chromedriverInfo':"/DATA/INPUT/LSH0602/chromedriver-linux64/chromedriver",

                # 지연 시간
                'pageTimeout': 120,
                'loadTimeout': 60,
                'defTimeout': 30,

                # 로그인 기능
                'loginId': "18333208671",
                'loginPw': "world&peace",

                # 검색 목록
                'sectorList': ['交通', '住宅', '建筑', '电力', '工业'],
                'keyList': ["碳排放", "低碳", "减碳", "温室气体", "节能", "能源效率", "能源消耗", "产能过剩", "碳中和", "可再生能源", "清洁能源", "绿色能源", "能源转型", "减排", "绿色建筑", "非化石能源", "碳足迹"],

                # 자료 저장
                'saveFileList': '/DATA/OUTPUT/LSH0605/*_{cityMat}.xlsx',
                'saveFile': '/DATA/OUTPUT/LSH0605/%Y%m%d_{cityMat}.xlsx',
            }

            # ==========================================================================================================
            # 구글 트렌드 기반 실시간 검색어 웹
            # 동적 크롤링
            # https://trends.google.co.kr/trending?geo=KR
            # ==========================================================================================================
            dataL1 = pd.DataFrame()

            pytrends = TrendReq(hl='ko-KR', tz=540)
            orgData = pytrends.trending_searches(pn='south_korea')

            orgDataL1 = orgData.rename(columns={0: 'keyword'})
            orgDataL1['no'] = orgDataL1.index + 1
            orgDataL1['dateTime'] = datetime.now(tz=tzKst)
            orgDataL1['type'] = 'google'

            data = orgDataL1[['type', 'dateTime', 'no', 'keyword']]
            if len(data) > 0:
                dataL1 = pd.concat([dataL1, data])

            # ==========================================================================================================
            # 웨어이즈포스트 기반 실시간 검색어
            # 정적 크롤링
            # ==========================================================================================================
            url = "https://whereispost.com/hot"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            lxml = etree.HTML(str(soup))

            try:
                tag = lxml.xpath('/html/body/content/div/div/div/div[1]/text()')[0]
                match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', tag.strip())
                sDateTime = None if match is None else match.group(0)
                dtDateTime = pd.to_datetime(sDateTime).tz_localize('Asia/Seoul')
            except Exception:
                dtDateTime = None
            log.info(f'[CHECK] dtDateTime : {dtDateTime}')

            noList = soup.find('ul', {'class': 'list-group bg-white'}).find_all("span", {'class': 'rank daum_color'})
            keywordList = soup.find('ul', {'class': 'list-group bg-white'}).find_all("span", {'class': 'keyword'})

            data = pd.DataFrame()
            for noInfo, keywordInfo in zip(noList, keywordList):
                try:
                    no = None if noInfo is None or len(noInfo) < 1 else noInfo.text.strip()
                    keyword = None if keywordInfo is None or len(keywordInfo) < 1 else keywordInfo.text.strip()

                    dict = {
                        'type': ['whereispost'],
                        'dateTime': [dtDateTime],
                        'no': [no],
                        'keyword': [keyword],
                    }

                    data = pd.concat([data, pd.DataFrame.from_dict(dict)])

                except Exception:
                    pass

            if len(data) > 0:
                dataL1 = pd.concat([dataL1, data])

            # ==========================================================================================================
            # 이지미넷 기반 실시간 검색어
            # 정적 크롤링
            # ==========================================================================================================
            url = "https://rank.ezme.net"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            try:
                tag = soup.find('div', {'id': 'content'}).find('small')
                sDateTime = None if tag is None or len(tag) < 1 else tag.text.strip()
                dtDateTime = pd.to_datetime(sDateTime).tz_localize('Asia/Seoul')
            except Exception:
                dtDateTime = None
            log.info(f'[CHECK] dtDateTime : {dtDateTime}')


            noList = soup.find('div', {'id': 'content'}).find_all("span", {'class': 'rank_no'})
            keywordList = soup.find('div', {'id': 'content'}).find_all("span", {'class': 'rank_word'})

            data = pd.DataFrame()
            for noInfo, keywordInfo in zip(noList, keywordList):
                try:
                    no = None if noInfo is None or len(noInfo) < 1 else noInfo.text.strip(".").strip()
                    keyword = None if keywordInfo is None or len(keywordInfo) < 1 else keywordInfo.find('a').text.strip()

                    dict = {
                        'type': ['ezme'],
                        'dateTime': [dtDateTime],
                        'no': [no],
                        'keyword': [keyword],
                    }

                    data = pd.concat([data, pd.DataFrame.from_dict(dict)])
                except Exception:
                    pass

            if len(data) > 0:
                dataL1 = pd.concat([dataL1, data])


            print(dataL1)

            dataL2 = dataL1.reset_index(drop=True)

            # ==========================================================================================================
            # 자료 저장
            # ==========================================================================================================
            # if len(data) > 0:
            #     saveFile = datetime.now().strftime(sysOpt['saveFile']).format(cityMat=cityMat)
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     data.to_excel(saveFile, index=False)
            #     log.info(f'[CHECK] saveFile : {saveFile}')

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
        inParams = { }
        print("[CHECK] inParams : {}".format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))