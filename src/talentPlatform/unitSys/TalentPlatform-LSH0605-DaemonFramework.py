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
from selenium.common.exceptions import NoSuchWindowException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
# import undetected_chromedriver as uc
from pathlib import Path
from selenium.webdriver.common.proxy import Proxy, ProxyType

if platform.system() == 'Windows':
    import chromedriver_autoinstaller

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

    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(saveLogFile, encoding='utf-8')

    streamHandler.setFormatter(format)
    fileHandler.setFormatter(format)

    log.addHandler(streamHandler)
    log.addHandler(fileHandler)

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

def initDriver(sysOpt):

    if platform.system() == 'Windows':
        # 자동 설치
        chromedriver_autoinstaller.install()

        # 무료 프록시
        # https://geonode.com/free-proxy-list
        # proxy = "207.180.234.234:3128"
        # proxyInfo = Proxy()
        # proxyInfo.proxy_type = ProxyType.MANUAL
        # proxyInfo.http_proxy = proxy
        # proxyInfo.ssl_proxy = proxy

        # Chrome 옵션 설정
        options = Options()
        # options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        # options.add_argument(f'--proxy-server={proxyInfo}')

        # WebDriver 실행
        driver = webdriver.Chrome(options=options)

    else:
        # Chrome 옵션 설정
        options = Options()
        options.headless = False  # 창을 띄우도록 설정 (True로 하면 백그라운드 실행)
        options.binary_location = sysOpt['chromeInfo']  # 사용자 지정 Chrome 경로
        options.add_argument("--window-size=1920,1080")  # 창 크기 설정
        options.add_experimental_option("detach", True)  # 실행 후 브라우저 종료 방지

        # 백그라운드 실행 관련 옵션
        options.add_argument("--no-sandbox")  # 샌드박스 비활성화 (Linux 환경에서 필수)
        options.add_argument("--disable-dev-shm-usage")  # /dev/shm 메모리 제한 방지
        options.add_argument("--headless")  # 브라우저를 백그라운드에서 실행 (UI 없음)
        options.add_argument("--remote-debugging-port=9222")  # 원격 디버깅 포트 설정
        options.add_argument("--disable-gpu")  # GPU 비활성화 (Linux 환경에서 필요)
        options.add_argument("--disable-blink-features=AutomationControlled")

        # User-Agent 설정 (봇 감지 우회)
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.6778.264 Safari/537.36"
        )

        # ChromeDriver 서비스 설정
        service = Service(sysOpt['chromedriverInfo'])

        # WebDriver 실행
        driver = webdriver.Chrome(service=service, options=options)

    # 페이지 로드 타임아웃 설정
    driver.set_page_load_timeout(sysOpt['pageTimeout'])

    return driver

def initLogin(driver, sysOpt):

    try:
        # 로그인 페이지 이동
        url = sysOpt['loginUrl']
        driver.get(url)

        # driver.save_screenshot("/tmp/screenshot.png")

        # 최대 timeout 대기
        wait = WebDriverWait(driver, sysOpt['loadTimeout'])

        # 회원 활성화 버튼
        btnId = wait.until(EC.presence_of_element_located((By.ID, "zhanghaodenglu")))
        btnId.click()

        # 계정 활성화 버튼
        btnId = wait.until(EC.presence_of_element_located((By.ID, "userPasswordDiv")))
        btnId.click()

        # 이메일/아이디 입력
        emailId = wait.until(EC.presence_of_element_located((By.ID, "email-phone")))
        emailId.send_keys(sysOpt['loginId'])

        # 비밀번호 입력
        passId = wait.until(EC.presence_of_element_located((By.ID, "passwordFront")))
        driver.execute_script("arguments[0].removeAttribute('disabled')", passId)
        passId.send_keys(sysOpt['loginPw'])

        # 로그인 버튼
        btnId = wait.until(EC.element_to_be_clickable((By.ID, "loginByUserName")))
        btnId.click()
        time.sleep(sysOpt['defTimeout'])

        # 광고 삭제
        try:
            isId = driver.find_element("id", "layui-layer1")
            if isId:
                btnId = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "layui-layer-close2")))
                btnId.click()
        except NoSuchElementException:
            pass

    except Exception as e:
        log.error(f"Exception : {e}")


def textProp(text):
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # ASCII 제어 문자 제거
    text = text.replace('\xa0', ' ')  # Non-breaking space 제거
    return text.strip()

def getSplitData(df, splitNum=5):
    n = len(df)
    partSize = n // splitNum
    remainder = n % splitNum

    result = []
    start = 0
    for i in range(splitNum):
        end = start + partSize
        if i < remainder:
            end += 1
        result.append(df.iloc[start:end])
        start = end

    return result

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

    # 크롬 다운로드
    # https://googlechromelabs.github.io/chrome-for-testing

    # /DATA/INPUT/LSH0605/chrome-linux64/chrome --version
    # Google Chrome for Testing 131.0.6778.264

    # /DATA/INPUT/LSH0605/chromedriver-linux64/chromedriver --version
    # ChromeDriver 131.0.6778.264 (2d05e31515360f4da764174f7c448b33e36da871-refs/branch-heads/6778@{#4323})

    # 프로그램 실행
    # cd /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
    # /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0605-DaemonFramework.py
    # nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0605-DaemonFramework.py &
    # tail -f nohup.out

    # 프로그램 실행 자동화
    # * * * * * bash /SYSTEMS/PROG/SHELL/PROC/RunShell-ProcAgentCheck.sh

    # 프로그램 종료
    # ps -ef | grep "TalentPlatform-LSH0605-DaemonFramework" | grep -v "grep" | awk '{print $2}' | xargs kill -9
    # ps -ef | grep "chrome" | grep -v "grep" | awk '{print $2}' | xargs kill -9

    # 원도우 환경
    # C:\Users\sangh\.conda\envs\py38\python.exe C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\talentPlatform\unitSys\TalentPlatform-LSH0605-DaemonFramework.py --selIdx 0 --splitNum 8
    # C:\Users\sangh\.conda\envs\py38\python.exe C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\talentPlatform\unitSys\TalentPlatform-LSH0605-DaemonFramework.py --selIdx 1 --splitNum 8
    # C:\Users\sangh\.conda\envs\py38\python.exe C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\talentPlatform\unitSys\TalentPlatform-LSH0605-DaemonFramework.py --selIdx 2 --splitNum 8
    # C:\Users\sangh\.conda\envs\py38\python.exe C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\talentPlatform\unitSys\TalentPlatform-LSH0605-DaemonFramework.py --selIdx 3 --splitNum 8
    # C:\Users\sangh\.conda\envs\py38\python.exe C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\talentPlatform\unitSys\TalentPlatform-LSH0605-DaemonFramework.py --selIdx 4 --splitNum 8
    # C:\Users\sangh\.conda\envs\py38\python.exe C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\talentPlatform\unitSys\TalentPlatform-LSH0605-DaemonFramework.py --selIdx 5 --splitNum 8
    # C:\Users\sangh\.conda\envs\py38\python.exe C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\talentPlatform\unitSys\TalentPlatform-LSH0605-DaemonFramework.py --selIdx 6 --splitNum 8
    # C:\Users\sangh\.conda\envs\py38\python.exe C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\talentPlatform\unitSys\TalentPlatform-LSH0605-DaemonFramework.py --selIdx 7 --splitNum 8

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
    serviceName = 'LSH0605'

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
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                'loginUrl': "https://www.pkulaw.com/login",
                'listUrl': "https://www.pkulaw.com",
                'verfUrl': "https://www.pkulaw.com/VerificationCode/Index?IsBlack=true",

                'chromeInfo': "/DATA/INPUT/LSH0602/chrome-linux64/chrome",
                'chromedriverInfo':"/DATA/INPUT/LSH0602/chromedriver-linux64/chromedriver",

                # 지연 시간
                # 'pageTimeout': 120,
                # 'loadTimeout': 60,
                # 'defTimeout': 30,
                'pageTimeout': 60,
                'loadTimeout': 30,
                'defTimeout': 15,

                # 'selIdx': 4,
                # 'splitNum': 5,
                # 'isRev': False,
                'isRev': globalVar['isRev'] == 'True',
                'selIdx': int(globalVar['selIdx']),
                'splitNum': int(globalVar['splitNum']),

                # 로그인 기능
                'loginId': "18333208671",
                'loginPw': "world&peace",

                # 검색 목록
                'sectorList': ['交通', '住宅', '建筑', '电力', '工业'],
                'keyList': ["碳排放", "低碳", "减碳", "温室气体", "节能", "能源效率", "能源消耗", "产能过剩", "碳中和", "可再生能源", "清洁能源", "绿色能源", "能源转型", "减排", "绿色建筑", "非化石能源", "碳足迹"],

                # 자료 저장
                # 'saveFileList': '/DATA/OUTPUT/LSH0605/*_{cityMat}.xlsx',
                # 'saveFile': '/DATA/OUTPUT/LSH0605/%Y%m%d_{cityMat}.xlsx',
                'saveFileList': 'G:/내 드라이브/shlee/04. TalentPlatform/[재능플랫폼] 최종납품/[완료] LSH0605. Python을 이용한 중국 빅데이터 사이트 조사 및 셀레늄 기반 로그인, 기본 및 부가정보 수집 및 추출/20250302_결과/*_{cityMat}.xlsx',
                'saveFile': 'G:/내 드라이브/shlee/04. TalentPlatform/[재능플랫폼] 최종납품/[완료] LSH0605. Python을 이용한 중국 빅데이터 사이트 조사 및 셀레늄 기반 로그인, 기본 및 부가정보 수집 및 추출/20250302_결과/%Y%m%d_{cityMat}.xlsx',
            }

            # ==========================================================================================================
            # 설정 파일
            # ==========================================================================================================
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'city_matched.xlsx')
            fileList = sorted(glob.glob(inpFile))
            if fileList is None or len(fileList) < 1:
                raise Exception(f"inpFile : {inpFile} : 설정 파일 검색 실패")

            cfgData = pd.read_excel(fileList[0])
            cfgDataL1 = cfgData.drop_duplicates(subset=['Matching_City_Column_2']).reset_index(drop=True)

            splitData = getSplitData(cfgDataL1, splitNum=sysOpt['splitNum'])
            cfgDataL2 = splitData[sysOpt['selIdx']].sort_index(ascending=sysOpt['isRev']).reset_index(drop=True)

            # ==========================================================================================================
            # 크롬드라이브 삭제
            # ==========================================================================================================
            # if platform.system() == 'Linux':
            #     os.system("ps -ef | grep 'chrome' | grep -v 'grep' | awk '{print $2}' | xargs kill -9")
            #     os.system("ps -ef | grep 'chrome' | grep -v 'grep' | awk '{print $2}' | xargs kill -9")

            # ==========================================================================================================
            # 전역 설정
            # ==========================================================================================================
            # 크롬드라이브 초기화
            driver = initDriver(sysOpt)
            wait = WebDriverWait(driver, sysOpt['loadTimeout'])

            # ==========================================================================================================
            # 로그인 기능
            # ==========================================================================================================
            # initLogin(driver, sysOpt)

            # ==========================================================================================================
            # 기본정보 수집
            # ==========================================================================================================
            for i, item in cfgDataL2.iterrows():
                # if i > 1: break

                city = item['City_Column_1']
                cityMat = item['Matching_City_Column_2']
                per = round(i / len(cfgDataL2) * 100, 1)
                log.info(f'[CHECK] cityMat : {cityMat} / {per}%')
                # print(f'[CHECK] cityMat : {cityMat} / {per}%')

                # sector = sysOpt['sectorList'][0]
                # key = sysOpt['keyList'][0]

                saveFilePattern = sysOpt['saveFileList'].format(cityMat=cityMat)
                # saveFileList = sorted(glob.glob(str(Path(saveFilePattern))), reverse=True)
                saveFileList = list(Path(os.path.dirname(saveFilePattern)).glob(os.path.basename(saveFilePattern)))

                # 파일 존재
                if len(saveFileList) > 0: continue

                data = pd.DataFrame()
                for j, sector in enumerate(sysOpt['sectorList']):
                    for k, key in enumerate(sysOpt['keyList']):

                        try:
                            # keyword = f'广东省 茂名市 交通 碳排放'
                            keyword = f'{cityMat} {sector} {key}'
                            # log.info(f'[CHECK] keyword : {keyword}')
                            # print(f'[CHECK] keyword : {keyword}')

                            # 검색 화면
                            url = sysOpt['listUrl']
                            driver.get(url)
                            time.sleep(sysOpt['defTimeout'])

                            # 광고 삭제
                            try:
                                isId = driver.find_element("id", "layui-layer1")
                                if isId:
                                    btnId = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "layui-layer-close2")))
                                    btnId.click()
                            except Exception as e:
                                pass

                            # driver.save_screenshot("/tmp/screenshot.png")

                            # 검색어 입력
                            inputId = wait.until(EC.presence_of_element_located((By.ID, "txtSearch")))
                            inputId.send_keys(keyword)

                            # 검색어 버튼
                            btnId = wait.until(EC.element_to_be_clickable((By.ID, "btnSearch")))
                            btnId.click()
                            time.sleep(sysOpt['defTimeout'])

                            # eleList = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".accompanying-wrap > .item")))
                            eleList = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".grouping-title, .accompanying-wrap > .item")))

                            # ele = eleList[0]
                            for index, ele in enumerate(eleList):
                                clsName = ele.get_attribute('class')

                                # 정책 분류
                                if re.search('grouping-title', clsName, re.IGNORECASE):
                                    try:
                                        text = ele.text.strip()
                                        polType = re.sub(r'（\d+）', '', text) if text else None
                                    except Exception:
                                        polType = None
                                    continue
                                # log.info(f'[CHECK] polType : {polType}')

                                # 公布日期=공표날짜=Announcement_Year
                                # 施行日期=실행날짜=Starting_Year
                                try:
                                    yearList = ele.find_elements(By.CSS_SELECTOR, ".info .text")
                                    for year in yearList:
                                        text = year.text.strip()
                                        if re.search('公布', text, re.IGNORECASE):
                                            annYear = re.sub(r'公布', '', text) if text else None
                                        elif re.search('施行', text, re.IGNORECASE):
                                            srtYear = re.sub(r'施行', '', text) if text else None
                                        else:
                                            continue
                                except Exception:
                                    annYear = None
                                    srtYear = None
                                # log.info(f'[CHECK] annYear : {annYear}')
                                # log.info(f'[CHECK] srtYear : {srtYear}')

                                # 时效性=실효성=Ending_year （만약 값이 现行有效이면 현재까지 유효한 거이므로 Active를 넣어주면 좋을 거 같습니다)
                                try:
                                    tagList = ele.find_elements(By.CSS_SELECTOR, ".info > a")
                                    for idx, tag in enumerate(tagList):
                                        if idx > 0: continue
                                        text = tag.text.strip()
                                        endYear = "Active" if re.search('现行有效', text, re.IGNORECASE) else text
                                except Exception:
                                    endYear = None
                                # log.info(f'[CHECK] endYear : {endYear}')

                                # Policy_title, Web_link
                                try:
                                    tagCss = ele.find_element(By.CSS_SELECTOR, "h4 a")
                                    polTitle = tagCss.text.strip() if tagCss else None
                                    webLink = tagCss.get_attribute("href") if tagCss else None
                                except Exception:
                                    polTitle = None
                                    webLink = None
                                # log.info(f'[CHECK] polTitle : {polTitle}')
                                # log.info(f'[CHECK] webLink : {webLink}')

                                dict = {
                                    'City_Column_1': [city],
                                    'Matching_City_Column_2': [cityMat],
                                    'Sector': [sector],
                                    'key': [key],
                                    'keyword': [keyword],
                                    'Announcement_Year': [annYear],
                                    'Starting_Year': [srtYear],
                                    'Ending_year': [endYear],
                                    'Policy_title': [polTitle],
                                    'Policy_type': [polType],
                                    'Web_link': [webLink],
                                    'Full_Article': [None],
                                }
                                # log.info(f'[CHECK] dict : {dict}')

                                data = pd.concat([data, pd.DataFrame.from_dict(dict)], ignore_index=True)

                            log.info(f'[CHECK] keyword : {keyword} : {len(data)}')
                            # print(f'[CHECK] keyword : {keyword} : {len(data)}')

                        except Exception as e:
                            log.error(f"Exception : {e}")

                # ==========================================================================================================
                # 상세정보 추출
                # ==========================================================================================================
                # for idx, info in data.iterrows():
                #     webLink = info['Web_link']
                #
                #     try:
                #         driver.get(webLink)
                #
                #         if driver.current_url == sysOpt['verfUrl']:
                #             initLogin(driver, sysOpt)
                #             driver.get(webLink)
                #
                #         divId = wait.until(EC.presence_of_element_located((By.ID, "divFullText")))
                #         fullArt = textProp(divId.text) if divId else None
                #     except Exception:
                #         fullArt = None
                #     data.loc[idx, 'Full_Article'] = fullArt
                #     # log.info(f'[CHECK] fullArt : {len(fullArt)}')

                # ==========================================================================================================
                # 자료 저장
                # ==========================================================================================================
                if len(data) > 0:
                    saveFile = datetime.now().strftime(sysOpt['saveFile']).format(cityMat=cityMat)
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    data.to_excel(saveFile, index=False)
                    log.info(f'[CHECK] saveFile : {saveFile}')

                # if driver:
                #     driver.quit()
                #     driver.service.stop()

                # sys.exit(0)

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