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

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 셀레늄 기반 유니콘팩토리 스타트업 정보 수집 및 추출

    # 원도우 X11 (X Window System) 프로토콜 지원
    # xming

    # 리눅스 CLI 실행
    # google-chrome --no-sandbo

    # 크롬 다운로드
    # https://googlechromelabs.github.io/chrome-for-testing

    # /DATA/INPUT/LSH0602/chrome-linux64/chrome --version
    # Google Chrome for Testing 131.0.6778.264

    # /DATA/INPUT/LSH0602/chromedriver-linux64/chromedriver --version
    # ChromeDriver 131.0.6778.264 (2d05e31515360f4da764174f7c448b33e36da871-refs/branch-heads/6778@{#4323})

    # cd /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
    # nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0602-DaemonFramework.py &

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
    serviceName = 'LSH0570'

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
                'loginUrl': "https://www.unicornfactory.co.kr/login",
                'listDefUrl': "https://www.unicornfactory.co.kr/datalab/startup/company-search",
                'listUrl': "https://www.unicornfactory.co.kr/datalab/startup/company-search?search=&unicorn=on&business=&tech=&cert=&stage=&amount=&location=&revenue=&history=&sort=A&limit={perPage}&page={page}",
                'listDtlUrl': "https://www.unicornfactory.co.kr/login",

                'chromeInfo': "/DATA/INPUT/LSH0602/chrome-linux64/chrome",
                'chromedriverInfo':"/DATA/INPUT/LSH0602/chromedriver-linux64/chromedriver",

                # 지연 시간
                'timeout': 60,

                # 로그인 기능
                'loginId': "backjoi@naver.com",
                'loginPw': "cjswo124!Q",

                # 자료 저장
                'saveFile': '/DATA/OUTPUT/LSH0602/%Y%m%d_data.csv',
            }

            # ==========================================================================================================
            # 전역 설정
            # ==========================================================================================================
            # 크롬 실행
            options = Options()
            options.headless = False
            options.binary_location = sysOpt['chromeInfo']
            options.add_argument("--window-size=1920,1080")

            # 백그라운드
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--headless")
            options.add_argument("--remote-debugging-port=9222")
            options.add_argument("--disable-gpu")

            # 크롬드라이브 실행
            # service = Service(ChromeDriverManager().install())
            service = Service(sysOpt['chromedriverInfo'])
            driver = webdriver.Chrome(service=service, options=options)

            # ==========================================================================================================
            # 로그인 기능
            # ==========================================================================================================
            url = sysOpt['loginUrl']
            driver.get(url)

            # 최대 timeout 대기
            wait = WebDriverWait(driver, sysOpt['timeout'])

            # 이메일 입력
            emailTag = wait.until(EC.presence_of_element_located((By.ID, "email")))
            emailTag.send_keys(sysOpt['loginId'])

            # 비밀번호 입력
            tagPassword = wait.until(EC.presence_of_element_located((By.ID, "password")))
            driver.execute_script("arguments[0].removeAttribute('disabled')", tagPassword)
            tagPassword.send_keys(sysOpt['loginPw'])

            # 로그인 버튼
            btnLogin = wait.until(EC.element_to_be_clickable((By.ID, "join_submit_btn")))
            btnLogin.click()

            # ==========================================================================================================
            # 기업정보 수집
            # ==========================================================================================================
            # 최대 개수
            url = sysOpt['listDefUrl']
            driver.get(url)
            wait = WebDriverWait(driver, sysOpt['timeout'])

            time.sleep(1)

            tagCnt = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "total")))
            maxCnt = int(re.search(r'\d+', tagCnt.text).group())

            perPage = 100
            maxPage = (maxCnt + perPage - 1) // perPage

            # 페이지 목록
            pageList = list(range(1, maxPage + 1))

            dataL1 = []
            for page in pageList:
                url = sysOpt['listUrl'].format(perPage = perPage, page=page)
                log.info(f'[CHECK] url : {url}')
                driver.get(url)

                wait = WebDriverWait(driver, sysOpt['timeout'])
                tagTable = wait.until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
                itemList = tagTable.find_elements(By.TAG_NAME, "tr")

                for item in itemList:
                    cols = item.find_elements(By.TAG_NAME, "td")
                    data = [col.text for col in cols]
                    href_value = item.find_element(By.TAG_NAME, "a").get_attribute("href")
                    data.append(href_value)

                    dataL1.append(data)

            # 테이블 헤더 가져오기
            tagTableHead = driver.find_element(By.TAG_NAME, "thead")
            headList = [th.text for th in tagTableHead.find_elements(By.TAG_NAME, "th")]
            headList.append('세부URL')

            dataL2 = pd.DataFrame(dataL1, columns=headList)

            # ==========================================================================================================
            # 기본정보 및 상세정보 추출
            # ==========================================================================================================
            for i, item in dataL2.iterrows():
                urlDtl = item['세부URL']
                log.info(f'[CHECK] urlDtl : {urlDtl}')

                driver.get(urlDtl)

                # 기본정보 추출
                try:
                    text = driver.find_element(By.CLASS_NAME, "homepage").get_attribute("href")
                    website = None if text is None or len(text) < 1 else text
                except Exception:
                    website = None

                try:
                    text = driver.find_element(By.CLASS_NAME, "address").text
                    address = None if text is None or len(text) < 1 else text
                except Exception:
                    address = None

                try:
                    intro = driver.find_element(By.CLASS_NAME, "realtxt").text
                    text = None if text is None or len(text) < 1 else text
                except Exception:
                    intro = None

                try:
                    text = driver.find_element(By.XPATH, "//strong[text()='설립일자(업력)']/following-sibling::div/p").text
                    estDate = None if text is None or len(text) < 1 else text
                except Exception:
                    estDate = None

                try:
                    text = driver.find_element(By.XPATH, "//strong[text()='사업분야']/following-sibling::div/p").text
                    businArea = None if text is None or len(text) < 1 else text
                except Exception:
                    businArea = None

                dataL2.loc[i, '웹사이트'] = website
                dataL2.loc[i, '주소'] = address
                dataL2.loc[i, '회사소개'] = intro
                dataL2.loc[i, '설립일자'] = estDate
                dataL2.loc[i, '사업분야'] = businArea

                # 상세정보 추출
                try:
                    driver.get(website)
                    pageSrc = driver.page_source
                    emailPattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
                    emailList = re.findall(emailPattern, pageSrc)

                    emailInfo = []
                    for emailInfo in emailList:
                        domain = emailInfo.split('@')[1]
                        if domain.count('.') < 3:
                            emailInfo.append(emailInfo)

                    email = None if emailInfo is None or len(emailInfo) < 1 else emailInfo
                except Exception:
                    email = None

                try:
                    phonePattern = r'\d{2,3}-\d{3,4}-\d{4}'
                    text = re.findall(phonePattern, pageSrc)
                    phone = None if text is None or len(text) < 1 else text
                except Exception:
                    phone = None

                dataL2.loc[i, '이메일'] = email
                dataL2.loc[i, '연락처'] = phone

            # ==========================================================================================================
            # 자료 저장
            # ==========================================================================================================
            saveFile = datetime.now().strftime(sysOpt['saveFile'])
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            dataL2.to_csv(saveFile, index=False)
            log.info(f'[CHECK] saveFile : {saveFile}')

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