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
    # Python을 이용한 셀레늄 기반 동적 엑셀 다운로드 및 데이터 가공

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
                # 구글맵 상위 주소
                'rootUrl': 'https://www.google.com/maps'

                # 크롬 지연 설정
                , 'chrDelayOpt': 10

                # 인증 초기화 여부
                , 'isInit': False

                # 수행 목록
                , 'procList': {
                    'ConsumptionTrendState': 'Consumption Trend State*.xlsx'
                    , 'State-wisePetroleumProductsConsumptionTrend': 'State-wise Petroleum Products Consumption Trend (for All States).xlsx'
                    , 'State-wisePetroleumProductsConsumptionTrend2': 'State-wise Petroleum Products Consumption Trend2 (for All States).xlsx'
                }
            }

            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'HN_M.csv')
            # fileList = sorted(glob.glob(inpFile))

            # ======================================================================================================
            # 데이터 전처리
            # ======================================================================================================
            # procInfo = 'ConsumptionTrendState'
            # procInfo = 'State-wisePetroleumProductsConsumptionTrend'
            # procInfo = 'State-wisePetroleumProductsConsumptionTrend2'
            for procInfo in sysOpt['procList']:
                log.info(f'[CHECK] procInfo / {procInfo}')

                inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, sysOpt['procList'][procInfo])
                fileList = sorted(glob.glob(inpFile))

                if fileList is None or len(fileList) < 1:
                    # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                    continue

                # fileInfo = fileList[0]
                dataL2 = pd.DataFrame()
                for fileInfo in fileList:
                    log.info(f'[CHECK] fileInfo : {fileInfo}')
                    data = pd.read_excel(fileInfo)
                    dataL1 = data.dropna()
                    # fileName = os.path.basename(fileInfo)

                    dataL2 = pd.concat([dataL2, dataL1])

                # dataL2.columns
                if re.search('ConsumptionTrendState', procInfo, re.IGNORECASE):
                    uniqData = dataL2.drop_duplicates().reset_index(drop=True)
                    dataL3 = uniqData.pivot_table(index=['State', 'Year'], columns='Sectors', values='Quantity in Million Tonnes').reset_index(drop=False)
                else:
                    dataL3 = dataL2[['State', 'Year', 'Quantity (in 000Tonne)']]

                saveFile = '{}/{}/{}.xlsx'.format(globalVar['outPath'], serviceName, procInfo)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL3.to_excel(saveFile, index=False)
                print(f'[CHECK] saveFile : {saveFile}')

            # ======================================================================================================
            # 크롤링 전역 설정
            # ======================================================================================================
            # # 크롬 설정
            # options = Options()
            # options.headless = False
            # options.add_argument("--window-size=1920,1080")
            #
            # # 백그라운드 화면 여부
            # # options.add_argument('--headless')
            # # options.add_argument('--no-sandbox')
            # # options.add_argument('--disable-dev-shm-usage')
            #
            # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

            # ======================================================================================================
            # 크롤링 1번
            # ======================================================================================================
            # try:
            #     urlInfo = 'https://iced.niti.gov.in/energy/fuel-sources/coal/consumption#state'
            #     driver.implicitly_wait(sysOpt['chrDelayOpt'])
            #     driver.get(urlInfo)
            #
            #     WebDriverWait(driver, sysOpt['chrDelayOpt']).until(
            #         EC.element_to_be_clickable((By.XPATH, '/html/body/app-root/div/app-consumption/section/div/div/div[1]/app-fuel-sources-accordion/accordion/accordion-group[5]/div/div[2]/div/p[3]'))
            #     ).click()
            #
            #     selTag = Select(driver.find_element(By.XPATH,'//*[@id="download-consumption-trend-state-section"]/div[1]/div/div/select'))
            #     for i, opt in  enumerate(selTag.options):
            #         # if i < 24: continue
            #         value = opt.get_attribute('value')
            #         log.info(f'[CHECK] i / {i} / text : {opt.text} / value : {value}')
            #
            #         # State 선택
            #         selTag.select_by_value(value)
            #
            #         time.sleep(2)
            #
            #         # 엑셀 아이콘 선택
            #         WebDriverWait(driver, sysOpt['chrDelayOpt']).until(
            #             EC.element_to_be_clickable((By.XPATH, '//*[@id="download-consumption-trend-state-section"]/div[1]/div/div/div/li[2]/a/img'))
            #         ).click()
            #
            #         if (sysOpt['isInit'] == False):
            #             nameEle = driver.find_element(By.ID, 'name')
            #             nameEle.clear()
            #             nameEle.send_keys('test')
            #
            #             emailEle = driver.find_element(By.ID, 'email')
            #             emailEle.clear()
            #             emailEle.send_keys('test@gmail.com')
            #
            #             orgEle = driver.find_element(By.ID, 'organization')
            #             orgEle.clear()
            #             orgEle.send_keys('test')
            #
            #             # 리캡차 프레임으로 전환
            #             frames = driver.find_elements(By.TAG_NAME, 'iframe')
            #             for frame in frames:
            #                 if 'recaptcha' in frame.get_attribute('src'):
            #                     driver.switch_to.frame(frame)
            #                     break
            #
            #             # 리캡차 체크박스 클릭
            #             WebDriverWait(driver, 10).until(
            #                 EC.element_to_be_clickable((By.CLASS_NAME, 'recaptcha-checkbox-border'))
            #             ).click()
            #
            #             # 프레임 밖으로 전환
            #             driver.switch_to.default_content()
            #
            #             sysOpt['isInit'] = True
            #
            #         time.sleep(1)
            #
            #         # 다운로드 요청
            #         WebDriverWait(driver, sysOpt['chrDelayOpt']).until(
            #             EC.element_to_be_clickable((By.XPATH, '//*[@id="enquiry-form"]/div[4]/div/button'))
            #         ).click()
            #
            #         time.sleep(1)
            #
            #         # 다운로드 완료
            #         WebDriverWait(driver, sysOpt['chrDelayOpt']).until(
            #             EC.element_to_be_clickable((By.XPATH, '/html/body/div[3]/div/div[6]/button[1]'))
            #         ).click()
            #
            # except Exception as e:
            #     log.error(f"Exception : {str(e)}")
            #
            # finally:
            #     driver.quit()

            # ======================================================================================================
            # 크롤링 2번
            # ======================================================================================================
            # try:
            #     urlInfo = 'https://iced.niti.gov.in/energy/fuel-sources/oil/consumption'
            #     driver.implicitly_wait(sysOpt['chrDelayOpt'])
            #     driver.get(urlInfo)
            #
            #     selEle = driver.find_element(By.XPATH, '//*[@id="download-state-wise-cons-trend-section"]/app-chart-option-menu-strip/div/div/div[1]/select')
            #     driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", selEle)
            #     selTag = Select(selEle)
            #     for i, opt in  enumerate(selTag.options):
            #         if i > 0: continue
            #         value = opt.get_attribute('value')
            #         log.info(f'[CHECK] i / {i} / text : {opt.text} / value : {value}')
            #
            #         # State 선택
            #         selTag.select_by_value(value)
            #
            #         time.sleep(2)
            #
            #         # 엑셀 아이콘 선택
            #         WebDriverWait(driver, sysOpt['chrDelayOpt']).until(
            #             EC.element_to_be_clickable((By.XPATH, '//*[@id="download-state-wise-cons-trend-section"]/app-chart-option-menu-strip/div/div/div[2]/li[2]/a/img'))
            #         ).click()
            #
            #         if (sysOpt['isInit'] == False):
            #             nameEle = driver.find_element(By.ID, 'name')
            #             nameEle.clear()
            #             nameEle.send_keys('test')
            #
            #             emailEle = driver.find_element(By.ID, 'email')
            #             emailEle.clear()
            #             emailEle.send_keys('test@gmail.com')
            #
            #             orgEle = driver.find_element(By.ID, 'organization')
            #             orgEle.clear()
            #             orgEle.send_keys('test')
            #
            #             # 리캡차 프레임으로 전환
            #             frames = driver.find_elements(By.TAG_NAME, 'iframe')
            #             for frame in frames:
            #                 if 'recaptcha' in frame.get_attribute('src'):
            #                     driver.switch_to.frame(frame)
            #                     break
            #
            #             # 리캡차 체크박스 클릭
            #             WebDriverWait(driver, 10).until(
            #                 EC.element_to_be_clickable((By.CLASS_NAME, 'recaptcha-checkbox-border'))
            #             ).click()
            #
            #             # 프레임 밖으로 전환
            #             driver.switch_to.default_content()
            #
            #             sysOpt['isInit'] = True
            #
            #         time.sleep(1)
            #
            #         # 다운로드 요청
            #         WebDriverWait(driver, sysOpt['chrDelayOpt']).until(
            #             EC.element_to_be_clickable((By.XPATH, '//*[@id="enquiry-form"]/div[4]/div/button'))
            #         ).click()
            #
            #         time.sleep(1)
            #
            #         # 다운로드 완료
            #         WebDriverWait(driver, sysOpt['chrDelayOpt']).until(
            #             EC.element_to_be_clickable((By.XPATH, '/html/body/div[3]/div/div[6]/button[1]'))
            #         ).click()
            #
            # except Exception as e:
            #     log.error(f"Exception : {str(e)}")
            #
            # finally:
            #     driver.quit()

            # ======================================================================================================
            # 크롤링 3번
            # ======================================================================================================
            # try:
            #     urlInfo = 'https://iced.niti.gov.in/energy/fuel-sources/oil/consumption'
            #     driver.implicitly_wait(sysOpt['chrDelayOpt'])
            #     driver.get(urlInfo)
            #
            #     WebDriverWait(driver, sysOpt['chrDelayOpt']).until(
            #         EC.element_to_be_clickable((By.XPATH, '/html/body/app-root/div/app-consumption/section/div/div/div[1]/app-fuel-sources-accordion/accordion/accordion-group[5]/div/div[2]/div/p[2]'))
            #     ).click()
            #
            #     selEle = driver.find_element(By.XPATH, '//*[@id="download-sector-wise-cons-trend-section"]/app-chart-option-menu-strip/div/div/div[1]/select')
            #     driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", selEle)
            #     selTag = Select(selEle)
            #     for i, opt in  enumerate(selTag.options):
            #         if i > 0: continue
            #         value = opt.get_attribute('value')
            #         log.info(f'[CHECK] i / {i} / text : {opt.text} / value : {value}')
            #
            #         # State 선택
            #         selTag.select_by_value(value)
            #
            #         time.sleep(2)
            #
            #         # 엑셀 아이콘 선택
            #         WebDriverWait(driver, sysOpt['chrDelayOpt']).until(
            #             EC.element_to_be_clickable((By.XPATH, '//*[@id="download-sector-wise-cons-trend-section"]/app-chart-option-menu-strip/div/div/div[2]/li[2]/a/img'))
            #         ).click()
            #
            #         if (sysOpt['isInit'] == False):
            #             nameEle = driver.find_element(By.ID, 'name')
            #             nameEle.clear()
            #             nameEle.send_keys('test')
            #
            #             emailEle = driver.find_element(By.ID, 'email')
            #             emailEle.clear()
            #             emailEle.send_keys('test@gmail.com')
            #
            #             orgEle = driver.find_element(By.ID, 'organization')
            #             orgEle.clear()
            #             orgEle.send_keys('test')
            #
            #             # 리캡차 프레임으로 전환
            #             frames = driver.find_elements(By.TAG_NAME, 'iframe')
            #             for frame in frames:
            #                 if 'recaptcha' in frame.get_attribute('src'):
            #                     driver.switch_to.frame(frame)
            #                     break
            #
            #             # 리캡차 체크박스 클릭
            #             WebDriverWait(driver, 10).until(
            #                 EC.element_to_be_clickable((By.CLASS_NAME, 'recaptcha-checkbox-border'))
            #             ).click()
            #
            #             # 프레임 밖으로 전환
            #             driver.switch_to.default_content()
            #
            #             sysOpt['isInit'] = True
            #
            #         time.sleep(1)
            #
            #         # 다운로드 요청
            #         WebDriverWait(driver, sysOpt['chrDelayOpt']).until(
            #             EC.element_to_be_clickable((By.XPATH, '//*[@id="enquiry-form"]/div[4]/div/button'))
            #         ).click()
            #
            #         time.sleep(1)
            #
            #         # 다운로드 완료
            #         WebDriverWait(driver, sysOpt['chrDelayOpt']).until(
            #             EC.element_to_be_clickable((By.XPATH, '/html/body/div[3]/div/div[6]/button[1]'))
            #         ).click()
            #
            # except Exception as e:
            #     log.error(f"Exception : {str(e)}")
            #
            # finally:
            #     driver.quit()

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