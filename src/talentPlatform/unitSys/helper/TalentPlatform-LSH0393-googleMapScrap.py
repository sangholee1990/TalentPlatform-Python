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

import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import googlemaps
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from urllib.request import urlopen
from urllib import parse
from urllib.request import Request
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import json
import math
from scipy import spatial
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

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

        # 글꼴 설정
        fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        plt.rcParams['font.family'] = fontName

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
    # Python을 이용한 주요 나라 및 도시를 기준으로 위경도 매칭 (구글맵 스크래핑 활용)

    # nohup python3 TalentPlatform-LSH0393-googleMapScrap.py &

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0393'

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

                # 옵션 설정
                sysOpt = {
                    # 구글맵 상위 주소
                    'rootUrl': 'https://www.google.com/maps/place/'

                    # 크롬 지연 설정
                    , 'chrDelayOpt': 10
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 구글맵 상위 주소
                    'rootUrl' : 'https://www.google.com/maps'
                    # 'rootUrl' : 'https://www.google.com/maps/place'

                    # 크롬 지연 설정
                    , 'chrDelayOpt' : 10
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 크롬 설정
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            driver.implicitly_wait(sysOpt['chrDelayOpt'])

            # ********************************************************************
            # 파일 읽기
            # ********************************************************************
            # inpFile = '{}/{}/{}.xlsx'.format(globalVar['inpPath'], serviceName, 'trafficindex')
            # inpFile = '{}/{}/{}.xlsx'.format(globalVar['inpPath'], serviceName, 'trafficindexupdatewithouttres')
            # inpFile = '{}/{}/{}.xlsx'.format(globalVar['inpPath'], serviceName, 'trafficindexwithmexico')
            inpFile = '{}/{}/{}.xlsx'.format(globalVar['inpPath'], serviceName, '0206columcambopanama')

            # 파일 검사
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

            # 파일 읽기
            # countryInfo.Country = 'Burundi'
            countryData = pd.read_excel(fileList[0], sheet_name=0)
            for i, countryInfo in countryData.iterrows():

                if pd.isna(countryInfo.Country): continue
                # if re.search('Argentina|Bolivia|Burundi|Cambodia|Chile', countryInfo.Country): continue
                # if re.search('Cambodia|Cuba', countryInfo.Country): continue
                # if re.search('Columbia|Cambodia', countryInfo.Country): continue

                log.info(f'[CHECK] countryInfo : {countryInfo.Country}')
                data = pd.read_excel(fileList[0], sheet_name = countryInfo.Country, skiprows=1, index_col=False)

                # 동적 컬럼 선택
                if pd.Series(['City', 'Province']).isin(data.columns).all():
                    data['addr'] = data['City'] + ', ' + data['Province'] + ', ' + countryInfo.Country
                elif pd.Series(['City']).isin(data.columns).all():
                    data['addr'] = data['City'] + ', ' + countryInfo.Country
                elif pd.Series(['Province']).isin(data.columns).all():
                    data['addr'] = data['Province'] + ', ' + countryInfo.Country
                else:
                    data['addr'] = countryInfo.Country

                # ********************************************************************
                # 주요 나라 시트를 기준으로 구글 위경도 환산
                # ********************************************************************
                # 중복없는 주소 목록
                addrList = set(data['addr'])

                # addrInfo = 'Reconquista, Argentina'
                # addrInfo = 'Colón, Putumayo, Cambodia'
                # addrInfo = 'Esperanza, Argentina'
                matData = pd.DataFrame()
                for i, addrInfo in enumerate(addrList):

                    log.info(f'[CHECK] [{round((i / len(addrList)) * 100.0, 2)}] addrInfo : {addrInfo}')

                    # 초기값 설정
                    matData.loc[i, 'addr'] = addrInfo
                    matData.loc[i, 'glat'] = None
                    matData.loc[i, 'glon'] = None

                    try:

                        urlInfo = '{}/{}/{}'.format(sysOpt['rootUrl'], 'place', parse.quote(addrInfo))

                        # URL 진행 요청
                        driver.get(urlInfo)

                        # 검색 선택
                        WebDriverWait(driver, sysOpt['chrDelayOpt']).until(
                            EC.element_to_be_clickable((By.ID, 'searchbox-searchbutton'))
                        ).click()

                        # URL 진행 완료
                        while True:
                            if len(driver.current_url.replace(urlInfo, '')) < 1: continue
                            break

                        # 위/경도 반환
                        geoInfo = parse.unquote(driver.current_url).replace(sysOpt['rootUrl'], '').split('/')[3].split(',')
                        matData.loc[i, 'glat'] = float(geoInfo[0].replace('@', ''))
                        matData.loc[i, 'glon'] = float(geoInfo[1])

                    except Exception as e:
                        log.error("Exception : {}".format(e))

                # addr를 기준으로 병합
                dataL1 = data.merge(matData, left_on=['addr'], right_on=['addr'], how='inner')

                # 파일 저장
                saveFile = '{}/{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, 'sheet', countryInfo.Country)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL1.to_csv(saveFile, index=False)
                log.info('[CHECK] saveFile : {}'.format(saveFile))

                # 엑셀 저장
                # saveFile = '{}/{}/{}.xlsx'.format(globalVar['outPath'], serviceName, 'trafficindexupdatewithouttres-geo')
                # saveFile = '{}/{}/{}.xlsx'.format(globalVar['outPath'], serviceName, 'trafficindexwithmexico-geo')
                saveFile = '{}/{}/{}.xlsx'.format(globalVar['outPath'], serviceName, '0206columcambopanama-geo')

                try:
                    if not os.path.exists(saveFile):
                        with pd.ExcelWriter(saveFile, mode='w', engine='openpyxl') as writer:
                            dataL1.to_excel(writer, sheet_name=countryInfo.Country, index=False)
                    else:
                        with pd.ExcelWriter(saveFile, mode='a', engine='openpyxl') as writer:
                            dataL1.to_excel(writer, sheet_name=countryInfo.Country, index=False)
                except Exception as e:
                    log.error("Exception : {}".format(e))

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