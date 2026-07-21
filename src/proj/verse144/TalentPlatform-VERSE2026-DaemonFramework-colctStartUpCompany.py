# ================================================
# 요구사항
# ================================================
# Python을 이용한 서울캠퍼스타운 창입기업 수집

# cd /SYSTEMS/PROG/PYTHON/IDE/src/proj/verse144
# /SYSTEMS/LIB/anaconda3/envs/py39/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/proj/verse144/TalentPlatform-VERSE2026-DaemonFramework-colctStartUpCompany.py
# nohup /SYSTEMS/LIB/anaconda3/envs/py39/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/proj/verse144/TalentPlatform-VERSE2026-DaemonFramework-colctStartUpCompany.py &
# tail -f nohup.out

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
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import time
import urllib.parse as urlparse

from tensorflow.python.grappler import item
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
from bs4 import BeautifulSoup
import re

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

    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    streamHandler = logging.StreamHandler()
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1, backupCount=30, encoding='utf-8')

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

    prjName = 'colctStartUpCompany'
    serviceName = 'VERSE2026'

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
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                },
                'url': 'https://campustown.seoul.go.kr/site/main/startup/list?cp={page}&pageSize=15&sortOrder=COMP_NM&sortDirection=ASC&listType=list',
                'urlDtl': 'https://campustown.seoul.go.kr/site/main/startup/intro?compId={compId}',
                'urlRoot': 'https://campustown.seoul.go.kr',
                'saveFile': '/DATA/OUTPUT/VERSE2026/서울캠퍼스타운 창입기업.csv',
            }

            # ==========================================================================================================
            # 기본정보 수집
            # ==========================================================================================================
            # itemList = []
            # pageList = np.arange(1, 400, 1)
            #
            # # 기본 정보
            # for i, page in enumerate(pageList):
            #     percent = ((i + 1) / len(pageList)) * 100
            #     log.info(f"{page}, {percent:.1f}%")
            #
            #     urlList = sysOpt['url'].format(page=page)
            #     urlRes = requests.get(urlList, headers=sysOpt['headers'])
            #     urlRes.raise_for_status()
            #     urlSoup = BeautifulSoup(urlRes.text, 'html.parser')
            #
            #     trList = urlSoup.select('table > tbody > tr')
            #     for tr in trList:
            #         try:
            #             linkTag = tr.select_one('a')
            #             if not linkTag: continue
            #             compName = linkTag.text.strip()
            #             compId = None
            #
            #             if 'href' in linkTag.attrs:
            #                 href = linkTag['href']
            #                 parsedUrl = urlparse.urlparse(href)
            #                 compId = urlparse.parse_qs(parsedUrl.query).get('compId', [None])[0]
            #
            #             if not compId: continue
            #
            #             # 상세 정보
            #             detailUrl = sysOpt['urlDtl'].format(compId=compId)
            #
            #             detailResponse = requests.get(detailUrl, headers=sysOpt['headers'])
            #             detailResponse.raise_for_status()
            #             detailSoup = BeautifulSoup(detailResponse.text, 'html.parser')
            #
            #             item = {
            #                 '기업명': compName,
            #                 '기업ID': compId,
            #                 '상세URL': detailUrl,
            #             }
            #
            #             logoTag = detailSoup.select_one('.startup_logo img')
            #             if logoTag and 'src' in logoTag.attrs:
            #                 item['로고'] = sysOpt['urlRoot'] + logoTag['src']
            #             else:
            #                 item['로고'] = ''
            #
            #             infoList = detailSoup.select('.cont_right li')
            #             for li in infoList:
            #                 keyTag = li.select_one('h3')
            #                 if not keyTag: continue
            #
            #                 key = keyTag.text.strip()
            #
            #                 valTagA = li.select_one('a')
            #                 valTagP = li.select_one('p')
            #
            #                 if valTagA:
            #                     val = valTagA.text.strip()
            #                 elif valTagP:
            #                     val = valTagP.text.strip()
            #                 else:
            #                     val = ''
            #                 item[key] = val
            #
            #             itemList.append(item)
            #             # time.sleep(1.5)
            #         except Exception as e:
            #             log.error(f"Exception : {e}")
            #
            # data = pd.DataFrame(itemList)
            # saveFile = sysOpt['saveFile']
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # data.to_csv(saveFile, index=False, encoding='utf-8')
            # log.info(f'saveFile : {saveFile}')

            # ==========================================================================================================
            # 부가정보 수집
            # ==========================================================================================================
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            targetUrl = "http://gausslab.co.kr/"
            contactInfo = {
                '이메일': '',
                '대표번호': '',
                '주소': ''
            }

            if targetUrl:
                if not targetUrl.startswith(('http://', 'https://')):
                    targetUrl = 'http://' + targetUrl

                session = requests.Session()
                retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
                adapter = HTTPAdapter(max_retries=retry)
                session.mount('http://', adapter)
                session.mount('https://', adapter)

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }

                try:
                    response = session.get(targetUrl, headers=headers, timeout=10, verify=False)
                    response.raise_for_status()
                    response.encoding = response.apparent_encoding
                    soup = BeautifulSoup(response.text, 'html.parser')
                    textContent = soup.get_text(separator=' ', strip=True)

                    emailMatch = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', textContent)
                    if emailMatch:
                        contactInfo['이메일'] = emailMatch.group(0)

                    phoneMatch = re.search(r'(?:전화번호|대표전화|대표번호|Tel|TEL|T)\.?\s*[:]?\s*([\d]{2,3}[-\s]?\d{3,4}[-\s]?\d{4})', textContent, re.IGNORECASE)
                    if phoneMatch:
                        contactInfo['대표번호'] = phoneMatch.group(1).strip()

                    addrMatch = re.search(r'(?:주소|본점|ADDRESS)\s*[:]?\s*(.+?)(?=\s+(?:전화번호|대표번호|대표전화|Tel|Fax|이메일|사업자|COPYRIGHT|ⓒ|©|$))', textContent, re.IGNORECASE)
                    if addrMatch:
                        contactInfo['주소'] = addrMatch.group(1).strip()[:100]

                except Exception as e:
                    log.error(f"Exception : {e}")

                finally:
                    session.close()

            log.info(f"이메일: {contactInfo['이메일']}")
            log.info(f"대표번호: {contactInfo['대표번호']}")
            log.info(f"주소: {contactInfo['주소']}")

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