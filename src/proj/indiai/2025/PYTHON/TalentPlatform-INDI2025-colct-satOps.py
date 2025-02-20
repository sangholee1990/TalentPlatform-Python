# ================================================
# 요구사항
# ================================================
# Python을 이용한 NOAA/EUMETSAT 운영공지 수집

# ps -ef | grep "TalentPlatform-INDI2025-colct-kmaApiHub.py" | awk '{print $2}' | xargs kill -9

# cd /vol01/SYSTEMS/INDIAI/PROG/PYTHON
# /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-prop.py --modelList 'UMKR' --cpuCoreNum '5' --srtDate '2019-01-01' --endDate '2021-01-01'
# nohup /vol01/SYSTEMS/INDIAI/LIB/anaconda3/envs/py38/bin/python /vol01/SYSTEMS/INDIAI/PROG/PYTHON/TalentPlatform-INDI2025-prop.py --modelList 'UMKR' --cpuCoreNum '10' --srtDate '2019-01-01' --endDate '2021-01-01' &

import argparse
import glob
import logging
import logging.handlers
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
import xarray as xr
from pandas.tseries.offsets import Hour
import yaml
from multiprocessing import Pool
import multiprocessing as mp
from retrying import retry
# import cdsapi
import shutil

import requests
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime
import subprocess
from isodate import parse_duration
from pandas.tseries.offsets import DateOffset
from sklearn.neighbors import BallTree
from matplotlib import font_manager, rc
import urllib.request
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from lxml import etree

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

    saveLogFile = "{}/{}_{}_{}_{}_{}_{}.log".format(
        os.path.join(contextPath, 'log') if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
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
    log.info(f"[CHECK] inParInfo : {inParInfo}")

    for key, val in inParInfo.items():
        if val is None: continue
        # 전역 변수에 할당
        globalVar[key] = val

    # 전역 변수
    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)

        globalVar[key] = val.replace('\\', '/')

    return globalVar

def parseNoaaDtl(textDtl):

    result = None

    try:
        pattern = re.compile(
            r"Topic:\s*(?P<Topic>.*?)\s*Date/Time Issued:\s*(?P<Issued>.*?)\s*"
            r"Product\(s\) or Data Impacted:\s*(?P<Product>.*?)\s*Requested Center Point:\s*(?P<CenterPoint>.*?)\s*"
            r"Date/Time Initial Impact:\s*(?P<InitialImpact>.*?)\s*J/DAY\s*(?P<InitialImpact_JDAY>\d+)\s*"
            r"Date/Time of Expected End:\s*(?P<ExpectedEnd>.*?)\s*J/DAY\s*(?P<ExpectedEnd_JDAY>\d+)\s*"
            r"Length of Event:\s*(?P<Length>.*?)\s*Requester:\s*(?P<Requester>.*?)\s*"
            r"Priority:\s*(?P<Priority>.*?)\s*Details/Specifics:\s*(?P<Details>.*?)\s*"
            r"Web Site\(s\) for Applicable Information:\s*(?P<WebSite>.*)",
            re.DOTALL | re.MULTILINE
        )

        match = pattern.search(textDtl)

        if match:
            grpItem = match.groupdict()

            for key, val in grpItem.items():
                if not key in ['Issued', 'InitialImpact', 'ExpectedEnd']: continue
                try:
                    grpItem[key] = pd.to_datetime(grpItem[key], errors='coerce')
                except ValueError:
                    pass

            result = grpItem

    except Exception as e:
        log.error(f'Exception : {e}')

    return result

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'INDI2025'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info(f"[START] __init__ : init")

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

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
                # 예보시간 시작일, 종료일, 시간 간격 (연 1y, 월 1m, 일 1d, 시간 1h, 분 1t, 초 1s)
                # 'srtDate': globalVar['srtDate'],
                # 'endDate': globalVar['endDate'],
                'srtDate': '2019-01-01',
                'endDate': '2025-03-01',
                'invDate': '1m',

                # 수행 목록
                # 'modelList': [globalVar['modelList']],
                'modelList': ['UMKR'],

                # 비동기 다중 프로세스 개수
                # 'cpuCoreNum': globalVar['cpuCoreNum'],
                'cpuCoreNum': '5',

                # 설정 파일
                'CFG': {
                    'siteInfo': '/DATA/PROP/SAMPLE/site_info.csv',
                    'umkrFileInfo': '/DATA/COLCT/UMKR/201901/01/UMKR_l015_unis_H00_201901011200.grb2',
                },

                'UMKR': {
                    'fileList': '/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d*.grb2',
                    'saveFile': '/DATA/PROP/UMKR/%Y%m/UMKR_FOR_%Y%m%d.nc',
                },
                # 'ACT': {
                #     'ASOS': {
                #         'searchFileList': f"/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d%H%M.grb2",
                #         'invDate': '6h',
                #     },
                #     'AWS': {
                #         'searchFileList': f"/DATA/COLCT/UMKR/%Y%m/%d/UMKR_l015_unis_H*_%Y%m%d%H%M.grb2",
                #         'invDate': '6h',
                #     },
                # },
            }

            # **************************************************************************************************************
            # NOAA 운영공지
            # **************************************************************************************************************
            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

            rootUrl = 'https://www.ospo.noaa.gov'
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            }

            for dtDateInfo in dtDateList:
                log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                urlPattern = f"{rootUrl}/data/messages/%Y/%Y-%m-include.html"
                url = dtDateInfo.strftime(urlPattern)

                response = requests.get(url, headers=headers)
                if not (response.status_code == 200): return

                soup = BeautifulSoup(response.text, 'html.parser')

                tagList = soup.find('div', {'id': 'msg-list'}).find_all('a')
                for tagInfo in tagList:
                    # log.info(f'[CHECK] tagInfo : {tagInfo}')

                    try:
                        title = None if tagInfo is None or len(tagInfo) < 1 else tagInfo.text.strip()
                        href = None if tagInfo is None or len(tagInfo.get('href')) < 1 else tagInfo.get('href')
                    except Exception:
                        title = None
                        href = None
                    # log.info(f'[CHECK] title : {title}')
                    # log.info(f'[CHECK] href : {href}')

                    urlDtl = f"{rootUrl}/{href}"
                    response = requests.get(urlDtl, headers=headers)
                    if not (response.status_code == 200): return

                    soupDtl = BeautifulSoup(response.text, 'html.parser')
                    textDtl = soupDtl.text.strip()

                    data = pd.DataFrame({
                        'title': [title],
                        'url': [url],
                        'urlDtl': [urlDtl],
                    })

                    dictDtl = parseNoaaDtl(textDtl)
                    dataL1 = pd.concat([data, pd.DataFrame.from_dict([dictDtl])], axis=1)

                    from urllib.parse import urlparse

                    # aa = 'https://www.ospo.noaa.gov//data/messages/2025/02/MSG_20250201_1427.html'

                    parsed_url = urlparse(url)
                    path_parts = parsed_url.path.split('/')
                    filename = path_parts[-1]




                    # dictDtl['title'] = [title]
                    # dictDtl['url'] = [url]
                    # dictDtl['urlDtl'] = [urlDtl]

                    data = pd.DataFrame(dictDtl)


                    # https://www.ospo.noaa.gov/data/messages/2025/02/MSG_20250201_1427.html
                    # lxml = etree.HTML(str(soup))

                # try:
                #     tag = lxml.xpath('/html/body/content/div/div/div/div[1]/text()')[0]
                #     match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', tag.strip())
                #     sDateTime = None if match is None else match.group(0)
                #     dtDateTime = pd.to_datetime(sDateTime).tz_localize('Asia/Seoul')
                # except Exception:
                #     dtDateTime = None
                # log.info(f'[CHECK] dtDateTime : {dtDateTime}')
                #
                # noList = soup.find('ul', {'class': 'list-group bg-white'}).find_all("span", {'class': 'rank daum_color'})
                # keywordList = soup.find('ul', {'class': 'list-group bg-white'}).find_all("span", {'class': 'keyword'})
                #
                # data = pd.DataFrame()
                # for noInfo, keywordInfo in zip(noList, keywordList):
                #     try:
                #         no = None if noInfo is None or len(noInfo) < 1 else noInfo.text.strip()
                #         keyword = None if keywordInfo is None or len(keywordInfo) < 1 else keywordInfo.text.strip()
                #
                #         dict = {
                #             'type': ['whereispost'],
                #             'cate': '전체',
                #             'dateTime': [dtDateTime],
                #             'no': [no],
                #             'keyword': [keyword],
                #         }
                #
                #         data = pd.concat([data, pd.DataFrame.from_dict(dict)])
                #
                #     except Exception:
                #         pass
                #
                # if len(data) > 0:
                #     dataL1 = pd.concat([dataL1, data])

            # **************************************************************************************************************
            # EUMETSAT 운영공지
            # **************************************************************************************************************


            # filePattern = sysOpt['CFG']['siteInfo']
            # fileList = sorted(glob.glob(filePattern))
            # if fileList is None or len(fileList) < 1:
            #     log.error(f"filePattern : {filePattern} / 파일을 확인해주세요.")
            #     raise Exception(f"filePattern : {filePattern} / 파일을 확인해주세요.")
            # cfgData = pd.read_csv(fileList[0])
            # cfgDataL1 = matchStnFor(sysOpt['CFG'], cfgData)
            #
            # # **************************************************************************************************************
            # # 비동기 다중 프로세스 수행
            # # **************************************************************************************************************
            # # 비동기 다중 프로세스 개수
            # pool = Pool(int(sysOpt['cpuCoreNum']))
            #
            # for modelType in sysOpt['modelList']:
            #     modelInfo = sysOpt.get(modelType)
            #     if modelInfo is None: continue
            #
            #     # 시작일/종료일 설정
            #     dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            #     dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            #     dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])
            #
            #     for dtDateInfo in dtDateList:
            #         # log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
            #         pool.apply_async(propUmkr, args=(modelInfo, cfgDataL1, dtDateInfo))
            #
            #     pool.close()
            #     pool.join()

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
        print('[END] main')
