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
import re
from typing import List, Dict, Set
import re
from collections import defaultdict
from pathlib import Path
import os
import sys
import urllib.request
from urllib.parse import urlencode

from konlpy.tag import Okt
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

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
                # 수집 설정
                'colct': {
                    'naver': {
                        'baseUrl': "https://datalab.naver.com/shoppingInsight/getKeywordRank.naver",
                        'cateList': [
                            {"name": "패션의류", "param": ["50000000"]},
                            {"name": "패션잡화", "param": ["50000001"]},
                            {"name": "화장품/미용", "param": ["50000002"]},
                            {"name": "디지털/가전", "param": ["50000003"]},
                            {"name": "가구/인테리어", "param": ["50000004"]},
                            {"name": "출산/육아", "param": ["50000005"]},
                            {"name": "식품", "param": ["50000006"]},
                            {"name": "스포츠/레저", "param": ["50000007"]},
                            {"name": "생활/건강", "param": ["50000008"]},
                            {"name": "여가/생활편의", "param": ["50000009"]},
                            {"name": "도서", "param": ["50005542"]},
                        ],
                        'headers': {
                            "Content-Type": "application/x-www-form-urlencoded",
                            "Referer": "https://datalab.naver.com",
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        },
                    },

                    'whereispost': {
                        'baseUrl': "https://whereispost.com/hot",
                        'headers': {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                        },
                    },
                    'ezme': {
                        'baseUrl': "https://rank.ezme.net",
                        'headers': {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                        },
                    },
                },

                # 가공 설정
                'filter': {
                    'stopWordFileInfo': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/word/stopwords-ko.txt',
                    'forbidWordList': ["시신", "거지", "야사", "의사", "자지", "보지", "아다", "씹고", "음탕", "후장", "병원", "환자", "진단",
                                       "증상", "증세", "재발", "방지", "시술", "본원", "상담", "고자", "충동", "후회", "고비", "인내", "참아",
                                       "자살", "음부", "고환", "오빠가", "후다", "니미", "애널", "에널", "해적", "몰래", "재생", "유발", "만족",
                                       "무시", "네요", "하더라", "품절", "매진", "마감", "의아", "의문", "의심", "가격", "정가", "구매", "판매",
                                       "매입", "지저분함", "요가", "체형", "등빨", "탈출"]
                },
            }

            # ==========================================================================================================
            # 블로그 지수에 영향을 주는 금지어 위반 목록 찾기
            # https://github.com/keunyop/BadWordCheck
            # ==========================================================================================================

            # 파일 대신 텍스트 입력
            text = """
요즘 몸도 힘들고 마음도 힘들고 이래저래 기운없는 나날들을 보내고 있어요. 몸이 피곤하니 마음도 기분도 울적한가 봐요. 뒤늦게 가을을 타는 걸까요?
하고 싶은 제 머리는 아니지만 오늘을 딸의 생애 첫 커트에 대한 간단한 일기를 포스팅하겠습니다.

지난주 금요일 오후,
함께 누워있던 딸아이가
갑자기 '엄마, 나 머리카락이 너무 귀찮아요...
나 이제 머리카락 자르고 싶어요.'라고 해서
(무려 4년 만에... 우리 딸 4살!!)

그 말과 동시에
주섬주섬 옷을 입고
미용실로 직행!! 했습니다.

무려 4년 만에 자르는 것이라
작년부터 부쩍 길어진 머리에
아침마다 빗질도, 묶는 것도 일이기에
조금만 자르자고 꼬셔도
절대 안 자르겠다고,
아빠랑 오빠 미용실에 따라가서도
절대 안 자른다고 해왔어서
머리를 자른다는 말이
너무너무 반가웠어요. :)

미리 예약을 하고 간 것이
아니라 정말 급하게 왔더니
역시나 대기가 있었습니다.

아들은 아빠랑 아무데나 가서
커트를 하지만,
딸은 그래도 여자아이고
첫 커트니 만큼 예쁘게 잘라주고 싶은
마음에 제가 이용하는 미용실로 데리고 갔어요.
ㅎㅎㅎ
            """

            # 불용어 목록
            fileList = sorted(glob.glob(sysOpt['filter']['stopWordFileInfo']))
            stopWordData = pd.read_csv(fileList[0])
            stopWordList = stopWordData['word'].tolist()

            # 금지어 목록
            forbidWordList = sysOpt['filter']['forbidWordList']
            okt = Okt()
            posTagList = okt.pos(text, stem=True)

            # 명사 추출
            keywordList = [word for word, pos in posTagList if pos in ['Noun']]

            # 불용어 제거
            keywordList = [word for word in keywordList if word not in stopWordList and len(word) > 1]

            # 빈도수 계산
            keywordCnt = Counter(keywordList)
            data = pd.DataFrame(keywordCnt.items(), columns=['keyword', 'cnt'])

            pattern = re.compile("|".join(forbidWordList))
            data['type'] = data['keyword'].apply(lambda x: '금지어' if pattern.search(x) else '일반어')

            forbidData = data[data['type'] == '금지어'].sort_values(by='cnt', ascending=False)
            normalData = data[data['type'] == '일반어'].sort_values(by='cnt', ascending=False)
            forbidList = forbidData['keyword'].tolist()
            normalList = normalData['keyword'].tolist()

            log.info(f"[CHECK] 금지어 목록: {len(forbidList)} : {forbidList}")
            log.info(f"[CHECK] 일반어 목록: {len(normalList)} : {normalList}")

            # ==========================================================================================================
            # 네이버 트렌드 기반 실시간 검색어 (분야 선택 필연)
            # 정적 크롤링
            # https://datalab.naver.com

            # 통합 검색어 트렌드 https://openapi.naver.com/v1/datalab/search
            # 쇼핑인사이트 https://openapi.naver.com/v1/datalab/shopping/categories
            # ==========================================================================================================
            try:
                dataL1 = pd.DataFrame()
                for idx, cateInfo in enumerate(sysOpt['colct']['naver']['cateList']):
                    params = {
                        "timeUnit": "date",
                        "cid": cateInfo['param'][0],
                    }

                    queryStr = urlencode(params)
                    url = f"{sysOpt['colct']['naver']['baseUrl']}?{queryStr}"

                    response = requests.post(url, headers=sysOpt['colct']['naver']['headers'])
                    if not (response.status_code == 200): continue

                    resData = response.json()
                    resDataL1 = resData[-1]

                    orgData = pd.DataFrame(resDataL1['ranks']).rename(
                        columns={
                            'rank': 'no'
                        }
                    )

                    orgData['type'] = 'naver'
                    orgData['cate'] = cateInfo['name']
                    orgData['dateTime'] = pd.to_datetime(resDataL1['date']).tz_localize('Asia/Seoul')
                    data = orgData[['type', 'cate', 'dateTime', 'no', 'keyword']]

                    if len(data) > 0:
                        dataL1 = pd.concat([dataL1, data])
            except Exception as e:
                log.error(f"네이버 검색어 수집 실패 : {e}")

            # ==========================================================================================================
            # 구글 트렌드 기반 실시간 검색어 웹
            # 동적 크롤링
            # https://trends.google.co.kr/trending?geo=KR&hl=ko
            # ==========================================================================================================
            try:
                pytrends = TrendReq(geo='ko-KR', tz=540)
                orgData = pytrends.trending_searches(pn='south_korea')

                orgDataL1 = orgData.rename(columns={0: 'keyword'})
                orgDataL1['no'] = orgDataL1.index + 1
                orgDataL1['dateTime'] = datetime.now(tz=tzKst)
                orgDataL1['type'] = 'google'
                orgDataL1['cate'] = '전체'

                data = orgDataL1[['type', 'cate', 'dateTime', 'no', 'keyword']]
                if len(data) > 0:
                    dataL1 = pd.concat([dataL1, data])
            except Exception as e:
                log.error(f"구글 검색어 수집 실패 : {e}")

            # ==========================================================================================================
            # 웨어이즈포스트 기반 실시간 검색어
            # 정적 크롤링
            # ==========================================================================================================
            try:
                response = requests.get(sysOpt['colct']['whereispost']['baseUrl'], headers=sysOpt['colct']['whereispost']['headers'])
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
                            'cate': '전체',
                            'dateTime': [dtDateTime],
                            'no': [no],
                            'keyword': [keyword],
                        }

                        data = pd.concat([data, pd.DataFrame.from_dict(dict)])

                    except Exception:
                        pass

                if len(data) > 0:
                    dataL1 = pd.concat([dataL1, data])

            except Exception as e:
                log.error(f"웨어이즈포스트 검색어 수집 실패 : {e}")

            # ==========================================================================================================
            # 이지미넷 기반 실시간 검색어
            # 정적 크롤링
            # ==========================================================================================================
            try:
                response = requests.get(sysOpt['colct']['ezme']['baseUrl'], headers=sysOpt['colct']['ezme']['headers'])
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
                            'cate': '전체',
                            'dateTime': [dtDateTime],
                            'no': [no],
                            'keyword': [keyword],
                        }

                        data = pd.concat([data, pd.DataFrame.from_dict(dict)])
                    except Exception:
                        pass

                if len(data) > 0:
                    dataL1 = pd.concat([dataL1, data])
            except Exception as e:
                log.error(f"이지미넷 검색어 수집 실패 : {e}")

            # ==========================================================================================================
            # 자료 저장
            # ==========================================================================================================
            dataL2 = dataL1.reset_index(drop=True)

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

# ==========================================================================================================
# pipeline
# ==========================================================================================================
# import transformers
# import torch
#
# model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
#
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )
#
# PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 한국어로 답변해주세요.'''
# instruction = "대한민국의 역사 소개해줘 "
#
# messages = [
#     {"role": "system", "content": f"{PROMPT}"},
#     {"role": "user", "content": f"{instruction[:2000]}"}
# ]
#
# prompt = pipeline.tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
#
# terminators = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]
#
# outputs = pipeline(
#     prompt,
#     max_new_tokens=2048,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9
# )
#
# print(outputs[0]["generated_text"][len(prompt):])
#
# # ==========================================================================================================
# # 무료 GPT
# # ==========================================================================================================
# # # GPT4All	로컬에서 실행 가능한 경량 GPT	✅ 무료
# from gpt4all import GPT4All
# #
# # # GPT 모델 로드
# # # model = GPT4All("ggml-model-gpt4all-falcon-q4_0.bin")  # deprecated
# # # model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf") # 좀 더 가벼운 모델
# model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")  # 추천: 빠르고 성능 좋은 모델
# # model2 = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
#
# for token in model.generate("Tell me a story.", streaming=True):
#     print(token, end="", flush=True)
#
# # with model.chat_session():
# #     print(model.generate("quadratic formula"))
# #
# # with model2.chat_session():
# #     print(model2.generate("quadratic formula"))
# #
# # # 한번에 여러 메시지 처리 (챗봇 모드).  n_predict 조절
# # def generate_responses(messages, model, n_predict=128):
# #     with model.chat_session():
# #         responses = []
# #         for message in messages:
# #             response = model.generate(message, max_tokens=n_predict)
# #             responses.append(response)
# #     return responses
# #
# # #  대화 예제 (챗봇)
# # messages = [
# #     # "안녕, 넌 누구니?",
# #     # "한국의 수도는 어디야?",
# #     # "오늘 날씨 어때?",
# #     "금지어 목록을 알려줘",
# #     "금지어가 포함된 문장 예시를 만들어줘",
# # ]
# #
# # responses = generate_responses(messages, model)
# #
# # print("\n-- 챗봇 대화 --")
# # for i, (message, response) in enumerate(zip(messages, responses)):
# #     print(f"User {i + 1}: {message}")
# #     print(f"Bot  {i + 1}: {response}")
# #     print("-" * 20)
# #
# # # 단일 프롬프트 생성 (일반 모드)
# # prompt = "금지어 필터링 시스템을 만드는 방법에 대한 파이썬 코드를 작성해줘. "
# # prompt += "scikit-learn, konlpy를 사용하고, "
# # prompt += "금지어 목록은 ['바보', '멍청이', '나쁜놈']으로 해줘."
# #
# # print("\n-- 단일 프롬프트 생성 --")
# # output = model.generate(prompt, max_tokens=1024)  # max_tokens: 최대 생성 길이
# # print(output)
# #
# # #  금지어 필터링 (챗봇 모드 활용)
# # def filter_text(text, model):
# #     with model.chat_session():
# #         system_template = "You are a helpful assistant that filters forbidden words.  If the text contains a forbidden word, respond with 'Filtered', otherwise respond with 'OK'."  # system prompt
# #
# #         response = model.generate(f"{system_template}\nUser: {text}", max_tokens=10)
# #
# #     if "Filtered" in response:
# #         return "Filtered"
# #     else:
# #         return "OK"
# #
# # print("\n-- 금지어 필터링 --")
# #
# # test_sentences = [
# #     "이것은 테스트 문장입니다.",
# #     "저 녀석은 정말 나쁜놈이야.",
# #     "바보는 아니지만, 조금 멍청이 같아.",
# #     "좋은 하루 되세요!",
# # ]
# #
# # for sentence in test_sentences:
# #     result = filter_text(sentence, model)
# #     print(f"'{sentence}' -> {result}")
#
#
#
# # LLaMA (Meta AI)	Facebook AI에서 제공하는 LLM	✅ 무료
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
#
# from transformers import pipeline
#
# # messages = [
# #     {"role": "user", "content": "Who are you?"},
# # ]
# # pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
# # pipe(messages)
#
#
# # 모델 및 토크나이저 로드
# # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hugFaceToken)
# # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hugFaceToken)
#
# # hugFaceToken = None
#
# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
#
#
# import os
# from huggingface_hub import constants
#
# # 방법 2: huggingface_hub 라이브러리 상수 사용 (더 안정적)
# print(constants.HF_HUB_CACHE)
# print(constants.HUGGINGFACE_HUB_CACHE)  # 이전 버전과의 호환성을 위해 존재
#
#
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hugFaceToken)
# # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hugFaceToken)
# model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGM", token=hugFaceToken)
#
# # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hugFaceToken)
# # model = AutoModelForCausalLM.from_pretrained(
# #     "meta-llama/Llama-2-7b-chat-hf",
# #     token=hugFaceToken,
# #     device_map="auto",
# #     load_in_4bit=True,
# #     low_cpu_mem_usage=True,
# # )
#
# # 금지어 필터링 함수
# def predict_llama(text):
#     prompt = f"이 텍스트가 금지어를 포함하는지 판단해줘. 금지어 포함 시 '🚨 금지어 포함', 포함되지 않으면 '✅ 정상 텍스트'라고 답변해.\n\n{text}"
#     inputs = tokenizer(prompt, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model.generate(**inputs)
#     return tokenizer.decode(outputs[0])
#
# # 테스트 실행
# print(predict_llama("이거 완전 사기야!"))
# print(predict_llama("좋은 제품이네요!"))
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import torch
#
# # 모델 및 토크나이저 로드 (GPU 사용 가능하면 자동 사용)
# model_name = "EleutherAI/polyglot-ko-1.3b"  # 작은 모델.  더 큰 모델: EleutherAI/polyglot-ko-5.8b, EleutherAI/polyglot-ko-12.8b
# # model_name = "beomi/kollama-12.8b-v2"  # KoLLaMA (한국어, LLaMA 기반)
# # model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Mistral (영어, 한국어도 일부 가능)
#
# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto",  # GPU 사용 가능하면 자동으로 사용, 없으면 CPU
#         # torch_dtype=torch.float16,  # FP16 사용 (GPU 메모리 절약, 속도 향상) - transformers 버전에 따라 지원 안될 수 있음.
#         low_cpu_mem_usage=True,
#     )
#
# except Exception as e:
#     print(f"모델 로드 중 오류 발생: {e}")
#     print("모델 이름이 정확한지, transformers, torch, accelerate, sentencepiece 라이브러리가 설치되어 있는지 확인하세요.")
#     exit()
#
# # 프롬프트 생성 (예시)
# def generate_prompt(instruction, input_text=""):
#     # 한국어 모델에 맞는 프롬프트 템플릿 사용 (모델마다 다름)
#     if "polyglot-ko" in model_name.lower():  # polyglot
#         prompt = f"### 질문: {instruction}\n\n### 답변:"
#         if input_text:
#             prompt = f"### 질문: {instruction}\n\n### 입력: {input_text}\n\n### 답변:"
#     elif "kollama" in model_name.lower():  # KoLLaMA
#         prompt = f"""아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요.
#
# ### 명령어:
# {instruction}
# ### 응답:"""
#         if input_text:
#             prompt = f"""아래는 작업을 설명하는 명령어와 컨텍스트로 구성된 입력입니다. 요청을 적절히 완료하는 응답을 작성하세요.
#
# ### 명령어:
# {instruction}
#
# ### 입력:
# {input_text}
#
# ### 응답:"""
#
#     elif "mistral" in model_name.lower():  # Mistral
#         prompt = f"[INST] {instruction} [/INST]"
#         if input_text:
#             prompt = f"[INST] {instruction}\n\n{input_text} [/INST]"  # input 예시
#
#     else:
#         # 기본 템플릿 (영어 모델에 적합)
#         prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
#         if input_text:
#             prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
#
#     return prompt
#
# # 텍스트 생성 함수
# def generate_text(instruction, input_text="", max_new_tokens=128, temperature=0.7, top_p=0.9,
#                   repetition_penalty=1.2):
#
#     prompt = generate_prompt(instruction, input_text)
#
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#
#     with torch.no_grad():  # Gradient 계산 비활성화 (메모리 절약, 속도 향상)
#         generated_ids = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,  # 생성할 최대 토큰 수
#             temperature=temperature,  # 높을수록 다양한 결과, 낮을수록 결정론적 결과
#             top_p=top_p,  # Nucleus Sampling: 확률이 높은 토큰 중에서 선택
#             repetition_penalty=repetition_penalty,  # 반복 감소 (값이 클수록 반복 줄어듬)
#             do_sample=True,  # 샘플링 기반 생성
#             pad_token_id=tokenizer.eos_token_id,  # 패딩 토큰
#             # eos_token_id=tokenizer.eos_token_id, # <eos>토큰이 생성되면, 생성 종료. (모델에 따라 설정)
#             # early_stopping=True,  # eos 토큰 나오면 생성 early stop
#
#         )
#
#     generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#
#     # 프롬프트 이후의 텍스트만 반환 (모델에 따라 조정 필요)
#     answer = generated_text[len(prompt):]
#
#     return answer
#
# # --- 사용 예시 ---
#
# # 1. 간단한 질문-답변
# instruction = "한국의 수도는 어디인가요?"
# response = generate_text(instruction)
# print(f"질문: {instruction}\n답변: {response}\n")
#
# # 2. 텍스트 요약
# instruction = "다음 텍스트를 요약해 주세요."
# input_text = """
# 인공지능(AI)은 21세기 가장 혁신적인 기술 중 하나로, ... (긴 텍스트) ...
# """
# response = generate_text(instruction, input_text)
# print(f"요약:\n{response}\n")
#
# # 3. 텍스트 생성 (스토리, 시 등)
# instruction = "바닷가에서 해질녘 풍경을 묘사하는 시를 써 주세요."
# response = generate_text(instruction, max_new_tokens=256)  # max_new_tokens 늘림
# print(f"시:\n{response}\n")
#
# # 4. 번역 (한국어 -> 영어)
# instruction = "다음 문장을 영어로 번역해 주세요."
# input_text = "오늘 날씨가 정말 좋습니다."
# response = generate_text(instruction, input_text)
# print(f"번역: {response}\n")
#
# # 5. 코드 생성
# instruction = "파이썬으로 간단한 웹 서버를 만드는 코드를 작성해 주세요."
# response = generate_text(instruction, max_new_tokens=512)  # 코드 생성이므로 max_new_tokens을 늘림
# print(f"코드:\n{response}\n")
#
# # 6. 금지어 필터링 (분류)
# instruction = "다음 문장에 욕설이나 비속어가 포함되어 있는지 판별해 주세요. 포함되어 있다면 '유해함', 포함되어 있지 않다면 '안전함'이라고 출력하세요."
# input_text = "이 서비스는 정말 최고예요!"
# response = generate_text(instruction, input_text, max_new_tokens=10)  # 짧은 응답이므로 max_tokens 줄임
# print(f"'{input_text}' 판별: {response}")
#
# input_text = "이 서비스는 정말 개쓰레기같아요."  # 욕설 포함
# response = generate_text(instruction, input_text, max_new_tokens=10)
# print(f"'{input_text}' 판별: {response}")