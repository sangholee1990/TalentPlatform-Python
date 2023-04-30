# -*- coding: utf-8 -*-

import logging
import logging.handlers
import os
import sys
import logging
import platform
import sys
import traceback
import urllib
from datetime import datetime
from urllib import parse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
import dfply
from gnews.utils.utils import import_or_install
from plotnine.data import *
from plotnine import *
from sspipe import p, px

import urllib.request
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote_plus
from bs4 import BeautifulSoup
from lxml import etree
import xml.etree.ElementTree as et
import requests
from lxml import html
import urllib
import math
import glob
import warnings
import requests
import pandas as pd
import time
from gnews import GNews
from pandas import json_normalize
import json
from GoogleNews import GoogleNews
from newspaper import Article
import pandas as pd
from newspaper import Article
import nltk
from newspaper import Config
from expressvpn import wrapper

# =================================================
# 초기 설정
# =================================================
warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False


# =================================================
# 함수 정의
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
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

    globalVar = {
        "prjName": prjName
        , "sysOs": platform.system()
        , "contextPath": contextPath
        , "resPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , "cfgPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , "inpPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , "figPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , "outPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , "movPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , "logPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , "mapPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , "sysPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        , "seleniumPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , "fontPath": contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)
        globalVar[key] = val.replace("\\", "/")

    return globalVar


#  초기 전달인자 설정
def initArgument(globalVar, inParams):

    for i, key in enumerate(inParams):
        if globalVar['sysOs'] in 'Linux':
            if i >= len(sys.argv[1:]): continue
            if inParams[key] is None: continue
            val = inParams[key] if sys.argv[i + 1] is None else sys.argv[i + 1]

        if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
            if inParams[key] is None: continue
            val = inParams[key]

        # self 변수에 할당
        # setattr(self, key, val)

        # 전역 변수에 할당
        globalVar[key] = val
        log.info("[CHECK] {} / val : {}".format(key, val))

    return globalVar

def getFullArticle(url):

    try:
        import_or_install('newspaper')
        from newspaper import Article

        article = Article(url="%s" %url, language='en')
        article.download()
        article.parse()
    except Exception as e:
        log.error("Exception : {}".format(e))
        return None

    return article

def searchGoogleNews(unitGoogleNews, companyInfo, keywordInfo, searchMaxPage, Article, config, saveFile):

    try:
        searchInfo = '{} {}'.format(companyInfo, keywordInfo)
        log.info("[CHECK] searchInfo : {}".format(searchInfo))

        unitGoogleNews.search(searchInfo)

        for i in range(0, searchMaxPage):
            result = unitGoogleNews.page_at(i)
            if len(result) < 1: continue

            data = pd.DataFrame(result)
            if len(data) < 1: continue

            data.insert(0, 'companyName', companyInfo)
            data.insert(1, 'keyword', keywordInfo)
            data.insert(2, 'searchName', searchInfo)

            for j in data.index:
                dataDtl = getFullArticle(data.loc[j]['link'])
                if dataDtl == None: continue

                data.loc[j, 'description'] = None if len(dataDtl.text) < 1 else dataDtl.text
                data.loc[j, 'authors'] = None if len(dataDtl.authors) < 1 else json.dumps(dataDtl.authors)
                data.loc[j, 'images'] = None if len(dataDtl.images) < 1 else [dataDtl.images]

            if not os.path.exists(saveFile + '.csv'):
                data.to_csv(saveFile + '.csv', index=False, mode='w')
            else:
                data.to_csv(saveFile + '.csv', index=False, mode='a', header=False)

            # 1분 지연
            # time.sleep(60)

            # 10초 지연
            # time.sleep(10)

    except Exception as e:
        log.error("Exception : {}".format(e))
        # vpnIpChange()


def searchGoogleNewsAll(unitGoogleNews, companyInfo, keywordInfo, saveFile):

    try:

        searchInfo = '{} {}'.format(companyInfo, keywordInfo)
        log.info("[CHECK] searchInfo : {}".format(searchInfo))

        result = unitGoogleNews.get_news(searchInfo)
        if result is None or len(result) < 1: return None

        data = json_normalize(result)
        if len(data) < 1: return None

        data.insert(0, 'companyName', companyInfo)
        data.insert(1, 'keyword', keywordInfo)
        data.insert(2, 'searchName', searchInfo)

        for i in data.index:
            dataDtl = unitGoogleNews.get_full_article(data.loc[i]['url'])
            if dataDtl is None: continue

            data.loc[i, 'description'] = None if len(dataDtl.text) < 1 else dataDtl.text
            data.loc[i, 'authors'] = None if len(dataDtl.authors) < 1 else json.dumps(dataDtl.authors)
            data.loc[i, 'images'] = None if len(dataDtl.images) < 1 else [dataDtl.images]

            if not os.path.exists(saveFile + '.csv'):
                data.to_csv(saveFile + '.csv', index=False, mode='w')
            else:
                data.to_csv(saveFile + '.csv', index=False, mode='a', header=False)

    except Exception as e:
        log.error("Exception : {}".format(e))

def vpnIpChange():
    maxAttempts = 100
    attempts = 0
    while True:
        attempts += 1
        try:
            logging.info('GETTING NEW IP')
            wrapper.random_connect()
            log.info('SUCCESS')
            return
        except Exception as e:
            log.error("Exception : {}".format(e))
            if attempts > maxAttempts:
                log.error('Max attempts reached for VPN. Check its configuration.')
                log.error('Browse https://github.com/philipperemy/expressvpn-python.')
                log.error('Program will exit.')
                exit(1)

class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # LSH0216. Python을 이용한 Google News 크롤러 개발

    # 지난번 트위터 크롤러랑 비슷한데 이번엔 조금더 간단한 작업입니다.
    # GoogleNews에 엑셀파일에 있는 기업명과 정해진 몇개의 키워드를 포함한 키워드 서칭, 그리고 정해진 날짜 스크래핑하는 crawler 제작 문의드립니다.
    # 지난번 트위터 크롤러랑 원하는 내용은 동일합니다.
    # 비용은 8만원 생각 중이고 최대한 빨리 코드를 받아보았으면 합니다.
    # 필요한 내용은 "기업명", "키워드", "제목", "내용", "링크", "날짜" 정도가 되겠습니다. 제작 가능할지요? 감사합니다.
    # 칼럼B에 있는 companyName 명과 아래의 키워드가 함께 검색되었으면 합니다.

    # 'MeToo', 'Me Too', '#MeToo', 'tarana burke', 'harvey weinstein', 'alyssa milano', 'feminism', 'feminist'
    # , 'gender', 'sexual assault', 'sexual harassment', 'sexual abuse', 'movement', 'violence', 'female', 'women', 'equality'

    # 검색 기간은 2017.10.5~2018.1.3 의 90일이 되었으면 하고요.
    # GoogleNews는 한국이 아닌, United States (English)로 설정하여 서칭 부탁드립니다.
    # 파일은 csv에 말씀드린 내용을 칼럼으로 저장되었으면 합니다.

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0216'

    log = initLog(env, contextPath, prjName)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):
        log.info("[START] __init__ : {}".format("init"))

        try:
            log.info("[CHECK] inParams : {}".format(inParams))

            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

            for key, val in globalVar.items():
                log.info("[CHECK] globalVar key / val : {} / {}".format(key, val))

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info("[END] __init__ : {}".format("init"))

    # ================================================================================================
    # 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        try:
            log.info('[START] {}'.format("exec"))

            # breakpoint()

            nltk.download('punkt')
            config = Config()
            config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'

            # ********************************************
            # 옵션 설정
            # ********************************************
            # 초기 옵션 설정
            self.sysOpt = {
                # ++++++++++++++++++++++++++++++++++++++++++
                #  [단위 시스템] 구글 뉴스 크롤러 선택
                # +++++++++++++++++++++++++++++++++++++++++
                # 시작/종료 날짜 설정 O
                'crawler' : 'A'

                # 날짜 설정 X
                # 'crawler' : 'B'

                #++++++++++++++++++++++++++++++++++++++++++
                # 공통 옵션
                # +++++++++++++++++++++++++++++++++++++++++
                # 언어 설정
                , 'language' : 'en'

                # 국가 설정
                , 'country' : 'US'

                # ++++++++++++++++++++++++++++++++++++++++++
                # 크롤러 A 옵션
                # +++++++++++++++++++++++++++++++++++++++++
                # 시간 설정
                , 'srtDate' : '10/05/2017'
                , 'endDate' : '01/03/2018'

                # 검색 최대 페이지 (페이지 당 10개)
                # , 'searchMaxPage': 2  # 테스트
                # , 'searchMaxPage': 10
                , 'searchMaxPage': 99

                # ++++++++++++++++++++++++++++++++++++++++++
                # 크롤러 B 옵션
                # +++++++++++++++++++++++++++++++++++++++++
                # 최대 검색 개수
                , 'searchMaxCnt': 2  # 테스트
                # , searchMaxCnt = 100
            }

            crawler = self.sysOpt['crawler']

            if crawler == 'A':
                # [단위 시스템] 구글 뉴스 크롤러 (시작/종료 날짜 설정 O)
                unitGoogleNews = GoogleNews(lang=self.sysOpt['language'], region=self.sysOpt['country'], start=self.sysOpt['srtDate'], end=self.sysOpt['endDate'], encode='UTF-8')
            elif crawler == 'B':
                # [단위 시스템] 구글 뉴스 크롤러 (날짜 설정 X)
                unitGoogleNewsAll = GNews(language=self.sysOpt['language'], country=self.sysOpt['country'], max_results=self.sysOpt['searchMaxCnt'])
            else:
                log.error('[ERROR] crawler : {} / {}'.format(crawler, '크롤러를 선택해주세요.'))
                raise Exception('[ERROR] crawler : {} / {}'.format(crawler, '크롤러를 선택해주세요.'))

            # 파일 정보
            inpFile = '{}/{}_{}'.format(globalVar['inpPath'], serviceName, 'Fortune1000_TwitterAccounts_final_v1.csv')

            fileInfo = glob.glob(inpFile)
            if fileInfo is None or len(fileInfo) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

            data = pd.read_csv(fileInfo[0], encoding='EUC-KR')
            companyList = data['companyName'].tolist()
            keywordList = ['MeToo', 'Me Too', '#MeToo', 'tarana burke', 'harvey weinstein', 'alyssa milano', 'feminism',
                           'feminist', 'gender', 'sexual assault', 'sexual harassment', 'sexual abuse', 'movement',
                           'violence', 'female', 'women', 'equality']

            # companyInfo = 'Walmart'
            # keywordInfo = 'gender'

            saveFile = '{}/{}_{}_{}'.format(globalVar['outPath'], serviceName, '회사 및 키워드에 따른 구글 뉴스 크롤링', crawler)

            for companyInfo in companyList[:5]: # 테스트
            # for companyInfo in companyList:
                for keywordInfo in keywordList:

                    if crawler == 'A':
                        searchGoogleNews(unitGoogleNews, companyInfo, keywordInfo, self.sysOpt['searchMaxPage'], Article, config, saveFile)
                    elif crawler == 'B':
                        searchGoogleNewsAll(unitGoogleNewsAll, companyInfo, keywordInfo, saveFile)

            # 크롤링 데이터
            dataL1 = pd.read_csv(saveFile + '.csv')
            if len(dataL1) < 1:
                log.error('[ERROR] dataL1 : {} / {}'.format(len(dataL1), '데이터가 없습니다.'))
                raise Exception('[ERROR] dataL1 : {} / {}'.format(len(dataL1), '데이터가 없습니다.'))

            # 중복 데이터 삭제
            dataL2 = dataL1.drop_duplicates()

            dataL2.to_csv(saveFile + '_FNL' + '.csv', index=False)
            dataL2.to_excel(saveFile + '_FNL' + '.xlsx', index=False)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info('[END] {}'.format("exec"))

    # 수행 프로그램 (단일 코어, 다중 코어 멀티프로세싱)
    def runPython(self):
        try:
            log.info('[START] {}'.format("runPython"))

            DtaProcess.exec(self)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e

        finally:
            log.info('[END] {}'.format("runPython"))


if __name__ == '__main__':

    try:
        print('[START] {}'.format("main"))

        # 파이썬 실행 시 전달인자를 초기 환경변수 설정
        # 입력 자료 : inpPath
        # 그림 자료 : figPath
        # 출력 자료 : outPath
        # 로그 자료 : logPath
        inParams = {
            # 'inpPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
            # , 'figPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
            # , 'outPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
            # , 'logPath': 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/test'
        }

        print("[CHECK] inParams : {}".format(inParams))

        callDtaProcess = DtaProcess(inParams)

        callDtaProcess.runPython()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
