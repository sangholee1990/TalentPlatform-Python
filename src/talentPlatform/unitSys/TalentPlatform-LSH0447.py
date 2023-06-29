# -*- coding: utf-8 -*-

import argparse
import glob
import json
import logging
import logging.handlers
import os
import platform
import re
import sys
import traceback
import warnings
from collections import Counter
from datetime import datetime

import nltk
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from GoogleNews import GoogleNews
from gnews import GNews
from gnews.utils.utils import import_or_install
from konlpy.tag import Twitter
from newspaper import Article
from newspaper import Config
from pandas import json_normalize
from pandas.api.types import CategoricalDtype
from plotnine import *
from wordcloud import WordCloud

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
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
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

        log.info("[CHECK] {} / val : {}".format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar


def getFullArticle(url):
    try:
        import_or_install('newspaper')
        from newspaper import Article

        article = Article(url="%s" % url, language='en')
        article.download()
        article.parse()
    except Exception as e:
        log.error("Exception : {}".format(e))
        return None

    return article


def subCrawler(sysOpt):
    log.info('[START] {}'.format('subCrawler'))

    try:
        nltk.download('punkt')
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'

        subOpt = sysOpt['subCrawler']
        crawler = subOpt['crawler']

        if crawler == 'A':
            # [단위 시스템] 구글 뉴스 크롤러 (시작/종료 날짜 설정 O)
            unitGoogleNews = GoogleNews(lang=subOpt['language'], region=subOpt['country'],
                                        start=subOpt['srtDate'], end=subOpt['endDate'], encode='UTF-8')
        elif crawler == 'B':
            # [단위 시스템] 구글 뉴스 크롤러 (날짜 설정 X)
            unitGoogleNewsAll = GNews(language=subOpt['language'], country=subOpt['country'],
                                      max_results=subOpt['searchMaxCnt'])
        else:
            log.error('[ERROR] crawler : {} / {}'.format(crawler, '크롤러를 선택해주세요.'))
            raise Exception('[ERROR] crawler : {} / {}'.format(crawler, '크롤러를 선택해주세요.'))

        domainList = subOpt['domainList']
        keywordList = subOpt['keywordList']
        # domainInfo = 'Walmart'
        # keywordInfo = 'gender'

        saveCsvFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, '지역 및 키워드에 따른 구글 뉴스 크롤링', crawler)
        os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)

        for domainInfo in domainList:
            for keywordInfo in keywordList:

                if crawler == 'A':
                    result = searchGoogleNews(unitGoogleNews, domainInfo, keywordInfo, subOpt['searchMaxPage'], Article, config, saveCsvFile)
                elif crawler == 'B':
                    result = searchGoogleNewsAll(unitGoogleNewsAll, domainInfo, keywordInfo, saveCsvFile)

        # log.info(f'[CHECK] result : {result}')
        log.info(f'[CHECK] saveCsvFile : {saveCsvFile}')

        # 크롤링 데이터
        dataL1 = pd.read_csv(saveCsvFile)
        if (len(dataL1) < 1):
            log.error('dataL1 : {} / {}'.format(len(dataL1), '입력 자료를 확인해주세요.'))
            raise Exception('dataL1 : {} / {}'.format(len(dataL1), '입력 자료를 확인해주세요.'))

        # 중복 데이터 삭제
        dataL2 = dataL1.drop_duplicates()

        # 시작일/종료일에 대한 필터
        dataL2['datetime'] = pd.to_datetime(dataL2['date'])
        dataL3 = dataL2.loc[
            (dataL2['datetime'] >= pd.to_datetime(subOpt['srtDate'])) & (dataL2['datetime'] <= pd.to_datetime(subOpt['endDate']))
        ].reset_index(drop=True)

        saveCsvFnlFile = '{}/{}/{}_{}_FNL.csv'.format(globalVar['outPath'], serviceName, '지역 및 키워드에 따른 구글 뉴스 크롤링', crawler)
        os.makedirs(os.path.dirname(saveCsvFnlFile), exist_ok=True)
        dataL3.to_csv(saveCsvFnlFile, index=False)
        log.info(f'[CHECK] saveCsvFnlFile : {saveCsvFnlFile}')

        saveXlsxFnlFile = '{}/{}/{}_{}_FNL.xlsx'.format(globalVar['outPath'], serviceName, '지역 및 키워드에 따른 구글 뉴스 크롤링', crawler)
        os.makedirs(os.path.dirname(saveXlsxFnlFile), exist_ok=True)
        dataL3.to_excel(saveXlsxFnlFile, index=False)
        log.info(f'[CHECK] saveXlsxFnlFile : {saveXlsxFnlFile}')

    except Exception as e:
        log.error("Exception : {}".format(e))

    finally:
        log.info('[END] {}'.format('subCrawler'))


def searchGoogleNews(unitGoogleNews, domainInfo, keywordInfo, searchMaxPage, Article, config, saveFile):

    log.info('[START] {}'.format('searchGoogleNews'))

    result = None

    try:
        searchInfo = '{} {}'.format(domainInfo, keywordInfo)
        log.info("[CHECK] searchInfo : {}".format(searchInfo))

        unitGoogleNews.search(searchInfo)

        for i in range(0, searchMaxPage):
            result = unitGoogleNews.page_at(i)
            if len(result) < 1: continue

            data = pd.DataFrame(result)
            if len(data) < 1: continue

            data.insert(0, 'domainName', domainInfo)
            data.insert(1, 'keyword', keywordInfo)
            data.insert(2, 'searchName', searchInfo)

            for j in data.index:
                dataDtl = getFullArticle(data.loc[j]['link'])
                if dataDtl == None: continue

                data.loc[j, 'description'] = None if len(dataDtl.text) < 1 else dataDtl.text
                data.loc[j, 'authors'] = None if len(dataDtl.authors) < 1 else json.dumps(dataDtl.authors)
                data.loc[j, 'images'] = None if len(dataDtl.images) < 1 else [dataDtl.images]

            if not os.path.exists(saveFile):
                data.to_csv(saveFile, index=False, mode='w')
            else:
                data.to_csv(saveFile, index=False, mode='a', header=False)

            # 1분 지연
            # time.sleep(60)

            # 10초 지연
            # time.sleep(10)

        result = {
            'msg': 'succ'
            , 'saveFile': saveFile
            , 'isExist': os.path.exists(saveFile)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('searchGoogleNews'))


def webTextPrep(text):

    log.info('[START] {}'.format('webTextPrep'))

    result = None

    try:

        resText = text.strip()
        resText = resText.strip('""')
        resText = re.sub('[a-zA-Z]', '', resText)
        resText = re.sub('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]', '', resText)

        return resText

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('webTextPrep'))

def searchGoogleNewsAll(unitGoogleNews, domainInfo, keywordInfo, saveFile):

    log.info('[START] {}'.format('searchGoogleNewsAll'))

    result = None

    try:

        searchInfo = '{} {}'.format(domainInfo, keywordInfo)
        log.info("[CHECK] searchInfo : {}".format(searchInfo))

        result = unitGoogleNews.get_news(searchInfo)
        if result is None or len(result) < 1: return None

        data = json_normalize(result)
        if len(data) < 1: return None

        data.insert(0, 'domainName', domainInfo)
        data.insert(1, 'keyword', keywordInfo)
        data.insert(2, 'searchName', searchInfo)

        for i in data.index:
            dataDtl = unitGoogleNews.get_full_article(data.loc[i]['url'])
            if dataDtl is None: continue

            data.loc[i, 'description'] = None if len(dataDtl.text) < 1 else dataDtl.text
            data.loc[i, 'authors'] = None if len(dataDtl.authors) < 1 else json.dumps(dataDtl.authors)
            data.loc[i, 'images'] = None if len(dataDtl.images) < 1 else [dataDtl.images]

            if not os.path.exists(saveFile):
                data.to_csv(saveFile, index=False, mode='w')
            else:
                data.to_csv(saveFile, index=False, mode='a', header=False)

        result = {
            'msg': 'succ'
            , 'saveFile': saveFile
            , 'isExist': os.path.exists(saveFile)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('searchGoogleNewsAll'))

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 구글 뉴스에서 5종 범죄 키워드 수집체계

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0447'

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
                    # +++++++++++++++++++++++++++++++++++++++++
                    #  [단위 시스템] 구글 뉴스 크롤러 선택
                    # +++++++++++++++++++++++++++++++++++++++++
                    'subCrawler': {
                        # 시작/종료 날짜 설정 O
                        'crawler': 'A'

                        # 날짜 설정 X
                        # 'crawler': 'B'

                        # ++++++++++++++++++++++++++++++++++++++++++
                        # 공통 옵션
                        # +++++++++++++++++++++++++++++++++++++++++
                        # 언어 설정
                        # , 'language' : 'en'
                        , 'language': 'ko'

                        # 국가 설정
                        # , 'country' : 'US'
                        , 'country': 'KR'

                        # 지역 설정
                        , 'domainList':  ['강도', '절도', '성폭력', '살인', '사기']

                        # 키워드 설정
                        , 'keywordList': ['범죄']

                        # ++++++++++++++++++++++++++++++++++++++++++
                        # 크롤러 A 옵션
                        # +++++++++++++++++++++++++++++++++++++++++
                        # 시간 설정
                        , 'srtDate': '10/05/2017'
                        , 'endDate': '01/03/2018'

                        # 검색 최대 페이지 (페이지 당 10개)
                        , 'searchMaxPage': 2  # 테스트
                        # , 'searchMaxPage': 10
                        # , 'searchMaxPage': 99

                        # ++++++++++++++++++++++++++++++++++++++++++
                        # 크롤러 B 옵션
                        # +++++++++++++++++++++++++++++++++++++++++
                        # 최대 검색 개수
                        , 'searchMaxCnt': 10  # 테스트
                        # , searchMaxCnt = 100
                    }
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # +++++++++++++++++++++++++++++++++++++++++
                    #  [단위 시스템] 구글 뉴스 크롤러 선택
                    # +++++++++++++++++++++++++++++++++++++++++
                    'subCrawler': {
                        # 시작/종료 날짜 설정 O
                        'crawler': 'A'

                        # 날짜 설정 X
                        # 'crawler': 'B'

                        # ++++++++++++++++++++++++++++++++++++++++++
                        # 공통 옵션
                        # +++++++++++++++++++++++++++++++++++++++++
                        # 언어 설정
                        # , 'language' : 'en'
                        , 'language': 'ko'

                        # 국가 설정
                        # , 'country' : 'US'
                        , 'country': 'KR'

                        # 지역 설정
                        , 'domainList': ['강도', '절도', '성폭력', '살인', '사기']

                        # 키워드 설정
                        , 'keywordList': ['범죄']

                        # ++++++++++++++++++++++++++++++++++++++++++
                        # 크롤러 A 옵션
                        # +++++++++++++++++++++++++++++++++++++++++
                        # 시간 설정
                        , 'srtDate': '10/05/2017'
                        , 'endDate': '01/03/2018'

                        # 검색 최대 페이지 (페이지 당 10개)
                        , 'searchMaxPage': 2  # 테스트
                        # , 'searchMaxPage': 10
                        # , 'searchMaxPage': 99

                        # ++++++++++++++++++++++++++++++++++++++++++
                        # 크롤러 B 옵션
                        # +++++++++++++++++++++++++++++++++++++++++
                        # 최대 검색 개수
                        , 'searchMaxCnt': 10  # 테스트
                        # , searchMaxCnt = 100
                    }
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # [서브 시스템] 구글 뉴스 크롤링
            subCrawler(sysOpt)

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
        inParams = {}

        print("[CHECK] inParams : {}".format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))
