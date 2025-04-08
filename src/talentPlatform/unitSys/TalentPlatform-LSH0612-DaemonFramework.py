# ================================================
# 요구사항
# ================================================
# Python을 이용한 청소년 인터넷 게임 중독 관련 소셜데이터 수집과 분석을 위한 한국형 온톨로지 개발 및 평가

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
# from matplotlib import font_manager, rc
# from dbfread import DBF, FieldParser
import csv
import os
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re
# from GoogleNews import GoogleNews
from gnews import GNews
from newspaper import Article

import json
from datetime import datetime, timedelta
from googlenewsdecoder import gnewsdecoder

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

    log.propagate = False

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
        ,
        'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
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

def searchGoogleNews(unitGoogleNews, sysOpt, dtDateInfo):

    try:
        dataL1 = pd.DataFrame()
        for keywordInfo in sysOpt['keywordList']:
            # searchInfo = '{} {}'.format(domainInfo, keywordInfo)
            searchKeyword = '{}'.format(keywordInfo)

            unitGoogleNews.search(searchKeyword)
            for i in range(0, sysOpt['searchMaxPage']):
                result = unitGoogleNews.page_at(i)
                if len(result) < 1: continue

                data = pd.DataFrame(result)
                if len(data) < 1: continue

                data.insert(0, 'searchKeyword', searchKeyword)
                dataL1 = pd.concat([dataL1, data], axis=0)

        log.info(f"[CHECK] searchKeyword : {searchKeyword} : {len(dataL1)}")
        if len(dataL1) > 0:
            saveFile = dtDateInfo.strftime(sysOpt['saveFile']).format(searchKeyword=searchKeyword)
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            dataL1.to_csv(saveFile, index=False)
            log.info(f'[CHECK] saveFile : {saveFile}')

        time.sleep(10)

    except Exception as e:
        log.error("Exception : {}".format(e))

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

    prjName = 'test'
    serviceName = 'LSH0612'

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
                # 시작/종료 시간
                'srtDate': '2025-01-01',
                'endDate': '2025-01-05',
                'invDate': '1d',
                'searchMaxPage': 99,

                # 언어 설정
                # , 'language' : 'en'
                'language': 'ko',

                # 국가 설정
                # , 'country' : 'US'
                'country': 'KR',

                # 키워드 설정
                'keywordList': ['청소년 게임 중독'],

                # 저장 경로
                'saveFile': '/DATA/OUTPUT/LSH0612/gnews_%Y%m%d.csv',
            }

            # =================================================================
            # from gnews import GNews
            # from newspaper import Article
            # =================================================================
            unitGoogleNews = GNews(language='ko', country='KR')
            searchList = unitGoogleNews.get_news('청소년 게임 중독')
            log.info(f'[CHECK] searchList : {len(searchList)}')

            flatList = []
            for data in searchList:
                flatData = {
                    'title': data['title'],
                    'description': data['description'],
                    'publishedDate': data['published date'],
                    'url': data['url'],
                    'publisherTitle': data['publisher']['title'],
                    'publisherHref': data['publisher']['href']
                }

                flatList.append(flatData)

            data = pd.DataFrame.from_dict(flatList)

            # title                                    [기획] 청소년 게임중독 문제 심각 - 매일일보
            # description                               [기획] 청소년 게임중독 문제 심각  매일일보
            # publishedDate                         Thu, 30 May 2024 07:00:00 GMT
            # url               https://news.google.com/rss/articles/CBMiZEFVX...
            # publisherTitle                                                 매일일보
            # publisherHref                                    https://www.m-i.kr

            for i, row in data.iterrows():
                log.info(f'[CHECK] i : {i}')

                # https://www.m-i.kr/news/articleView.html?idxno=1125607
                decInfo = gnewsdecoder(row['url'])
                if not (decInfo['status'] == True): continue

                articleInfo = Article(decInfo['decoded_url'], language='ko')

                # 뉴스 다운로드/파싱/자연어 처리
                articleInfo.download()
                articleInfo.parse()
                articleInfo.nlp()

                from konlpy.tag import Okt
                okt = Okt()

                keywordList = None if articleInfo.keywords is None or len(articleInfo.keywords) < 1 else articleInfo.keywords
                nouns_in_phrase = okt.nouns(set(keywordList))

                nouns_only_list = []  #
                for word in keywordList:
                    # 각 단어(또는 구)의 품사 태깅 시도
                        pos_tags = okt.pos(word, stem=False, norm=True)

                        # 품사 태깅 결과가 있고, 첫 번째 태그가 'Noun'(명사)이면
                        if pos_tags and pos_tags[0][1] == 'Noun':
                            nouns_only_list.append(word)
                            # print(f"   >> 명사로 판단됨: '{word}'")

                from collections import Counter
                okt = Okt()
                text = None if articleInfo.text is None or len(articleInfo.text) < 1 else articleInfo.text
                posTagList = okt.pos(text, stem=True)

                # 명사/동사/형용사 추출
                keyList = ['Noun', 'Verb', 'Adjective']
                for keyInfo in keyList:
                    log.info(f'[CHECK] keyInfo : {keyInfo}')

                    keywordList = [word for word, pos in posTagList if pos in keyInfo]

                    # 불용어 제거
                    # keywordList = [word for word in keywordList if word not in stopWordList and len(word) > 1]

                    # 빈도수 계산
                    keywordCnt = Counter(keywordList).most_common(20)
                    keywordData = pd.DataFrame(keywordCnt.items(), columns=['keyword', 'cnt']).sort_values(by='cnt', ascending=False)
                    keywordDataL1 = keywordData[keywordData['keyword'].str.len() >= 2].reset_index(drop=True)
                    keywordDataL1['keyCnt'] = ''


                list_of_lists = keywordDataL1.values.tolist()
                # list_of_dicts = keywordDataL1.to_dict(orient='records')



                data.loc[i, f'text'] = None if articleInfo.text is None or len(articleInfo.text) < 1 else articleInfo.text
                data.loc[i, f'summary'] = None if articleInfo.summary is None or len(articleInfo.summary) < 1 else articleInfo.summary
                data.loc[i, f'keywords'] = None if articleInfo.keywords is None or len(articleInfo.keywords) < 1 else articleInfo.keywords
                data.loc[i, f'authors'] = None if articleInfo.authors is None or len(articleInfo.authors) < 1 else articleInfo.authors
                data.loc[i, f'publish_date'] = None if articleInfo.publish_date is None or len(articleInfo.publish_date) < 1 else articleInfo.publish_date
                data.loc[i, f'top_image'] = None if articleInfo.top_image is None or len(articleInfo.top_image) < 1 else articleInfo.top_image
                data.loc[i, f'images'] = None if articleInfo.images is None or len(articleInfo.images) < 1 else articleInfo.images

                # articleInfo.title
                # articleInfo.text
                # articleInfo.images
                # articleInfo.authors
                # articleInfo.publish_date
                # articleInfo.top_image
                # articleInfo.movies
                # articleInfo.keywords
                # articleInfo.summary




                # matDataL1.loc[i, f'tas-dist-{j}'] = cloDist


            # =================================================================
            # from GoogleNews import GoogleNews
            # =================================================================
            # from GoogleNews import GoogleNews
            # 시작일/종료일 설정
            # dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d %H:%M')
            # dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d %H:%M')
            # dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])
            # for dtDateInfo in dtDateList:
            #     log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
            #
            #     unitGoogleNews = GoogleNews(
            #         lang=sysOpt['language'],
            #         region=sysOpt['country'],
            #         start=dtDateInfo.strftime('%m/%d/%Y'),
            #         end=(dtDateInfo + timedelta(days=1)).strftime('%m/%d/%Y'),
            #         encode='UTF-8'
            #     )
            #
            #     searchGoogleNews(unitGoogleNews, sysOpt, dtDateInfo)

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
