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
from collections import Counter
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import nagisa
import nltk
import pandas as pd
from konlpy.tag import Okt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import itertools

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
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'mapInfo')
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

        log.info("[CHECK] {} : {}".format(key, val))

        # self 변수에 할당
        # setattr(self, key, val)

    return globalVar

def getTermFreqIdf(corpus):

    result = None

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        result = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

        return result

    except Exception as e:
        log.error(f'Exception : {e}')
        return result


def getProcTag(text, regionCode):

    result = None
    stop_words = set(stopwords.words())

    try:
        if regionCode == 'JP':
            tagged = nagisa.tagging(text)
            nouns = [word for word, tag in zip(tagged.words, tagged.postags) if tag == '名詞']
        elif regionCode == 'KR':
            nouns = Okt().nouns(text)
        else:
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in stop_words]
            tagged = nltk.pos_tag(words)
            nouns = [word for word, pos in tagged if pos.startswith('NN')]

        result = ' '.join(nouns)

        return result

    except Exception as e:
        log.error(f'Exception : {e}')
        return result

def getTopTermFreqIdf(tfidf_result, topNum=10):

    result = None

    try:
        mean_scores = tfidf_result.mean(axis=0)
        sorted_scores = mean_scores.sort_values(ascending=False)

        result = sorted_scores.head(topNum)
        return result

    except Exception as e:
        log.error(f'Exception : {e}')
        return result


def makeTermProc(sysOpt, dtDateInfo, modelType, period, data, key, regionCode):

    try:
        srtDate = pd.to_datetime(sysOpt['analy']['srtDate']).strftime('%Y%m%d')
        endDate = pd.to_datetime(sysOpt['analy']['endDate']).strftime('%Y%m%d')
        keyJoin = '-'.join(key)

        # joined_string = ', '.join(a)

        # 중복 제거
        # textList = data[key].drop_duplicates().tolist()
        # textList = data[key].tolist()

        # 병합 처리
        comTextList = [data[k].tolist() for k in key]
        procTextList = list(itertools.chain(*comTextList))

        # 결측값 제거
        textList = [item for item in procTextList if not pd.isna(item)]

        # 전처리 및 품사 태깅
        procText = [getProcTag(text, regionCode=regionCode) for text in textList]

        # 공백 제거
        procTextL1 = [item for item in procText if item.strip()]

        # TF 단어
        termList = ' '.join(procTextL1).split()

        # TF 단어 빈도
        termFreq = Counter(termList)

        # TF 저장
        saveFilePattern = '{}/{}'.format(sysOpt['analy']['savePath'], sysOpt['analy']['saveTfName'])
        saveFile = dtDateInfo.strftime(saveFilePattern).format(regionCode, modelType, period, keyJoin, srtDate, endDate)
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)

        saveData = pd.DataFrame.from_dict(termFreq.items()).rename({0: 'term', 1: 'freq'}, axis=1).sort_values(by='freq', ascending=False)
        saveData['cum'] = saveData['freq'].cumsum() / saveData['freq'].sum() * 100
        saveData.to_csv(saveFile, index=False)
        log.info(f'[CHECK] saveFile : {saveFile}')

        # TF 시각화
        saveImgPattern = '{}/{}'.format(sysOpt['analy']['savePath'], sysOpt['analy']['saveTfImg'])
        saveImg = dtDateInfo.strftime(saveImgPattern).format(regionCode, modelType, period, keyJoin, srtDate, endDate)
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)

        wordcloud = WordCloud(
            width=1500
            , height=1500
            , background_color=None
            , mode='RGBA'
            , font_path=globalVar['fontPath']
        ).generate_from_frequencies(termFreq)

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
        # plt.show()
        plt.close()
        log.info(f'[CHECK] saveImg : {saveImg}')

        # TF-IDF 계산
        termFreqIdfData = getTermFreqIdf(procText)
        topTermFreqIdf = getTopTermFreqIdf(termFreqIdfData, topNum=sysOpt['analy']['topNum'])

        # TF-IDF 저장
        saveFilePattern = '{}/{}'.format(sysOpt['analy']['savePath'], sysOpt['analy']['saveTfIdfName'])
        saveFile = dtDateInfo.strftime(saveFilePattern).format(regionCode, modelType, period, keyJoin, srtDate, endDate)
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)

        saveData = pd.DataFrame.from_dict(topTermFreqIdf.items()).rename({0: 'term', 1: 'freq'}, axis=1)
        saveData['cum'] = saveData['freq'].cumsum() / saveData['freq'].sum() * 100
        saveData.to_csv(saveFile, index=False)
        log.info(f'[CHECK] saveFile : {saveFile}')

        # TF 시각화
        saveImgPattern = '{}/{}'.format(sysOpt['analy']['savePath'], sysOpt['analy']['saveTfIdfImg'])
        saveImg = dtDateInfo.strftime(saveImgPattern).format(regionCode, modelType, period, keyJoin, srtDate, endDate)
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)

        # TF-IDF 시각화
        wordcloud = WordCloud(
            width=1000
            , height=1000
            , background_color=None
            , mode='RGBA'
            , font_path=globalVar['fontPath']
        ).generate_from_frequencies(topTermFreqIdf)

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
        # plt.show()
        plt.close()
        log.info(f'[CHECK] saveImg : {saveImg}')

    except Exception as e:
        log.error(f'Exception : {e}')

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 유튜브 급상승 영상 및 댓글 수집 및 텍스트 분석

    # 유튜브 영상의 제목/설명/댓글 등을 크롤링해서 R로 명사추출 후 tf-idf 적용해서 키워드 빈도 랭킹을 뽑고 있습니다.
    #
    # 큰 구조를 설명드리면,
    # (1) 아래 사이트에서 유튜브 인기급상승 동영상 크롤링 데이터 다운 받기 + 해당 영상의 댓글 파이썬 크롤링하기
    # https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset/data
    # (2) 영상 제목/설명 등과 댓글 텍스트를 파이썬에서 인코딩 & 슬라이싱
    # (3) R로 가져와서 명사 추출, 빈도 분석 진행

    # 이렇게 진행 중인데, 코드 자체를 오래된 버전을 쓰고 있기도 하고 (1) 사이트가 막히면 인급동 크롤링을 못하는 상황이어서요.
    # 코드를 좀 더 간단하고 최신 버전으로 업그레이드해서 여러 사이트와 툴을 오가는 작업을 좀 간추려보고 싶습니다.

    # 오 네! 1번은 제가 지금 가지고 있는 api로 현재 코드를 실행했을 때 하루치 인급동(50개 영상 정보)은 여유롭게 됐던 것 같은데, 24시간 리셋을 기다려야 하니.
    # 그러면 하루에 인급동 2회 정도만 (오전/오후) 크롤링 할 수 있게 되면 좋을 것 같아요.

    # 댓글 수집은 제가 코드를 가지고 있는 게 있는데, 대댓글까지 완벽히 수집되지는 않아서 수정 필요한 부분이 있는지만 봐주시면 될 것 같습니다!
    # 2번은 파이썬으로 통합 부탁드립니다!
    # - 6. 3에서 뽑은 영상의 title, description 등 텍스트 정보 추출해서 유튜브 영상 키워드의 빈도 순위 (tf & tf-idf)
    # - 7. 4 댓글들의 키워드 빈도 순위 (tf & tf-idf)
    # *빈도는 업무에서는 tf-idf를 주로 활용하나, 단순 tf도 참고를 위해 확인하고 있습니다

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
    serviceName = 'LSH0536'

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
                globalVar['fontPath'] = 'C:/Windows/Fonts/malgun.ttf'
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'
                globalVar['fontPath'] = '/SYSTEMS/PROG/PYTHON/PyCharm/resources/config/fontInfo/malgun.ttf'

            # 옵션 설정
            sysOpt = {
                'analy': {
                    # 시작일, 종료일, 시간 간격 (시간 1h)
                    # 'srtDate': '2024-01-01'
                    # , 'endDate': '2024-01-04'
                    'srtDate': '2024-01-09'
                    , 'endDate': '2024-01-10'
                    , 'invDate': '1h'

                    # 국가 코드 목록
                    # , 'regionCodeList': ['BR', 'CA', 'DE', 'FR', 'GB', 'IN', 'JP', 'MX', 'RU', 'US', 'KR']
                    , 'regionCodeList': ['KR']

                    # 수행 목록
                    , 'modelList': ['API']

                    # 중요 키워드
                    , 'topNum': 9999

                    # 입력 파일경로/파일명
                    , 'inpPath': '/DATA/OUTPUT/LSH0536/colct/%Y%m/%d'
                    , 'inpName': '{}_{}_youtube_trending_data_%Y%m%d%H.csv'

                    # 저장 파일경로/파일명
                    , 'savePath': '/DATA/OUTPUT/LSH0536/analy/%Y%m/%d'
                    , 'saveTfName': '{}_{}_{}_{}_termFreq_{}_{}_%Y%m%d%H%M.csv'
                    , 'saveTfIdfName': '{}_{}_{}_{}_termFreqIdf_{}_{}_%Y%m%d%H%M.csv'
                    , 'saveTfImg': '{}_{}_{}_{}_termFreq_{}_{}_%Y%m%d%H%M.png'
                    , 'saveTfIdfImg': '{}_{}_{}_{}_termFreqIdf_{}_{}_%Y%m%d%H%M.png'
                }
            }

            # ===================================================================================
            # Python을 이용한 유튜브 급상승 영상 및 댓글 분석
            # ===================================================================================
            log.info(f'[CHECK] sysOpt(analy) : {sysOpt["analy"]}')

            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['analy']['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['analy']['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['analy']['invDate'])

            # NLTK Stopwords 다운로드
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('stopwords', quiet=True)

            for modelType in sysOpt['analy']['modelList']:
                log.info(f'[CHECK] modelType : {modelType}')

                for regionCode in sysOpt['analy']['regionCodeList']:
                    log.info(f'[CHECK] regionCode : {regionCode}')

                    periodData = pd.DataFrame()
                    for dtDateInfo in dtDateList:
                        log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                        inpFilePattern = '{}/{}'.format(sysOpt['analy']['inpPath'], sysOpt['analy']['inpName'])
                        inpFile = dtDateInfo.strftime(inpFilePattern).format(regionCode, modelType)

                        fileList = sorted(glob.glob(inpFile))
                        if fileList is None or len(fileList) < 1: continue

                        # 파일 읽기
                        dailyData = pd.DataFrame()
                        for fileInfo in fileList:
                            hourlyData = pd.read_csv(fileInfo)
                            dailyData = pd.concat([dailyData, hourlyData], ignore_index=True)
                            periodData = pd.concat([periodData, hourlyData], ignore_index=True)

                            # 매 국가코드/시간별에 따른 텍스트 분석
                            # if len(hourlyData) > 0:
                            #     makeTermProc(sysOpt, dtDateInfo, modelType, 'hourly', hourlyData, 'title', regionCode)
                            #     makeTermProc(sysOpt, dtDateInfo, modelType, 'hourly', hourlyData, 'description', regionCode)
                            #     makeTermProc(sysOpt, dtDateInfo, modelType, 'hourly', hourlyData, 'replyText', regionCode)
                            #     makeTermProc(sysOpt, dtDateInfo, modelType, 'hourly', hourlyData, 'replyDtlText', regionCode)

                        # 매 국가코드/일별에 따른 텍스트 분석
                        if len(dailyData) > 0:
                            makeTermProc(sysOpt, dtDateInfo.normalize(), modelType, 'daily', dailyData, ['title'], regionCode)
                            makeTermProc(sysOpt, dtDateInfo.normalize(), modelType, 'daily', dailyData, ['description', 'replyText', 'replyDtlText'], regionCode)


                    # 매 국가코드에 따른 텍스트 분석
                    if len(periodData) > 0:
                        makeTermProc(sysOpt, datetime.now(), modelType, 'period', periodData, ['title'], regionCode)
                        makeTermProc(sysOpt, datetime.now(), modelType, 'period', periodData, ['description', 'replyText', 'replyDtlText'], regionCode)

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
