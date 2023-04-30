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

import contextily as ctx
import geopandas as gpd
import geoplot as gplt
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from GoogleNews import GoogleNews
from gnews import GNews
from gnews.utils.utils import import_or_install
from konlpy.tag import Twitter
from newspaper import Article
from newspaper import Config
from pandas import json_normalize
from pandas.api.types import CategoricalDtype
# import geoplot.crs as gcrs
# import geoplot as gplt #conda install -c conda-forge geoplot
from plotnine import *
from shapely.geometry import Point
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

        saveCsvFile = '{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '지역 및 키워드에 따른 구글 뉴스 크롤링', crawler)

        for domainInfo in domainList:
            for keywordInfo in keywordList:

                if crawler == 'A':
                    result = searchGoogleNews(unitGoogleNews, domainInfo, keywordInfo, subOpt['searchMaxPage'], Article, config, saveCsvFile)
                elif crawler == 'B':
                    result = searchGoogleNewsAll(unitGoogleNewsAll, domainInfo, keywordInfo, saveCsvFile)

                log.info('[CHECK] result : {}'.format(result))

        # breakpoint()

        # 크롤링 데이터
        dataL1 = pd.read_csv(saveCsvFile)
        if (len(dataL1) < 1):
            log.error('dataL1 : {} / {}'.format(len(dataL1), '입력 자료를 확인해주세요.'))
            raise Exception('dataL1 : {} / {}'.format(len(dataL1), '입력 자료를 확인해주세요.'))

        # 중복 데이터 삭제
        dataL2 = dataL1.drop_duplicates()

        saveCsvFnlFile = '{}/{}_{}_{}_FNL.csv'.format(globalVar['outPath'], serviceName, '지역 및 키워드에 따른 구글 뉴스 크롤링', crawler)
        dataL2.to_csv(saveCsvFnlFile, index=False)

        saveXlsxFnlFile = '{}/{}_{}_{}_FNL.xlsx'.format(globalVar['outPath'], serviceName, '지역 및 키워드에 따른 구글 뉴스 크롤링', crawler)
        dataL2.to_excel(saveXlsxFnlFile, index=False)

        # ********************************************
        # 워드 클라우드
        # ********************************************
        domainList = dataL2['domainName'].unique().tolist()
        keywordList = dataL2['keyword'].unique().tolist()

        for domainInfo in domainList:
            for keywordInfo in keywordList:
                keyInfo = '{} {}'.format(domainInfo, keywordInfo)

                selData = dataL2.loc[
                    ((dataL2['domainName'] == domainInfo) & (dataL2['keyword'] == keywordInfo))
                ]

                # 뉴스 제목
                saveImg = '{}/{}_{}_{}_{}.png'.format(globalVar['figPath'], serviceName, keyInfo, 'title', '워드클라우드')
                saveFile = '{}/{}_{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, keyInfo, 'title', '워드클라우드')
                result = makePlotWordCloud(selData, 'title', saveImg, saveFile)
                log.info('[CHECK] result : {}'.format(result))

                # 뉴스 내용
                saveImg = '{}/{}_{}_{}_{}.png'.format(globalVar['figPath'], serviceName, keyInfo, 'description', '워드클라우드')
                saveFile = '{}/{}_{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, keyInfo, 'description', '워드클라우드')
                result = makePlotWordCloud(selData, 'description', saveImg, saveFile)
                log.info('[CHECK] result : {}'.format(result))

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


def makePlotWordCloud(data, key, saveImg, saveFile):

    log.info('[START] {}'.format('makePlotWordCloud'))

    result = None

    try:

        nlpy = Twitter()

        getData = data[key]
        getDataList = getData.to_list()
        # getDataTextAll = ' '.join(getDataList)
        getDataTextAll = ' '.join([str(x) for x in getDataList])

        # 명사만 추출
        nounList = nlpy.nouns(getDataTextAll)

        # 빈도 계산
        countList = Counter(nounList)

        dictData = {}

        # 상위 100개 선정
        for none, cnt in countList.most_common(100):
            # 빈도수 2 이상
            if (cnt < 2): continue
            # 명사  2 글자 이상
            if (len(none) < 2): continue

            dictData[none] = cnt


        # 워드클라우드 입력 데이터
        saveData = pd.DataFrame.from_dict(dictData.items()).rename(
            {
                0 : 'none'
                , 1 : 'cnt'
              }
            , axis=1
        )
        saveData.to_csv(saveFile, index=False)

        # 워드 클라우드 생성
        wordcloud = WordCloud(
            font_path='font/malgun.ttf'
            , width=1000
            , height=1000
            , background_color="white"
        ).generate_from_frequencies(dictData)

        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent = True)
        plt.show()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makePlotWordCloud'))


def readDataPrep(subOpt, inpFile):

    log.info('[START] {}'.format('readDataPrep'))

    result_data = None

    try:

        fileList = glob.glob(inpFile)
        if fileList is None or len(fileList) < 1:
            log.error('[ERROR] inpFile : {} / {}'.format(fileList, '입력 자료를 확인해주세요.'))
            raise Exception('[ERROR] inpFile : {} / {}'.format(fileList, '입력 자료를 확인해주세요.'))

        # gmaps = googlemaps.Client(googleApiKey)
        #
        # data = pd.DataFrame()
        # for fileInfo in fileList:
        #     data_pat = pd.read_excel(fileInfo, skiprows=16)
        #     data = pd.concat([data, data_pat], ignore_index=True)

        # dataGeo = data[['시군구', '도로명']]
        # dataGeoL1 = dataGeo.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=True)
        #
        # for i, item in dataGeoL1.iterrows():
        #     addr = item['시군구'] + ' ' + item['도로명']
        #     rtnGeo = gmaps.geocode((addr), language='ko')
        #
        #     # 위/경도 반환
        #     dataGeoL1.loc[i, 'latitude'] = rtnGeo[0]['geometry']['location']['lat']
        #     dataGeoL1.loc[i, 'longitude'] = rtnGeo[0]['geometry']['location']['lng']

        # result_data = pd.merge(data, dataGeoL1, left_on=['시군구', '도로명'], right_on=['시군구', '도로명'], how='left')

        # breakpoint()

        fileInfo = fileList[0]
        data = pd.read_csv(fileInfo)
        data.drop(['Unnamed: 0', '해제사유발생일'], axis=1, inplace=True)

        # 컬럼별 결측값 확인
        # data.isnull().sum()
        dataL1 = data.dropna().reset_index(drop=True)

        result_data = dataL1.rename(
            {
                'lat' : 'latitude'
                , 'lon' : 'longitude'
             }
            , axis='columns'
        )

        result_data['date'] = pd.to_datetime(result_data['계약년월'], format='%Y%m')
        result_data['거래금액(만원)'] = pd.Series(result_data['거래금액(만원)']).str.replace(',', '', regex=True)
        result_data["거래금액(만원)"] = pd.to_numeric(result_data["거래금액(만원)"])
        result_data['구'] = result_data['시군구'].str.split(' ').str[1]
        result_data['동'] = result_data['시군구'].str.split(' ').str[2]
        result_data['연'] = result_data['계약년월'].astype(str).str.slice(0, 4)
        result_data['월'] = result_data['계약년월'].astype(str).str.slice(4, 6)
        result_data['단지명'] = result_data['단지명'] + '(' + result_data['도로명'] + ')'

        saveFile = '{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, '서울특별시 강북구 아파트 전월세가_20111101_20201101_PROP')
        result_data.to_csv(saveFile,index=False)
        log.info('[CHECK] saveFile : {}'.format(saveFile))

        return result_data

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result_data

    finally:
        log.info('[END] {}'.format('readDataPrep'))


def readGeoData(inpShpFile):

    log.info('[START] {}'.format('readGeoData'))

    result = None

    try:

        # breakpoint()

        fileList = glob.glob(inpShpFile)
        log.info('[CHECK] fileList : {}'.format(fileList))

        if fileList is None or len(fileList) < 1:
            log.error('[ERROR] inpFile : {} / {}'.format(inpShpFile, '입력 자료를 확인해 주세요.'))
            raise Exception('[ERROR] inpFile : {} / {}'.format(inpShpFile, '입력 자료를 확인해 주세요.'))

        # df = px.data.election()
        # geojson = px.data.election_geojson()

        fileInfo = fileList[0]

        try:
            shp_file = gpd.read_file(fileInfo)
        except Warning as w:
            log.warn("[WARN] Warning : {}".format(w))

        shp_file.to_crs(epsg=4162, inplace=True)
        data_seoul = shp_file[0:25]

        return data_seoul

    except Exception as e:
        log.error("[ERROR] Exception : {}".format(e))
        return result

    finally:
        log.info('[END] {}'.format('readGeoData'))


def subVis(sysOpt):

    log.info('[START] {}'.format('subVis'))

    try:

        subOpt = sysOpt['subVis']

        aptCmplxList = subOpt['aptCmplxList']
        aptList = subOpt['aptList']

        # 실거래가 파일 읽기
        # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*.xlsx')
        inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '서울특별시 강북구 아파트 실거래가_20111101_20201101.csv')
        data = readDataPrep(subOpt, inpFile)

        # 서울특별시 지도 데이터 읽기
        inpShpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'bnd_sigungu_00_2019_2019_2Q.shp')
        dataGeo = readGeoData(inpShpFile)

        # 교육기관 데이터 읽기
        inpPosFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '전국초중등학교위치표준데이터_20220316.csv')
        eduData = pd.read_csv(inpPosFile, encoding='EUC-KR')

        # 부동산 중개소 데이터 읽기
        inpAgencyFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '서울특별시 부동산 중개업소 위경도 정보.csv')
        agencyData = pd.read_csv(inpAgencyFile, encoding='UTF-8')

        # 강북구 주요 아파트 거래 지역 위치 현황
        saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '강북구 주요 아파트 거래 지역 위치 현황')
        result = makePlotCase1(data, dataGeo, saveImg)
        log.info('[CHECK] result : {}'.format(result))

        # 강북구 주요 아파트 및 교육기관/부동산 중개소 거래 지역 위치 현황
        saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '강북구 주요 아파트 및 교육기관 그리고 중개소 거래 지역 위치 현황')
        result = makePlotNewCase1(data, dataGeo, eduData, agencyData, saveImg)
        log.info('[CHECK] result : {}'.format(result))

        # 강북구 평균 아파트 거래가 현황
        saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '강북구 평균 아파트 거래가 현황')
        result = makePlotCase2(data, dataGeo, saveImg)
        log.info('[CHECK] result : {}'.format(result))

        # 연도별 동에 따른 아파트 실거래가 분포
        for year in range(2011, 2021, 1):
            saveImg = '{}/{}_{} ({}).png'.format(globalVar['figPath'], serviceName, '연도별 동에 따른 아파트 실거래가 분포', year)
            result = makePlotCase3(data, dataGeo, year, saveImg)
            log.info('[CHECK] result : {}'.format(result))

        # 아파트 실거래가 추이
        saveImg = '{}/{}_{}.png'.format(globalVar['figPath'], serviceName, '아파트 단지별로 실거래가 추이')
        result = makePlotCase4(data, dataGeo, aptCmplxList, saveImg)
        log.info('[CHECK] result : {}'.format(result))

        # 아파트에 따른 이름
        for i, aptName in enumerate(aptList):
            log.info('[CHECK] aptName : {}'.format(aptName))

            # 아파트 실거래가 및 거래량 추이
            saveImg = '{}/{}_{} ({}).png'.format(globalVar['figPath'], serviceName, '아파트 실거래가 및 거래량 추이', aptName)
            result = makePlotCase5(data, dataGeo, aptName, saveImg)
            log.info('[CHECK] result : {}'.format(result))

            # 날짜별 해당 아파트 거래량 분포 (경남이너스빌)
            saveImg = '{}/{}_{} ({}).png'.format(globalVar['figPath'], serviceName, '날짜별 해당 아파트 거래량 분포', aptName)
            result = makePlotCase6(data, dataGeo, aptName, saveImg)
            log.info('[CHECK] result : {}'.format(result))

            # 층별 해당 아파트 거래량 분포 (경남이너스빌)
            saveImg = '{}/{}_{} ({}).png'.format(globalVar['figPath'], serviceName, '층별 해당 아파트 거래량 분포', aptName)
            result = makePlotCase7(data, dataGeo, aptName, saveImg)
            log.info('[CHECK] result : {}'.format(result))

            # 면적별 해당 아파트 거래량 분포 (경남이너스빌)
            saveImg = '{}/{}_{} ({}).png'.format(globalVar['figPath'], serviceName, '면적별 해당 아파트 거래량 분포', aptName)
            result = makePlotCase8(data, dataGeo, aptName, saveImg)
            log.info('[CHECK] result : {}'.format(result))

    except Exception as e:
        log.error("Exception : {}".format(e))
    finally:
        log.info('[END] {}'.format('subVis'))

def makePlotCase1(result_data, data_seoul, saveImg):

    log.info('[START] {}'.format('makePlotCase1'))

    result = None

    try:
        # breakpoint()

        fig, ax = plt.subplots(figsize=(10, 8))

        crs = {'init': 'epsg:4162'}
        geometry = [Point(xy) for xy in zip(result_data["longitude"], result_data["latitude"])]
        geodata1 = gpd.GeoDataFrame(result_data, crs=crs, geometry=geometry)

        data_seoul['coords'] = data_seoul['geometry'].apply(lambda x: x.representative_point().coords[:])
        data_seoul['coords'] = [coords[0] for coords in data_seoul['coords']]

        # 컬러바 표시
        # gplt.kdeplot(geodata1, cmap='rainbow', zorder=0, cbar=True, shade=True, alpha=0.5, ax=ax)
        gplt.kdeplot(geodata1, cmap='rainbow', shade=True, alpha=0.5, ax=ax)
        gplt.polyplot(data_seoul, ax=ax)

        # 서울 시군구 표시
        for i, row in data_seoul.iterrows():
            ax.annotate(size=10, text=row['sigungu_nm'], xy=row['coords'], horizontalalignment='center')

        plt.gcf()
        plt.savefig(saveImg, width=10, height=8, dpi=600, bbox_inches='tight', transparent = True)
        plt.show()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makePlotCase1'))

def makePlotNewCase1(result_data, data_seoul, eduData, agencyData, saveImg):

    log.info('[START] {}'.format('makePlotNewCase1'))

    result = None

    try:

        # breakpoint()

        fig, ax = plt.subplots(figsize=(10, 8))

        # crs = {'init': 'epsg:4162'}
        crs = {'init': 'epsg:4326'}
        geometry = [Point(xy) for xy in zip(result_data['longitude'], result_data['latitude'])]
        geodata1 = gpd.GeoDataFrame(result_data, crs=crs, geometry=geometry)

        data_seoul['coords'] = data_seoul['geometry'].apply(lambda x: x.representative_point().coords[:])
        data_seoul['coords'] = [coords[0] for coords in data_seoul['coords']]

        gplt.polyplot(data_seoul, ax=ax)
        gplt.kdeplot(geodata1, cmap='rainbow', shade=True, alpha=0.5, ax=ax)

        # 교육기관 표시
        eduDataL1 = eduData[['위도', '경도', '학교급구분']].rename(
            columns={
                '위도': 'lat'
                , '경도': 'lon'
                , '학교급구분': 'type'
            }
        )

        # 부동산 중개소
        agencyData['type'] = '중개소'
        agencyDataL1 = agencyData[['lon', 'lat', 'type']]

        posData = pd.concat([eduDataL1, agencyDataL1], axis=0)
        posData['type'] = posData['type'].astype(CategoricalDtype(categories=['고등학교', '중학교', '초등학교', '중개소'], ordered=False))

        lon = posData['lon'].values
        lat = posData['lat'].values

        geometry = [Point(xy) for xy in zip(lon, lat)]
        posDataL1 = gpd.GeoDataFrame(posData, geometry=geometry)
        posDataL1.crs = {'init': "epsg:4326"}
        posDataL1.plot(column='type', categorical=True, legend=True, ax=ax, cmap='Set2', legend_kwds={'loc':'upper right', 'fontsize':14, 'frameon':True}, markersize=20, marker='o', zorder=3)

        ctx.add_basemap(ax, crs=posDataL1.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        # ctx.add_basemap(ax, crs=posDataL1.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zoom=19)
        # ax.set_axis_off()

        # 서울 시군구 표시
        # for i, row in data_seoul.iterrows():
        #     ax.annotate(size=10, text=row['sigungu_nm'], xy=row['coords'], horizontalalignment='center')

        plt.gcf()
        plt.savefig(saveImg, width=20, height=16, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makePlotNewCase1'))


def makePlotCase2(result_data, data_seoul, saveImg):

    log.info('[START] {}'.format('makePlotCase2'))

    result = None

    try:
        # breakpoint()

        data_mapo = result_data[result_data['구'] == '강북구']
        data_mapoL1 = data_mapo.groupby(['date']).mean()
        data_mapoL1['date'] = data_mapoL1.index

        plot = (
                ggplot(data=data_mapoL1) +
                theme_bw() +
                theme(axis_text_x=element_text(angle=45, hjust=1)) +
                theme(text=element_text(family="Malgun gothic", size=16)) +
                geom_line(aes(x='date', y='거래금액(만원)')) +
                geom_point(aes(x='date', y='거래금액(만원)')) +
                scale_x_datetime(date_labels='%Y') +
                labs(title='[강북구] 평균 거래가격') +
                xlab('날짜') +
                ylab('평균 거래가격')
        )

        fig = plot.draw()
        # plot.save(saveImg, width=10, height=8, dpi=600, transparent = True)
        plot.save(saveImg, width=10, height=8, dpi=600)
        fig.show()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makePlotCase2'))


def makePlotCase3(result_data, data_seoul, year, saveImg):

    log.info('[START] {}'.format('makePlotCase3'))

    result = None

    try:
        dataL1 = result_data.groupby(['연', '동']).mean()
        dataL1 = dataL1.reset_index()

        dataL2 = dataL1[dataL1['연'] == str(year)]
        if (len(dataL2) < 1): return result

        plot = (
                ggplot(data=dataL2) +
                theme_bw() +
                geom_bar(aes(x='동', y='거래금액(만원)', fill='거래금액(만원)'), stat="identity") +
                coord_flip() +
                labs(title='[{}] 연도별 동에 따른 아파트 실거래가 분포'.format(year)) +
                theme(text=element_text(family="Malgun gothic", size=16)) +
                xlab('동') +
                ylab('가격 (만원)')
        )

        fig = plot.draw()
        # plot.save(saveImg, width=10, height=8, dpi=600, transparent = True)
        plot.save(saveImg, width=10, height=8, dpi=600)
        fig.show()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makePlotCase3'))


def makePlotCase4(result_data, data_seoul, aptCmplxList, saveImg):

    log.info('[START] {}'.format('makePlotCase4'))

    result = None

    try:

        data = pd.DataFrame()
        for i, aptName in enumerate(aptCmplxList):
            tmpData = result_data[result_data['단지명'] == aptName]
            data = data.append(tmpData)

        dataL2 = data.groupby(['단지명', 'date']).mean()
        dataL2 = dataL2.reset_index()

        log.info('[CHECK] aptCmplxList : {}'.format(aptCmplxList))

        plot = (
                ggplot(data=dataL2) +
                theme_bw() +
                geom_line(aes(y='거래금액(만원)', x='date', color='단지명', group='단지명')) +
                geom_point(aes(y='거래금액(만원)', x='date', color='단지명', group='단지명')) +
                labs(title='아파트 단지별로 실거래가 추이') +
                scale_x_datetime(date_labels='%Y') +
                theme(
                    text=element_text(family="Malgun gothic", size=16)
                    , axis_text_x=element_text(angle=45, hjust=1)
                    # , legend_position='top'
                    # , legend_position=[0, 0.96]
                    ) +
                xlab('날짜') +
                ylab('가격(만원)')
        )

        fig = plot.draw()
        # plot.save(saveImg, width=10, height=8, dpi=600, bbox_inches='tight', transparent = True)
        plot.save(saveImg, width=10, height=8, dpi=600, bbox_inches='tight')
        fig.show()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makePlotCase4'))


def makePlotCase5(result_data, data_seoul, aptName, saveImg):

    log.info('[START] {}'.format('makePlotCase5'))

    result = None

    try:

        dataL2_price = result_data[result_data['단지명'] == aptName].groupby(['단지명', 'date']).mean().reset_index()
        dataL2_count = result_data[result_data['단지명'] == aptName].groupby(['단지명', 'date']).count().reset_index()

        plot = (
                ggplot(data=dataL2_price) +
                theme_bw() +
                geom_line(aes(y=dataL2_price['거래금액(만원)'] / 1000, x='date', color='단지명', group='단지명')) +
                geom_point(aes(y=dataL2_price['거래금액(만원)'] / 1000, x='date', color='단지명', group='단지명')) +
                geom_bar(aes(x='date', y=dataL2_count['시군구']), data=dataL2_count, color='blue', fill='blue', alpha=0.7,
                         stat="identity") +
                labs(title='[{}] 아파트 실거래가 추이 + 거래량'.format(aptName)) +
                scale_x_datetime(date_labels='%Y') +
                theme(
                    text=element_text(family="Malgun gothic", size=16)
                    , axis_text_x=element_text(angle=45, hjust=1)
                    # , legend_position='top'
                ) +
                xlab('날짜') +
                ylab('가격(천만원)')
        )

        fig = plot.draw()
        # plot.save(saveImg, width=10, height=8, dpi=600, bbox_inches='tight', transparent = True)
        plot.save(saveImg, width=10, height=8, dpi=600, bbox_inches='tight')
        fig.show()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makePlotCase5'))


def makePlotCase6(result_data, data_seoul, aptName, saveImg):
    log.info('[START] {}'.format('makePlotCase6'))

    result = None

    try:

        dataL2_date = result_data[result_data['단지명'] == aptName].groupby(['단지명', 'date']).agg(['count']).reset_index()

        plot = (
                ggplot(data=dataL2_date) +
                theme_bw() +
                geom_bar(aes(x='date', y='거래금액(만원)', fill='거래금액(만원)'), stat="identity") +
                # geom_bar(aes(x = 'date', y = '거래금액(만원)',fill = '거래금액(만원)',group='단지명',color = '단지명'),stat = "identity")+
                coord_flip() +
                labs(title='[{}] 날짜별 해당 아파트 거래량 분포'.format(aptName)) +
                theme(text=element_text(family="Malgun gothic", size=16)) +
                labs(fill="거래량") +
                xlab('날짜') +
                ylab('거래량')
        )

        fig = plot.draw()
        # plot.save(saveImg, width=10, height=8, dpi=600, bbox_inches='tight', transparent = True)
        plot.save(saveImg, width=10, height=8, dpi=600, bbox_inches='tight')
        fig.show()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makePlotCase6'))


def makePlotCase7(result_data, data_seoul, aptName, saveImg):
    log.info('[START] {}'.format('makePlotCase7'))

    result = None

    try:

        dataL2_level = result_data[result_data['단지명'] == aptName].groupby(['단지명', '층']).agg(['count']).reset_index()
        dataL2_level['전용면적(㎡)'] = dataL2_level['전용면적(㎡)'].apply(str)

        plot = (
                ggplot(data=dataL2_level) +
                theme_bw() +
                geom_bar(aes(x='층', y='거래금액(만원)', fill='거래금액(만원)'), stat="identity") +
                coord_flip() +
                labs(title='[{}] 층별 아파트 거래량 분포'.format(aptName)) +
                theme(text=element_text(family="Malgun gothic", size=16)) +
                labs(fill="거래량") +
                xlab('층') +
                ylab('거래량')
        )


        fig = plot.draw()
        # plot.save(saveImg, width=10, height=8, dpi=600, bbox_inches='tight', transparent = True)
        plot.save(saveImg, width=10, height=8, dpi=600, bbox_inches='tight')
        fig.show()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makePlotCase7'))


def makePlotCase8(result_data, data_seoul, aptName, saveImg):
    log.info('[START] {}'.format('makePlotCase7'))

    result = None

    try:

        dataL2_m2 = result_data[result_data['단지명'] == aptName].groupby(['단지명', '전용면적(㎡)']).agg(['count']).reset_index()
        # dataL2_m2['전용면적(㎡)'] = dataL2_m2['전용면적(㎡)'].apply(str)
        # dataL2_m2['전용면적(㎡)'] = pd.factorize(dataL2_m2['전용면적(㎡)'], sort=True)
        dataL2_m2['전용면적(㎡)'] = dataL2_m2['전용면적(㎡)'].astype('category')

        plot = (
                ggplot(data=dataL2_m2) +
                theme_bw() +
                geom_bar(aes(x='전용면적(㎡)', y='거래금액(만원)', fill='거래금액(만원)'), stat="identity") +
                coord_flip() +
                labs(title='[{}] 면적별 아파트 거래량 분포'.format(aptName)) +
                theme(text=element_text(family="Malgun gothic", size=16)) +
                labs(fill="거래량") +
                xlab('전용면적 (㎡)') +
                ylab('거래량')
        )


        fig = plot.draw()
        # plot.save(saveImg, width=10, height=8, dpi=600, bbox_inches='tight', transparent = True)
        plot.save(saveImg, width=10, height=8, dpi=600, bbox_inches='tight')
        fig.show()

        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makePlotCase7'))


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 부동산 데이터 분석 및 가격 예측

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0250'

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

        try:
            log.info('[START] {}'.format("exec"))

            # breakpoint()

            # ********************************************
            # 옵션 설정
            # ********************************************
            # 초기 옵션 설정
            sysOpt = {
                # +++++++++++++++++++++++++++++++++++++++++
                #  [단위 시스템] 구글 뉴스 크롤러 선택
                # +++++++++++++++++++++++++++++++++++++++++
                'subCrawler': {
                    # 시작/종료 날짜 설정 O
                    # 'crawler' : 'A'

                    # 날짜 설정 X
                    'crawler': 'B'

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
                    , 'domainList': ['강북구']

                    # 키워드 설정
                    , 'keywordList': ['아파트 매매', '토지 매매']

                    # ++++++++++++++++++++++++++++++++++++++++++
                    # 크롤러 A 옵션
                    # +++++++++++++++++++++++++++++++++++++++++
                    # 시간 설정
                    # , 'srtDate' : '10/05/2017'
                    # , 'endDate' : '01/03/2018'

                    # 검색 최대 페이지 (페이지 당 10개)
                    # , 'searchMaxPage': 2  # 테스트
                    # , 'searchMaxPage': 10
                    # , 'searchMaxPage': 99

                    # ++++++++++++++++++++++++++++++++++++++++++
                    # 크롤러 B 옵션
                    # +++++++++++++++++++++++++++++++++++++++++
                    # 최대 검색 개수
                    , 'searchMaxCnt': 10  # 테스트
                    # , searchMaxCnt = 100
                }

                # +++++++++++++++++++++++++++++++++++++++++
                #  [단위 시스템] 시각화 선택
                # +++++++++++++++++++++++++++++++++++++++++
                , 'subVis': {
                    # 아파트 단지 설정
                    # , 'aptCmplxList': ['미아동부센트레빌', '송천센트레빌', '에스케이북한산시티']
                    # , 'aptCmplxList': ['미아동부센트레빌(숭인로7가길 37)', '송천센트레빌(숭인로 39)', '에스케이북한산시티(솔샘로 174)']
                    'aptCmplxList': ['미아동부센트레빌(숭인로7가길 37)']

                    # 아파트 설정
                    # , 'aptList': ['미아동부센트레빌', '송천센트레빌', '에스케이북한산시티']
                    # , 'aptList': ['미아동부센트레빌(숭인로7가길 37)', '송천센트레빌(숭인로 39)', '에스케이북한산시티(솔샘로 174)']
                    , 'aptList': ['미아동부센트레빌(숭인로7가길 37)']

                }

            }

            # [서브 시스템] 구글 뉴스 크롤링
            # subCrawler(sysOpt)

            # [서브 시스템] 시각화 subVis(sysOpt)
            subVis(sysOpt)

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
