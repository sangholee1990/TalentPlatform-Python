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
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import xarray as xr

import urllib.request
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote_plus
from bs4 import BeautifulSoup
from lxml import etree
import xml.etree.ElementTree as et
import requests
from lxml import html
import urllib
import unicodedata2

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


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 증발산 2종 결과에 대한 과거대비 변화율 산정, 통계 계산

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/pycharm_project_83'
        # contextPath = os.getcwd() if env in 'local' else '/home/dxinyu/SYSTEMS/PROG/PYTHON'

    prjName = 'test'
    serviceName = 'LSH0352'

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
                    # 최대 페이지 설정
                    'maxPage': 2

                    # 기본 주소
                    , 'prefixUrl': 'http://oneroma6.com/shop'

                    # 상세 주소
                    # , 'prefixDtlUrl': 'https://land.bizmk.kr/memul/detail.php'
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 최대 페이지 설정
                    'maxPage': 2

                    # 기본 주소
                    , 'prefixUrl': 'http://oneroma6.com'

                    # 상세 주소
                    # , 'prefixDtlUrl': 'https://land.bizmk.kr/memul/detail.php'
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # ********************************************************************************************
            # 기본 정보
            # ********************************************************************************************

            res = urllib.request.urlopen(sysOpt['prefixUrl'])
            # res = urllib.request.urlopen('https://www.naver.com/')
            # res = urllib.request.urlopen('http://roma6789.com/')
            # res = urllib.request.urlopen('http://oneroma6.com/')
            # res = urllib.request.urlopen('https://oneroma6.com/shop/list.php?ca_id=80')
            # res = urllib.request.urlopen('http://oneroma6.com/shop/item.php?it_id=1587289154')
            # beautSoup = BeautifulSoup(res, 'html.parser')
            # html = etree.HTML(str(beautSoup))

            session = requests.Session()
            headers = {
                'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit 537.36 (KHTML, like Gecko) Chrome'
                , 'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
            res = session.get(sysOpt['prefixUrl'], headers=headers).content
            beautSoup = BeautifulSoup(res, 'html.parser')
            html = etree.HTML(str(beautSoup))

            # 메뉴 목록
            data = pd.DataFrame()

            menuList = html.xpath('//*[@id="categoryNavigation5bc1ea6f871a6"]/ul/li[*]/a')
            for i, menuInfo in enumerate(menuList):

                title = menuInfo.text.strip() if (len(menuInfo.text) > 0) else None
                menuUrl = menuInfo.attrib['href']

                dict = {
                    'title': [title]
                    , 'idx': i
                    , 'menuUrl': menuUrl
                }

                # 1043 / 44
                # //*[@id="sct_sort"]/ul/li[1]/div/b
                tt = html.xpath('//*[@id="sct_sort"]/ul/li[1]/div/b')[0].text.strip()

                import re
                re.sub(r'[^0-9]', '', tt)

                np.ceil(1043 / (4 * 10))



                # 23.704545454545453
                # http://oneroma6.com/shop/list.php?ca_id=c0&sort=&sortodr=&page=1

                menuUrl= 'http://oneroma6.com/shop/list.php?ca_id=c0&sort=&sortodr=&page=1'
                res = session.get(menuUrl, headers=headers).content
                beautSoup = BeautifulSoup(res, 'html.parser')
                html = etree.HTML(str(beautSoup))
                itemList = html.xpath('//*[@id="sct"]/ul/li[*]/div[1]/a')
                itemList[0].attrib['href']

                imgThnTag = html.xpath('//*[@id="sct"]/ul/li[1]/div[1]/a/img')
                imgThnTag[0].attrib['src']
                # http: // oneroma6.com / data / item / 1662458519 / thumb - thumbia_100000519_500x500_500x500.jpg

                for j, itemInfo in enumerate(itemList):
                    print(itemInfo.attrib['href'])

                itemDrlList = 'http://oneroma6.com/shop/item.php?it_id=1662996376'
                resDtl = session.get(itemDrlList, headers=headers).content
                beautSoupDtl = BeautifulSoup(resDtl, 'html.parser')
                htmlDtl = etree.HTML(str(beautSoupDtl))

                # 상품 제목
                dtlTitleTag = htmlDtl.xpath('//*[@id="sit_title"]')
                dtlTitle = dtlTitleTag[0].text.strip() if (len(dtlTitleTag) > 0) else None

                # 상품 썸네일
                dtlImgThmTag = htmlDtl.xpath('//*[@id="sit_pvi_big"]/div[1]/div/div/div/a/img')
                dtlTitle = dtlTitleTag[0].text.strip() if (len(dtlTitleTag) > 0) else None
                d = beautSoupDtl('#sit_pvi_big img')
                d = htmlDtl.xpath('img')
                # [0].attrib['src']
                # for j, mseqTagInfo in data:
            #     tagInfo = mseqTagInfo
            #     tagAttrInfo = tagInfo.attrib
            #
            #     title = tagAttrInfo['title'].strip() if (len(tagAttrInfo['title']) > 0) else None
            #     mseq = tagAttrInfo['href'].split("'")[1].strip() if (len(tagAttrInfo['href']) > 0) else None
            #
            #     dict = {
            #         'title': [title]
            #         , 'mseq': [mseq]
            #         , 'page': pageInfo
            #         , 'idx': j
            #         , 'urlInfo': sysOpt['prefixUrl'] + urlInfo
            #     }

                data = pd.concat([data, pd.DataFrame.from_dict(dict)], axis=0, ignore_index=True)


            pageList = range(0, 9999, 1)
            data = pd.DataFrame()
            # pageInfo = 1
            for i, pageInfo in enumerate(pageList):
                log.info('[CHECK] pageInfo : {}'.format(pageInfo))

                urlInfo = '?mgroup=A&mclass=A01,A02,A03&bdiv=A&tab=1&' + urlencode(
                    {
                        quote_plus('page'): pageInfo
                        # , quote_plus('bubcode'): admCode
                    }
                )

                res = urllib.request.urlopen(sysOpt['prefixUrl'] + urlInfo)
                beautSoup = BeautifulSoup(res, 'html.parser')
                html = etree.HTML(str(beautSoup))

                if (i > sysOpt['maxPage']): break

                # 매물 목록
                mseqTagList = html.xpath('//*[@id="SY"]/div/div/div[1]/div[3]/table/tbody/tr[*]/td[4]/div/a[1]')

                for j, mseqTagInfo in enumerate(mseqTagList):
                    tagInfo = mseqTagInfo
                    tagAttrInfo = tagInfo.attrib

                    title = tagAttrInfo['title'].strip() if (len(tagAttrInfo['title']) > 0) else None
                    mseq = tagAttrInfo['href'].split("'")[1].strip() if (len(tagAttrInfo['href']) > 0) else None

                    dict = {
                        'title': [title]
                        , 'mseq': [mseq]
                        , 'page': pageInfo
                        , 'idx': j
                        , 'urlInfo': sysOpt['prefixUrl'] + urlInfo
                    }

                    data = pd.concat([data, pd.DataFrame.from_dict(dict)], axis=0, ignore_index=True)


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
