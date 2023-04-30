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
import glob
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from dfply import *
from plotnine.data import *
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
import re

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

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 NetCDF 파일 병합 및 경도 환산

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
    serviceName = 'LSH0321'

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

            # ********************************************
            # 옵션 설정
            # ********************************************
            sysOpt = {
                # 최대 페이지 설정
                'maxPage': 2

                # 기본 주소
                , 'prefixUrl' : 'https://land.bizmk.kr/memul/list.php'

                # 상세 주소
                , 'prefixDtlUrl' : 'https://land.bizmk.kr/memul/detail.php'
            }

            # ********************************************************************************************
            # 법정동 코드 읽기
            # ********************************************************************************************
            inpFile = '{}/{}'.format(globalVar['mapPath'], 'admCode/법정동코드_전체자료.txt')
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(fileList, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(fileList, '입력 자료를 확인해주세요.'))

            admData = pd.read_csv(fileList[0], encoding='EUC-KR', sep='\t')
            admData['법정동코드'] = admData['법정동코드'].astype(str)
            admData['filterType'] = admData['법정동코드'].str.slice(2, 10)
            admData['filterType2'] = admData['법정동코드'].str.slice(6, 10)

            admDataL1 = admData.loc[
                (admData['filterType'] != '00000000')
                & (admData['filterType2'] == '0000')
                & (admData['폐지여부'] == '존재')
                & (admData['법정동명'].str.match('서울특별시'))
            ].reset_index()

            dataL1 = pd.DataFrame()
            for idx, admInfo in admDataL1.iterrows():
                admCode = admInfo['법정동코드']
                admAddr = admInfo['법정동명']

                log.info('[CHECK] admAddr : {}'.format(admAddr))

                # ********************************************************************************************
                # 기본 정보
                # ********************************************************************************************
                pageList = range(0, 9999, 1)
                data = pd.DataFrame()
                # pageInfo = 1
                for i, pageInfo in enumerate(pageList):
                    log.info('[CHECK] pageInfo : {}'.format(pageInfo))

                    urlInfo = '?mgroup=A&mclass=A01,A02,A03&bdiv=A&tab=1&' + urlencode(
                        {
                            quote_plus('page'): pageInfo
                            ,  quote_plus('bubcode'): admCode
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
                            , 'urlInfo' : sysOpt['prefixUrl'] + urlInfo
                        }

                        data = pd.concat([data, pd.DataFrame.from_dict(dict)], axis=0, ignore_index=True)

                # ********************************************************************************************
                # 상세 정보
                # ********************************************************************************************
                for k, row in data.iterrows():
                    log.info('[CHECK] percent : {} : {} %'.format(admAddr, round(k / len(data) * 100, 1)))

                    urlDtlInfo = '?mgroup=A&mclass=A01%2CA02%2CA03&bdiv=A&areadiv=&aptcode=&scalecode=&xpos=&ypos=&tab=1&listOrder=&siteOrder=&JMJ=&' + urlencode(
                        {
                            quote_plus('mseq'): row['mseq']
                            , quote_plus('bubcode'): admCode
                        }
                    )

                    dtlAgencyName = None
                    dtlAgencyNum = None
                    dtlName = None
                    dtlPhoneNum = None
                    dtlRegNum = None
                    dtlAddr = None

                    try:
                        resDtl = urllib.request.urlopen(sysOpt['prefixDtlUrl'] + urlDtlInfo)
                        beautSoupDtl = BeautifulSoup(resDtl, 'html.parser')
                        htmlDtl = etree.HTML(str(beautSoupDtl))

                        # 중개소 이름
                        dtlAgencyNameTag = htmlDtl.xpath('//*[@id="SY"]/div/div[2]/div[1]/div[2]/div/div[2]/div[1]/table/tbody/tr[4]/td[1]/div')
                        dtlAgencyName = unicodedata2.normalize("NFC", dtlAgencyNameTag[0].text).strip() if (len(dtlAgencyNameTag) > 0) else None

                        # 중개소 번호
                        dtlAgencyNumTag = htmlDtl.xpath('//*[@id="SY"]/div/div[2]/div[1]/div[2]/div/div[2]/div[1]/table/tbody/tr[4]/td[1]/div/strong')
                        dtlAgencyNum = dtlAgencyNumTag[0].text.strip() if (len(dtlAgencyNumTag) > 0) else None

                        # 이름
                        dtlNameTag = htmlDtl.xpath('//*[@id="SY"]/div/div[2]/div[1]/div[2]/div/div[2]/div[1]/table/tbody/tr[4]/td[2]/div')
                        dtlName = unicodedata2.normalize("NFC", dtlNameTag[0].text).strip() if (len(dtlNameTag) > 0) else None

                        # 휴대폰번호
                        dtlPhoNumTag = htmlDtl.xpath('//*[@id="SY"]/div/div[2]/div[1]/div[2]/div/div[2]/div[1]/table/tbody/tr[4]/td[2]/div/strong')
                        dtlPhoneNum = dtlPhoNumTag[0].text.strip() if (len(dtlPhoNumTag) > 0) else None

                        # 개설등록번호
                        dtlRegNumTag = htmlDtl.xpath('//*[@id="SY"]/div/div[2]/div[1]/div[2]/div/div[2]/div[1]/table/tbody/tr[4]/td[3]/div')
                        dtlRegNum = dtlRegNumTag[0].text.strip() if (len(dtlRegNumTag) > 0) else None

                        # 소재지
                        dtlAddrTag = htmlDtl.xpath('//*[@id="jusoDiv"]/div/p')
                        dtlAddr = dtlAddrTag[0].text.strip() if (len(dtlAddrTag) > 0) else None

                    except Exception as e:
                        log.error("Exception : {}".format(e))

                    data.loc[k, 'dtlAgencyName'] = dtlAgencyName
                    data.loc[k, 'dtlAgencyNum'] = dtlAgencyNum
                    data.loc[k, 'dtlName'] = dtlName
                    data.loc[k, 'dtlPhoneNum'] = dtlPhoneNum
                    data.loc[k, 'dtlRegNum'] = dtlRegNum
                    data.loc[k, 'dtlAddr'] = dtlAddr
                    data.loc[k, 'admCode'] =  admInfo['법정동코드']
                    data.loc[k, 'admAddr'] = admInfo['법정동명']

                    data.loc[k, 'urlDtlInfo'] = sysOpt['prefixDtlUrl'] + urlDtlInfo

                dataL1 = pd.concat([dataL1, data], axis=0, ignore_index=True)
            
            saveFile = '{}/{}_{}_{}.xlsx'.format(globalVar['outPath'], serviceName, '서울특별시 시군구에 따른 매물 및 부동산 중개업소 목록', datetime.now().strftime('%Y%m%d%H'))
            log.info('[CHECK] saveFile : {}'.format(saveFile))
            dataL1.to_excel(saveFile)

            saveFile = '{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, '서울특별시 시군구에 따른 매물 및 부동산 중개업소 목록', datetime.now().strftime('%Y%m%d%H'))
            log.info('[CHECK] saveFile : {}'.format(saveFile))
            dataL1.to_csv(saveFile, index=False)

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