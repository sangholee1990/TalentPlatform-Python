# -*- coding: utf-8 -*-
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
from collections import Counter
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
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


def makePlotWordCloud(data, key, topKeyword, saveFile, saveImg):

    log.info('[START] {}'.format('makePlotWordCloud'))

    result = None

    try:

        nlpy = Okt()

        getData = data[key]
        getDataList = getData.to_list()
        getDataTextAll = " ".join(getDataList)

        # 명사만 추출
        nounList = nlpy.nouns(getDataTextAll)

        # 빈도 계산
        countList = Counter(nounList)

        # 상위 키워드 선정
        dictData = {}
        for none, cnt in countList.most_common():
            # 빈도수 2 이상
            if (cnt < 2): continue
            # 명사  2 글자 이상
            if (len(none) < 2): continue
            if (re.match('환경|교육|환경교육|학교|기반|조례|세부목표|사업|운영|프로그램|관리|지원', none)): continue
            if (len(dictData) >= topKeyword): continue
            dictData[none] = cnt

        # 엑셀 파일 저장
        saveData = pd.DataFrame()
        for key, val in dictData.items():
            saveDict = {
                '키워드': [key]
                , '빈도': [val]
                , '비율': [val / sum(dictData.values()) * 100]
            }

            tmpData = pd.DataFrame.from_dict(saveDict)
            saveData = saveData.append(tmpData)
        saveData.to_excel(saveFile, index=False)

        dictWordCloudData = {}
        for none, cnt in countList.most_common(len(countList)):
            # 빈도수 2 이상
            if (cnt < 2): continue
            # 명사  2 글자 이상
            if (len(none) < 2): continue
            if (re.match('환경|교육|환경교육|학교|기반|조례|세부목표|사업|운영|프로그램|관리|지원', none)): continue

            dictWordCloudData[none] = cnt

        # 워드 클라우드 생성
        wordcloud = WordCloud(
            font_path='font/malgun.ttf'
            , width=1000
            , height=1000
            , background_color="white"
        ).generate_from_frequencies(dictWordCloudData)

        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(saveImg, dpi=600, bbox_inches='tight')
        plt.show()

        result = {
            'msg': 'succ'
            , 'saveFile': saveFile
            , 'isFileExist': os.path.exists(saveFile)
            , 'saveImg': saveImg
            , 'isImgExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info('[END] {}'.format('makePlotWordCloud'))

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # LSH0292. Python을 이용한 키워드 분석 (빈도, 비율) 및 워드클라우드 시각화

    # 1. 비젼을 : 키워드별 빈도수 상위 10개 키워드 별 빈도수 및 퍼센트도 엑셀 파일
    # 2. 목표 : 키워드별 빈도수 상위 10개 키워드 별 빈도수 및 퍼센트도 엑셀 파일
    # 3. 추진전략을 큰번호 1/ 2/3/4에 맞추어 키워드별 빈도수 상위 20개 키워드 별 빈도수 및 퍼센트도 엑셀 파일
    # 4. 영역별 큰번호 1/ 2/3/4 에 맞추어 키워드별 빈도수 상위 20개 키워드 별 빈도수 및 퍼센트도 엑셀 파일

    # 비젼, 목표를 상위 키워드 20개로 시각화
    # 영역별 1/2/3/4 를 각각 추진전략과 방향 , 추진과제 상위 키워드 20개로 시각화 해 주시면 감사드리겠습니다.
    # 비용 추가가 있을 경우 말씀주세요.

    # 교육청은 빼주세요

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0292'

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
            log.info("[CHECK] inParams : {}".format(inParams))

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

            # 파일 패턴
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*.xlsx')

            # 파일 찾기
            fileList = glob.glob(inpFile)

            # 파일 없을 경우 예외 처리
            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

            fileInfo = fileList[0]

            # ************************************************************************
            # 1. 비젼 : 키워드별 빈도수 상위 10개 키워드 별 빈도수 및 퍼센트도 엑셀 파일
            # ************************************************************************
            # data = pd.read_excel(fileInfo, sheet_name='비전')
            # dataL1 = data[['구분', '비전']].dropna().reset_index(drop=True)
            #
            # dataL2 = dataL1.loc[
            #     ~dataL1['구분'].str.contains("교육청")
            #     ]
            #
            # saveImg = '{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, '비전', '워드클라우드')
            # saveFile = '{}/{}_{}_{}.xlsx'.format(globalVar['outPath'], serviceName, '비전', '키워드 분석')
            # result = makePlotWordCloud(dataL2, '비전', 20, saveFile, saveImg)
            # log.info('[CHECK] result : {}'.format(result))

            # ************************************************************************
            # 2. 목표 : 키워드별 빈도수 상위 10개 키워드 별 빈도수 및 퍼센트도 엑셀 파일
            # ************************************************************************
            # data = pd.read_excel(fileInfo, sheet_name='목표')
            # dataL1 = data[['구분', '목표']].dropna().reset_index(drop=True)
            #
            # dataL2 = dataL1.loc[
            #     ~dataL1['구분'].str.contains("교육청")
            #     ]
            #
            # saveImg = '{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, '목표', '워드클라우드')
            # saveFile = '{}/{}_{}_{}.xlsx'.format(globalVar['outPath'], serviceName, '목표', '키워드 분석')
            # result = makePlotWordCloud(dataL2, '목표', 20, saveFile, saveImg)
            # log.info('[CHECK] result : {}'.format(result))


            # ************************************************************************
            # 1. 비젼 : 키워드별 빈도수 상위 10개 키워드 별 빈도수 및 퍼센트도 엑셀 파일
            # 2. 목표 : 키워드별 빈도수 상위 10개 키워드 별 빈도수 및 퍼센트도 엑셀 파일
            # ************************************************************************
            tmpData = pd.read_excel(fileInfo, sheet_name='비전')
            tmpData2 = pd.read_excel(fileInfo, sheet_name='목표')

            dataL1 = pd.concat(
                [
                    tmpData[['구분', '비전']].dropna().reset_index(drop=True).rename( { '비전':'비전목표' } ,axis=1)
                    ,  tmpData2[['구분', '목표']].dropna().reset_index(drop=True).rename( { '목표':'비전목표' } ,axis=1)
                ]
                , axis=0
            )

            dataL2 = dataL1.loc[
                (~dataL1['구분'].str.contains("교육청"))
                # & (~dataL1['비전목표'].str.contains("환경|교육|환경|환경교육|학교|기반|조례|세부목표|사업|운영|프로그램|관리|지원"))
                ]

            saveImg = '{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, '비전목표', '워드클라우드')
            saveFile = '{}/{}_{}_{}.xlsx'.format(globalVar['outPath'], serviceName, '비전목표', '키워드 분석')
            result = makePlotWordCloud(dataL2, '비전목표', 20, saveFile, saveImg)
            log.info('[CHECK] result : {}'.format(result))

            # ************************************************************************
            # 3. 추진전략을 큰번호 1/ 2/3/4에 맞추어 키워드별 빈도수 상위 20개 키워드 별 빈도수 및 퍼센트도 엑셀 파일
            # ************************************************************************
            data = pd.read_excel(fileInfo, sheet_name='추진전략 및 방향')
            dataL1 = data[['구분', '영역', '추진전략']].dropna().reset_index(drop=True)

            for i in range(1, 5):

                dataL2 = dataL1.loc[
                    (~dataL1['구분'].str.contains("교육청"))
                    & (dataL1['영역'] == i)
                    # & (~dataL1['추진전략'].str.contains("환경|교육|환경|환경교육|학교|기반|조례|세부목표|사업|운영|프로그램|관리|지원"))
                    ]

                if (len(dataL2) < 1): continue

                saveImg = '{}/{}_{}_{}_{}.png'.format(globalVar['figPath'], serviceName, '추진전략 및 방향', i, '워드클라우드')
                saveFile = '{}/{}_{}_{}_{}.xlsx'.format(globalVar['outPath'], serviceName, '추진전략 및 방향', i, '키워드 분석')
                result = makePlotWordCloud(dataL2, '추진전략', 20, saveFile, saveImg)
                log.info('[CHECK] result : {}'.format(result))


            # ************************************************************************
            # 4. 영역별 큰번호 1/ 2/3/4 에 맞추어 키워드별 빈도수 상위 20개 키워드 별 빈도수 및 퍼센트도 엑셀 파일
            # ************************************************************************
            data = pd.read_excel(fileInfo, sheet_name='영역별 추진과제')
            dataL1 = data[['구분', '영역', '추진과제2']].dropna().reset_index(drop=True)

            for i in range(1, 5):

                dataL2 = dataL1.loc[
                    (~dataL1['구분'].str.contains("교육청"))
                    & (dataL1['영역'] == i)
                    # & (~dataL1['추진과제2'].str.contains("환경|교육|환경|환경교육|학교|기반|조례|세부목표|사업|운영|프로그램|관리|지원"))
                    ]

                if (len(dataL2) < 1): continue

                saveImg = '{}/{}_{}_{}_{}.png'.format(globalVar['figPath'], serviceName, '영역별 추진과제', i, '워드클라우드')
                saveFile = '{}/{}_{}_{}_{}.xlsx'.format(globalVar['outPath'], serviceName, '영역별 추진과제', i, '키워드 분석')
                result = makePlotWordCloud(dataL2, '추진과제2', 20, saveFile, saveImg)
                log.info('[CHECK] result : {}'.format(result))

            # ************************************************************************
            # 5. 세부추진과제
            # ************************************************************************
            data = pd.read_excel(fileInfo, sheet_name='세부추진과제')
            dataL1 = data[['구분', '영역', '세부추진과제']].dropna().reset_index(drop=True)

            dataL2 = dataL1.loc[
                (~dataL1['구분'].str.contains("교육청"))
                # & (~dataL1['세부추진과제'].str.contains("환경|교육|환경|환경교육|학교|기반|조례|세부목표|사업|운영|프로그램|관리|지원"))
                ]

            saveImg = '{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, '세부추진과제', '워드클라우드')
            saveFile = '{}/{}_{}_{}.xlsx'.format(globalVar['outPath'], serviceName, '세부추진과제', '키워드 분석')
            result = makePlotWordCloud(dataL2, '세부추진과제', 20, saveFile, saveImg)
            log.info('[CHECK] result : {}'.format(result))

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
