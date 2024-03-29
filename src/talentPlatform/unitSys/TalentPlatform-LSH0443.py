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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns

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

# 맵 시각화
def makeProc(data, saveFile, saveImg):

    log.info(f'[START] makeProc')

    result = None

    try:

        # 명사만 추출
        # nounList = nlpy.nouns(getDataTextAll)
        # nounList = data['상권업종중분류명'].tolist()
        # nounList = data[0].tolist()
        nounList = data.tolist()

        # 빈도 계산
        countList = Counter(nounList)

        # for i, topPer in enumerate(sysOpt['topPerList']):
        #     log.info(f'[CHECK] topPer: {topPer}')

        # 상위 20% 선정
        # maxCnt = int(len(countList) * sysOpt['topPerInfo'] / 100)
        # log.info(f'[CHECK] maxCnt : {maxCnt}')

        dictData = {}
        for none, cnt in countList.most_common():
            # 빈도수 2 이상
            if (cnt < 2): continue
            # 명사  2 글자 이상
            if (len(none) < 2): continue

            dictData[none] = cnt

        # 빈도분포
        saveData = pd.DataFrame.from_dict(dictData.items()).rename(
            {
                0: 'none'
                , 1: 'cnt'
            }
            , axis=1
        )
        saveData['cum'] = saveData['cnt'].cumsum() / saveData['cnt'].sum() * 100
        maxCnt = (saveData['cum'] > 20).idxmax()

        # saveFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, fileNameNoExt)
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        saveData.to_csv(saveFile, index=False)
        # log.info(f'[CHECK] saveFile : {saveFile}')

        # *********************************************************
        # 그래프 병합
        # *********************************************************
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # 단어 구름
        wordcloud = WordCloud(
            width=1500
            , height=1500
            , background_color=None
            , mode='RGBA'
            , font_path="NanumGothic.ttf"
        ).generate_from_frequencies(dictData)

        ax1 = axs[0]
        ax1.imshow(wordcloud, interpolation="bilinear")
        ax1.axis("off")

        # 빈도 분포
        ax2 = axs[1]
        bar = sns.barplot(x='none', y='cnt', data=saveData, ax=ax2, linewidth=0)
        ax2.set_title('업종 빈도 분포도')
        ax2.set_xlabel(None)
        ax2.set_ylabel('빈도 개수')
        # ax2.set_xlim([-1.0, len(countList)])
        ax2.set_xlim([-1.0, len(dictData)])
        # ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=7, rotation=45, horizontalalignment='right')
        ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=6, rotation=90)
        line = ax2.twinx()
        line.plot(saveData.index, saveData['cum'], color='black', marker='o', linewidth=1)
        line.set_ylabel('누적 비율', color='black')
        line.set_ylim(0, 101)

        # 20% 누적비율에 해당하는 가로줄 추가
        line.axhline(y=20, color='r', linestyle='-')

        # 7번째 막대에 대한 세로줄 추가
        ax2.axvline(x=maxCnt, color='r', linestyle='-')

        # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, fileNameNoExt)
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        plt.tight_layout()
        # plt.subplots_adjust(hspace=1)
        # plt.subplots_adjust(hspace=0, left=0, right=1)
        # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
        # plt.show()
        plt.close()

        result = {
            'msg': 'succ'
            , 'saveFile': saveFile
            , 'isFileExist': os.path.exists(saveFile)
            , 'saveImg': saveImg
            , 'isImgExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        print("Exception : {}".format(e))
        return result
    finally:
        # try, catch 구문이 종료되기 전에 무조건 실행
        log.info(f'[END] makeProc')

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 점포정보에 대한 키워드 추출 및 상위 20% 팔레트 및 워드클라우드 시각화

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
    serviceName = 'LSH0443'

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
                    # 상위 비율
                    'topPerInfo': 20
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 상위 비율
                    'topPerInfo': 20
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 전역 설정
            plt.rcParams['font.family'] = 'NanumGothic'

            # 데이터 읽기
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*')
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '키워드 입력 자료.txt')
            fileList = sorted(glob.glob(inpFile))

            # fileInfo = fileList[1]
            for i, fileInfo in enumerate(fileList):
                log.info(f'[CHECK] fileInfo: {fileInfo}')

                fileNameNoExt = os.path.basename(fileInfo).split('.')[0]
                # data = pd.read_csv(fileInfo, encoding='UTF-8')
                data = pd.read_csv(fileInfo, header=None)

                saveFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, fileNameNoExt)
                saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, fileNameNoExt)
                result = makeProc(data=data[0], saveFile=saveFile, saveImg=saveImg)
                log.info(f'[CHECK] result : {result}')

                # # 명사만 추출
                # # nounList = nlpy.nouns(getDataTextAll)
                # # nounList = data['상권업종중분류명'].tolist()
                # nounList = data[0].tolist()
                #
                # # 빈도 계산
                # countList = Counter(nounList)
                #
                # # for i, topPer in enumerate(sysOpt['topPerList']):
                # #     log.info(f'[CHECK] topPer: {topPer}')
                #
                # # 상위 20% 선정
                # # maxCnt = int(len(countList) * sysOpt['topPerInfo'] / 100)
                # # log.info(f'[CHECK] maxCnt : {maxCnt}')
                #
                # dictData = {}
                # for none, cnt in countList.most_common():
                #     # 빈도수 2 이상
                #     if (cnt < 2): continue
                #     # 명사  2 글자 이상
                #     if (len(none) < 2): continue
                #
                #     dictData[none] = cnt
                #
                # # 빈도분포
                # saveData = pd.DataFrame.from_dict(dictData.items()).rename(
                #     {
                #         0: 'none'
                #         , 1: 'cnt'
                #     }
                #     , axis=1
                # )
                # saveData['cum'] = saveData['cnt'].cumsum() / saveData['cnt'].sum() * 100
                # maxCnt = (saveData['cum'] > 20).idxmax()
                #
                # # saveFile = '{}/{}/{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, fileNameNoExt, topPer, 'cnt')
                # # saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, fileNameNoExt, 'cnt')
                # saveFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, fileNameNoExt)
                # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                # saveData.to_csv(saveFile, index=False)
                # log.info(f'[CHECK] saveFile : {saveFile}')
                #
                # # *********************************************************
                # # 빈도 분포
                # # *********************************************************
                # # saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, fileNameNoExt, 'cnt')
                # # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                # # bar = sns.barplot(x='none', y='cnt', data=saveData)
                # # # plt.title(f'상위{topPer}% 빈도 분포')
                # # plt.title(f'업종 빈도 분포도')
                # # plt.xlabel(None)
                # # plt.ylabel('빈도 개수')
                # # plt.xticks(rotation=45, ha='right')
                # # line = bar.twinx()
                # # line.plot(saveData.index, saveData['cum'], color='black', marker='o', linewidth=1)
                # # line.set_ylabel('누적 비율', color='black')
                # # plt.tight_layout()
                # # plt.savefig(saveImg, dpi=600, width=1000, height=800, bbox_inches='tight', transparent=True)
                # # plt.show()
                # # plt.close()
                # # log.info(f'[CHECK] saveImg : {saveImg}')
                #
                # # *********************************************************
                # # 단어 구름
                # # *********************************************************
                # # wordcloud = WordCloud(
                # #     width=1000
                # #     , height=1000
                # #     , background_color = None
                # #     , mode = 'RGBA'
                # #     , font_path="NanumGothic.ttf"
                # # ).generate_from_frequencies(dictData)
                # #
                # # saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, fileNameNoExt, 'wordCloud')
                # # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                # # plt.imshow(wordcloud, interpolation="bilinear")
                # # plt.axis("off")
                # # # plt.title(f'상위{topPer}% 단어 구름')
                # # plt.tight_layout()
                # # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                # # plt.show()
                # # plt.close()
                # # log.info(f'[CHECK] saveImg : {saveImg}')

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
