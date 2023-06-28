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
import pandas as pd
import seaborn as sns
from dfply import *
from plotnine.data import *
from sspipe import p, px

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
            if inParams[key] == None: continue
            val = inParams[key] if sys.argv[i + 1] == None else sys.argv[i + 1]

        if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
            if inParams[key] == None: continue
            val = inParams[key]

        # self 변수에 할당
        # setattr(self, key, val)

        # 전역 변수에 할당
        globalVar[key] = val
        log.info("[CHECK] {} / val : {}".format(key, val))

    return globalVar

def reqHwpFileDown(inHwpFile, inCaSvephy):

    prefixUrl = 'https://gnews.gg.go.kr/Operator/reporter_room/notice/download.do?'

    reqHwpUrl = (
            '{}file={}&BS_CODE=s017&CA_SAVEPHY={}'.format(prefixUrl, inHwpFile, inCaSvephy)
            | p(parse.urlparse).query
            | p(parse.parse_qs)
            | p(parse.urlencode, doseq=True)
            | prefixUrl + px
    )

    saveHwpFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, inHwpFile)

    # 디렉터리 없을 시 생성
    if not os.path.exists(os.path.dirname(saveHwpFile)):
        os.makedirs(os.path.dirname(saveHwpFile))

    # 파일 존재 유무 판단
    isFile = os.path.exists(saveHwpFile)

    # if isFile: return Pa

    res = urllib.request.urlopen(reqHwpUrl)
    resCode = res.getcode()
    resSize = int(res.headers['content-length'])

    if resCode != 200:
        return False

    if resSize < 82:
        return False

    with open(saveHwpFile, mode="wb") as f:
        f.write(res.read())

    log.info('[CHECK] saveHwpFile : {} / {} / {}'.format(inCaSvephy, isFile, saveHwpFile))

    return True


class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # LSH0199. Python을 이용한 한글 파일 (경기도 코로나19 발생 현황) 다운로드 및 저장

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'   # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    prjName = 'test'
    serviceName = 'LSH0199'

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

            # 초기 옵션 설정
            self.sysOpt = {
                'srtDate': '2021-01-01'
                , 'endDate': '2021-08-25'
            }

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

            # 시작 날짜 및 종료 날짜 설정
            srtDate = self.sysOpt['srtDate']
            endDate = self.sysOpt['endDate']

            dateList = pd.date_range(start=srtDate, end=endDate, freq="D")

            for i in dateList:
                inCaSvephy = i.strftime("%Y-%m-%d")

                # breakpoint()

                inHwpFileList = [
                    '경기도+코로나19+발생+현황({} 10시).hwp'.format(i.strftime("%Y.%#m.%#d."))
                    , '경기도+코로나19+발생+현황({}10시).hwp'.format(i.strftime("%Y.%#m.%#d."))
                    , '경기도+코로나19+발생+현황({} 10시 배포).hwp'.format(i.strftime("%Y.%#m.%#d."))
                    , '★경기도+코로나19+발생+현황({} 10시 배포).hwp'.format(i.strftime("%Y.%#m.%#d."))
                    , '경기도+코로나19+발생+현황({}10시 배포).hwp'.format(i.strftime("%Y.%#m.%#d."))
                ]

                for j, inHwpFileInfo in enumerate(inHwpFileList):
                    isResult = reqHwpFileDown(inHwpFileInfo, inCaSvephy)

                    if isResult == True: break
                        # log.info("[CHECK] isResult : {} : {}".format(isResult, inHwpFileInfo))


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
        log.info('[START] {}'.format("main"))

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

        log.info("[CHECK] inParams : {}".format(inParams))

        callDtaProcess = DtaProcess(inParams)

        callDtaProcess.runPython()

    except Exception as e:
        log.error(traceback.format_exc())
        sys.exit(1)

    finally:
        log.info('[END] {}'.format("main"))
