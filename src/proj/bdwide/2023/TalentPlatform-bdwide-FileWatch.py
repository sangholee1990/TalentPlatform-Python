import argparse
import glob
import json
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import urllib.parse
import warnings
from builtins import enumerate
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import pytz
from datetime import timedelta

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import fnmatch
import re
import tempfile
import subprocess
import shutil
import asyncio

import argparse
import base64
import json
import os
import os.path as osp

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils


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
# font_manager._rebuild()

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')
dtKst = timedelta(hours=9)


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

        # 글꼴 설정
        plt.rc('font', family='Malgun Gothic')

    # 리눅스 환경
    if globalVar['sysOs'] in 'Linux':
        parser = argparse.ArgumentParser()

        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)

        inParInfo = vars(parser.parse_args())

        # 글꼴 설정
        #fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        #fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        #plt.rcParams['font.family'] = fontName

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


def makeFileProc(fileInfo):

    # log.info(f'[START] makeFileProc')

    try:
        if not os.path.exists(fileInfo): return
        log.info(f'[CHECK] fileInfo : {fileInfo}')

        fileRegDate = fileInfo.replace(globalVar['orgPath'], '').split('/')[1]
        fileName = os.path.basename(fileInfo)
        fileNameNoExt = fileName.split('.')[0]
        fileExt = fileName.split('.')[1]

        # 자료 처리
        if re.search('jpg', fileExt, re.IGNORECASE):
            saveFile = "{}/{}/{}.jpg".format(globalVar['oldPath'], fileRegDate, fileNameNoExt)
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)

            try:
                shutil.move(fileInfo, saveFile)
                log.info(f'[CHECK] saveFile : {saveFile}')
            except OSError as e:
                log.error(f'OSError : {e}')

        if re.search('json', fileExt, re.IGNORECASE):
            tmpPath = globalVar['tmpPath']
            os.makedirs(tmpPath, exist_ok=True)

            cmdProc = f"labelme_export_json '{fileInfo}' -o '{tmpPath}'"
            cmd = f"source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate py38 && {cmdProc}"
            log.info(f'[CHECK] cmd : {cmd}')

            res = subprocess.run(cmd, shell=True, executable='/bin/bash')
            if res.returncode != 0: log.info('[ERROR] cmd : {}'.format(cmd))

            oldFile = "{}/{}".format(tmpPath, 'label_viz.png')

            saveFile = "{}/{}/{}.png".format(globalVar['newPath'], fileRegDate, fileNameNoExt)
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)

            # 파일 이동
            try:
                shutil.move(oldFile, saveFile)
                log.info(f'[CHECK] saveFile : {saveFile}')
            except OSError as e:
                log.error(f'OSError : {e}')

            # 파일 삭제
            try:
                os.remove(fileInfo)
            except OSError as e:
                pass

            try:
                shutil.rmtree(tmpPath)
            except OSError as e:
                pass

    except Exception as e:
        log.error(f'Exception : {e}')

    # finally:
        # log.info(f'[END] makeFileProc')

class Handler(FileSystemEventHandler):
    def __init__(self, patterns):
        self.patterns = patterns

    def on_any_event(self, event):
        log.info(f'[CHECK] event : {event} / event_type : {event.event_type} / src_path : {event.src_path}')

        if not any(fnmatch.fnmatch(event.src_path, pattern) for pattern in self.patterns): return
        if not re.search('closed', event.event_type, re.IGNORECASE): return
        if not os.path.exists(event.src_path): return

        log.info(f'[CHECK] event : {event} / event_type : {event.event_type} / src_path : {event.src_path}')

        makeFileProc(event.src_path)

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 기존 파일 처리 및 신규 파일 감시

    # 프로그램 종료
    
    # ps -ef | grep python | grep TalentPlatform-bdwide-FileWatch.py | awk '{print $2}' | xargs kill -9

    # 프로그램 시작
    # conda activate py38
    # cd /SYSTEMS/PROG/PYTHON/PyCharm/src/proj/bdwide/2023
    # nohup python TalentPlatform-bdwide-FileWatch.py &

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
    serviceName = 'BDWIDE2023'

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

            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

                globalVar['orgPath'] = '/DATA/LABEL/ORG'
                globalVar['oldPath'] = '/DATA/LABEL/OLD'
                globalVar['newPath'] = '/DATA/LABEL/NEW'
                globalVar['tmpPath'] = tempfile.TemporaryDirectory().name

            # 옵션 설정
            sysOpt = {
                # 모니터링 파일
                'mntrgFileList': [
                    f'{globalVar["orgPath"]}/*/*.jpg'
                    , f'{globalVar["orgPath"]}/*/*.json'
                ]
            }

            mntrgFileList = sysOpt['mntrgFileList']
            log.info(f'[CHECK] mntrgFileList : {mntrgFileList}')

            filePathList = set(os.path.dirname(os.path.dirname(fileInfo)) for fileInfo in mntrgFileList)

            # 기존 파일 처리
            for mntrgFileInfo in mntrgFileList:
                fileList = glob.glob(mntrgFileInfo)
                for fileInfo in fileList:
                    makeFileProc(fileInfo)

            # 신규 파일 감시
            observer = Observer()
            eventHandler = Handler(mntrgFileList)

            for filePathInfo in filePathList:
                observer.schedule(eventHandler, filePathInfo, recursive=True)

            observer.start()

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt as e:
                log.error(f'KeyboardInterrupt : {e}')
                observer.stop()

            observer.join()

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