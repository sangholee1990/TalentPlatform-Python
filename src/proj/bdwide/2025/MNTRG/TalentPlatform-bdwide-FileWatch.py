# ================================================
# 요구사항
# ================================================
# Python을 이용한 파일이벤트 및 비동기 스케줄러 기반 라벨링 영상 생산

# 프로그램 종료
# ps -ef | grep python | grep TalentPlatform-bdwide-FileWatch.py | awk '{print $2}' | xargs kill -9
# pkill -f TalentPlatform-bdwide-FileWatch.py

# 프로그램 시작
# conda activate py38
# cd /SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2023
# cd /SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/MNTRG
# nohup python TalentPlatform-bdwide-FileWatch.py &
# tail -f nohup.out

# cd /SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/MNTRG
# nohup /SYSTEMS/LIB/anaconda3/envs/py38/bin/python TalentPlatform-bdwide-FileWatch.py &
# tail -f nohup.out

# 입력 자료
# /DATA/LABEL/ORG/생성일
# *.jpg, *.json

# 2024.03.18
# 연구원님 저번에 주신 라벨링 이미지 변환 프로그램상에 문제가 확인되어 공유드립니다.

# 1. 이미지 인식 문제
# ORG와 OLD 폴더에는 정상적으로 이미지가 업로드되나 NEW 폴더에 있는 PNG 파일이 0바이트(빈 파일)로 생성되는 경우가 자주 발생하고 있습니다.
# json 파일에 따른 이미지 생산될 경우 기존 json 파일을 삭제하게끔 변경해놨습니다.
#
# 2. 라벨링 누락
# 실제 라벨링 프로그램(라벨미)에서는 확인되는 라벨링 작업이 변환된 PNG 파일에서는 없어지는 경우가 확인되고 있습니다.
# 라벨링 과정에서 음영 색칠을 위한 4개 꼭지점이 없을 경우 정상적으로 표출이 안되는 것 같더라구요ㅠㅠ
# 혹시 관련 샘플파일을 보내주시면 확인해보겠습니다.
#
# 2가지 사항에 대한 의견을 여쭤보고자 합니다.

# /etc/security/limits.conf
# * soft nofile 65536
# * hard nofile 65536

# 20250816_Python을 이용한 라벨링 영상 개선 (폴리곤, 경계선박스)
# 최상위 경로 (/HDD/DATA/LABEL/LABEL/ORG)에서 키워드 (bbox) 유무에 따라 폴리곤/경계선박스 생산 처리

# [폴리곤 생산]
# 입력경로 /HDD/DATA/LABEL/LABEL/ORG/20250816
# JPG 출력경로 /HDD/DATA/LABEL/LABEL/OLD/20250816
# JSON 출력경로 /HDD/DATA/LABEL/LABEL/NEW/20250816

# [경계선박스 생산] 폴더 키워드 추가 (bbox)
# 입력경로 /HDD/DATA/LABEL/LABEL/ORG/20250816_bbox
# JPG 출력경로 /HDD/DATA/LABEL/LABEL/OLD/20250816_bbox
# JSON 출력경로 /HDD/DATA/LABEL/LABEL/NEW/20250816_bbox

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

from sqlalchemy.util import await_only
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

# from labelme.logger import logger
from labelme import utils
from retrying import retry
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor
import threading
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# plt.rc('font', family='Malgun Gothic')
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
def initArgument(globalVar):
    parser = argparse.ArgumentParser()

    for i, argv in enumerate(sys.argv[1:]):
        if not argv.__contains__('--'): continue
        parser.add_argument(argv)

    inParInfo = vars(parser.parse_args())
    log.info(f"inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        if env not in 'local' and key.__contains__('Path'):
            os.makedirs(val, exist_ok=True)
        globalVar[key] = val

    return globalVar

@retry(stop_max_attempt_number=10)
def makeFileProc(fileInfo):
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
                raise ValueError(f'파일 관리 실패 : {e}')

        if re.search('json', fileExt, re.IGNORECASE):
            # tmpPath = globalVar['tmpPath']
            tmpPath = tempfile.TemporaryDirectory().name
            os.makedirs(tmpPath, exist_ok=True)

            # # cmdProc = f"labelme_export_json '{fileInfo}' -o '{tmpPath}'"
            # # cmd = f"source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate py38 && {cmdProc}"
            # # cmd = f"source /HDD/SYSTEMS/LIB/anaconda3/etc/profile.d/conda.sh && conda activate py38 && {cmdProc}"
            # cmd = f"/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/labelme_export_json '{fileInfo}' -o '{tmpPath}'"
            # log.info(f'[CHECK] cmd : {cmd}')
            #
            # try:
            #     subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
            # except subprocess.CalledProcessError as e:
            #     raise ValueError(f'실행 프로그램 실패 : {e}')

            # /SYSTEMS/LIB/anaconda3/envs/py38/lib/python3.8/site-packages/labelme/cli/export_json.py 수행
            try:
                if re.search('bbox', fileRegDate, re.IGNORECASE):
                    makeLabelmeBbox(json_file=fileInfo, out_dir=tmpPath, font_size='medium')
                else:
                    makeLabelmePolygon(json_file=fileInfo, out_dir=tmpPath, font_size=20)
            except Exception as e:
                raise ValueError(f'실행 프로그램 실패 : {e}')

            oldFile = "{}/{}".format(tmpPath, 'label_viz.png')

            if not os.path.exists(oldFile):
                raise ValueError(f'파일 존재 검사')

            if os.path.getsize(oldFile) < 1:
                raise ValueError(f'파일 용량 검사')

            saveFile = "{}/{}/{}.png".format(globalVar['newPath'], fileRegDate, fileNameNoExt)
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)

            try:
                # 파일 이동
                shutil.move(oldFile, saveFile)
                log.info(f'[CHECK] saveFile : {saveFile}')

                # 임시 폴더 삭제
                shutil.rmtree(tmpPath)

                # 파일 삭제
                if os.path.exists(saveFile):
                    os.remove(fileInfo)
            except OSError as e:
                raise ValueError(f'실행 프로그램 실패 : {e}')

    except Exception as e:
        log.error(f'Exception : {e}')
        raise e

class handler(FileSystemEventHandler):
    def __init__(self, patterns):
        self.patterns = patterns

    def on_any_event(self, event):
        if not any(fnmatch.fnmatch(event.src_path, pattern) for pattern in self.patterns): return
        if not re.search('closed', event.event_type, re.IGNORECASE): return
        if not os.path.exists(event.src_path): return

        log.info(f'[CHECK] event : {event} / event_type : {event.event_type} / src_path : {event.src_path}')

        try:
            makeFileProc(event.src_path)
        except Exception as e:
            log.error(f'Exception : {e}')

def fileWatch(sysOpt):
    observer = Observer()
    eventHandler = handler(sysOpt['mntrgFileList'])
    filePathList = set(os.path.dirname(os.path.dirname(fileInfo)) for fileInfo in sysOpt['mntrgFileList'])
    for filePathInfo in filePathList:
        observer.schedule(eventHandler, filePathInfo, recursive=True)
    observer.start()

    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except Exception as e:
        log.error(f'Exception : {e}')
    finally:
        if observer.is_alive():
            observer.stop()
        observer.join()

def makeFileList(mntrgFileInfo):
    fileList = glob.glob(mntrgFileInfo)
    # log.info(f"[CHECK] fileList : {fileList}")

    for fileInfo in fileList:
        try:
            makeFileProc(fileInfo)
        except Exception as e:
            log.error(f'Exception : {e}')

    # with ProcessPoolExecutor() as executor:
    #     list(executor.map(makeFileProc, fileList))

async def asyncSchdl(sysOpt):
    scheduler = AsyncIOScheduler()

    jobList = [
        (makeFileList, 'cron', {'second': '0'}, {'args': [sysOpt['mntrgFileList'][0]]}),
        (makeFileList, 'cron', {'second': '30'}, {'args': [sysOpt['mntrgFileList'][1]]}),
    ]

    for fun, trigger, triggerArgs, kwargs in jobList:
        try:
            scheduler.add_job(fun, trigger, **triggerArgs, **kwargs)
        except Exception as e:
            log.error(f"Exception : {e}")

    scheduler.start()
    asyncEvent = asyncio.Event()

    try:
        await asyncEvent.wait()
    except Exception as e:
        log.error(f"Exception : {e}")
    finally:
        if scheduler.running:
            scheduler.shutdown()

# /SYSTEMS/LIB/anaconda3/envs/py38/lib/python3.8/site-packages/labelme/cli/export_json.py 코드 참조
def makeLabelmePolygon(json_file, out_dir, font_size=30):

    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    data = json.load(open(json_file))
    imageData = data.get("imageData")

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {"_background_": 0}
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    # 33% 크기 감소
    lbl_viz = imgviz.label2rgb(
        lbl, imgviz.asgray(img), label_names=label_names, loc="rb", font_size=font_size
    )

    # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.png"))
    saveImg = osp.join(out_dir, "label_viz.png")
    PIL.Image.fromarray(lbl_viz).save(saveImg)

    # PIL.Image.fromarray(img).save(osp.join(out_dir, "img.png"))
    # utils.lblsave(osp.join(out_dir, "label.png"), lbl)
    # with open(osp.join(out_dir, "label_names.txt"), "w") as f:
    #     for lbl_name in label_names:
    #         f.write(lbl_name + "\n")

def makeLabelmeBbox(json_file, out_dir, font_size):

    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    data = json.load(open(json_file))
    imageData = data.get("imageData")

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = utils.img_b64_to_arr(imageData)
    imgObj = PIL.Image.fromarray(img)

    label_names = sorted(list(set([shape['label'] for shape in data['shapes']])))
    label_names.insert(0, '_background_')
    colormap = imgviz.label_colormap(len(label_names))

    # 영상 생산
    dpi = 100
    figsizeInch = (imgObj.width / dpi, imgObj.height / dpi)

    fig, ax = plt.subplots(1, figsize=figsizeInch, dpi=dpi)
    ax.imshow(img)
    ax.axis('off')

    legendList = set()
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        labelName = shape["label"]
        if not labelName or labelName == "_background_":
            continue

        points = np.array(shape["points"])
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)

        width = xmax - xmin
        height = ymax - ymin

        label_index = label_names.index(labelName)
        color = np.array(colormap[label_index]) / 255.0

        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )

        ax.add_patch(rect)
        # ax.text(xmin, ymin - 10, labelName, color=color, fontsize=12, weight='bold')

        if labelName not in legendList:
            rect.set_label(labelName)
            legendList.add(labelName)

    ax.legend(loc='lower right', fontsize=font_size)
    saveImg = osp.join(out_dir, 'label_viz.png')
    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.savefig(saveImg, dpi=dpi, pad_inches=0, transparent=True)
    log.info(f'[CHECK] saveImg : {saveImg}')
    # plt.show()

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
    serviceName = 'BDWIDE2025'

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

            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/HDD/DATA/INPUT'
                globalVar['outPath'] = '/HDD/DATA/OUTPUT'
                globalVar['figPath'] = '/HDD/DATA/FIG'

                # globalVar['orgPath'] = '/DATA/TEST/LABEL/ORG'
                # globalVar['oldPath'] = '/DATA/TEST/LABEL/OLD'
                # globalVar['newPath'] = '/DATA/TEST/LABEL/NEW'
                globalVar['orgPath'] = '/HDD/DATA/LABEL/LABEL/ORG'
                globalVar['oldPath'] = '/HDD/DATA/LABEL/LABEL/OLD'
                globalVar['newPath'] = '/HDD/DATA/LABEL/LABEL/NEW'
                # globalVar['tmpPath'] = tempfile.TemporaryDirectory().name

            # 옵션 설정
            sysOpt = {
                'mntrgFileList': [
                    f'{globalVar["orgPath"]}/*/*.jpg',
                    f'{globalVar["orgPath"]}/*/*.json',
                ],
            }

            # 파일 감시
            # fileWatch(sysOpt)
            fileWachThr = threading.Thread(target=fileWatch, args=(sysOpt,), daemon=True)
            fileWachThr.start()

            # 파일 스케줄러
            asyncio.run(asyncSchdl(sysOpt))

        except Exception as e:
            log.error(f"Exception : {e}")
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