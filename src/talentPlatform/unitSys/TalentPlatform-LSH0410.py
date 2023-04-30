# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import re
import sys
import traceback
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from webcolors import name_to_rgb

# import xarray as xr

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

def getColorNameToRgb(name):
    rgb = name_to_rgb(name)
    return rgb[0], rgb[1], rgb[2]

def getColorNameToRgba(name, alpha=1.0):
    rgb = mcolors.CSS4_COLORS[name]
    rgba = tuple(map(lambda x: int(x * 255), mcolors.to_rgba(rgb, alpha)))
    return rgba

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'

    prjName = 'test'
    serviceName = 'LSH0410'

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
                    'fontStyle': {'기본': 'malgun.ttf'}
                    , 'fontColor': {'기본': 'black'}
                    , 'fontSize': {'기본': 10}
                    , 'backColor': {'기본': 'white'}
                    , 'backAlpha': {'기본': 1.0}
                    , 'borderColor': {'기본': 'black'}
                    , 'borderSize': {'기본': 2}
                    , 'borderAlpha': {'기본': 1.0}
                    , 'fontRowPos': {'기본': 0}
                    , 'fontColPos': {'기본': 0}
                    , 'topMargin': {'기본': 0}
                    , 'botMargin': {'기본': 0}
                    , 'leftMargin': {'기본': 0}
                    , 'rightMargin': {'기본': 0}
                    , 'widSize': {'기본': 100}
                    , 'heiSize': {'기본': 100}
                    , 'dpi': {'기본': 100}
                    , 'isVet': {'기본': 'N'}
                }

            else:

                # 옵션 설정
                sysOpt = {
                    'fontStyle': {'기본': 'malgun.ttf'}
                    , 'fontColor': {'기본': 'black'}
                    , 'fontSize': {'기본': 10}
                    , 'backColor': {'기본': 'white'}
                    , 'backAlpha': {'기본': 1.0}
                    , 'borderColor': {'기본': 'black'}
                    , 'borderSize': {'기본': 2}
                    , 'borderAlpha': {'기본': 1.0}
                    , 'fontRowPos': {'기본': 0}
                    , 'fontColPos': {'기본': 0}
                    , 'topMargin': {'기본': 0}
                    , 'botMargin': {'기본': 0}
                    , 'leftMargin': {'기본': 0}
                    , 'rightMargin': {'기본': 0}
                    , 'widSize': {'기본': 100}
                    , 'heiSize': {'기본': 100}
                    , 'dpi': {'기본': 100}
                    , 'isVet' : {'기본': 'N'}
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'


            # ********************************************************************************************
            # 메타 데이터 정보
            # ********************************************************************************************
            # 글꼴 목록
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'FONT/*.ttf')
            # inp2File = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'FONT/*.TTF')
            # fileList = sorted(glob.glob(inpFile) + glob.glob(inp2File))
            #
            # # fileInfo = fileList[0]
            # fontData = pd.DataFrame()
            # for i, fileInfo in enumerate(fileList):
            #     log.info(f'[CHECK] fileInfo : {fileInfo}')
            #
            #     fontInfo = TTFont(fileInfo)
            #
            #     fileName = os.path.basename(fileInfo)
            #     fileNameNoExt = fileName.split('.')[0]
            #
            #     # 국문/영문 정보 가져오기
            #     fontEnName = fontInfo['name'].getName(4, 3, 1, 1033).toUnicode() if (fontInfo['name'].getName(4, 3, 1, 1033) is not None) else np.nan
            #     fontKoName = fontInfo['name'].getName(4, 3, 1, 1042).toUnicode() if (fontInfo['name'].getName(4, 3, 1, 1042) is not None) else None
            #
            #     dict = {
            #         'fileName': [fileName]
            #         , 'fileNameNoExt': [fileNameNoExt]
            #         , 'fontEnName': [fontEnName]
            #         , 'fontKoName': [fontKoName]
            #     }
            #
            #     fontData = pd.concat([fontData, pd.DataFrame.from_dict(dict)], ignore_index=True)

            saveXlsxFile = '{}/{}/{}.xlsx'.format(globalVar['outPath'], serviceName, 'fontData')
            # os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)
            # fontData.to_excel(saveXlsxFile, index=False)
            # log.info(f'[CHECK] saveXlsxFile : {saveXlsxFile}')
            fontData = pd.read_excel(saveXlsxFile)
            fontData = fontData.astype('str')

            # ********************************************************************************************
            # 기본 정보
            # ********************************************************************************************
            # inpFile = '{}/{}/{}.xlsx'.format(globalVar['inpPath'], serviceName, '텍스트변환')
            # inpFile = '{}/{}/{}.xlsx'.format(globalVar['inpPath'], serviceName, '20230317_텍스트변환')
            # inpFile = '{}/{}/{}.xlsx'.format(globalVar['inpPath'], serviceName, '20230318_텍스트변환')
            inpFile = '{}/{}/{}.xlsx'.format(globalVar['inpPath'], serviceName, '20230403_텍스트변환')
            fileList = sorted(glob.glob(inpFile))

            if (len(fileList) < 1):
                raise Exception("[ERROR] fileInfo : {} : {}".format("입력 자료를 확인해주세요.", inpFile))

            fileInfo = fileList[0]

            # 엑셀 파일 읽기
            data = pd.read_excel(fileInfo, sheet_name='설정 정보')
            # data.columns = ['fileName', 'dpi', 'text', 'widSize', 'heiSize', 'leftMargin', 'rightMargin', 'topMargin', 'botMargin', 'fontRowPos', 'fontColPos', 'fontColor', 'fontSize', 'fontStyle', 'backColor', 'isVet']
            # data.columns = ['fileName', 'dpi', 'text', 'widSize', 'heiSize', 'leftMargin', 'rightMargin', 'topMargin', 'botMargin', 'fontRowPos', 'fontColPos', 'fontColor', 'fontSize', 'fontStyle', 'backColor', 'backAlpha', 'isVet']
            # data.columns = ['fileName', 'dpi', 'text', 'widSize', 'heiSize', 'leftMargin', 'rightMargin', 'topMargin', 'botMargin', 'fontRowPos', 'fontColPos', 'fontColor', 'fontSize', 'fontStyle', 'backColor', 'backAlpha', 'borderColor', 'borderSize', 'borderAlpha', 'isVet']
            data.columns = ['fileName', 'text', 'widSize', 'heiSize', 'leftMargin', 'rightMargin', 'topMargin', 'botMargin', 'fontColor', 'fontStyle', 'backColor', 'backAlpha', 'borderColor', 'borderSize', 'borderAlpha', 'isVet']

            # info = data.iloc[0]
            # info = data.iloc[4]
            for i, info in data.iterrows():
                log.info(f'[CHECK] info : {info.values}')

                try:
                    # 폰트 크기
                    # fontSize = int(info['fontSize']) if ((info['fontSize'] is not np.nan) and (info['fontSize'] is not None) and (info['fontSize'] > 0)) else sysOpt['fontSize']['기본']
                    fontSize = sysOpt['fontSize']['기본']

                    # 폰트 설정
                    fontStyle = sysOpt['fontStyle']['기본']

                    if ((info['fontStyle'] is not np.nan) and (info['fontStyle'] is not None) and (len(info['fontStyle']) > 0)):
                        regexPattern = re.compile(info['fontStyle'])
                        findData = fontData.applymap(lambda x: bool(regexPattern.findall(x))).sum(axis=1)
                        if (findData.max() > 0): fontStyle = fontData.iloc[findData.idxmax(), ].fileName

                    fontStyleInfo = f"{globalVar['inpPath']}/{serviceName}/FONT/{fontStyle}"

                    # 폰트 컬러
                    fontColor = info['fontColor'] if ((info['fontColor'] is not np.nan) and (info['fontColor'] is not None) and (len(info['fontColor']) > 0)) else sysOpt['fontColor']['기본']
                    fontColorInfo = getColorNameToRgba(fontColor)

                    # 텍스트 위치 설정
                    # fontRowPos = int(info['fontRowPos']) if ((info['fontRowPos'] is not np.nan) and (info['fontRowPos'] is not None) and (info['fontRowPos'] > 0)) else sysOpt['fontRowPos']['기본']
                    # fontColPos = int(info['fontColPos']) if ((info['fontColPos'] is not np.nan) and (info['fontColPos'] is not None) and (info['fontColPos'] > 0)) else sysOpt['fontColPos']['기본']

                    # 이미지 크기 설정
                    widSize = int(info['widSize']) if ((info['widSize'] is not np.nan) and (info['widSize'] is not None) and (info['widSize'] > 0)) else sysOpt['widSize']['기본']
                    heiSize = int(info['heiSize']) if ((info['heiSize'] is not np.nan) and (info['heiSize'] is not None) and (info['heiSize'] > 0)) else sysOpt['heiSize']['기본']

                    # 배경 투명도 설정
                    backAlpha = info['backAlpha'] if ((info['backAlpha'] is not np.nan) and (info['backAlpha'] is not None) and (info['backAlpha'] >= 0)) else sysOpt['backAlpha']['기본']

                    # 배경 색상 설정
                    backColor = info['backColor'] if ((info['backColor'] is not np.nan) and (info['backColor'] is not None) and (len(info['backColor']) > 0)) else sysOpt['backColor']['기본']
                    backColorInfo = getColorNameToRgba(backColor, backAlpha)

                    # 테두리 투명도 설정
                    borderAlpha = info['borderAlpha'] if ((info['borderAlpha'] is not np.nan) and (info['borderAlpha'] is not None) and (info['borderAlpha'] > 0)) else sysOpt['borderAlpha']['기본']

                    # 테두리 크기
                    borderSize = int(info['borderSize']) if ((info['borderSize'] is not np.nan) and (info['borderSize'] is not None) and (info['borderSize'] > 0)) else sysOpt['borderSize']['기본']

                    # 테두리 컬러
                    borderColor = info['backColor'] if ((info['backColor'] is not np.nan) and (info['backColor'] is not None) and (len(info['backColor']) > 0)) else sysOpt['backColor']['기본']
                    borderColorInfo = getColorNameToRgba(borderColor, borderAlpha)

                    # 이미지 생성
                    image = Image.new('RGBA', (widSize, heiSize), backColorInfo)

                    # 이미지에 텍스트 그리기
                    draw = ImageDraw.Draw(image)

                    # 텍스트 가로/세로 쓰기
                    text = info['text']
                    isVet = info['isVet'] if ((info['isVet'] is not np.nan) and (info['isVet'] is not None) and (len(info['isVet']) > 0)) else sysOpt['isVet']['기본']

                    # 폰트 설정
                    font = ImageFont.truetype(fontStyleInfo, fontSize)
                    textWidSize, textHeiSize = font.getsize(text)

                    # 폰트 크기 자동 조정
                    if (isVet == 'Y'):
                        limitHeiSize = heiSize / len(text)

                        while textHeiSize < limitHeiSize:
                            fontSize += 1
                            font = ImageFont.truetype(fontStyleInfo, fontSize)
                            textWidSize, textHeiSize = font.getsize(text)

                    else:
                        limitHeiSize = heiSize

                        while textWidSize < widSize:
                            fontSize += 1
                            font = ImageFont.truetype(fontStyleInfo, fontSize)
                            textWidSize, textHeiSize = font.getsize(text)

                    log.info(f'[CHECK] fontSize : {fontSize}')

                    # fontSize = fontSize * 0.98
                    # fontSize = fontSize * 0.99
                    fontSize -= 1
                    font = ImageFont.truetype(fontStyleInfo, int(fontSize))
                    textWidSize, textHeiSize = font.getsize(text)
                    log.info(f'[CHECK] fontSize : {fontSize}')
                    log.info(f'[CHECK] textWidSize / textHeiSize : {textWidSize} / {textHeiSize}')

                    if (isVet == 'Y'):
                        for i, char in enumerate(text):
                            # x = fontRowPos
                            # y = fontColPos + (i * heiSize/25)
                            # y = (i * textHeiSize)
                            # x = (widSize - textWidSize) / 2.0
                            # y = ((heiSize - textHeiSize) / 2.0) + (i * textHeiSize)
                            x = (widSize / 2.0) - (textHeiSize / 2.0)
                            y = (i * textHeiSize)

                            log.info(f'[CHECK] x / y : {x} / {y}')

                            # 폰트 경계선
                            for ii in range(-borderSize, borderSize + 1):
                                for jj in range(-borderSize, borderSize + 1):
                                    if ii == x and jj == y: continue
                                    draw.text((x + ii, y + jj), char, font=font, fill=borderColorInfo, angle=-90, align='center')
                                    # draw.text((x + ii, y + jj), char, font=font, fill=borderColorInfo, angle=-90)

                            draw.text((x, y), char, font=font, fill=fontColorInfo, angle=-90, align='center')
                            # draw.text((x, y), char, font=font, fill=fontColorInfo, angle=-90)

                    else:
                        # draw.text((fontRowPos, fontColPos), text, font=font, fill=fontColorInfo, align='center')
                        # draw.text((widSize / 2.0, heiSize / 2.0), text, font=font, fill=fontColorInfo, align='center')

                        x = (widSize - textWidSize) / 2.0
                        # y = (heiSize - textHeiSize) / 2.0
                        y = ((heiSize - textHeiSize) / 2.0)
                        # x = 0
                        # y = 0

                        log.info(f'[CHECK] x / y : {x} / {y}')

                        # 폰트 경계선
                        for ii in range(-borderSize, borderSize + 1):
                            for jj in range(-borderSize, borderSize + 1):
                                if ii == x and jj == y: continue
                                draw.text((x + ii, y + jj), text, font=font, fill=borderColorInfo, align='center')
                                # draw.text((x + ii, y + jj), text, font=font, fill=borderColorInfo)

                        draw.text((x, y), text, font=font, fill=fontColorInfo, align='center')
                        # draw.text((x, y), text, font=font, fill=fontColorInfo)

                    # 테두리 설정
                    # for j in range(borderSize):
                    #     draw.rectangle((j, j, widSize - j, heiSize - j), outline=fontColorInfo)

                    # 여백 설정
                    topMargin = int(info['topMargin']) if ((info['topMargin'] is not np.nan) and (info['topMargin'] is not None) and (info['topMargin'] > 0)) else sysOpt['topMargin']['기본']
                    botMargin = int(info['botMargin']) if ((info['botMargin'] is not np.nan) and (info['botMargin'] is not None) and (info['botMargin'] > 0)) else sysOpt['botMargin']['기본']
                    leftMargin = int(info['leftMargin']) if ((info['leftMargin'] is not np.nan) and (info['leftMargin'] is not None) and (info['leftMargin'] > 0)) else sysOpt['leftMargin']['기본']
                    rightMargin = int(info['rightMargin']) if ((info['rightMargin'] is not np.nan) and (info['rightMargin'] is not None) and (info['rightMargin'] > 0)) else sysOpt['rightMargin']['기본']
                    #
                    # # 이미지에 여백 추가
                    finalWidth = widSize + leftMargin + rightMargin
                    finalHeight = heiSize + topMargin + botMargin
                    # finalImage = Image.new('RGB', (finalWidth, finalHeight), backColorInfo)
                    finalImage = Image.new('RGBA', (finalWidth, finalHeight), backColorInfo)
                    finalImage.paste(image, (leftMargin, topMargin))

                    # 해상도 설정
                    # dpi = int(info['dpi']) if ((info['dpi'] is not np.nan) and (info['dpi'] is not None) and (info['dpi'] > 0)) else sysOpt['dpi']['기본']

                    # 이미지 저장
                    mainTitle = '{}'.format(info['fileName'])
                    saveImg = '{}/{}/{}'.format(globalVar['figPath'], serviceName, mainTitle)
                    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                    # finalImage.save(saveImg, dpi=(dpi, dpi))
                    # image.save(saveImg, dpi=(dpi, dpi))
                    finalImage.save(saveImg, dpi=(100, 100))
                    # image.save(saveImg, dpi=(100, 100))
                    log.info('[CHECK] saveImg : {}'.format(saveImg))

                except Exception as e:
                    log.error("Exception : {}".format(e))

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
