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
# from _ast import expr
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pyart
import gzip
import shutil

import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# from pyproj import Proj, Transformer
import re
# import xarray as xr
# from sklearn.neighbors import BallTree
import matplotlib.cm as cm
from multiprocessing import Pool
import multiprocessing as mp
from retrying import retry
# from pymap3d import enu2geodetic
import requests
from datetime import datetime
import time
import pandas as pd
from tqdm import tqdm
import time
import pytz
import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor

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

kst = pytz.timezone("Asia/Seoul")

# plt.rc('font', family='Malgun Gothic')
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
        , 'sysCfg': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.json')
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

        # 글꼴 설정
        # fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        # fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        # plt.rcParams['font.family'] = fontName

    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        globalVar[key] = val

    # 전역 변수
    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)

        globalVar[key] = val.replace('\\', '/')

        log.info(f"[CHECK] {key} : {val}")

    return globalVar


# def JPL_qpe(kdp, drf, ref):
#
#     try:
#         # 데이터 크기 설정
#         nx, ny = ref.shape
#         RZh = np.zeros((nx, ny))
#         RKdp = np.zeros((nx, ny))
#         ZdrF1 = np.zeros((nx, ny))
#         ZdrF2 = np.zeros((nx, ny))
#         RJPL = np.zeros((nx, ny))
#         Rcas = np.zeros((nx, ny))
#
#         # 단일편파 및 이중편파 레이더 데이터 변환
#         # zh: mm^6 m^-3, linear scale value
#         # zh = np.power(10.0, ref / 10.0)
#         zh = 10.0 ** (ref / 10.0)
#
#         # zdr: mm^6 m^-3, linear scale value
#         # zdr = np.power(10.0, drf / 10.0)
#         zdr = 10.0 ** (drf / 10.0)
#
#         # R(Zh) 계산
#         # RZh = 1.70 * 10 ** (-2) * zh ** 0.714
#         RZh = 1.70e-2 * zh ** 0.714
#
#         # R(Kdp) 계산 (Ryzhkov et al., 2005)
#         RKdp = 44.0 * (np.abs(kdp) ** 0.822) * np.sign(kdp)
#
#         # Zdr 보정 인자 계산
#         ZdrF1 = 0.4 + 5.0 * np.abs(zdr - 1.0) ** 1.3
#         ZdrF2 = 0.4 + 3.5 * np.abs(zdr - 1.0) ** 1.7
#
#         # CASE I: R(Zh,Zdr) [mm/h]
#         ix = (RZh > 0) & (RZh < 6)
#         RJPL[ix] = RZh[ix] / ZdrF1[ix]
#         Rcas[ix] = 1
#
#         # CASE II: R(Kdp,Zdr)
#         ix = (RZh > 6) & (RZh < 50)
#         RJPL[ix] = RKdp[ix] / ZdrF2[ix]
#         Rcas[ix] = 2
#
#         # CASE III: R(Kdp)
#         ix = RZh > 50  # CASE III 조건
#         RJPL[ix] = RKdp[ix]
#         Rcas[ix] = 3
#
#         idx = ((RJPL < 0) | np.isnan(ref) | (ref < 10) | (RJPL > 150) | (drf < -3) | (drf > 5))
#         Rcas[idx] = 0
#         RJPL[idx] = np.nan
#
#         return RJPL, Rcas
#
#     except Exception as e:
#         log.error(f"Exception : {str(e)}")
#         return None, None
#
# def CSU_qpe(kdp, drf, ref):
#
#     try:
#         # 데이터 크기 설정
#         nx, ny = ref.shape
#         RCSU = np.zeros((nx, ny))
#         Rcas = np.zeros((nx, ny))
#
#         # 단일편파 및 이중편파 레이더 데이터 변환
#         # zh: mm^6 m^-3, linear unit
#         # zh = np.power(10.0, ref / 10.0)
#         zh = 10.0 ** (ref / 10.0)
#
#         # zdr: dB, log unit
#         zdr = drf
#
#         # CASE I: R(Kdp, Zdr)
#         ix = (kdp >= 0.3) & (ref >= 38.0) & (zdr >= 0.5)
#         RCSU[ix] = 90.8 * (kdp[ix] ** 0.93) * (10 ** (-0.169 * zdr[ix]))
#         Rcas[ix] = 1
#
#         # CASE II: R(Kdp)
#         ix = (kdp >= 0.3) & (ref >= 38.0) & (zdr < 0.5)
#         RCSU[ix] = 40.5 * (kdp[ix] ** 0.85)
#         Rcas[ix] = 2
#
#         # CASE III: R(Zh, Zdr)
#         ix = ((kdp < 0.3) | (ref < 38.0)) & (zdr >= 0.5)
#         # RCSU[ix] = 6.7 * 10 ** (-3) * (zh[ix] ** 0.927) * (10 ** (-0.343 * zdr[ix]))
#         RCSU[ix] = 6.7e-3 * (zh[ix] ** 0.927) * (10 ** (-0.343 * zdr[ix]))
#         Rcas[ix] = 3
#
#         # CASE IV: R(Zh)
#         ix = ((kdp < 0.3) | (ref < 38.0)) & (zdr < 0.5)
#         RCSU[ix] = 0.017 * (zh[ix] ** 0.7143)
#         Rcas[ix] = 4
#
#         idx = ((RCSU < 0) | np.isnan(ref) | (ref < 10) | (RCSU > 150) | (drf < -3) | (drf > 5))
#         Rcas[idx] = 0
#         RCSU[idx] = np.nan
#
#         return RCSU, Rcas
#
#     except Exception as e:
#         log.error(f"Exception : {str(e)}")
#         return None, None
#
#
# def calRain_SN(rVarf, rVark, rVard, Rtyp, dt, aZh):
#
#     try:
#         # 초기화
#         appzdrOffset = 'no'
#         zdrOffset = 0
#
#         # 1. R(Zh): 단일편파 강우량 계산
#         # zh = np.power(10.0, rVarf / 10.0)
#         zh = 10.0 ** (rVarf / 10.0)
#         # RintZH = 1.70 * 10 ** (-2) * (zh + aZh) ** 0.714
#         RintZH = 1.70e-2 * (zh + aZh) ** 0.714
#         # RintZH[rVarf == -327.6800] = np.nan
#         # RintZH[RintZH <= 0] = np.nan
#         RintZH = np.where(rVarf == -327.6800, np.nan, RintZH)
#         RintZH = np.where(RintZH <= 0, np.nan, RintZH)
#         Rcas = 1
#
#         # 2. R(Kdp): Ryzhkov et al., 2005의 Kdp 기반 강우량 계산
#         RintKD = 44.0 * (np.abs(rVark) ** 0.822) * np.sign(rVark)
#         # RintKD[rVark == -327.6800] = np.nan
#         # RintKD[RintKD <= 0] = np.nan
#         RintKD = np.where(rVark == -327.6800, np.nan, RintKD)
#         RintKD = np.where(RintKD <= 0, np.nan, RintKD)
#         Rcas = 1
#
#         # 3. R(Zh, Zdr): Bringi and Chandraseker, 2001의 Zh, Zdr 기반 강우량 계산
#         # Zdr 계산
#         if appzdrOffset == 'yes':
#             # zdr = np.power(10.0, (rVard + zdrOffset) / 10.0)
#             zdr = 10.0 ** ((rVard + zdrOffset) / 10.0)
#         else:
#             # zdr = np.power(10.0, rVard / 10.0)
#             zdr = 10.0 ** (rVard / 10.0)
#
#         # S-band 기준
#         RintZD = 0.0067 * (zh ** 0.927) * (zdr ** -3.43)
#         # RintZD[(rVarf == -327.6800) | (rVard == -327.6800)] = np.nan
#         # RintZD[RintZD <= 0] = np.nan
#         RintZD = np.where((rVarf == -327.6800) | (rVard == -327.6800), np.nan, RintZD)
#         RintZD = np.where(RintZD <= 0, np.nan, RintZD)
#         Rcas = 1
#
#         # 4. JPL 강우강도 계산
#         if appzdrOffset == 'yes':
#             RintJP, Rcas = JPL_qpe(rVark, rVard + zdrOffset, rVarf)
#         else:
#             RintJP, Rcas = JPL_qpe(rVark, rVard, rVarf)
#
#         # RintJP[(rVark == -327.6800) | (rVard == -327.6800) | (rVarf == -327.6800)] = np.nan
#         # RintJP[RintJP <= 0] = np.nan
#         RintJP = np.where((rVark == -327.6800) | (rVard == -327.6800) | (rVarf == -327.6800), np.nan, RintJP)
#         RintJP = np.where(RintJP <= 0, np.nan, RintJP)
#
#         # 5. CSU 강우강도 계산
#         if appzdrOffset == 'yes':
#             RintCS, Rcas = CSU_qpe(rVark, rVard + zdrOffset, rVarf)
#         else:
#             RintCS, Rcas = CSU_qpe(rVark, rVard, rVarf)
#
#         # RintCS[(rVark == -327.6800) | (rVard == -327.6800) | (rVarf == -327.6800)] = np.nan
#         # RintCS[RintCS <= 0] = np.nan
#         RintCS = np.where((rVark == -327.6800) | (rVard == -327.6800) | (rVarf == -327.6800), np.nan, RintCS)
#         RintCS = np.where(RintCS <= 0, np.nan, RintCS)
#
#         # Rtyp에 따른 강우량 계산
#         if Rtyp == 'int':
#             # Rcal에 단위 시간당 강우강도 (mm/h)
#             # Rcal = [RintZH, RintKD, RintZD, RintJP, RintCS]
#             # Rcal = [RintZH, RintKD, RintZD, RintJP, RintCS]
#             Rcal = {
#                 'RintZH': RintZH,
#                 'RintKD': RintKD,
#                 'RintZD': RintZD,
#                 'RintJP': RintJP,
#                 'RintCS': RintCS
#             }
#
#         elif Rtyp == 'ran':
#             # Rcal에 누적 강우강도 (mm)
#             factor = dt / 3600.0
#             # Rcal = [RintZH * factor, RintKD * factor, RintZD * factor, RintJP * factor, RintCS * factor]
#             Rcal = {
#                 'RintZH': RintZH * factor,
#                 'RintKD': RintKD * factor,
#                 'RintZD': RintZD * factor,
#                 'RintJP': RintJP * factor,
#                 'RintCS': RintCS * factor
#             }
#
#         else:
#             Rcal = None
#
#         return Rcal, Rcas
#
#     except Exception as e:
#         log.error(f"Exception : {str(e)}")
#         return None, None
#
# def makeProcVis(modelInfo, code, dtDateInfo, data):
#
#     try:
#         # plot sigmet data
#         display = pyart.graph.RadarDisplay(data)
#         fig = plt.figure(figsize=(35, 8))
#
#         # ----------------------
#         # nEL=0 # GNG 0.2
#         # nEL=3 # GDK 0.8
#         # nEL=1 # GSN 5.2
#         # ----------------
#         nEL = 0  # FCO SBS
#         # ----------------------
#         plotList = [
#             {'field': 'reflectivity', 'vmin': -5, 'vmax': 40}
#             , {'field': 'corrected_differential_reflectivity', 'vmin': -2, 'vmax': 5}
#             , {'field': 'cross_correlation_ratio', 'vmin': 0.5, 'vmax': 1.0}
#         ]
#
#         for i, plotInfo in enumerate(plotList):
#             ax = fig.add_subplot(1, 3, i + 1)
#             display.plot(plotInfo['field'], nEL, vmin=plotInfo['vmin'], vmax=plotInfo['vmax'])
#             # display.plot_range_rings([50, 100, 150, 200, 250]) # KMA 설정
#             display.plot_range_rings([25, 50, 75, 100, 125])  # FCO 설정
#             display.plot_cross_hair(5.0)
#
#         saveImgPattern = '{}/{}'.format(modelInfo['figPath'], modelInfo['figName'])
#         saveImg = dtDateInfo.strftime(saveImgPattern).format(code)
#         os.makedirs(os.path.dirname(saveImg), exist_ok=True)
#         # plt.savefig(saveImg, dpi=600, bbox_inches='tight')
#         plt.savefig(saveImg, dpi=100, bbox_inches='tight')
#         plt.close()
#         log.info(f"[CHECK] saveImg : {saveImg}")
#
#     except Exception as e:
#         log.error(f"Exception : {str(e)}")
#
# def matchStnRadar(sysOpt, modelInfo, code, dtDateList):
#
#     try:
#         # ==========================================================================================================
#         # 융합 ASOS/AWS 지상 관측소을 기준으로 최근접 레이더 가공파일 화소 찾기 (posRow, posCol, posLat, posLon, posDistKm)
#         # ==========================================================================================================
#         # 매 5분 순간마다 가공파일 검색/병합
#         procFilePattern = '{}/{}'.format(modelInfo['procPath'], modelInfo['procName'])
#         # dtHourList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invHour'])
#
#         searchList = []
#         for dtDateInfo in dtDateList:
#             procFile = dtDateInfo.strftime(procFilePattern).format(code)
#             fileList = sorted(glob.glob(procFile))
#             if fileList is None or len(fileList) < 1: continue
#             searchList.append(fileList[0])
#             break
#
#         if searchList is None or len(searchList) < 1:
#             log.error(f"[ERROR] procFilePattern : {procFilePattern} / 가공파일을 확인해주세요.")
#             return
#
#         # 레이더 가공 파일 일부
#         fileInfo = searchList[0]
#         cfgData = xr.open_dataset(fileInfo)
#         cfgDataL1 = cfgData.to_dataframe().reset_index(drop=False)
#
#         # ASOS/AWS 융합 지상관측소
#         inpFilePattern = '{}/{}'.format(sysOpt['stnInfo']['filePath'], sysOpt['stnInfo']['fileName'])
#         fileList = sorted(glob.glob(inpFilePattern))
#         if fileList is None or len(fileList) < 1:
#             log.error(f"[ERROR] inpFilePattern : {inpFilePattern} / 융합 지상관측소를 확인해주세요.")
#             return
#
#         fileInfo = fileList[0]
#         allStnData = pd.read_csv(fileInfo)
#         allStnDataL1 = allStnData[['STN', 'STN_KO', 'LON', 'LAT']]
#
#         # 2024.10.24 수정
#         # allStnDataL2 = allStnDataL1[allStnDataL1['STN'].isin(sysOpt['stnInfo']['list'])]
#         allStnDataL2 = allStnDataL1
#
#         # 융합 ASOS/AWS 지상 관측소을 기준으로 최근접 레이더 가공파일 화소 찾기 (posRow, posCol, posLat, posLon, posDistKm)
#         #      STN STN_KO        LON       LAT  ...  posCol      posLat     posLon  posDistKm
#         # 0     90     속초  128.56473  38.25085  ...   456.0  128.565921  38.251865   0.024091
#         # 10   104    북강릉  128.85535  37.80456  ...   482.0  128.850344  37.805816   0.072428
#         # 11   105     강릉  128.89099  37.75147  ...   486.0  128.894198  37.750990   0.045061
#         # 12   106     동해  129.12433  37.50709  ...   507.0  129.124659  37.503421   0.064192
#         # 289  520    설악동  128.51818  38.16705  ...   452.0  128.518319  38.171507   0.077807
#         # 292  523    주문진  128.82139  37.89848  ...   479.0  128.818774  37.896447   0.050570
#         # 424  661     현내  128.40191  38.54251  ...   441.0  128.401035  38.542505   0.011947
#         # 432  670     양양  128.62954  38.08874  ...   462.0  128.630338  38.088726   0.010963
#         # 433  671     청호  128.59360  38.19091  ...   459.0  128.598611  38.188309   0.082373
#         baTree = BallTree(np.deg2rad(cfgDataL1[['lat', 'lon']].values), metric='haversine')
#         for i, posInfo in allStnDataL2.iterrows():
#             if (pd.isna(posInfo['LAT']) or pd.isna(posInfo['LON'])): continue
#
#             closest = baTree.query(np.deg2rad(np.c_[posInfo['LAT'], posInfo['LON']]), k=1)
#             cloDist = closest[0][0][0] * 1000.0
#             cloIdx = closest[1][0][0]
#             cfgInfo = cfgDataL1.loc[cloIdx]
#
#             allStnDataL2.loc[i, 'posRow'] = cfgInfo['row']
#             allStnDataL2.loc[i, 'posCol'] = cfgInfo['col']
#             allStnDataL2.loc[i, 'posLat'] = cfgInfo['lon']
#             allStnDataL2.loc[i, 'posLon'] = cfgInfo['lat']
#             allStnDataL2.loc[i, 'posDistKm'] = cloDist
#
#         # log.info(f"[CHECK] allStnDataL2 : {allStnDataL2}")
#
#         saveFilePattern = '{}/{}'.format(sysOpt['stnInfo']['matchPath'], sysOpt['stnInfo']['matchName'])
#         saveFile = dtDateInfo.strftime(saveFilePattern).format(code)
#         os.makedirs(os.path.dirname(saveFile), exist_ok=True)
#         allStnDataL2.to_csv(saveFile, index=False)
#         log.info(f"[CHECK] saveFile : {saveFile}")
#
#     except Exception as e:
#         log.error(f'Exception : {str(e)}')
#
# @retry(stop_max_attempt_number=1)
# def radarProc(modelInfo, code, dtDateInfo):
#     try:
#         procInfo = mp.current_process()
#
#         # ==========================================================================================================
#         # KMA_GNG_Kang4_수정용2.py
#         # ==========================================================================================================
#         saveFilePattern = '{}/{}'.format(modelInfo['savePath'], modelInfo['saveName'])
#         saveFile = dtDateInfo.strftime(saveFilePattern).format(code)
#         # if os.path.exists(saveFile): return
#
#         inpFilePattern = '{}/{}'.format(modelInfo['filePath'], modelInfo['fileName'])
#         inpFile = dtDateInfo.strftime(inpFilePattern).format(code)
#         fileList = sorted(glob.glob(inpFile))
#
#         if fileList is None or len(fileList) < 1:
#             log.error(f"[ERROR] inpFile : {inpFile} / 입력파일을 확인해주세요.")
#             return
#
#         fileInfo = fileList[0]
#         log.info(f'[CHECK] fileInfo : {fileInfo}')
#
#         fileName = os.path.basename(fileInfo)
#         fileNameNotExt = fileName.split(".")[0]
#
#         # 자료 읽기
#         data = pyart.io.read(fileInfo)
#
#         rnam = data.metadata['instrument_name']
#         rlat = data.latitude['data']
#         rlon = data.longitude['data']
#         ralt = data.altitude['data']
#         styp = data.scan_type
#         fbwh = data.instrument_parameters['radar_beam_width_h']['data']
#         fprt = data.instrument_parameters['prt']['data']
#         fvel = data.instrument_parameters['nyquist_velocity']['data']
#         fpul = data.instrument_parameters['pulse_width']['data']
#         ffrq = data.instrument_parameters['frequency']['data']
#         nray = data.nrays
#         ngat = data.ngates
#         nswp = data.nsweeps
#         fang = data.fixed_angle['data']
#         fazm = data.azimuth['data']
#         frng = data.range['data']
#         felv = data.elevation['data']
#         fscn = data.scan_rate['data']
#         fswp = data.sweep_number['data']
#         fsws = data.sweep_start_ray_index['data']
#         fswe = data.sweep_end_ray_index['data']
#         ftme = data.time['data']
#         fdat_ref = data.fields['reflectivity']['data']
#         fdat_zdr = data.fields['corrected_differential_reflectivity']['data']
#         fdat_pdp = data.fields['differential_phase']['data']
#         fdat_kdp = data.fields['specific_differential_phase']['data']
#         fdat_vel = data.fields['velocity']['data']
#         fdat_phv = data.fields['cross_correlation_ratio']['data']
#         fdat_spw = data.fields['spectrum_width']['data']
#
#         # 자료 가공
#         dataL1 = {
#             'str_nam': [rnam]
#             , 'arr_lat_lon_alt_bwh': [rlat, rlon, ralt, fbwh]
#             , 'str_typ': [styp]
#             , 'arr_prt_prm_vel': [fprt, fvel]
#             , 'num_ray_gat_swp': [nray, ngat, nswp]
#             , 'fix_ang': fang
#             , 'arr_azm_rng_elv': [fazm, frng, felv]
#             , 'arr_etc': [fpul, ffrq, fscn, fswp, fsws, fswe, ftme]
#             , 'arr_ref': np.array(fdat_ref)
#             # , 'arr_crf': arr_crf
#             , 'arr_zdr': np.array(fdat_zdr)
#             , 'arr_pdp': np.array(fdat_pdp)
#             , 'arr_kdp': np.array(fdat_kdp)
#             , 'arr_vel': np.array(fdat_vel)
#             , 'arr_phv': np.array(fdat_phv)
#             # , 'arr_ecf': arr_ecf
#             # , 'arr_coh': arr_coh
#             , 'arr_spw': np.array(fdat_spw)
#         }
#
#         if modelInfo['isProcVis']:
#             makeProcVis(modelInfo, code, dtDateInfo, data)
#
#         # ==========================================================================================================
#         # snowC_GNG_Kang_20240923_KMA_GNG.m
#         # ==========================================================================================================
#         # 강수 유형
#         # int(mm/h)/ran(mm)
#         # Rtyp = 'int'
#         Rtyp = modelInfo['rainType']
#
#         # 강수 알고리즘 인덱스
#         # Rcal{RintZH;RintKD;RintZD;RintJP;RintCS}
#         # Ralg = 3
#         # Ralg = 'RintZD'
#         Ralg = modelInfo['rainAlg']
#
#         # 시간 간격 [초]
#         # 5분 단위
#         # [80s+70s=>2.5min] low sng 300km[-0.3 0.1 0.60]=80s, high dul 150km[1.4 2.7 4.8]=70s
#         # dt = 5.0 * 60
#         dt = modelInfo['rainDt']
#
#         # 시작 Elevation Angle
#         # srtEA = 3  # 시작 각도
#         srtEA = modelInfo['rainSrtEA']
#
#         # 변수 초기화
#         # ZcaloA = np.zeros((601, 601))
#         # FcaloA = np.zeros((601, 601))
#         # RcaloA = np.zeros((601, 601))
#         # sFcalA = np.zeros((nflst, 1))
#         # sRcalA = np.zeros((nflst, 1))
#         # xFcalA = np.zeros((nflst, 1))
#         # xRcalA = np.zeros((nflst, 1))
#
#         # 9 지점에 대해 데이터 저장
#         # aws_data = np.zeros((nflst, 9, 3))
#
#         # 방위각
#         azm_r = dataL1['arr_azm_rng_elv'][0].T.flatten()
#
#         # 거리
#         rng_r = dataL1['arr_azm_rng_elv'][1].T.flatten()
#
#         # 고도각
#         elv_r = dataL1['arr_azm_rng_elv'][2].T.flatten()
#
#         # 각도 정보
#         Tang = dataL1['fix_ang'].flatten()
#         #         # Tang[Tang > 180] -= 360
#         #         # Tang[Tang > 180] = Tang[Tang > 180] - 360
#         Tang = np.where(Tang > 180, Tang - 360, Tang)
#
#         # 고도각에 따른 인덱스
#         # pattern = r'D:/Data190/|D:/Data191/|D:/2022/X0810/|' + re.escape(datDRA)
#         matchPattern = r'Data190|Data191|2022/X0810|RDR_.*_FQC'
#         if re.search(matchPattern, fileInfo, re.IGNORECASE):
#             didxs = dataL1['arr_etc'][4].flatten().astype(int)
#             didxe = dataL1['arr_etc'][5].flatten().astype(int)
#         else:
#             didxs = dataL1['arr_etc'][2].flatten().astype(int)
#             didxe = dataL1['arr_etc'][3].flatten().astype(int)
#
#         # log.info(f"[CHECK] didxs : {didxs}")
#         # log.info(f"[CHECK] didxe : {didxe}")
#
#         # 인덱스 값 변경 (0-based indexing 보정)
#         # didxs = didxs + 1  # set '0' to '1'
#         # didxe = didxe + 1
#
#         # didxs와 didxe를 합쳐서 배열 생성
#         # didX2 = np.column_stack((didxs, didxe)).ravel()
#         # log.info(f"[CHECK] didX2 : {didX2}")
#
#         # didxs = didxs + 1
#         # didxe = didxe + 1
#
#         # didX2 = np.vstack((didxs, didxe)).ravel()
#         # didX2 = np.column_stack((didxs, didxe)).ravel()
#
#         # elv_r 배열에서 인덱스 값 추출 (인덱스는 0-based이므로 조정 필요)
#         Fang = elv_r[didxs]
#         # log.info(f"[CHECK] Fang : {Fang}")
#
#         # dual para
#         if dataL1['arr_prt_prm_vel'][0][0] == dataL1['arr_prt_prm_vel'][0][-1]:
#             Tprf = 'sing'
#         else:
#             Tprf = 'dual'
#         # log.info(f"[CHECK] Tprf : {Tprf}")
#
#         # log.info(f"[CHECK] dataL1['arr_prt_prm_vel'] : {len(dataL1['arr_prt_prm_vel'])} : {dataL1['arr_prt_prm_vel']}")
#         # log.info(f"[CHECK] dataL1['arr_prt_prm_vel'][0] : {dataL1['arr_prt_prm_vel'][0].shape} : {dataL1['arr_prt_prm_vel'][0]}")
#         # log.info(f"[CHECK] dataL1['arr_prt_prm_vel'][0][0] : {dataL1['arr_prt_prm_vel'][0][0]}")
#         # log.info(f"[CHECK] dataL1['arr_prt_prm_vel'][0][-1] : {dataL1['arr_prt_prm_vel'][0][-1]}")
#
#         # log.info(f"[CHECK] dataL1['arr_ref'] : {dataL1['arr_ref'].shape} : {dataL1['arr_ref']}")
#
#         bw = dataL1['arr_lat_lon_alt_bwh'][3]
#         # log.info(f"[CHECK] bw : {bw}")
#
#         # Elev. ang
#         # Arng = np.arange(didxs[srtEA - 1], didxe[srtEA - 1] + 1)
#         Arng = np.arange(didxs[srtEA - 2], didxe[srtEA - 2] + 1)
#         # log.info(f"[CHECK] Arng : {Arng}")
#
#         # Var. info
#         rVar_rf = dataL1['arr_ref'].T
#         rVar_rk = dataL1['arr_kdp'].T
#         rVar_rd = dataL1['arr_zdr'].T
#         rVar_rc = dataL1['arr_phv'].T
#         rVar_rp = dataL1['arr_pdp'].T
#         rVar_rv = dataL1['arr_vel'].T
#
#         # rVar_rf[rVar_rf < 0] = 0
#         rVar_rf = np.where(rVar_rf < 0, 0, rVar_rf)
#         rVarf = rVar_rf[:, Arng]
#         rVark = rVar_rk[:, Arng]
#         rVard = rVar_rd[:, Arng]
#         rVarc = rVar_rc[:, Arng]
#         rVarp = rVar_rp[:, Arng]
#         rVarv = rVar_rv[:, Arng]
#
#         # cal. rain
#         # rainfall(mm)
#         [Rcalo, Rcas] = calRain_SN(rVarf, rVark, rVard, Rtyp, dt, 10)
#         if Rcalo is None: return
#
#         # grid
#         gw = 1
#
#         azm = azm_r[Arng]
#         rng = rng_r
#         elv = elv_r[Arng]
#
#         # 1196 x 1080 -> 960 x 360
#         # xrEle = rng[:, None] * (np.sin(np.deg2rad(azm.T)) * np.cos(np.deg2rad(elv.T))) / 1000
#         # 1196 x 1080 -> 960 x 360
#         # yrEle = rng[:, None] * (np.cos(np.deg2rad(azm.T)) * np.cos(np.deg2rad(elv.T))) / 1000
#
#         azm_rad = np.deg2rad(azm)
#         elv_rad = np.deg2rad(elv)
#
#         # 원본 불규칙 격자의 레이더 위경도 좌표 (xrEle/yrEle, 960 x 360 배열)
#         xrEle = (rng.reshape(-1, 1) * np.sin(azm_rad) * np.cos(elv_rad)) / 1000
#         yrEle = (rng.reshape(-1, 1) * np.cos(azm_rad) * np.cos(elv_rad)) / 1000
#
#         dxr = xrEle.flatten()
#         dyr = yrEle.flatten()
#
#         # 가공 규칙 격자의 레이더 인덱스 좌표 (xi/yi, 601 x 601 배열)
#         # 즉 중심 기준점 (0,0)을 기준으로 반경 300 km (동/서/남/북쪽 300 km)을 생성
#         xi, yi = np.meshgrid(np.arange(-300, 301, gw), np.arange(-300, 301, gw))
#
#         # 가공 규칙 격자를 기준으로 선형 내삽 수행
#         # zh.linear unit in mm6 m-3
#         # zh = np.power(10.0, rVarf / 10.0)
#         zh = 10.0 ** (rVarf / 10.0)
#         zhh = griddata((dxr, dyr), zh.flatten(), (xi, yi), method='linear')
#         # zhh = griddata(np.column_stack((dxr, dyr)), zh.flatten(), (xi, yi), method='linear')
#
#         # refl.
#         rVarf[np.isnan(rVarf)] = 0
#         ziR = griddata((dxr, dyr), rVarf.flatten(), (xi, yi), method='linear')
#         # ziR = griddata(np.column_stack((dxr, dyr)), rVarf.flatten(), (xi, yi), method='linear')
#
#         # rain
#         # dzr = Rcalo[Ralg]
#         # dzr = Rcalo[Ralg - 1].astype(np.float64)
#         dzr = Rcalo[Ralg].astype(np.float64)
#         dzr[np.isnan(dzr)] = 0
#         Rcal = griddata((dxr, dyr), dzr.flatten(), (xi, yi), method='linear')
#         # Rcal = griddata(np.column_stack((dxr, dyr)), dzr.flatten(),(xi, yi), method='linear')
#
#         # 가공 규칙 격자 (xi, yi) 및 중심 기준점 (lat0, lon0)를 이용한 위경도 변환
#         # - 기준점 (127.43, 38.11)에서 가공 규칙 격자 (0, 0) -> 위경도 (124.13, 35.37)
#         # - 기준점 (127.43, 38.11)에서 가공 규칙 격자 (300, 300) -> 위경도 (127.43, 38.11)
#         # - 기준점 (127.43, 38.11)에서 가공 규칙 격자 (600, 600) -> 위경도 (130.98, 40.76)
#         # xy -> lonlat
#         lat0 = dataL1['arr_lat_lon_alt_bwh'][0][0]
#         lon0 = dataL1['arr_lat_lon_alt_bwh'][1][0]
#         elv0 = dataL1['arr_lat_lon_alt_bwh'][2][0]
#
#         ylatg, xlong, h0 = enu2geodetic(xi * 1000, yi * 1000, np.zeros_like(xi), lat0, lon0, 0, deg=True)
#
#         # projEnu = Proj(proj='tmerc', lat_0=lat0, lon_0=lon0, ellps='WGS84', units='km')
#         # projWgs84 = Proj(proj='latlong', datum='WGS84')
#         # transformer = Transformer.from_proj(projEnu, projWgs84)
#         # xlong, ylatg = transformer.transform(xi, yi)
#         # h0 = np.zeros_like(xi)
#
#         # xlong[0][0]
#         # ylatg[0][0]
#         # xlong[300][300]
#         # ylatg[300][300]
#         # xlong[600][600]
#         # ylatg[600][600]
#
#
#         # 누적 계산 반사도 팩터
#         # ZcaloA = ZcaloA + zhh
#
#         # 누적 계산 반사도
#         # FcaloA = FcaloA + ziR
#
#         # 누적 계산 강우강도, mm/hr
#         # RcaloA = RcaloA + Rcal
#
#         # 1time당 sumFcalA 반사도
#         # sFcalA[j] = np.nansum(ziR)
#
#         # 1time당 sumFcalA 강우강도
#         # sRcalA[j] = np.nansum(Rcal)
#
#         # 1time당 maxFcalA
#         # xFcalA[j] = np.nanmax(ziR)
#
#         # 1time당 sumFcalA
#         # xRcalA[j] = np.nanmax(Rcal)
#
#         # NetCDF 생산
#         lon2D = xlong
#         lat2D = ylatg
#
#         xdim = lon2D.shape[0]
#         ydim = lon2D.shape[1]
#
#         # zhh 반사도 팩터
#         # ziR 반사도
#         # Rcal 강우강도 mm/hr
#         dataL2 = xr.Dataset(
#             {
#                 'zhh': (('time', 'row', 'col'), (zhh).reshape(1, xdim, ydim))
#                 , 'ziR': (('time', 'row', 'col'), (ziR).reshape(1, xdim, ydim))
#                 , 'Rcal': (('time', 'row', 'col'), (Rcal).reshape(1, xdim, ydim))
#             }
#             , coords={
#                 'row': np.arange(xdim)
#                 , 'col': np.arange(ydim)
#                 , 'lon': (('row', 'col'), lon2D)
#                 , 'lat': (('row', 'col'), lat2D)
#                 , 'time': pd.date_range(dtDateInfo, periods=1)
#             }
#         )
#
#         # import scipy.io as sio
#         # refData = sio.loadmat(f"/DATA/INPUT/LSH0579/KMA_GNG_sel_OUT/RDR_GDK_FQC_202302100000.mat")
#         # refData = sio.loadmat(f"/HDD/DATA/INPUT/LSH0579/KMA_GNG_sel_OUT/_CMU/dat.mat")
#
#         # NetCDF 저장
#         os.makedirs(os.path.dirname(saveFile), exist_ok=True)
#         dataL2.to_netcdf(saveFile)
#         log.info(f"[CHECK] saveFile : {saveFile}")
#
#         log.info(f'[END] radarProc : {dtDateInfo} / pid : {procInfo.pid}')
#
#     except Exception as e:
#         log.error(f'Exception : {str(e)}')
#         raise e
#
# @retry(stop_max_attempt_number=1)
# def radarValid(sysOpt, modelInfo, code, dtDateList):
#     try:
#         # ==========================================================================================================
#         # 매 5분 순간마다 가공파일 검색/병합
#         # 융합 ASOS/AWS 지상 관측소 및 레이더 가공파일의 매칭 파일 읽기
#         # 매 5분 순간마다 가공파일을 이용하여 매 1시간 누적 계산
#         # 매 1시간 누적마다 지상 관측소를 기준으로 최근접/선형내삽 화소 추출 그리고 엑셀 저장
#         # 매 1시간 누적마다 반사도/강우강도 시각화
#         # ==========================================================================================================
#         # 매 5분 순간마다 가공파일 검색/병합
#         procFilePattern = '{}/{}'.format(modelInfo['procPath'], modelInfo['procName'])
#         # dtHourList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invHour'])
#
#         searchList = []
#         for dtDateInfo in dtDateList:
#             procFile = dtDateInfo.strftime(procFilePattern).format(code)
#             fileList = sorted(glob.glob(procFile))
#             if fileList is None or len(fileList) < 1: continue
#             searchList.append(fileList[0])
#
#         if searchList is None or len(searchList) < 1:
#             log.error(f"[ERROR] procFilePattern : {procFilePattern} / 가공파일을 확인해주세요.")
#             return
#
#         dataL3 = xr.open_mfdataset(searchList).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
#
#         matchFilePattern = '{}/{}'.format(sysOpt['stnInfo']['matchPath'], sysOpt['stnInfo']['matchName'])
#         matchFile = dtDateInfo.strftime(matchFilePattern).format(code)
#         fileList = sorted(glob.glob(matchFile))
#         if fileList is None or len(fileList) < 1:
#             log.error(f"[ERROR] matchFile : {matchFile} / 지상관측소-레이더 매칭파일을 확인해주세요.")
#             return
#
#         allStnDataL2 = pd.read_csv(fileList[0])
#         # log.info(f"[CHECK] allStnDataL2 : {allStnDataL2}")
#
#         # 엑셀 저장
#         # Rst.xlsx
#         # RstH.xlsx
#         # 매 5분 순간마다 가공파일을 이용하여 매 1시간 누적 계산
#         dataL4 = dataL3.resample(time='1H').sum(dim=['time'], skipna=False)
#         # dataL4 = dataL3.resample(time='1H').sum(dim=['time'], skipna=True)
#
#         # 매 1시간 누적마다 지상 관측소를 기준으로 최근접/선형내삽 화소 추출 그리고 엑셀 저장
#         posDataL3 = pd.DataFrame()
#         for i, posInfo in allStnDataL2.iterrows():
#             if (pd.isna(posInfo['posRow']) or pd.isna(posInfo['posCol'])): continue
#             log.info(f"[CHECK] posInfo : {posInfo.to_frame().T}")
#
#             # 최근접 화소 추출
#             posData = dataL4.interp({'row': posInfo['posRow'], 'col': posInfo['posCol']}, method='nearest')
#             # posData = dataL4.sel({'row': posInfo['posRow'], 'col': posInfo['posCol']})
#
#             # 선형내삽 화소 추출
#             # posData = dataL4[varInfo].interp({'row': posInfo['posRow'], 'col': posInfo['posCol']}, method='linear')
#
#             posDataL1 = pd.DataFrame({
#                 'time': posData['time'].values
#                 , 'ziR': posData['ziR'].values
#                 , 'Rcal': posData['Rcal'].values
#             })
#
#             if len(posDataL1) < 1: continue
#             posDataL2 = posDataL1.rename(columns={'ziR': f"누적반사도{posInfo['STN']}", "Rcal": f"누적강도{posInfo['STN']}"})
#
#             if len(posDataL3) == 0:
#                 posDataL3 = posDataL2
#             else:
#                 posDataL3 = pd.merge(posDataL3, posDataL2, how='left', on='time')
#
#         saveXlsxPattern = '{}/{}'.format(modelInfo['xlsxPath'], modelInfo['xlsxName'])
#         saveXlsxFile = saveXlsxPattern.format(code, dtDateList.min().strftime('%Y%m%d%H%M'), dtDateList.max().strftime('%Y%m%d%H%M'))
#         os.makedirs(os.path.dirname(saveXlsxFile), exist_ok=True)
#         posDataL3.to_excel(saveXlsxFile, index=False)
#         log.info(f"[CHECK] saveXlsxFile : {saveXlsxFile}")
#
#         # 매 1시간 누적 반사도/강우강도 시각화
#         timeList = dataL4['time'].values
#         for timeInfo in timeList:
#             selData = dataL4.sel(time=timeInfo)
#             dtDateInfo = pd.to_datetime(selData['time'].values)
#
#             # 누적 반사도
#             saveImgPattern = '{}/{}'.format(modelInfo['cumPath'], modelInfo['cumName'])
#             saveImg = dtDateInfo.strftime(saveImgPattern).format(code, 'cf')
#             mainTitle = os.path.basename(saveImg).split(".")[0]
#             os.makedirs(os.path.dirname(saveImg), exist_ok=True)
#
#             lon2D = selData['lon'].values
#             lat2D = selData['lat'].values
#             val2D = selData['ziR'].values
#
#             plt.pcolormesh(lon2D, lat2D, val2D, cmap=cm.get_cmap('jet'), vmin=500, vmax=6000)
#             plt.colorbar()
#             plt.title(mainTitle)
#             plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
#             # plt.show()
#             plt.close()
#             log.info(f"[CHECK] saveImg : {saveImg}")
#
#             # 누적 강우강도 mm/hr
#             saveImgPattern = '{}/{}'.format(modelInfo['cumPath'], modelInfo['cumName'])
#             saveImg = dtDateInfo.strftime(saveImgPattern).format(code, 'cr')
#             mainTitle = os.path.basename(saveImg).split(".")[0]
#             os.makedirs(os.path.dirname(saveImg), exist_ok=True)
#
#             lon2D = selData['lon'].values
#             lat2D = selData['lat'].values
#             val2D = selData['Rcal'].values
#
#             plt.pcolormesh(lon2D, lat2D, val2D, cmap=cm.get_cmap('jet'), vmin=50, vmax=500)
#             plt.colorbar()
#             plt.title(mainTitle)
#             # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
#             plt.savefig(saveImg, dpi=100, bbox_inches='tight', transparent=False)
#             # plt.show()
#             plt.close()
#             log.info(f"[CHECK] saveImg : {saveImg}")
#
#     except Exception as e:
#         log.error(f'Exception : {str(e)}')
#         raise e

@retry(stop_max_attempt_number=1)
def cfgProc(sysOpt):
    try:
        res = requests.get(sysOpt['cfgUrl'])
        if not (res.status_code == 200): return None

        resJson = res.json()
        if resJson is None or len(resJson) < 1: return None

        resData = pd.DataFrame(resJson)
        if resData is None or len(resData) < 1: return None

        resDataL1 = resData[resData['symbol'].str.endswith('USDT')]

        return resDataL1

    except Exception as e:
        log.error(f"Exception : {str(e)}")
        return None

@retry(stop_max_attempt_number=5)
def colctTrendVideo(sysOpt, funName):
    print(f"[{datetime.now()}] {funName} : {sysOpt}")

    try:
        print(f"[{datetime.now()}] {funName} : {sysOpt}")
        raise Exception("예외 발생")
    except Exception as e:
        log.error(f"Exception : {str(e)}")
        raise e

async def asyncSchdl(sysOpt):

    scheduler = AsyncIOScheduler()
    scheduler.add_executor(AsyncIOExecutor(), 'default')

    # 정적 스케줄 등록
    # scheduler.add_job(colctTrendVideo, 'cron', second=0, args=[sysOpt, 'colctTrendVideo'])

    # 동적 스케줄 등록
    jobList = [
        (colctTrendVideo, 'cron', {'second': '0'}, {'args': [sysOpt, 'colctTrendVideo']})
        , (colctTrendVideo, 'cron', {'second': '30'}, {'args': [sysOpt, 'colctTrendVideo2']})
        , (colctTrendVideo, 'cron', {'second': '*/15'}, {'args': [None, 'colctTrendVideo3']})
    ]
    
    for fun, trigger, triggerArgs, kwargs in jobList:
        try:
            scheduler.add_job(fun, trigger, **triggerArgs, **kwargs)
        except Exception as e:
            log.error(f"Exception : {str(e)}")

    scheduler.start()

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        asyncio.Event().close()
    finally:
        scheduler.shutdown()

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 비트코인 데이터 기반 AI 지능형 체계 구축

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
    serviceName = 'LSH0586'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info('[START] {}'.format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

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

            if (platform.system() == 'Windows'):
                pass
            else:
                pass

            # globalVar['inpPath'] = '/DATA/INPUT'
            # globalVar['outPath'] = '/DATA/OUTPUT'
            # globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작일, 종료일, 시간 간격 (연 1y, 월 1h, 일 1d, 시간 1h, 분 1t)
                'srtDate': '2024-01-10 00:00'
                # , 'endDate': '2024-01-10 01:00'
                , 'endDate': '2024-01-10 06:00'
                , 'invDate': '1t'

                # 비동기 다중 프로세스 개수
                , 'cpuCoreNum': '2'

                # 비동기 True, 동기 False
                # , 'isAsync': True
                , 'isAsync': False

                # 설정 정보
                , 'cfgUrl': 'https://api.binance.com/api/v3/ticker/price'

                # 수행 목록
                , 'modelList': ['USDT']

                # 세부 정보
                , 'USDT': {
                    # 'filePath': '/DATA/INPUT/LSH0579/uf'
                    # 'filePath': f'{contextPath}/uf'
                    # , 'fileName': 'RDR_{}_FQC_%Y%m%d%H%M.uf'

                    # 가공파일 시각화 여부
                    # , 'isProcVis': True
                    'isProcVis': False

                    # 수집 파일
                    , 'colctUrl': 'https://api.binance.com/api/v3/klines'
                    , 'colctColList': ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore']
                    , 'colctPath': '/DATA/OUTPUT/LSH0579/COLCT/%Y%m/%d/%H/{symbol}'
                    , 'colctName': '{symbol}_%Y%m%d%H%M.csv'

                    , 'savePath': '/DATA/OUTPUT/LSH0579/PROC'
                    , 'saveName': 'RDR_{}_FQC_%Y%m%d%H%M.nc'

                    # 저장 영상
                    , 'figPath': '/DATA/FIG/LSH0579'
                    , 'figName': 'RDR_{}_FQC_%Y%m%d%H%M.png'

                    # 가공 파일
                    , 'procPath': '/DATA/OUTPUT/LSH0579/PROC'
                    , 'procName': 'RDR_{}_FQC_%Y%m%d%H%M.nc'

                    # 엑셀 파일
                    , 'xlsxPath': '/DATA/OUTPUT/LSH0579'
                    , 'xlsxName': 'RDR_{}_FQC_{}-{}.xlsx'

                    # 누적 영상
                    , 'cumPath': '/DATA/FIG/LSH0579'
                    , 'cumName': 'RDR_{}_FQC-{}_%Y%m%d%H%M.png'
                }
            }

            # **********************************************************************************************************
            # 자동화
            # **********************************************************************************************************
            asyncio.run(asyncSchdl(sysOpt))

            # # **********************************************************************************************************
            # # 수동화
            # # **********************************************************************************************************
            # # 시작일/종료일 설정
            # # dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            # # dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            # dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d %H:%M')
            # dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d %H:%M')
            # dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])
            #
            # # **************************************************************************************************************
            # # 테스트
            # # **************************************************************************************************************
            # # endDate = datetime.now()
            #
            #
            #
            # cfgData = cfgProc(sysOpt)
            # if cfgData is None or len(cfgData) < 1:
            #     log.error(f"[ERROR] cfgData['cfgUrl'] : {cfgData['cfgUrl']} / 설정 정보 URL을 확인해주세요.")
            #     raise Exception("오류 발생")
            #
            # # int(kst.localize(dtDateList[0]).timestamp() * 1000)
            #
            # for modelType in sysOpt['modelList']:
            #     log.info(f'[CHECK] modelType : {modelType}')
            #
            #     modelInfo = sysOpt.get(modelType)
            #     if modelInfo is None: continue
            #
            #     # symbol = 'BTCUSDT'
            #     for symbol in cfgData['symbol'].tolist():
            #         # log.info(f'[CHECK] symbol : {symbol}')
            #
            #         # 테스트
            #         if not 'BTCUSDT' == symbol: continue
            #
            #         for dtDateInfo in dtDateList:
            #             # log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
            #
            #             colctFilePattern = '{}/{}'.format(modelInfo['colctPath'], modelInfo['colctName'])
            #             colctFile = dtDateInfo.strftime(colctFilePattern).format(symbol=symbol)
            #             if os.path.exists(colctFile): continue
            #
            #             dtUnixTimeMs = int(dtDateInfo.timestamp() * 1000)
            #             params = {
            #                 'symbol': symbol
            #                 , 'interval': '1m'
            #                 , 'limit': 1000
            #                 , 'startTime': dtUnixTimeMs
            #                 , 'endTime': dtUnixTimeMs
            #             }
            #
            #             res = requests.get(modelInfo['colctUrl'], params=params)
            #             if not (res.status_code == 200): return None
            #
            #             resJson = res.json()
            #             if resJson is None or len(resJson) < 1: return None
            #
            #             resData = pd.DataFrame(resJson)
            #             if resData is None or len(resData) < 1: return None
            #
            #             resDataL1 = resData
            #             resDataL1.columns = modelInfo['colctColList']
            #             resDataL1['Open_time'] = resDataL1.apply(lambda x: datetime.fromtimestamp(resDataL1['Open_time'] // 1000), axis=1)
            #             resDataL1 = resDataL1.drop(columns=['Close_time', 'ignore'])
            #             resDataL1['Symbol'] = symbol
            #             resDataL1.loc[:, 'Open':'tb_quote_av'] = resDataL1.loc[:, 'Open':'tb_quote_av'].astype(float)
            #             resDataL1['trades'] = resDataL1['trades'].astype(int)
            #
            #             # 파일 저장
            #             os.makedirs(os.path.dirname(colctFile), exist_ok=True)
            #             resDataL1.to_csv(colctFile, index=False)






            # **************************************************************************************************************
            # 비동기 다중 프로세스 수행
            # **************************************************************************************************************
            # 비동기 다중 프로세스 개수
            # pool = Pool(int(sysOpt['cpuCoreNum'])) if sysOpt['isAsync'] else None
            #
            # for modelType in sysOpt['modelList']:
            #     log.info(f'[CHECK] modelType : {modelType}')
            #
            #     modelInfo = sysOpt.get(modelType)
            #     if modelInfo is None: continue
            #
            #     for code in modelInfo['codeList']:
            #         log.info(f'[CHECK] code : {code}')
            #
            #         for dtDateInfo in dtDateList:
            #             if sysOpt['isAsync']:
            #                 # 비동기 자료 가공
            #                 pool.apply_async(radarProc, args=(modelInfo, code, dtDateInfo))
            #             else:
            #                 # 단일 자료 가공
            #                 radarProc(modelInfo, code, dtDateInfo)
            #         if pool:
            #             pool.close()
            #             pool.join()
            #
            #         # 지상관측소 및 레이더 간의 최근접 화소 찾기
            #         matchStnRadar(sysOpt, modelInfo, code, dtDateList)
            #
            #         # 자료 검증
            #         radarValid(sysOpt, modelInfo, code, dtDateList)

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            # if pool:
            #     pool.terminate()
            raise e
        finally:
            # if pool:
            #     pool.close()
            #     pool.join()
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
