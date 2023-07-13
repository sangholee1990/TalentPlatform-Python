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
# import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
# from pyproj import Proj
# import pymannkendall as mk

import sys
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pyart
from scipy.fft import fft, fftfreq

# import xarray as xr
# import dask.array as da
# from dask.diagnostics import ProgressBar
# from xarrayMannKendall import *
# import dask.array as da
# import dask
# from dask.distributed import Client

# from scipy.stats import kendalltau
# from plotnine import ggplot, aes, geom_boxplot
import gc
import numpy as np
import pandas as pd
from docutils.nodes import description
from scipy.signal import find_peaks

# 반복 추세 적합
from scipy.optimize import curve_fit
import os
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.seasonal import STL
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
from scipy.ndimage import generic_filter
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.stats.mstats import mquantiles
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence

from sklearn.ensemble import GradientBoostingRegressor

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


def prepcss_new(refFlt, phvFlt, appPDPelim, refThz, phvThz, pdpThrd, ntex, mtex, rVarf, rVarc, rVarp, rVarT):
    # (1) apply low value(const.) threshold
    vidx = np.zeros_like(rVarT)
    if refFlt == 'yes':
        rVarT[rVarf < refThz] = np.nan
        vidx[rVarf < refThz] = 1

    # (2) phv filtering
    if phvFlt == 'yes':
        rVarT[rVarc < phvThz] = np.nan
        vidx[rVarc < phvThz] = 1

    # (3) Tex(pdp) filtering
    # ftex 정의되지 않음
    # if appPDPelim == 'yes':
    #     Rcalp, vidx_p = ftex(appPDPelim, pdpThrd, rVarp, ntex, mtex)  # Need to define 'ftex' function in Python
    #     rVarT[vidx_p == 1] = np.nan
    #     vidx = np.logical_or(vidx, vidx_p)

    return rVarT, vidx


def getVarInfo191013(data, vnam):
    rVar = None
    rVarI = None

    if vnam in ['ref', 'Zh']:
        rVarI = 'Reflectivity [dBZ]'
        rVar = np.transpose(data['arr_ref'])
    elif vnam in ['zdr', 'ZDR']:
        rVarI = 'Differential reflectivity [dB]'
        rVar = np.transpose(data['arr_zdr'])
    elif vnam in ['pdp', 'PDP']:
        rVarI = 'Differential phase [degrees]'
        rVar = np.transpose(data['arr_pdp'])
    elif vnam in ['kdp', 'KDP']:
        rVarI = 'Specific differential phase [degrees/km]'
        rVar = np.transpose(data['arr_kdp'])
    elif vnam in ['phv', 'PHV']:
        rVarI = 'Cross correlation ratio [ratio]'
        rVar = np.transpose(data['arr_phv'])
    elif vnam in ['vel', 'VEL']:
        rVarI = 'Velocity [meters/second]'
        rVar = np.transpose(data['arr_vel'])
    elif vnam in ['ecf', 'ECF']:
        rVarI = 'Radar echo classification'
        rVar = np.transpose(data['arr_ecf'])
    elif vnam in ['coh', 'COH']:
        rVarI = 'Normalized coherent power [ratio]'
        rVar = np.transpose(data['arr_coh'])
    elif vnam in ['spw', 'SPW']:
        rVarI = 'Spectrum width [meters/second]'
        rVar = np.transpose(data['arr_spw'])
    elif vnam in ['tpw', 'TPW']:
        rVarI = 'Total power [dBZ]'
        rVar = np.transpose(data['arr_tpw'])
    elif vnam in ['k-z', 'KDP/Zh']:
        rVarI = '10log10(Kdp[deg./km]/Zh[mm^{6}mm^{-3}])'
        rVarz = np.power(10, np.transpose(data['arr_ref']) / 10)
        rVark = np.transpose(data['arr_kdp'])
        rVark[rVark <= 0] = np.nan
        rVarz[rVarz <= 0] = np.nan
        rVar = 10 * np.log10(np.divide(rVark, rVarz, where=(rVarz != 0)))
        rVar[np.isinf(rVar)] = np.nan
        rVar[np.isnan(rVar)] = np.nan

    return rVar, rVarI


def func(x, offs, amp, f, phi):
    return offs + amp * np.sin(2 * np.pi * f * x + phi)


def sineFit(x, y, isPlot=True):
    # 샘플 테스트
    # x = np.linspace(-4, 5, 100)
    # y = 1 + 2 * (np.sin(2 * np.pi * 0.1 * x + 2) + 0.3 * np.random.normal(size=len(x)))  # Sine + noise
    # SineParams = sineFit(x, y)
    # print(SineParams)

    n = len(x)
    frq = np.fft.fftfreq(n, np.nanmean(np.diff(x)))
    Fyy = abs(np.fft.fft(y) / n)

    # guess_freq = abs(frq[np.argmax(Fyy[1:n // 2])])
    # guess_amp = np.std(y) * 2.**0.5
    # guess_offset = np.mean(y)
    # initParams = np.array([guess_offset, guess_amp, guess_freq, 0])
    initParams = [np.nanmean(y), np.nanstd(y), 1.0 / np.ptp(x), 0.0]

    # Curve fitting
    popt, pcov = curve_fit(func, x, y, p0=initParams, maxfev=10000)
    mse = np.mean((y - func(x, *popt))**2)
    SineParams = np.append(popt, mse)
    yOut = func(x, *popt)

    # Plotting
    if isPlot:
        plt.figure()
        plt.plot(x, y, 'b-', label='data')
        plt.plot(x, yOut, 'r-', label='fit: offs=%5.3f, amp=%5.3f, f=%5.3f, phi=%5.3f' % tuple(popt))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        # plt.show()

    return SineParams


def power1(x, a, b):
    return a * x ** b


def power2(x, a, b, c):
    return a * x ** b + c


def dspCycl(fwDir, vnam_b, mrVarT_btA, mrVarT_btAs, dtrnTyp, dspSel, j, rVarI_bA):

    # mrVarT_btA = None
    # mrVarT_btAs = ta
    # dtrnTyp = 'fix'
    # dspSel = 'each'
    # j =  fileIdx
    # rVarI_bA = rVarI_bA[ip]
    #
    # mrVarT_btA = mrVarT_btA
    # mrVarT_btAs = mrVarT_btAs
    # dtrnTyp = 'fix'
    # dspSel = 'total'
    # j =  None
    # rVarI_bA = rVarI_bA[ipi]

    if dspSel == 'total':
        mrVarT_btAm = np.nanmean(mrVarT_btAs, axis=0)

    elif dspSel == 'each':
        mrVarT_btAm = mrVarT_btAs

    tha = np.arange(1, 361)

    if dtrnTyp == 'fix':
        mvStep = 90
        mrVarT_btAmMV = pd.Series(mrVarT_btAm).rolling(window=mvStep).mean().values
        # mrVarT_btAmMV = np.zeros((360, )) + mrVarT_btAm
        # mrVarT_btAmMV = np.convolve(mrVarT_btAm, np.ones(mvStep) / mvStep, mode='valid')

    elif dtrnTyp == 'aut1':
        trnPol = 6
        # DTt = signal.detrend(mrVarT_btAm, type='linear', bp=trnPol)
        # mrVarT_btAmMV = mrVarT_btAm - DTt
        DTt = signal.detrend(mrVarT_btAm, type='constant')
        mrVarT_btAmMV = mrVarT_btAm - DTt
    elif dtrnTyp == 'aut2':
        SinePO = sineFit(tha, mrVarT_btAm, 0)
        mvStep = round(1 / SinePO[2])
        mrVarT_btAmMV = pd.Series(mrVarT_btAm).rolling(window=mvStep).mean().values
        # mvStep = round(1 / SinePO[2])
        # mrVarT_btAmMV = np.convolve(mrVarT_btAm, np.ones(mvStep) / mvStep, mode='valid')

    # xr.DataArray(mrVarT_btAmMV).plot()
    # # plt.show()

    mrVarT_btAmDT = mrVarT_btAm - mrVarT_btAmMV
    spa1 = np.nanmean(np.abs(mrVarT_btAmDT)) * 2
    spa2 = np.nanmean(mrVarT_btAmDT)

    # plt.plot(mrVarT_btAmDT)
    # # plt.show()

    if dspSel == 'total':
        # plot 1
        plt.figure()
        plt.plot(mrVarT_btAs)
        # plt.plot(mrVarT_btAs.T)
        plt.plot(mrVarT_btAm, 'k-', linewidth=3)
        plt.plot(mrVarT_btAmMV, 'w-', linewidth=2)
        plt.xlim([0, 360])
        plt.xticks(np.arange(0, 361, 30))
        plt.grid(True)
        plt.box(True)
        fig = plt.gcf()
        fig.set_size_inches(20 / 2.54, 20 / 2.54)  # Converting from centimeters to inches
        saveImg = f"{fwDir}{vnam_b.lower()}_dom_all.png"
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        plt.savefig(saveImg, dpi=600)
        log.info(f'[CHECK] saveImg : {saveImg}')
        # plt.show()
        plt.close()

        # plot 2
        plt.figure()
        plt.plot(mrVarT_btAm, 'k-', linewidth=3)
        plt.plot(mrVarT_btAmMV, 'b-', linewidth=2)
        plt.xlim([0, 360])
        plt.xticks(np.arange(0, 361, 30))
        plt.grid(True)
        plt.box(True)
        fig = plt.gcf()
        # fig.set_size_inches(20 / 2.54, 20 / 2.54)  # Converting from centimeters to inches

        # plt.savefig(f"{fwDir}{vnam_b.lower()}_dom_all2.png", dpi=600)
        saveImg = f"{fwDir}{vnam_b.lower()}_dom_all2.png"
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        plt.savefig(saveImg, dpi=600)
        log.info(f'[CHECK] saveImg : {saveImg}')
        # plt.show()
        plt.close()

        # plot 3
        plt.figure()
        plt.plot(mrVarT_btAmDT, 'r-', linewidth=2)
        plt.xlim([0, 360])
        plt.xticks(np.arange(0, 361, 30))
        plt.grid(True)
        plt.box(True)
        fig = plt.gcf()
        # fig.set_size_inches(20 / 2.54, 20 / 2.54)  # Converting from centimeters to inches

        # plt.savefig(f"{fwDir}{vnam_b.lower()}_dom_all3.png", dpi=600)
        saveImg = f"{fwDir}{vnam_b.lower()}_dom_all3.png"
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        plt.savefig(saveImg, dpi=600)
        log.info(f'[CHECK] saveImg : {saveImg}')
        # plt.show()
        plt.close()

    mrVarT_btAmDT[np.isnan(mrVarT_btAmDT)] = 0
    # SineP = sineFit(tha, mrVarT_btAmDT, 0)
    # SineP = sineFit(tha, mrVarT_btAmDT, isPlot=True)
    SineP = sineFit(tha, mrVarT_btAmDT, isPlot=True)

    spa2 = SineP[0]
    spa1 = SineP[1]
    spa3 = SineP[2]
    spa4 = SineP[3]
    spa5 = SineP[4]

    pdp1 = spa2 + spa1 * np.sin(2 * np.pi * spa3 * tha + spa4)
    pdp2 = pdp1 + mrVarT_btAmMV
    nos = mrVarT_btAmDT - pdp1
    pdp3 = mrVarT_btAmMV + nos

    if dspSel == 'total':
        fig, axs = plt.subplots(4, 1)
        # fig, axs = plt.subplots(4, 1, figsize=(7, 24))
        axs[0].plot(mrVarT_btAmMV + spa1, 'c-', linewidth=0.5)
        axs[0].plot(mrVarT_btAmMV - spa1, 'c-', linewidth=0.5)
        axs[0].fill_between(tha, mrVarT_btAmMV - spa1, mrVarT_btAmMV + spa1, color='m', linewidth=0.5)
        axs[0].plot(mrVarT_btAm, 'k-', linewidth=2)
        axs[0].plot(mrVarT_btAmMV, 'b-', linewidth=1)
        axs[0].set_xlim([0, 360])
        axs[0].xaxis.set_major_locator(MultipleLocator(30))
        axs[0].grid(True)
        axs[0].set_ylabel(rVarI_bA)

        axs[1].plot(np.full_like(pdp1, +spa1), 'c-', linewidth=0.5)
        axs[1].plot(np.full_like(pdp1, -spa1), 'c-', linewidth=0.5)
        axs[1].fill_between(tha, -spa1, +spa1, color='m', linewidth=0.5)
        axs[1].plot(mrVarT_btAmDT, 'k-', linewidth=1)
        axs[1].plot(pdp1, 'r-', linewidth=1)
        # text_str1 = f'Y = {spa2} + {spa1} x sin(2 x pi x {spa3} x X + {spa4}, MSE = {spa5}'
        text_str1 = f'Y = {spa2:.2f} + {spa1:.2f} x sin(2 x pi x {spa3:.2f} x X + {spa4:.2f}, MSE = {spa5:.2f}'
        text_str2 = f'Period = {round(1 / spa3)}'
        axs[1].text(5, +spa1 * 1.5, text_str1, fontsize=8)
        axs[1].text(5, -spa1 * 0.9, text_str2, fontsize=8)
        axs[1].set_xlim([0, 360])
        axs[1].xaxis.set_major_locator(MultipleLocator(30))
        axs[1].grid(True)
        axs[1].set_ylabel(f'Detrended {rVarI_bA}')

        axs[2].plot(mrVarT_btAmMV + spa1, 'c-', linewidth=0.5)
        axs[2].plot(mrVarT_btAmMV - spa1, 'c-', linewidth=0.5)
        axs[2].fill_between(tha, mrVarT_btAmMV - spa1, mrVarT_btAmMV + spa1, color='m', linewidth=0.5)
        axs[2].plot(nos, 'r-', linewidth=1)
        axs[2].plot(mrVarT_btAmMV, 'b-', linewidth=1)
        axs[2].set_xlim([0, 360])
        axs[2].xaxis.set_major_locator(MultipleLocator(30))
        axs[2].grid(True)
        axs[2].set_ylabel(f'Detrended {rVarI_bA}')

        axs[3].plot(pdp3 + spa1, 'c-', linewidth=0.5)
        axs[3].plot(pdp3 - spa1, 'c-', linewidth=0.5)
        axs[3].fill_between(tha, pdp3 - spa1, pdp3 + spa1, color='m', linewidth=0.5)
        axs[3].plot(pdp3, 'r-', linewidth=2)
        axs[3].plot(mrVarT_btAm, 'k-', linewidth=2)
        axs[3].set_xlim([0, 360])
        axs[3].xaxis.set_major_locator(MultipleLocator(30))
        axs[3].grid(True)
        axs[3].set_xlabel('Azimuth (degree)')
        axs[3].set_ylabel(rVarI_bA)

        fig.tight_layout()
        # plt.savefig(f"{fwDir}{vnam_b.lower()}_dom_all4.png", dpi=600)
        saveImg = f"{fwDir}{vnam_b.lower()}_dom_all4.png"
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        plt.savefig(saveImg, dpi=600)
        log.info(f'[CHECK] saveImg : {saveImg}')
        # plt.show()
        plt.close(fig)

    if dspSel == 'each':
        if ((vnam_b == 'zdr' and spa1 > 0.15) or (vnam_b == 'pdp' and spa1 > 2)):

            plt.figure()
            plt.plot(np.full_like(pdp1, +spa1), 'c-', linewidth=0.5)
            plt.plot(np.full_like(pdp1, -spa1), 'c-', linewidth=0.5)
            plt.fill_between(tha, -spa1, +spa1, color='m', linewidth=0.5)
            plt.plot(mrVarT_btAmDT, 'k-', linewidth=1)
            plt.plot(pdp1, 'r-', linewidth=1)
            # text_str1 = f'Y = {spa2} + {spa1} x sin(2 x pi x {spa3} x X + {spa4}, MSE = {spa5}'
            text_str1 = f'Y = {spa2:.2f} + {spa1:.2f} x sin(2 x pi x {spa3:.2f} x X + {spa4:.2f}, MSE = {spa5:.2f}'
            text_str2 = f'Period = {round(1 / spa3)}'
            plt.text(5, +spa1 * 1.5, text_str1, fontsize=8)
            plt.text(5, -spa1 * 0.9, text_str2, fontsize=8)
            plt.xlim([0, 360])
            plt.xticks(np.arange(0, 361, 30))
            plt.grid(True)
            plt.ylabel(f'Detrended {rVarI_bA}')
            plt.tight_layout()

            fwDirEc = os.path.join(fwDir, 'Ech/')
            os.makedirs(os.path.dirname(fwDirEc), exist_ok=True)
            # plt.savefig(f"{fwDirEc}{vnam_b.lower()}_dom_{j}a.png", dpi=600)
            saveImg = f"{fwDirEc}{vnam_b.lower()}_dom_{j}a.png"
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            plt.savefig(saveImg, dpi=600)
            log.info(f'[CHECK] saveImg : {saveImg}')
            # plt.show()
            plt.close()

def ccpltXY_191013_n(vardataX, vardataY, Xthd, Ythd,
                     vnam_a, vnam_b,
                     rVarI_a, rVarI_b, Tang,
                     sp, fn,
                     vidx,
                     refFlt, phvFlt, appPDPelim,
                     refThz, phvThz, pdpThz, ntex, mtex):

    vardataX[vidx == 1] = np.nan
    vardataY[vidx == 1] = np.nan

    idxXY = np.where((vardataX < Xthd) | (vardataY < Ythd))
    vardataX[idxXY] = np.nan
    vardataY[idxXY] = np.nan

    vardataO = vardataX[~np.isnan(vardataX)]
    vardataA = vardataY[~np.isnan(vardataY)]

    X = vardataO.ravel()
    Y = vardataA.ravel()

    if vnam_a == 'Zh' and vnam_b == 'KDP':
        xL, yL, dL = [-10, 70], [-4, 4], [0.5, 0.09]
    elif vnam_a == 'Zh' and vnam_b == 'ZDR':
        xL, yL, dL = [-10, 70], [-6, 8], [0.5, 0.13]
    elif vnam_a == 'ZDR' and vnam_b == 'KDP/Zh':
        xL, yL, dL = [-6, 8], [-60, 0], [0.13, 0.5]
    elif vnam_a == 'SPW' and vnam_b == 'ZDR':
        xL, yL, dL = [0, 4], [0, 2], [0.01, 0.005]
    elif vnam_a == 'SPW' and vnam_b == 'PDP':
        xL, yL, dL = [0, 4], [0, 12], [0.01, 0.03]
    else:
        xL, yL, dL = [-np.inf, np.inf], [-30, 30], [0.5, 0.5]

    H, xedges, yedges = np.histogram2d(X, Y, bins=[np.arange(xL[0], xL[1], dL[0]), np.arange(yL[0], yL[1], dL[1])], density=True)

    # 값 0의 경우 NaN 설정
    H = np.ma.masked_where(H == 0, H)

    setCmap = 'jet'
    # setCmap = 'coolwarm'
    # setCmap = 'hot'
    # setCmap = 'hot_r'

    # fig, ax = plt.subplots(figsize = (10, 6))
    fig, ax = plt.subplots(figsize=(10, 5))
    pc = ax.pcolormesh(xedges, yedges, H.T, cmap=setCmap)
    cb = fig.colorbar(pc, ax=ax)
    # # plt.show()

    eXfit = vardataO.copy()
    eYfit = vardataA.copy()

    nanIdxE = np.isnan(eXfit) | np.isnan(eYfit) | np.isinf(eXfit) | np.isinf(eYfit)
    eXfit = eXfit[~nanIdxE]
    eYfit = eYfit[~nanIdxE]

    arXdata = eXfit.flatten()
    arYdata = eYfit.flatten()

    qmed = 0.50
    xqmed = np.quantile(arXdata, qmed)
    yqmed = np.quantile(arYdata, qmed)

    arYdata = arYdata[arXdata > 0]
    arXdata = arXdata[arXdata > 0]

    qsrt = 0.05
    qend = 0.95
    qdt = 0.05
    qmedA = np.arange(qsrt, qend + qdt, qdt)
    dq = np.hstack([np.arange(0.5, qend + qdt, qdt), np.arange(qend - qdt, 0.5 - qdt, -qdt)]) / qend

    xqmedA = np.quantile(arXdata, qmedA)
    yqmedA = np.quantile(arYdata, qmedA)

    xqmedIdx = xqmedA <= 0
    xqmedA = xqmedA[~xqmedIdx]
    yqmedA = yqmedA[~xqmedIdx]

    Tbl = pd.DataFrame({'arXdata': arXdata, 'arYdata': arYdata})

    predT = np.linspace(xqmedA[0], xqmedA[-1], len(qmedA))

    # 25, 50, 75% 예측
    quartiles = []
    tauList = [0.25, 0.50, 0.75]
    for tau in tauList:  # Loop over the list of quantiles
        # Fit the model for each quantile
        gbr = GradientBoostingRegressor(loss='quantile', alpha=tau)
        gbr.fit(Tbl['arXdata'].values.reshape(-1, 1), Tbl['arYdata'])
        quartiles.append(gbr.predict(predT.reshape(-1, 1)))

    # 평균 예측
    Mdl = GradientBoostingRegressor()
    Mdl.fit(Tbl['arXdata'].values.reshape(-1, 1), Tbl['arYdata'])
    meanY = Mdl.predict(predT.reshape(-1, 1))
    meanY[np.isnan(meanY)] = 0

    # Tbl['prd'] = Mdl.predict(Tbl[['arXdata']])
    # plt.plot(Tbl['arXdata'], Tbl['arYdata'])
    # plt.plot(Tbl['arXdata'], Tbl['prd'], 'o', c='red')
    # # plt.show()

    fmed = [0, 0]
    fmed2 = [0, 0, 0]
    fmed25 = [0, 0]
    fmed75 = [0, 0]
    fmed225 = [0, 0, 0]
    fmed275 = [0, 0, 0]
    fmea = [0, 0]
    fmea2 = [0, 0, 0]

    # 중간값
    try:
        fmed, _ = curve_fit(power1, predT, quartiles[1], maxfev=10000)
    except:
        pass
    try:
        fmed2, _ = curve_fit(power2, predT, quartiles[1], maxfev=10000)
    except:
        pass

    # 25%
    try:
        fmed25, _ = curve_fit(power1, predT, quartiles[0], maxfev=10000)
    except:
        pass
    try:
        fmed225, _ = curve_fit(power2, predT, quartiles[0], maxfev=10000)
    except:
        pass

    # 75%
    try:
        fmed75, _ = curve_fit(power1, predT, quartiles[2], maxfev=10000)
    except:
        pass
    try:
        fmed275, _ = curve_fit(power2, predT, quartiles[2], maxfev=10000)
    except:
        pass

    # 평균
    try:
        fmea, _ = curve_fit(power1, predT, meanY, maxfev=10000)
    except:
        pass
    try:
        fmea2, _ = curve_fit(power2, predT, meanY, maxfev=10000)
    except:
        pass

    if vnam_a == 'Zh':
        xVal = np.arange(1.0, xL[1], 0.1)
    else:
        xVal = np.arange(0.1, xL[1], 0.1)

    # 중간값
    yValmed = power1(xVal, *fmed)
    yValmed2 = power2(xVal, *fmed2)

    # 25%
    yValmed25 = power1(xVal, *fmed25)
    yValmed225 = power2(xVal, *fmed225)

    # 75%
    yValmed75 = power1(xVal, *fmed75)
    yValmed275 = power2(xVal, *fmed275)

    # 평균
    yValmea = power1(xVal, *fmea)
    yValmea2 = power2(xVal, *fmea2)

    # get fPar and mPar
    fPar = fmed
    fPar2 = fmed2
    mPar = [xqmed, yqmed]

    # Plotting section
    fitTyp = 'poly2'
    if fitTyp == 'poly1':
        plt.plot(xVal, yValmed, '-', color='m', linewidth=4.0)
        plt.plot(xVal, yValmed25, '--', color='m', linewidth=1.0)
        plt.plot(xVal, yValmed75, '--', color='m', linewidth=1.0)
        plt.plot(xVal, yValmea, '-', color='b', linewidth=1.5)
    elif fitTyp == 'poly2':
        plt.plot(xVal, yValmed2, '-', color='m', linewidth=4.0)
        plt.plot(xVal, yValmed225, '--', color='m', linewidth=1.0)
        plt.plot(xVal, yValmed275, '--', color='m', linewidth=1.0)

    # For every predT plot median quartile
    for i in range(len(predT)):
        # plt.plot(predT[i], quartiles[i, 2], 'k+', markersize=9 * dq[i] ** 2, linewidth=1.5 * dq[i])
        plt.plot(predT[i], quartiles[1][i], 'k+', markersize=9 * dq[i] ** 2, linewidth=1.5 * dq[i])

    # median point
    plt.plot(xqmed, yqmed, 'w+', markersize=17, linewidth=3.0)
    plt.plot(xqmed, yqmed, 'r+', markersize=15, linewidth=2.0)

    # srtCidx = 135
    # cname = '../mapINF/cmap/precip2_17lev.rgb'
    # ncr = 256

    # Assuming you have a function mkRGBmap in Python which is equivalent to MATLAB version
    # RGBmap = mkRGBmap(cname, ncr)

    # cmap = RGBmap[srtCidx - 1:]  # note that Python uses 0-based indexing
    # cmap = plt.get_cmap('jet', ncr)

    # srtCidx = 0
    # cmap = ListedColormap(cmap(np.arange(srtCidx, ncr)))
    # plt.set_cmap(cmap)

    # colorbar
    cb.set_label('Probability density function estimate')

    # setting limits based on string comparison
    # ax = plt.gca()
    if vnam_a == 'Zh' and vnam_b == 'KDP':
        cb.mappable.set_clim([0.0, 0.015])
    elif vnam_a == 'Zh' and vnam_b == 'ZDR':
        cb.mappable.set_clim([0.0, 0.07])
    elif vnam_a == 'ZDR' and vnam_b == 'KDP/Zh':
        cb.mappable.set_clim([0.0, 0.07])
    elif vnam_a == 'SPW' and vnam_b == 'ZDR':
        cb.mappable.set_clim([0.0, 0.4])
    elif vnam_a == 'SPW' and vnam_b == 'PDP':
        cb.mappable.set_clim([0.0, 0.07])
    else:
        cb.mappable.set_clim([0.0, 0.07])

    if vnam_b in ['KDP', 'ZDR', 'KDP/Zh', 'PDP']:
        ax.set_xlim([xL[0], xL[1]])
        ax.set_ylim([yL[0], yL[1]])

    plt.grid(True)
    plt.xlabel(rVarI_a, fontsize=11, fontweight='normal', color='k')
    plt.ylabel(rVarI_b, fontsize=11, fontweight='normal', color='k')

    # calculating correlation and p-value
    if len(X) + len(Y) > 0:
        r, p = pearsonr(X, Y)
        log.info(f"corr: {r:.3f}, pval: {p:.3f}")
    else:
        r, p = 0, 0

    # Grid settings
    plt.grid(True)

    # Axes settings
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Title settings
    title_text = (
        f"{fn[0][0] + fn[1]} corr={r:.3f} (pval={p:.3f}) in {vnam_a} vs {vnam_b} "
        f"(Zh({refFlt}){refThz:d}, PHV({phvFlt}){phvThz:.2f}, PDP({appPDPelim}){pdpThz:d}), "
        f"elimd={np.sum(vidx) / np.prod(vidx.shape) * 100:.1f}%"
    )
    plt.title(title_text, fontsize=7)

    os.makedirs(os.path.dirname(sp), exist_ok=True)

    # X and Y labels
    plt.xlabel(rVarI_a, fontsize=11, fontweight='normal', color='k')
    plt.ylabel(rVarI_b, fontsize=11, fontweight='normal', color='k')
    plt.savefig(sp, dpi=600, bbox_inches='tight', transparent=False)
    # plt.show()
    plt.close()
    log.info(f'[CHECK] saveImg : {sp}')

    return fPar, fPar2, mPar


def readUfRadarData(fileInfo):
    log.info(f'[START] readUfRadarData')

    result = None

    try:
        data = pyart.io.read(fileInfo)

        rnam = data.metadata['instrument_name']
        rlat = data.latitude['data']
        rlon = data.longitude['data']
        ralt = data.altitude['data']

        styp = data.scan_type

        fbwh = data.instrument_parameters['radar_beam_width_h']['data']
        fprt = data.instrument_parameters['prt']['data']
        fvel = data.instrument_parameters['nyquist_velocity']['data']

        nray = data.nrays
        ngat = data.ngates
        nswp = data.nsweeps

        fang = data.fixed_angle['data']

        fazm = data.azimuth['data']
        frng = data.range['data']
        felv = data.elevation['data']

        fpul = data.instrument_parameters['pulse_width']['data']
        ffrq = data.instrument_parameters['frequency']['data']
        fscn = data.scan_rate['data']

        fswp = data.sweep_number['data']
        fsws = data.sweep_start_ray_index['data']
        fswe = data.sweep_end_ray_index['data']
        ftme = data.time['data']

        fdat_ref = data.fields['reflectivity']['data']
        fdat_crf = data.fields['corrected_reflectivity']['data']
        fdat_vel = data.fields['velocity']['data']
        fdat_spw = data.fields['spectrum_width']['data']
        fdat_zdr = data.fields['corrected_differential_reflectivity']['data']
        fdat_kdp = data.fields['specific_differential_phase']['data']
        fdat_pdp = data.fields['differential_phase']['data']
        fdat_ncp = data.fields['normalized_coherent_power']['data']
        fdat_phv = data.fields['cross_correlation_ratio']['data']
        fdat_ecf = data.fields['radar_echo_classification']['data']

        # Construct the dictionary with the extracted information
        result = {
            'str_nam': [rnam],
            'arr_lat_lon_alt_bwh': [rlat, rlon, ralt, fbwh],
            'str_typ': [styp],
            'arr_prt_prm_vel': [fprt, fvel],
            'num_ray_gat_swp': [nray, ngat, nswp],
            'fix_ang': fang,
            'arr_azm_rng_elv': [fazm, frng, felv],
            'arr_etc': [fpul, fswp, fsws, fswe, ftme, ffrq, fscn],
            'arr_ref': fdat_ref,
            'arr_crf': fdat_crf,
            'arr_vel': fdat_vel,
            'arr_spw': fdat_spw,
            'arr_zdr': fdat_zdr,
            'arr_kdp': fdat_kdp,
            'arr_pdp': fdat_pdp,
            'arr_ncp': fdat_ncp,
            'arr_phv': fdat_phv,
            'arr_ecf': fdat_ecf
        }

        return result

    except Exception as e:
        log.error(f'Exception : {e}')
        return result

    finally:
        log.info(f'[END] readUfRadarData')


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Matlab을 이용한 레이더 자료처리 및 시각화 프로그램 변환

    # <레이더 변수 읽어오기>
    # (현재) 파이선(pyart 이용)으로 uf 읽어 변수를 매트랩 mat 파일로 저장(NAK_2022.py)한 후 아래 프로그램에서 읽어 분석함
    # 샘플 uf 파일
    # Output02BSL210703234802.RAWP4XN (비슬산)
    # Output08SBS210703233502.RAW0M77 (소백산)
    # (금회 작성) 작성 프로그램에서는 매트랩 거치지 않고 uf에서 직접 변수 읽어 분석하면 됨

    # <분석 알고리즘 설명>
    # 첨부한 한글화일에 분석 방법, 절차, 결과 예시 등 나와 있음
    # 1. 첫번째 프로그램
    # 라. (차등반사도의 방위각 종속성) 특정 사상에 대하여 방위각 방향의 차등반사도 변화를 모니터링 하는 방법'
    # 참고 매트랩 프로그램 :
    # zdr_tst2_pdp_2022.m
    # dspCycl.m
    # sineFit201208.zip
    # 정리1.pptx

    # 2. 두번째 프로그램 (변수가 ZDR, PDP로 2가지 일뿐 방법은 같음)
    # 편파 매개변수의 측정 오류 추정치를 이용하여 레이더 하드웨어 및 데이터 수집 시스템의 품질 평가
    # 나. 스펙트럼 폭 기준 ZDR 정상성 검토 방법
    # 다. 스펙트럼 폭 기준 ΦDP 정상성 검토 방법
    # 참고 매트랩 프로그램 :
    # zdr_ts2_2022.m
    # ccpltXY_191013_n.m

    # 참고 원본 문헌
    # OPERA_2006_05_Evaluation_of_dual_polarization_technology_GOOD_GOOD.pdf

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
    serviceName = 'LSH0430'

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
                    # 시작/종료 시간
                    'srtDate': '2001-01-01'
                    , 'endDate': '2018-01-01'
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2010-01-01'
                    , 'endDate': '2015-01-01'
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # ======================================================================================
            # 테스트 파일
            # ======================================================================================
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'zdr_dom_all.mat')
            # fileList = sorted(glob.glob(inpFile))
            #
            # if fileList is None or len(fileList) < 1:
            #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            # fileInfo = fileList[0]
            # matData = io.loadmat(fileInfo)
            # matData.keys()

            # ======================================================================================
            # 파일 검색 및 조회
            # ======================================================================================
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*.*')
            fileList = sorted(glob.glob(inpFile))

            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
                exit(0)

            # ======================================================================================
            # (차등반사도의 방위각 종속성) 특정 사상에 대하여 방위각 방향의 차등반사도 변화를 모니터링
            # ======================================================================================
            # ***************************************************
            # 단일 파일에서 주요 변수 (vnamB) 자료 처리
            # ***************************************************
            # 3차원 배열 초기화
            mrVarT_btA = np.nan * np.ones((len(fileList) * 4, len(fileList) * 1, 360))
            # fileInfo = fileList[0]
            for fileIdx, fileInfo in enumerate(fileList):
                log.info(f'[CHECK] fileInfo: {fileInfo}')

                fileNameNoExt = os.path.basename(fileInfo).split('.')[0]

                # uf 레이더 파일 읽기
                data = readUfRadarData(fileInfo)

                # Reinitialize variables
                refFlt = 'no'  # ref filter
                refThz = 0  # 10 dBZ
                phvFlt = 'no'  # phv filter
                phvThz = 0.95  # 0.95 0.65
                appPDPelim = 'no'
                pdpThz = 15  # 15
                ntex = 7  # odd, OPERA 9
                mtex = 7
                spwFlt = 'no'  # ((yes))
                spwThz = 0.1  # 0.4m/s
                vnamB = ['ZDR', 'PDP', 'PHV', 'REF']
                pco = 0.9925
                srtEA = 3
                endEA = srtEA

                # azm_r = np.transpose(data['arr_azm_rng_elv'][0])
                # rng_r = np.transpose(data['arr_azm_rng_elv'][1])
                # elv_r = np.transpose(data['arr_azm_rng_elv'][2])
                azm_r = data['arr_azm_rng_elv'][0].T
                rng_r = data['arr_azm_rng_elv'][1].T
                elv_r = data['arr_azm_rng_elv'][2].T

                Tang = data['fix_ang']
                Tang[Tang > 180] -= 360

                # if datDR == ':\\Data190\\' or datDR == ':\\Data191\\':
                #     didxs = data['arr_etc'][4].astype(int)
                #     didxe = data['arr_etc'][5].astype(int)
                # else:
                didxs = data['arr_etc'][2].astype(int)
                didxe = data['arr_etc'][3].astype(int)

                # set '0' to '1'
                didxs += 1
                didxe += 1
                didX2 = np.vstack((didxs, didxe))

                Fang = elv_r[didxs].T
                log.info(f'[CHECK] Fang : {Fang}')

                if data['arr_prt_prm_vel'][0][0] == data['arr_prt_prm_vel'][0][-1]:
                    Tprf = 'sing'
                else:
                    Tprf = 'dual'

                bw = data['arr_lat_lon_alt_bwh'][3]

                rVarI_bA = {}
                # mrVarT_btA = (ip_max, j_max, ta_len)
                for ip, vnam_b in enumerate(vnamB):
                    vnam_b = vnam_b.lower()
                    log.info(f'[CHECK] vnam_b : {vnam_b}')

                    # for i in range(srtEA - 1, endEA):
                    for i in range(srtEA, endEA + 1):
                        # Arng = np.arange(didxs[i], didxe[i] + 1)
                        Arng = range(didxs[i - 1], didxe[i - 1])
                        # log.info(f'[CHECK] Arng : {Arng}')

                        # 아래에 있는 getVarInfo191013()는 MATLAB 코드에 정의된 함수로,
                        # 해당 파이썬 버전이 필요하며 이는 상황에 맞게 정의해야 합니다.
                        rVar_b, rVarI_b = getVarInfo191013(data, vnam_b)
                        rVarI_bA[ip] = rVarI_b

                        # rVarf_R = np.transpose(data['arr_ref'])
                        # rVarc_R = np.transpose(data['arr_phv'])
                        # rVarp_R = np.transpose(data['arr_pdp'])
                        rVarf_R = data['arr_ref'].T
                        rVarc_R = data['arr_phv'].T
                        rVarp_R = data['arr_pdp'].T

                        rVar_bT = rVar_b[:, Arng]
                        rVarf = rVarf_R[:, Arng]
                        rVarc = rVarc_R[:, Arng]
                        rVarp = rVarp_R[:, Arng]

                        rVarT_b, vidx_b = prepcss_new(refFlt, phvFlt, appPDPelim, refThz, phvThz, pdpThz, ntex, mtex, rVarf, rVarc, rVarp, rVar_bT)

                        # xr.DataArray(rVarT_b).plot()
                        # # plt.show()
                        #
                        # xr.DataArray(vidx_b).plot()
                        # # plt.show()

                        rVarT_bt = rVarT_b

                        if vnam_b == 'zdr':
                            rVarT_bt[np.logical_or(rVarT_bt < -10, rVarT_bt > 10)] = np.nan
                        elif vnam_b == 'pdp':
                            rVarT_bt[np.logical_or(rVarT_bt < -300, rVarT_bt > 300)] = np.nan
                        elif vnam_b == 'phv':
                            rVarT_bt[np.logical_or(rVarT_bt < 0, rVarT_bt > 1)] = np.nan
                        elif vnam_b == 'ref':
                            rVarT_bt[np.logical_or(rVarT_bt < -100, rVarT_bt > 100)] = np.nan

                        # xr.DataArray(rVarT_bt).plot()
                        # # plt.show()

                        # mrVarT_bt = np.nanmean(rVarT_bt)
                        mrVarT_bt = np.nanmean(rVarT_bt, axis=0)
                        # mrVarT_bt = np.convolve(mrVarT_bt, np.ones((3,)) / 3, mode='same')
                        mrVarT_bt = pd.Series(mrVarT_bt).rolling(window=3).mean()

                        ta = np.nan * np.ones((360,))
                        # ta = np.nan * np.ones((,360))
                        ta[:len(mrVarT_bt)] = mrVarT_bt

                        # mrVarT_btA[ip, i, :] = ta
                        # mrVarT_btA = np.vstack((mrVarT_btA, ta))
                        # mrVarT_btA = np.vstack((mrVarT_btA, ta))
                        mrVarT_btA[ip, i, :] = ta

                        # xr.DataArray(ta).plot()
                        # xr.DataArray(mrVarT_btA).plot()
                        # # plt.show()

                        # fwDir = '{}/{}/{}/'.format(globalVar['figPath'], serviceName, fileNameNoExt)
                        fwDir = '{}/{}/'.format(globalVar['figPath'], serviceName)
                        os.makedirs(os.path.dirname(fwDir), exist_ok=True)
                        if vnam_b in ['zdr', 'pdp']:
                            dspCycl(fwDir, vnam_b, None, ta, 'fix', 'each', fileIdx, rVarI_bA[ip])

            # ***************************************************
            # 전체 파일 목록에서 주요 변수 (vnamB) 자료 처리
            # ***************************************************
            for ipi, vnam_b in enumerate(vnamB):
                log.info(f'[CHECK] vnam_b : {vnam_b}')

                # mrVarT_btAs = np.squeeze(mrVarT_btA[ipi, :])
                mrVarT_btAs = np.squeeze(mrVarT_btA[ipi, :, :])

                # fwDir = '{}/{}/{}/'.format(globalVar['figPath'], serviceName, fileNameNoExt)
                fwDir = '{}/{}/'.format(globalVar['figPath'], serviceName)
                dspCycl(fwDir, vnam_b, mrVarT_btA, mrVarT_btAs, 'fix', 'total', None, rVarI_bA[ipi])

            # ======================================================================================
            # 편파 매개변수의 측정 오류 추정치를 이용하여 레이더 하드웨어 및 데이터 수집 시스템의 품질 평가
            # ======================================================================================
            # ***************************************************
            # 단일 파일에서 주요 변수 (vnamA) 자료 처리
            # ***************************************************
            mrVarT_btA = np.empty((0, 360))
            # fileInfo = fileList[0]
            for fileIdx, fileInfo in enumerate(fileList):
                log.info(f'[CHECK] fileInfo: {fileInfo}')

                # if (fileIdx > 10): continue

                # uf 레이더 파일 읽기
                dictData = readUfRadarData(fileInfo)

                # low (ref)
                refFlt = 'no'  # ref filter
                refThz = 0  # 10 dBZ

                # low (phv)
                phvFlt = 'no'  # phv filter
                phvThz = 0.95  # 0.95 0.65

                # large Tex(pdp)
                appPDPelim = 'no'
                pdpThz = 15  # 15

                ntex = 7  # odd, OPERA 9
                mtex = 7

                # low (spw)
                spwFlt = 'no'  # ((yes))
                spwThz = 0.1  # 0.4m/s

                vnamA = ['SPW', 'SPW']  # m/s
                vnamB = ['ZDR', 'PDP']  # dB
                # vnamA=['SPW']   # m/s
                # vnamB=['ZDR']   # dB
                pco = 0.9925
                srtEA = 2
                endEA = 2

                data = dictData

                azm_r = data['arr_azm_rng_elv'][0]  # 1080x1 (360x3)
                rng_r = data['arr_azm_rng_elv'][1]  # 1196x1
                elv_r = data['arr_azm_rng_elv'][2]  # 1080x1 (360x3)
                Tang = data['fix_ang']
                Tang[Tang > 180] = Tang[Tang > 180] - 360
                didxs = data['arr_etc'][2] + 1
                didxe = data['arr_etc'][3] + 1
                didX2 = [didxs, didxe]
                Fang = elv_r[didxs]

                if data['arr_prt_prm_vel'][0][0] == data['arr_prt_prm_vel'][0][-1]:
                    Tprf = 'sing'
                else:
                    Tprf = 'dual'

                bw = data['arr_lat_lon_alt_bwh'][3]

                # << dual para >>
                for ip in range(len(vnamA)):
                    vnam_a = vnamA[ip]
                    vnam_b = vnamB[ip]

                    for i in range(srtEA, endEA + 1):
                        Arng = np.arange(didxs[i] - 1, didxe[i])

                        rVar_a, rVarI_a = getVarInfo191013(data, vnam_a)  # need translation of function getVarInfo191013
                        rVar_b, rVarI_b = getVarInfo191013(data, vnam_b)  # need translation of function getVarInfo191013

                        rVarf_R = np.transpose(data['arr_ref'])
                        rVarc_R = np.transpose(data['arr_phv'])
                        rVarp_R = np.transpose(data['arr_pdp'])

                        rVar_aT = rVar_a[:, Arng]
                        rVar_bT = rVar_b[:, Arng]

                        rVarf = rVarf_R[:, Arng]
                        rVarc = rVarc_R[:, Arng]
                        rVarp = rVarp_R[:, Arng]

                        if ip == 1:
                            rVarct0 = rVarc
                            rVarct = np.where((rVarc < 0) | (rVarc > 1), np.nan, rVarc)

                            rVarft0 = rVarf
                            rVarft = np.where((rVarf < -100) | (rVarf > 100), np.nan, rVarf)

                            rVarpt0 = rVarp
                            rVarpt = np.where((rVarp < -300) | (rVarp > 300), np.nan, rVarp)

                            rVardt0 = rVar_bT
                            rVardt = np.where((rVar_bT < -10) | (rVar_bT > 10), np.nan, rVar_bT)

                        rVarT_a, vidx_a = prepcss_new(refFlt, phvFlt, appPDPelim, refThz, phvThz, pdpThz, ntex, mtex, rVarf, rVarc, rVarp, rVar_aT)
                        rVarT_b, vidx_b = prepcss_new(refFlt, phvFlt, appPDPelim, refThz, phvThz, pdpThz, ntex, mtex, rVarf, rVarc, rVarp, rVar_bT)

                        if vnam_b == 'ZDR':
                            rVarT_b = np.where(rVarc < pco, np.nan, rVarT_b)

                        if ip == 1:
                            rVarT_bt = rVarT_b
                            rVarT_bt = np.where((rVarT_bt < -10) | (rVarT_bt > 10), np.nan, rVarT_bt)

                            # mrVarT_bt = np.nanmean(rVarT_bt)
                            mrVarT_bt = np.nanmean(rVarT_bt, axis=0)

                            # mrVarT_bt = np.convolve(mrVarT_bt, np.ones(3), 'valid') / 3
                            mrVarT_bt = pd.Series(mrVarT_bt).rolling(window=3).mean()

                            ta = np.full(360, np.nan)
                            ta[:len(mrVarT_bt)] = mrVarT_bt

                            mrVarT_btA = np.vstack((mrVarT_btA, ta))

                            # std
                            texRng = np.ones((ntex, mtex))
                            rVarT_b[np.isnan(rVarT_b)] = 0
                            rVarT_b = generic_filter(rVarT_b, np.nanstd, footprint=texRng)

                            atyp = data['str_typ']

                            # if RdrNam in ['BSL', 'SBS']:
                            #     TLE_fname = fname[8:-4]
                            # else:
                            #     TLE_fname = fname[:-4]
                            TLE_fname = os.path.basename(fileInfo)[8:-4]

                            if Tang[i] < 0:
                                TLE_Tang = '-' + "{:.1f}".format(abs(Tang[i]))
                            else:
                                TLE_Tang = '+' + "{:.1f}".format(Tang[i])

                            TLE = [atyp, '(' + TLE_Tang[0:2] + ',' + TLE_Tang[3] + 'deg' + ')_' + Tprf + '_' + vnam_a + '-' + vnam_b + '_' + TLE_fname]

                            vidxAll = np.logical_or(vidx_a, vidx_b)

                            # Apar = {rVarT_a, rVarT_b, 0, 0}
                            Apar = [rVarT_a, rVarT_b, 0, 0]

                            fwDir = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, TLE[0][0] + TLE[1])
                            fPar, fPar2, mPar = ccpltXY_191013_n(Apar[0], Apar[1], Apar[2], Apar[3],
                                                                 vnam_a, vnam_b,
                                                                 rVarI_a, rVarI_b, Tang[i],
                                                                 fwDir, TLE,
                                                                 vidxAll,
                                                                 refFlt, phvFlt, appPDPelim,
                                                                 refThz, phvThz, pdpThz, ntex, mtex)

            # ***************************************************
            # 전체 파일 목록에서 주요 변수 (vnamA) 자료 처리
            # ***************************************************
            mvStep = 90
            fwDir = '{}/{}/'.format(globalVar['figPath'], serviceName)

            # Compute mean along the specified axis
            mrVarT_btAm = np.nanmean(mrVarT_btA, axis=0)

            # Compute moving average
            mrVarT_btAmMV = pd.Series(mrVarT_btAm).rolling(window=mvStep).mean().values

            plt.figure()
            plt.plot(mrVarT_btA.T)
            plt.plot(mrVarT_btAm, 'k-', linewidth=3)
            plt.plot(mrVarT_btAmMV, 'w-', linewidth=2)
            plt.xlim([0, 360])
            plt.xticks(np.arange(0, 361, 30))
            plt.grid(True)
            saveImg = f"{fwDir}zdr_dom_all.png"
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            plt.savefig(saveImg, dpi=600)
            log.info(f'[CHECK] saveImg : {saveImg}')
            # plt.show()
            plt.close()

            plt.figure()
            plt.plot(mrVarT_btAm, 'k-', linewidth=3)
            plt.plot(mrVarT_btAmMV, 'b-', linewidth=2)
            plt.xlim([0, 360])
            plt.xticks(np.arange(0, 361, 30))
            plt.grid(True)
            saveImg = f"{fwDir}zdr_dom_all2.png"
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            plt.savefig(saveImg, dpi=600)
            log.info(f'[CHECK] saveImg : {saveImg}')
            # plt.show()
            plt.close()

            mrVarT_btAmDT = mrVarT_btAm - mrVarT_btAmMV

            plt.figure()
            plt.plot(mrVarT_btAmDT, 'r-', linewidth=2)
            plt.xlim([0, 360])
            # plt.ylim([-0.2, 0.2])
            plt.xticks(np.arange(0, 361, 30))
            plt.grid(True)
            saveImg = f"{fwDir}zdr_dom_all3.png"
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            plt.savefig(saveImg, dpi=600)
            log.info(f'[CHECK] saveImg : {saveImg}')
            # plt.show()
            plt.close()

            # Save the results to a .npz file, which is numpy's file format
            # np.savez(fwDir + 'zdr_dom_all', mrVarT_btA=mrVarT_btA, mrVarT_btAm=mrVarT_btAm, mrVarT_btAmMV=mrVarT_btAmMV, mrVarT_btAmDT=mrVarT_btAmDT)

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
