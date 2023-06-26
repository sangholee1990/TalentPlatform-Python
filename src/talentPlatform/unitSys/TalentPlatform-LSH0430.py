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
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj
import pymannkendall as mk

import sys
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pyart

import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
from xarrayMannKendall import *
import dask.array as da
import dask
from dask.distributed import Client

from scipy.stats import kendalltau
from plotnine import ggplot, aes, geom_boxplot
import gc
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# 반복 추세 적합
from scipy.optimize import curve_fit
import os
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt


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


def calcMannKendall(x):
    try:
        result = mk.original_test(x)
        return result.Tau
        # return result.trend, result.p, result.Tau

    except Exception:
        return np.nan
        # return np.nan, np.nan, np.nan



def func(x, a, b, c):
    return a * np.sin(b * x + c)


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
    # dspCycl.m,
    # sineFit201208.zip,
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

                    # 경도 최소/최대/간격
                    , 'lonMin': -180
                    , 'lonMax': 180
                    , 'lonInv': 0.1
                    # , 'lonInv': 5

                    # 위도 최소/최대/간격
                    , 'latMin': -90
                    , 'latMax': 90
                    , 'latInv': 0.1
                    # , 'latInv': 5
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': '1950-01-01'
                    # , 'endDate': '2015-01-01'
                    'srtDate': '2010-01-01'
                    , 'endDate': '2015-01-01'

                    # 경도 최소/최대/간격
                    , 'lonMin': -180
                    , 'lonMax': 180
                    , 'lonInv': 0.1

                    # 위도 최소/최대/간격
                    , 'latMin': -90
                    , 'latMax': 90
                    , 'latInv': 0.1

                    , 'typeList': ['EC', 'GDP', 'Land_Cover_Type_1_Percent', 'landscan']
                    # , 'typeList': ['Land_Cover_Type_1_Percent']
                    # , 'keyList': ['CH4', 'CO2_excl', 'CO2_org', 'N2O', 'NH3', 'NMVOC', 'OC', 'NH3', 'SO2']
                    , 'keyList': ['emi_co', 'emi_n2o', 'emi_nh3', 'emi_nmvoc', 'emi_nox', 'emi_oc', 'emi_so2']
                    # , 'keyList': ['emi_nmvoc']
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*.uf')
            # inpFile = '{}/{}/{}.nc'.format(globalVar['inpPath'], serviceName, 'ACCESS-CM2*')
            fileList = sorted(glob.glob(inpFile))

            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

            fileInfo = fileList[0]
            data = pyart.io.read(fileInfo)

            fileNameNoExt = os.path.basename(fileInfo).split('.')[0]

            rnam = data.metadata['instrument_name']
            rlat = data.latitude['data']
            rlon = data.longitude['data']
            ralt = data.altitude['data']

            rnam = data.metadata['instrument_name']
            rlat = data.latitude['data']
            rlon = data.longitude['data']
            ralt = data.altitude['data']
            # -----------------------------------------
            styp = data.scan_type
            # -----------------------------------------
            fbwh = data.instrument_parameters['radar_beam_width_h']['data']
            fprt = data.instrument_parameters['prt']['data']
            fvel = data.instrument_parameters['nyquist_velocity']['data']
            # -----------------------------------------
            nray = data.nrays  # 1080
            ngat = data.ngates  # 1196 (125m)
            nswp = data.nsweeps  # 3
            # -----------------------------------------
            fang = data.fixed_angle['data']  # 1.4 2.7 4.8
            # -----------------------------------------
            fazm = data.azimuth['data']  #
            frng = data.range['data']  #
            felv = data.elevation['data']  #
            # ----------------------------------------------------------------------
            fpul = data.instrument_parameters['pulse_width']['data']  # [1.0e-6, seconds]
            # -----------------------------------------
            ffrq = data.instrument_parameters['frequency']['data']  # [2.88e+09, s-1]
            fscn = data.scan_rate['data']  # scan rate[deg/s]
            # -----------------------------------------
            fswp = data.sweep_number['data']  # 0,1,2...
            fsws = data.sweep_start_ray_index['data']  # [0 273 546]
            fswe = data.sweep_end_ray_index['data']  # [272 545 818]
            ftme = data.time['data']  # [seconds]
            # ----------------------------------------------------------------------
            fdat_ref = data.fields['reflectivity']['data']
            fdat_crf = data.fields['corrected_reflectivity']['data']
            fdat_vel = data.fields['velocity']['data']
            fdat_spw = data.fields['spectrum_width']['data']
            # fdat_zdr=data.fields['differential_reflectivity']['data']
            fdat_zdr = data.fields['corrected_differential_reflectivity']['data']
            fdat_kdp = data.fields['specific_differential_phase']['data']
            fdat_pdp = data.fields['differential_phase']['data']
            fdat_ncp = data.fields['normalized_coherent_power']['data']
            fdat_phv = data.fields['cross_correlation_ratio']['data']
            fdat_ecf = data.fields['radar_echo_classification']['data']
            # -----------------------------------------
            c = fdat_ref.shape
            # -----------------------------------------
            str_nam = [rnam]
            arr_lat_lon_alt_bwh = [rlat, rlon, ralt, fbwh]
            str_typ = [styp]
            arr_prt_prm_vel = [fprt, fvel]
            num_ray_gat_swp = [nray, ngat, nswp]
            fix_ang = fang
            arr_azm_rng_elv = [fazm, frng, felv]  # 3XN
            # ----------------------------------------------------------------------
            arr_etc = [fpul, fswp, fsws, fswe, ftme, ffrq, fscn]
            # arr_etc=[fpul,ffrq,fscn,fswp,fsws,fswe,ftme]
            # arr_etc=[fpul,fswp,fsws,fswe,ftme]
            # ----------------------------------------------------------------------
            arr_ref = np.array(fdat_ref)
            arr_crf = np.array(fdat_crf)
            arr_vel = np.array(fdat_vel)
            arr_spw = np.array(fdat_spw)
            arr_zdr = np.array(fdat_zdr)
            arr_kdp = np.array(fdat_kdp)
            arr_pdp = np.array(fdat_pdp)
            arr_ncp = np.array(fdat_ncp)
            arr_phv = np.array(fdat_phv)
            arr_ecf = np.array(fdat_ecf)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # --------------------------------------------------------
            _, ext = os.path.splitext(fileInfo)
            trm = 8 if ext == 'RAW' else 3
            # --------------------------------------------------------
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # sfmat = os.path.join(dirname + '_OUT', (fileInfo[0:len(fileInfo) - trm] + '.mat'))

            dict = {
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

            print('asdfasdfadsf')

            window_size = 5  # 이동평균에 사용될 윈도우 사이즈

            # 방위각 데이터
            fazm = dict['arr_azm_rng_elv'][0]


            # 차등반사도 데이터
            zdr = dict['arr_zdr']
            data = zdr.data

            # 초기 차등반사도 데이터를 거리 방향으로 평균
            mean_zdr = np.nanmean(data, axis=0)

            # 1. 초기 차등반사도 관측치 거리방향 평균에서 주기 성분만큼의 이동평균을 취하여 선형추세를 산정한다
            moving_avg = np.convolve(mean_zdr, np.ones(window_size), 'valid') / window_size

            # 2. 차등반사도 초기 관측치 평균에서 이동평균 선형추세를 빼서 선형추세가 제거된 성분의 반복 추세를 추출한다
            trend_removed = mean_zdr[window_size - 1:] - moving_avg

            # 3. 선형추세 제거된 반복 추세를 주기함수(예, sin 함수)를 이용하여 적합(fitting)한다
            x = np.linspace(0, 2 * np.pi, len(trend_removed))

            def func(x, a, b, c):
                return a * np.sin(b * x) + c

            popt, pcov = curve_fit(func, x, trend_removed)

            # 4. 반복 추세에서 반복 추세 적합을 빼면 반복 추세가 제거된 잔차 성분만을 추출할 수 있다
            residual = trend_removed - func(x, *popt)

            # 5. ①에서 분리한 호우에 의한 선형 추세와 ④에서 분리한 잔차 성분을 더하면 반복 추세가 제거된 차등반사도를 구할 수 있다
            final_diff_reflectivity = moving_avg + residual

            # Show the final result
            plt.plot(zdr)
            plt.plot(mean_zdr)
            plt.plot(residual)
            plt.plot(final_diff_reflectivity)
            plt.plot(fazm)
            plt.show()

            plt.plot(zdr[2])
            plt.show()








            # 방위각 데이터
            fazm = dict['arr_azm_rng_elv'][0]

            # 초기 차등반사도 데이터를 거리 방향으로 평균
            mean_zdr = np.mean(zdr, axis=1)

            # Define the window size for the moving average
            window_size = 10  # Adjust this value based on your data

            # 이동 평균
            moving_avg = np.convolve(mean_zdr, np.ones(window_size) / window_size, mode='valid')

            # Subtract the moving average from the original data to remove the linear trend
            detrended_diff_refl = mean_zdr[window_size - 1:] - moving_avg

            plt.figure(figsize=(10, 5))
            plt.plot(detrended_diff_refl)
            plt.title('Detrended Differential Reflectivity Change in Azimuth Direction')
            plt.xlabel('Azimuth Angle')
            plt.ylabel('Detrended Average Differential Reflectivity')
            plt.show()


            # Define a threshold for anomaly detection
            threshold = 0.2  # Adjust this value based on your data

            # Identify where the absolute value of the detrended data exceeds the threshold
            anomalies = np.abs(detrended_diff_refl) > threshold

            # Print the azimuth angles where anomalies were detected
            anomaly_angles = np.where(anomalies)[0]
            print(f"Anomalies detected at azimuth angles: {anomaly_angles}")






            # 방위각 방향으로 변동 관찰
            azimuth_change = np.diff(mean_zdr)

            # 차등반사도 변동의 주기성을 찾기 위해 peaks 찾기
            peaks, _ = find_peaks(azimuth_change)

            # 변동 주기성 시각화
            plt.figure(figsize=(10, 6))
            plt.plot(azimuth_change, label="Azimuth Change")
            plt.plot(peaks, azimuth_change[peaks], "x", label="Peaks")
            plt.title("Periodicity in Differential Reflectivity Change")
            plt.legend()
            plt.show()

            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.signal import detrend

            def remove_linear_trend(df, column_name):
                # ① 선형 추세 산정
                df['linear_trend'] = df[column_name].rolling(window=period).mean()

                # ② 선형 추세 제거
                df['detrended'] = df[column_name] - df['linear_trend']

                # ③ 주기 함수 적합
                sine_model = scipy.optimize.curve_fit(lambda t, a, b, c: a * np.sin(b * t + c),
                                                      df.index.values,
                                                      df['detrended'].values,
                                                      p0=[1, 2 * np.pi / period, 0])

                df['fit'] = sine_model[0][0] * np.sin(sine_model[0][1] * df.index.values + sine_model[0][2])

                # ④ 반복 추세 제거
                df['residual'] = df['detrended'] - df['fit']

                # ⑤ 반복 추세가 제거된 차등반사도
                df['corrected'] = df['linear_trend'] + df['residual']

                return df

            # 이 코드에서 'data'는 차등반사도 데이터이며, window는 이동평균 창의 크기입니다.
            # period는 주기적인 성분의 주기입니다.

            # 사용 예시
            column_name = 'column_name'  # 해당하는 열 이름
            period = 10  # 주기 성분
            df = pd.DataFrame(zdr)  # 차등반사도 데이터가 있는 DataFrame
            corrected_df = remove_linear_trend(df, column_name)

            # 결과를 그래프로 표시
            plt.figure(figsize=(12, 8))
            plt.subplot(211)
            plt.plot(data, label='Original')
            plt.plot(trend, label='Trend')
            plt.legend()

            plt.subplot(212)
            plt.plot(detrended, label='Detrended')
            plt.plot(fit_data, label='Fit')
            plt.plot(residual, label='Residual')
            plt.legend()

            plt.show()





            # 테스트

            # 가상의 데이터 생성
            # np.random.seed(0)
            # data = np.random.normal(0, 0.1, 360) + np.sin(np.linspace(0, 2. * np.pi, 360))  # 반사도 변화 모니터링 데이터
            # data_df = pd.DataFrame(data, columns=['Reflectivity'])

            # # 선형 추세 제거
            # data_df['Rolling_Mean'] = data_df['Reflectivity'].rolling(window=10).mean()
            # data_df['Deseasonalized'] = data_df['Reflectivity'] - data_df['Rolling_Mean']
            #
            # # x = np.array(range(len(data_df['Deseasonalized'].dropna())))
            # # y = np.array(data_df['Deseasonalized'].dropna())
            #
            # data_df2 = data_df.dropna()
            # x = np.array(range(len(data_df2['Deseasonalized'])))
            # y = np.array(data_df2['Deseasonalized'])
            # params, params_covariance = curve_fit(func, x, y, p0=[1, 1, 1])
            # data_df2['Fitted'] = func(x, params[0], params[1], params[2])
            #
            # # 잔차 성분 추출
            # data_df2['Residual'] = data_df2['Deseasonalized'] - data_df2['Fitted']
            #
            # # 잔차 성분에서 이상치 탐색
            # threshold = 0.2
            # data_df2['Anomaly'] = np.where(abs(data_df2['Residual']) > threshold, 1, 0)
            #
            # print(data_df2)
            #
            # plt.plot(data_df2['Anomaly'])
            # plt.show()






            # RdrNamA = ['BSL', 'SBS']
            # datDRA = [r'\Data303\']
            #           drvLet = 'j'
            # srtEA = 3
            # endEA = srtEA
            # vnamB = ['ZDR', 'PDP', 'PHV', 'REF']
            #
            # for datDR in datDRA:
            #     for
            # RdrNam in RdrNamA:
            # # Filter parameters
            # refFlt = 'no'
            # refThz = 0
            # phvFlt = 'no'
            # phvThz = 0.95
            # appPDPelim = 'no'
            # pdpThz = 15
            # ntex = 7
            # mtex = 7
            # spwFlt = 'no'
            # spwThz = 0.1
            # pco = 0.9925
            #
            # # Directories
            # frDir = drvLet + datDR + RdrNam + '_OUT\\'
            # fwDir = drvLet + datDR + RdrNam + '_OUT_COR_EA' + str(srtEA) + '\\'
            #
            # # Make directory if it doesn't exist
            # if not os.path.exists(fwDir):
            #     os.mkdir(fwDir)
            #
            # # File list
            # flist = [f for f in os.listdir(frDir) if f.endswith('.mat')]
            # nflst = len(flist)
            #
            # # Loop through all files
            # for j, fname in enumerate(flist):
            #     print(f"Processing file {j + 1} of {nflst}")
            #
            # # Load data
            # a = loadmat(frDir + fname)



            # sio.savemat(str(sfmat), mdict=dict_to_save, do_compression=True)











            #  data = xr.Dataset()
            # for i, fileInfo in enumerate(fileList):
            #     if fileInfo == '/DATA/INPUT/LSH0429/ACCESS-CM2 historical_195001-201412_pr.nc': continue
            #     log.info(f'[CHECK] fileInfo : {fileInfo}')
            #     # selData = xr.open_dataset(fileInfo).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
            #     selData = xr.open_mfdataset(fileInfo).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
            #
            #     # 날짜 변환 (연-월을 기준)
            #     selData['time'] = pd.to_datetime(pd.to_datetime(selData['time'].values).strftime("%Y-%m"), format='%Y-%m')
            #     data = xr.merge([data, selData])

            # data['hurs'].isel(time = 0).plot()
            # data['rsds'].isel(time = 0).plot()
            # data['sfcWind'].isel(time = 0).plot()
            # data['tas'].isel(time = 0).plot()
            # data['tasmax'].isel(time = 0).plot()
            # data['tasmin'].isel(time = 0).plot()
            # data['pr'].isel(time = 0).plot()
            # plt.show()

            # # CPU 활용
            # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            #
            # time = data['time']
            # latitude = data['lat']
            # longitude = data['lon']
            # independent_var1 = data['tas']
            # independent_var2 = data['tasmax']
            # independent_var3 = data['tasmin']
            # dependent_var = data['pr']
            #
            # # 재구성 (시간, 위도, 경도, 변수)
            # independent_var1 = np.reshape(independent_var1.values, (len(time), len(latitude), len(longitude), 1))
            # independent_var2 = np.reshape(independent_var2.values, (len(time), len(latitude), len(longitude), 1))
            # independent_var3 = np.reshape(independent_var3.values, (len(time), len(latitude), len(longitude), 1))
            # dependent_var = np.reshape(dependent_var.values, (len(time), len(latitude), len(longitude), 1))
            #
            # # 독립변수 결합
            # input_data = np.concatenate((independent_var1, independent_var2, independent_var3), axis=-1)
            #
            # # 학습 및 검증 데이터 분할
            # train_size = int(0.8 * len(time))
            # x_train, x_val = input_data[:train_size], input_data[train_size:]
            # y_train, y_val = dependent_var[:train_size], dependent_var[train_size:]
            #
            # # 배치 차원 추가 (ConvLSTM2D는 5D 입력을 요구함)
            # x_train = np.expand_dims(x_train, axis=0)
            # x_val = np.expand_dims(x_val, axis=0)
            # y_train = np.expand_dims(y_train, axis=0)
            # y_val = np.expand_dims(y_val, axis=0)
            #
            # # ConvLSTM 모델 정의
            # # model = Sequential([
            # #     ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=(None, len(latitude), len(longitude), 3)),
            # #     BatchNormalization(),
            # #     ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True),
            # # ])
            # model = Sequential([
            #     ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=(None, len(latitude), len(longitude), 3)),
            #     BatchNormalization(),
            #     ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True),
            # ])
            #
            # # 모델 컴파일
            # model.compile(loss='mse', optimizer='adam')
            #
            # # 모델 학습
            # # history = model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val))
            # # history = model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val), batch_size=32)
            #
            # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(2)
            # val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(2)
            #
            # history = model.fit(train_dataset, epochs=1, validation_data=val_dataset)
            # print('asdfasdf')

            # import xarray as xr
            # import numpy as np

            # xarray 데이터 로드
            # # 시간, 위도, 경도에 따른 독립변수와 종속변수 선택
            # independent_var1 = data['tas'].values
            # independent_var2 = data['tasmax'].values
            # independent_var3 = data['tasmin'].values
            # dependent_var = data['pr'].values  # 예를 들어 강수량 데이터
            #
            # # 데이터셋 준비
            # X = np.stack((independent_var1, independent_var2, independent_var3), axis=-1)
            # Y = dependent_var
            #
            # # 데이터셋 차원 변경 (샘플, 타임스텝, 행, 열, 채널)
            # X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2], X.shape[3]))
            # Y = Y.reshape((Y.shape[0], 1, Y.shape[1], Y.shape[2], 1))
            #
            # # ConvLSTM 모델 정의
            # model = tf.keras.models.Sequential([
            #     tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True,
            #                                input_shape=(None, X.shape[2], X.shape[3], X.shape[4])),
            #     tf.keras.layers.BatchNormalization(),
            #     tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True),
            #     tf.keras.layers.BatchNormalization(),
            #     tf.keras.layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last')
            # ])
            #
            # # 모델 컴파일
            # model.compile(loss='mae', optimizer='adam')
            #
            # # 모델 학습
            # model.fit(X, Y, batch_size=100, epochs=10)


            # 독립변수는 GPCC 입니다!
            # 종속변수는 ACCESS 데이터인데
            # 1950.01~2014.12  7:3
            # 아무래도 시계열이다보니까 RNN으로해야될거같습니다!


            # data = xr.open_mfdataset(fileList)
            # data = xr.open_dataset(fileList[0])
            # data = xr.open_dataset(fileList[7])



            # Dimensions:  (time: 780, lat: 300, lon: 720)
            # Coordinates:
            #   * lat      (lat) float64 -60.0 -59.5 -59.0 -58.5 -58.0 ... 88.0 88.5 89.0 89.5
            #   * lon      (lon) float64 0.0 0.5 1.0 1.5 2.0 ... 357.5 358.0 358.5 359.0 359.5
            #   * time     (time) datetime64[ns] 1950-01-01 1950-02-01 ... 2014-12-01

            # Dimensions:  (lat: 360, lon: 720, time: 1980)
            # Coordinates:
            #   * lat      (lat) float64 -90.0 -89.5 -89.0 -88.5 -88.0 ... 88.0 88.5 89.0 89.5
            #   * lon      (lon) float64 0.0 0.5 1.0 1.5 2.0 ... 357.5 358.0 358.5 359.0 359.5
            #   * time     (time) datetime64[ns] 1850-01-16T12:00:00 ... 2014-12-16T12:00:00
            # Data variables:
            #     hurs     (time, lat, lon) float64 dask.array<chunksize=(1980, 360, 720), meta=np.ndarray>
            #     pr       (time, lat, lon) float64 dask.array<chunksize=(1980, 360, 720), meta=np.ndarray>
            #     isLand   (time, lat, lon) float64 dask.array<chunksize=(1980, 360, 720), meta=np.ndarray>
            #     rsds     (time, lat, lon) float64 dask.array<chunksize=(1980, 360, 720), meta=np.ndarray>
            #     sfcWind  (time, lat, lon) float64 dask.array<chunksize=(1980, 360, 720), meta=np.ndarray>
            #     tas      (time, lat, lon) float64 dask.array<chunksize=(1980, 360, 720), meta=np.ndarray>
            #     tasmax   (time, lat, lon) float64 dask.array<chunksize=(1980, 360, 720), meta=np.ndarray>
            #     tasmin   (time, lat, lon) float64 dask.array<chunksize=(1980, 360, 720), meta=np.ndarray>





            # data = xr.open_mfdataset(fileList, chunks={'time': 10, 'lat': 10, 'lon': 10}).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
            # data = xr.open_mfdataset(fileList).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
            # data = xr.open_mfdataset(fileList)


            # **********************************************************************************************************
            # 피어슨 상관계수 계산
            # **********************************************************************************************************
            # for i, typeInfo in enumerate(sysOpt['typeList']):
            #     for j, keyInfo in enumerate(sysOpt['keyList']):
            #         log.info(f'[CHECK] typeInfo : {typeInfo} / keyInfo : {keyInfo}')
            #
            #         saveFile = '{}/{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, 'CORR', 'corr', typeInfo, keyInfo)
            #         fileChkList = glob.glob(saveFile)
            #         if (len(fileChkList) > 0): continue
            #
            #         var1 = data[typeInfo]
            #         var2 = data[keyInfo]
            #
            #         cov = ((var1 - var1.mean(dim='time', skipna=True)) * (var2 - var2.mean(dim='time', skipna=True))).mean(dim='time', skipna=True)
            #         stdVar1 = var1.std(dim='time', skipna=True)
            #         stdVar2 = var2.std(dim='time', skipna=True)
            #         peaCorr = cov / (stdVar1 * stdVar2)
            #         peaCorr = peaCorr.rename(f'{typeInfo}_{keyInfo}')
            #
            #         saveImg = '{}/{}/{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, 'CORR', 'corr', typeInfo, keyInfo)
            #         os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #         peaCorr.plot(vmin=-1.0, vmax=1.0)
            #         plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #         plt.tight_layout()
            #         # plt.show()
            #         # plt.close()
            #         log.info(f'[CHECK] saveImg : {saveImg}')
            #
            #         os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #         peaCorr.to_netcdf(saveFile)
            #         log.info(f'[CHECK] saveFile : {saveFile}')
            #
            #         # 데이터셋 닫기 및 메모리에서 제거
            #         var1.close(), var2.close(), cov.close(), stdVar1.close(), stdVar2.close(), peaCorr.close()
            #         del var1, var2, cov, stdVar1, stdVar2, peaCorr
            #
            #         # 가비지 수집기 강제 실행
            #         # gc.collect()

            # **********************************************************************************************************
            # 온실가스 배출량 계산
            # **********************************************************************************************************
            # for i, keyInfo in enumerate(sysOpt['keyList']):
            #     log.info(f'[CHECK] keyInfo : {keyInfo}')
            #
            #     var = data[keyInfo]
            #
            #     meanData = var.mean(dim=('time'), skipna=True)
            #     # meanData = meanData.where(meanData > 0)
            #     meanData = meanData.where(meanData != 0)
            #
            #     meanDataL1 = np.log10(meanData)
            #
            #     saveImg = '{}/{}/{}/{}.png'.format(globalVar['figPath'], serviceName, 'EMI', keyInfo)
            #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #     meanDataL1.plot()
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.tight_layout()
            #     # plt.show()
            #     plt.close()
            #     log.info(f'[CHECK] saveImg : {saveImg}')
            #
            #     saveFile = '{}/{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'EMI', keyInfo)
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     meanDataL1.to_netcdf(saveFile)
            #     log.info(f'[CHECK] saveFile : {saveFile}')

            # **********************************************************************************************************
            # Mann-Kendall 계산
            # **********************************************************************************************************
            # for i, keyInfo in enumerate(sysOpt['keyList']):
            #     log.info(f'[CHECK] keyInfo : {keyInfo}')
            #
            #     var = data[keyInfo]
            #
            #     client = Client(n_workers=os.cpu_count(), threads_per_worker=os.cpu_count())
            #     dask.config.set(scheduler='processes')
            #
            #     mannKendall = xr.apply_ufunc(
            #         calcMannKendall,
            #         var,
            #         input_core_dims=[['time']],
            #         output_core_dims=[[]],
            #         vectorize=True,
            #         dask='parallelized',
            #         output_dtypes=[np.float64],
            #         dask_gufunc_kwargs={'allow_rechunk': True}
            #     ).compute()
            #
            #     saveImg = '{}/{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, 'MANN', 'mann', keyInfo)
            #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #     mannKendall.plot(vmin=-1.0, vmax=1.0)
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.tight_layout()
            #     # plt.show()
            #     plt.close()
            #     log.info(f'[CHECK] saveImg : {saveImg}')
            #
            #     saveFile = '{}/{}/{}/{}_{}.nc'.format(globalVar['outPath'], serviceName, 'MANN', 'mann', keyInfo)
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     mannKendall.to_netcdf(saveFile)
            #     log.info(f'[CHECK] saveFile : {saveFile}')
            #
            #     client.close()

            # **********************************************************************************************************
            # Mann Kendall 상자 그림
            # **********************************************************************************************************
            # inpFile = '{}/{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, 'MANN', '*')
            # fileList = sorted(glob.glob(inpFile))
            #
            # if fileList is None or len(fileList) < 1:
            #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            # data = xr.open_mfdataset(fileList)
            # dataL1 = data.to_dataframe().reset_index(drop=True)
            # dataL1.columns = dataL1.columns.str.replace('emi_', '')
            #
            # dataL2 = pd.melt(dataL1, id_vars=[], var_name='key', value_name='val')
            #
            # mainTitle = '{}'.format('EDGAR Mann-Kendall Trend (2001~2018)')
            # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #
            # sns.set_style("whitegrid")
            # sns.set_palette(sns.color_palette("husl", len(dataL1.columns)))
            # sns.boxplot(x='key', y='val', data=dataL2, dodge=False, hue='key')
            # plt.xlabel(None)
            # plt.ylabel('Mann-Kendall Trend')
            # plt.title(mainTitle)
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, title=None)
            # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            # plt.tight_layout()
            # plt.show()
            # plt.close()
            # log.info(f'[CHECK] saveImg : {saveImg}')

            # **********************************************************************************************************
            # typeList에 따른 상자 그림
            # **********************************************************************************************************
            # for i, typeInfo in enumerate(sysOpt['typeList']):
            #     log.info(f'[CHECK] typeInfo : {typeInfo}')
            #
            #     inpFile = '{}/{}/{}/*{}*.nc'.format(globalVar['outPath'], serviceName, 'CORR', typeInfo)
            #     fileList = sorted(glob.glob(inpFile))
            #
            #     if fileList is None or len(fileList) < 1:
            #         log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            #     data = xr.open_mfdataset(fileList)
            #     dataL1 = data.to_dataframe().reset_index(drop=True)
            #     dataL1.columns = dataL1.columns.str.replace(f'{typeInfo}-emi_', '')
            #
            #     dataL2 = pd.melt(dataL1, id_vars=[], var_name='key', value_name='val')
            #
            #     mainTitle = f'EDGAR Pearson-Corr {typeInfo} (2001~2018)'
            #     saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #
            #     sns.set_style("whitegrid")
            #     sns.set_palette(sns.color_palette("husl", len(dataL1.columns)))
            #     sns.boxplot(x='key', y='val', data=dataL2, dodge=False, hue='key')
            #     plt.xlabel(None)
            #     plt.ylabel('Pearson-Corr')
            #     plt.title(mainTitle)
            #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, title=None)
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.tight_layout()
            #     # plt.show()
            #     # plt.close()
            #     log.info(f'[CHECK] saveImg : {saveImg}')

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
