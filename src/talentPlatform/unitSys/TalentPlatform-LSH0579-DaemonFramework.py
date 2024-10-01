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
from _ast import expr
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc

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


def JPL_qpe(kdp, drf, ref):
    # 데이터 크기 설정
    nx, ny = ref.shape
    RZh = np.zeros((nx, ny))
    RKdp = np.zeros((nx, ny))
    ZdrF1 = np.zeros((nx, ny))
    ZdrF2 = np.zeros((nx, ny))
    RJPL = np.zeros((nx, ny))
    Rcas = np.zeros((nx, ny))

    # 단일편파 및 이중편파 레이더 데이터 변환
    # zh: mm^6 m^-3, linear scale value
    zh = np.power(10.0, ref / 10.0)

    # zdr: mm^6 m^-3, linear scale value
    zdr = np.power(10.0, drf / 10.0)

    # R(Zh) 계산
    RZh = 1.70 * 10 ** (-2) * zh ** 0.714

    # R(Kdp) 계산 (Ryzhkov et al., 2005)
    RKdp = 44.0 * (np.abs(kdp) ** 0.822) * np.sign(kdp)

    # Zdr 보정 인자 계산
    ZdrF1 = 0.4 + 5.0 * np.abs(zdr - 1.0) ** 1.3
    ZdrF2 = 0.4 + 3.5 * np.abs(zdr - 1.0) ** 1.7

    # CASE I: R(Zh,Zdr) [mm/h]
    ix = (RZh > 0) & (RZh < 6)
    RJPL[ix] = RZh[ix] / ZdrF1[ix]
    Rcas[ix] = 1

    # CASE II: R(Kdp,Zdr)
    ix = (RZh > 6) & (RZh < 50)
    RJPL[ix] = RKdp[ix] / ZdrF2[ix]
    Rcas[ix] = 2

    # CASE III: R(Kdp)
    ix = RZh > 50  # CASE III 조건
    RJPL[ix] = RKdp[ix]
    Rcas[ix] = 3

    return RJPL, Rcas


def CSU_qpe(kdp, drf, ref):
    # 데이터 크기 설정
    nx, ny = ref.shape
    RCSU = np.zeros((nx, ny))
    Rcas = np.zeros((nx, ny))

    # 단일편파 및 이중편파 레이더 데이터 변환
    # zh: mm^6 m^-3, linear unit
    zh = np.power(10.0, ref / 10.0)

    # zdr: dB, log unit
    zdr = drf

    # CASE I: R(Kdp, Zdr)
    ix = (kdp >= 0.3) & (ref >= 38) & (zdr >= 0.5)
    RCSU[ix] = 90.8 * (kdp[ix] ** 0.93) * (10 ** (-0.169 * zdr[ix]))
    Rcas[ix] = 1

    # CASE II: R(Kdp)
    ix = (kdp >= 0.3) & (ref >= 38) & (zdr < 0.5)
    RCSU[ix] = 40.5 * (kdp[ix] ** 0.85)
    Rcas[ix] = 2

    # CASE III: R(Zh, Zdr)
    ix = ((kdp < 0.3) | (ref < 38)) & (zdr >= 0.5)
    RCSU[ix] = 6.7 * 10 ** (-3) * (zh[ix] ** 0.927) * (10 ** (-0.343 * zdr[ix]))
    Rcas[ix] = 3

    # CASE IV: R(Zh)
    ix = ((kdp < 0.3) | (ref < 38)) & (zdr < 0.5)
    RCSU[ix] = 0.017 * (zh[ix] ** 0.7143)
    Rcas[ix] = 4

    # ix = (RCSU < 0) | np.isnan(ref) | (ref < 10) | (RCSU > 150) | (drf < -3) | (drf > 5)
    # Rcas[ix] = 0
    # RCSU[ix] = np.nan

    return RCSU, Rcas


def calRain_SN(rVarf, rVark, rVard, Rtyp, dt, aZh):
    # 초기화
    appzdrOffset = 'no'
    zdrOffset = 0

    # 1. R(Zh): 단일편파 강우량 계산
    zh = np.power(10.0, rVarf / 10.0)
    RintZH = 1.70 * 10 ** (-2) * (zh + aZh) ** 0.714
    RintZH[rVarf == -327.6800] = np.nan
    RintZH[RintZH <= 0] = np.nan
    Rcas = 1

    # 2. R(Kdp): Ryzhkov et al., 2005의 Kdp 기반 강우량 계산
    RintKD = 44.0 * (np.abs(rVark) ** 0.822) * np.sign(rVark)
    RintKD[rVark == -327.6800] = np.nan
    RintKD[RintKD <= 0] = np.nan
    Rcas = 1

    # 3. R(Zh, Zdr): Bringi and Chandraseker, 2001의 Zh, Zdr 기반 강우량 계산
    # Zdr 계산
    if appzdrOffset == 'yes':
        zdr = np.power(10.0, (rVard + zdrOffset) / 10.0)
    else:
        zdr = np.power(10.0, rVard / 10.0)

    # S-band 기준
    RintZD = 0.0067 * (zh ** 0.927) * (zdr ** -3.43)
    RintZD[(rVarf == -327.6800) | (rVard == -327.6800)] = np.nan
    RintZD[RintZD <= 0] = np.nan
    Rcas = 1

    # 4. JPL 강우강도 계산
    if appzdrOffset == 'yes':
        RintJP, Rcas = JPL_qpe(rVark, rVard + zdrOffset, rVarf)
    else:
        RintJP, Rcas = JPL_qpe(rVark, rVard, rVarf)

    RintJP[(rVark == -327.6800) | (rVard == -327.6800) | (rVarf == -327.6800)] = np.nan
    RintJP[RintJP <= 0] = np.nan

    # 5. CSU 강우강도 계산
    if appzdrOffset == 'yes':
        RintCS, Rcas = CSU_qpe(rVark, rVard + zdrOffset, rVarf)
    else:
        RintCS, Rcas = CSU_qpe(rVark, rVard, rVarf)

    RintCS[(rVark == -327.6800) | (rVard == -327.6800) | (rVarf == -327.6800)] = np.nan
    RintCS[RintCS <= 0] = np.nan

    # Rtyp에 따른 강우량 계산
    Rcal = {}
    # Rcal에 단위 시간당 강우강도 (mm/h)
    if Rtyp == 'int':
        Rcal[1] = RintZH
        Rcal[2] = RintKD
        Rcal[3] = RintZD
        Rcal[4] = RintJP
        Rcal[5] = RintCS

    # Rcal에 누적 강우강도 (mm)
    elif Rtyp == 'ran':
        Rcal[1] = RintZH * dt / 3600
        Rcal[2] = RintKD * dt / 3600
        Rcal[3] = RintZD * dt / 3600
        Rcal[4] = RintJP * dt / 3600
        Rcal[5] = RintCS * dt / 3600

    return Rcal, Rcas

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 기상청 레이더 강우강도 산출 및 지상관측소 비교 검증 (기존 Python, Matlab 혼용)

    # 안녕하세요
    # 과장님, 유선상으로 말씀드린 프로그램 변환 건입니다.
    # 기상청 uf 자료 읽어서 변수별로 뽑아 저장하는 코드인데요.
    # 파이썬+매틀랩으로 분리되어 있고 시간이 많이 걸려 파이썬으로 통일하고자 합니다. (코드 정리 포함)
    # 현재 코드 설명은 다음과 같습니다.

    # KMA_GNG_Kang~
    # - uf 파일 읽어 필요한 정보를 mat, png파일로 저장하는 코드
    # - 정리 필요. 레이더 사이트 자료별로 중복없이 적용되게
    # - 전체 for문으로 돌릴 시, 오류발생으로 중간 멈춤 현상 있음(제 컴의 문제일수도..)
    # - mat 파일로 저장하는 것은 뒤 프로그램이 매틀랩으로 되어 있어서 저장하는 것임. 뒤 코드 파이선으로 구축시 단계만 구분하면 됨
    # - input : uf, Output : _OUT폴더, png. mat

    # 2. snowC_GNG_Kang~
    # - 1에서 저장한 변수들 이용해서 알고리즘별 적용하여 강우강도 산정한 후, 격자형태로 저장하는 코드
    # - 1와 연계하여 파이썬으로 변환 필요
    # - aws숫자: 지점별 값(주로 관측소)을 뽑기 위해 저장.
    # 별도로 지점에 해당하는 행렬(i,j)을 계산하여 넣었는데 지점에 해당하는 행렬(i,j)을 계산하는 모듈 필요.
    # 지점 정보 파일로 읽어 적용할 수 있었으면 함
    # - 현재 코드가 맞는지 의문임. 혹시 기존에 작업하는 과정과 다르다면 알려주셔요.
    # - input : MA_GNG_sel_OUT, output :＿CMU폴더, mat, Rst, RstH

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'LSH0579'

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

            globalVar['inpPath'] = '/DATA/INPUT'
            globalVar['outPath'] = '/DATA/OUTPUT'
            globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': '2019-01-01'
                , 'endDate': '2023-01-01'
            }

            # from __future__ import print_function
            # from __future__ import absolute_import
            # from __future__ import division

            import glob, os
            import numpy as np
            import matplotlib.pyplot as plt
            import scipy.io as sio
            import pyart
            import gzip
            import shutil

            import os
            import numpy as np
            # import scipy.io as sio
            import matplotlib.pyplot as plt
            from scipy.interpolate import griddata
            from pyproj import Proj, Transformer
            import re

            # ==========================================================================================================
            # 파이썬 전처리
            # ==========================================================================================================

            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'HN_M.csv')
            # fileList = sorted(glob.glob(inpFile))

            # 관악산(KWK), 오성산(KSN), 광덕산(GDK), 면봉산(MYN), 구덕산(PSN), 백령도(BRI), 영종도(IIA), 진도(JNI), 고산(GSN), 성산(SSP), 강릉(GNG)
            # KMA = ['KWK', 'KSN', 'GDK', 'MYN', 'PSN', 'BRI', 'IIA', 'JNI', 'GSN', 'SSP', 'GNG']
            KMA = ['KSN']
            # rDir = 'E:/uf/'
            # os.chdir('E:/uf/')
            rDir = '{}/{}/{}/'.format(globalVar['inpPath'], serviceName, 'uf')
            rrDir = rDir + 'RDR_FQC_20230215'
            file_lists = {code: glob.glob(f"{rrDir}\\*{code}*") for code in KMA}

            #  Print the count of files for each variable using enumerate
            for index, (code, files) in enumerate(file_lists.items()):
                print(f"{index + 1}. {code}: {len(files)} files")

            # KMA = ['GNG', 'JNI', 'KWK', 'KSN', 'MYN', 'PSN']
            # KMA = ['MYN', 'PSN']

            # rDir = 'E:/uf/'
            # os.chdir('E:/uf/')
            dir_list = os.listdir(rDir)
            # dir_list = dir_list[10:16]

            j = 0
            for j in range(len(dir_list)):
                rFDir = rDir + dir_list[j]
                print(rFDir)
                dirname = rFDir

                # code = 'KSN'
                list_files = []
                for code in KMA:
                    list_files.extend(glob.glob1(rFDir, f"*{code}*"))

                # 테스트
                list_files = [rFDir]

                    # list_files=list_files[:5]
                num_files = len(list_files)
                uffname = 'reflectivity'
                for f in list_files:
                    print(f)
                    filename = f
                    if re.search('.mat', filename, re.IGNORECASE): continue

                    readf = os.path.join(dirname, filename)
                    a = pyart.io.read(readf)

                    # ----------------------------------------------------------------------
                    rnam = a.metadata['instrument_name']  # radar_name
                    rlat = a.latitude['data']
                    rlon = a.longitude['data']
                    ralt = a.altitude['data']
                    # ----------------------------------------------------------------------
                    styp = a.scan_type
                    # ----------------------------------------------------------------------
                    fbwh = a.instrument_parameters['radar_beam_width_h']['data']
                    fprt = a.instrument_parameters['prt']['data']
                    # fprb=a.instrument_parameters['prt_ratio']['data']
                    ###fprf=a.instrument_parameters['prf_flag']['data'] # 0 for high PRF, 1 for low PRF
                    ###sprt=a.instrument_parameters['prt_mode']['data'] # 'fixed', 'staggered', 'dual'
                    fvel = a.instrument_parameters['nyquist_velocity']['data']
                    ###frng=a.instrument_parameters['unambiguous_range']
                    fpul = a.instrument_parameters['pulse_width']['data']  # [1.0e-6, seconds]
                    ffrq = a.instrument_parameters['frequency']['data']  # [2.88e+09, s-1]
                    # ----------------------------------------------------------------------
                    nray = a.nrays  # 1080
                    ngat = a.ngates  # 1196 (125m)
                    nswp = a.nsweeps  # 3
                    # ----------------------------------------------------------------------
                    fang = a.fixed_angle['data']  # -0.2(359.8) 0.1 0.5
                    # print(fang)
                    # ----------------------------------------------------------------------
                    fazm = a.azimuth['data']  #
                    frng = a.range['data']  #
                    felv = a.elevation['data']  #
                    # ----------------------------------------------------------------------
                    fscn = a.scan_rate['data']  # scan rate[deg/s]
                    fswp = a.sweep_number['data']  # 0,1,2...
                    fsws = a.sweep_start_ray_index['data']  # [0 273 546]
                    fswe = a.sweep_end_ray_index['data']  # [272 545 818]
                    ftme = a.time['data']  # [seconds]
                    # ----------------------------------------------------------------------
                    fdat_ref = a.fields['reflectivity']['data']
                    # fdat_crf=a.fields['corrected_reflectivity']['data']
                    fdat_zdr = a.fields['corrected_differential_reflectivity']['data']
                    # ----------------------------------------------------------------------
                    fdat_pdp = a.fields['differential_phase']['data']
                    fdat_kdp = a.fields['specific_differential_phase']['data']
                    fdat_vel = a.fields['velocity']['data']
                    fdat_phv = a.fields['cross_correlation_ratio']['data']
                    # fdat_ecf=a.fields['radar_echo_classification']['data']
                    # fdat_coh=a.fields['normalized_coherent_power']['data']
                    fdat_spw = a.fields['spectrum_width']['data']
                    ###    fdat_tpw=a.fields['total_power']['data']
                    # ----------------------------------------------------------------------
                    c = fdat_ref.shape
                    # ----------------------------------------------------------------------
                    str_nam = [rnam]
                    # arr_lat_lon_alt=[rlat,rlon,ralt]
                    arr_lat_lon_alt_bwh = [rlat, rlon, ralt, fbwh]
                    str_typ = [styp]
                    # arr_bwh_prt_prf_vel_rng=[fbwh,fprt,fprf,fvel,frng]
                    # arr_prt_prf_vel=[fprt,fprf,fvel]
                    # arr_prt_prm_vel=[fprt,sprt,fvel]
                    arr_prt_prm_vel = [fprt, fvel]
                    num_ray_gat_swp = [nray, ngat, nswp]
                    fix_ang = fang
                    arr_azm_rng_elv = [fazm, frng, felv]  # 3XN
                    # ----------------------------------------------------------------------
                    arr_etc = [fpul, ffrq, fscn, fswp, fsws, fswe, ftme]
                    # arr_etc=[fpul,fswp,fsws,fswe,ftme]
                    # ----------------------------------------------------------------------
                    arr_ref = np.array(fdat_ref)
                    # arr_crf=np.array(fdat_crf)
                    arr_zdr = np.array(fdat_zdr)
                    arr_pdp = np.array(fdat_pdp)
                    arr_kdp = np.array(fdat_kdp)
                    arr_vel = np.array(fdat_vel)
                    arr_phv = np.array(fdat_phv)
                    # arr_ecf=np.array(fdat_ecf)
                    # arr_coh=np.array(fdat_coh)
                    arr_spw = np.array(fdat_spw)
                    # ----------------------------------------------------------------------
                    # ----------------------------------------------------------------------
                    radar = a
                    # ----------------------------------------------------------------------
                    # # plot sigmet data

                    display = pyart.graph.RadarDisplay(radar)
                    fig = plt.figure(figsize=(35, 8))
                    # ----------------------
                    # nEL=0 # GNG 0.2
                    # nEL=3 # GDK 0.8
                    # nEL=1 # GSN 5.2
                    # ----------------
                    nEL = 0  # FCO SBS
                    # ----------------------
                    # fig.subplots_adjust(hspace=0.3)
                    for i in range(3):
                        ax = fig.add_subplot(1, 3, i + 1)
                        #
                        try:
                            if i == 0:
                                display.plot('reflectivity', nEL, vmin=-5, vmax=40)
                            elif i == 1:
                                # display.plot('differential_phase', 0, vmin=-30, vmax=120)
                                display.plot('corrected_differential_reflectivity', nEL, vmin=-2, vmax=5)
                            elif i == 2:
                                # display.plot('specific_differential_phase', 0, vmin=-1, vmax=5)
                                display.plot('cross_correlation_ratio', nEL, vmin=0.5, vmax=1.0)
                            #
                            # display.plot_range_rings([50, 100, 150, 200, 250]) #KMA
                            display.plot_range_rings([25, 50, 75, 100, 125])  # FCO
                            display.plot_cross_hair(5.)

                        except Exception as e:
                            print(f"Error plotting {i}: {e}")

                    # plt.show()

                    #----------------------------------------------------------------------
                    trm = 3

                    # savf1 = os.path.join(dirname + '/_OUT', (filename[0:len(f) - trm] + '.png'))
                    # plt.savefig(str(savf1))

                    fileName = os.path.basename(f)
                    fileNameNotExt = fileName.split(".")[0]

                    # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, fileNameNotExt)
                    # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                    # plt.savefig(saveImg, dpi=600, bbox_inches='tight')
                    # plt.close()
                    # log.info(f"[CHECK] saveImg : {saveImg}")

                    # ----------------------------------------------------------------------
                    # sfmat = os.path.join(dirname + '/_OUT', (filename[0:len(f) - trm] + '.mat'))
                    # sio.savemat(str(sfmat),
                    #             mdict={'str_nam': str_nam,
                    #                    'arr_lat_lon_alt_bwh': arr_lat_lon_alt_bwh,
                    #                    'str_typ': str_typ,
                    #                    'arr_prt_prm_vel': arr_prt_prm_vel,
                    #                    'num_ray_gat_swp': num_ray_gat_swp,
                    #                    'fix_ang': fix_ang,
                    #                    'arr_azm_rng_elv': arr_azm_rng_elv,
                    #                    'arr_etc': arr_etc,
                    #                    'arr_ref': arr_ref,
                    #                    # 'arr_crf': arr_crf,
                    #                    'arr_zdr': arr_zdr,
                    #                    'arr_pdp': arr_pdp,
                    #                    'arr_kdp': arr_kdp,
                    #                    'arr_vel': arr_vel,
                    #                    'arr_phv': arr_phv,
                    #                    # 'arr_ecf': arr_ecf,
                    #                    # 'arr_coh': arr_coh,
                    #                    'arr_spw': arr_spw},
                    #             do_compression=True)

                    # 자료 저장
                    data = {
                        'str_nam': str_nam,
                        'arr_lat_lon_alt_bwh': arr_lat_lon_alt_bwh,
                        'str_typ': str_typ,
                        'arr_prt_prm_vel': arr_prt_prm_vel,
                        'num_ray_gat_swp': num_ray_gat_swp,
                        'fix_ang': fix_ang,
                        'arr_azm_rng_elv': arr_azm_rng_elv,
                        'arr_etc': arr_etc,
                        'arr_ref': arr_ref,
                        # 'arr_crf': arr_crf,
                        'arr_zdr': arr_zdr,
                        'arr_pdp': arr_pdp,
                        'arr_kdp': arr_kdp,
                        'arr_vel': arr_vel,
                        'arr_phv': arr_phv,
                        # 'arr_ecf': arr_ecf,
                        # 'arr_coh': arr_coh,
                        'arr_spw': arr_spw
                    }

                    # ==================================================================================================
                    # 강우강도 산출
                    # ==================================================================================================
                    # 초기 변수 설정
                    # datDRA = ['D:/SNOW/KMA_GNG_sel_OUT']
                    # datDR = datDRA[0]
                    inpFilePattern = '{}/{}/{}/'.format(globalVar['inpPath'], serviceName, 'KMA_GNG_sel_OUT')
                    datDRA = [inpFilePattern]
                    datDR = datDRA[0]

                    # 강수 유형
                    # int(mm/h)/ran(mm)
                    Rtyp = 'int'

                    # 강수 알고리즘 인덱스
                    # Rcal{RintZH;RintKD;RintZD;RintJP;RintCS}
                    Ralg = 3

                    # 시간 간격 [초]
                    # 5분 단위
                    # [80s+70s=>2.5min] low sng 300km[-0.3 0.1 0.60]=80s, high dul 150km[1.4 2.7 4.8]=70s
                    dt = 5.0 * 60

                    # 시작 Elevation Angle
                    srtEA = 3  # 시작 각도
                    frDir = datDR
                    fwDir = f"{datDR}_CMU/"

                    # 결과 저장 디렉토리가 없으면 생성
                    # if not os.path.exists(fwDir):
                    #     os.makedirs(fwDir)

                    # 파일 리스트 가져오기
                    flist = [f for f in os.listdir(frDir) if f.endswith('.mat')]
                    nflst = len(flist)

                    # 변수 초기화
                    ZcaloA = np.zeros((601, 601))
                    FcaloA = np.zeros((601, 601))
                    RcaloA = np.zeros((601, 601))
                    sFcalA = np.zeros((nflst, 1))
                    sRcalA = np.zeros((nflst, 1))
                    xFcalA = np.zeros((nflst, 1))
                    xRcalA = np.zeros((nflst, 1))

                    # 9 지점에 대해 데이터 저장
                    aws_data = np.zeros((nflst, 9, 3))

                    # 파일별로 루프
                    j = 0
                    for j in range(nflst):
                        fname = flist[j]
                        print(f"Processing file {j + 1}/{nflst}: {fname}")

                        # MATLAB 파일 로드
                        # data = sio.loadmat(os.path.join(frDir, fname))

                        # 데이터 읽기

                        # 방위각
                        azm_r = data['arr_azm_rng_elv'][0].T

                        # 거리
                        rng_r = data['arr_azm_rng_elv'][1].T

                        # 고도각
                        elv_r = data['arr_azm_rng_elv'][2].T

                        # 각도 정보
                        Tang = data['fix_ang'].flatten()
                        Tang[Tang > 180] -= 360

                        # 고도각에 따른 인덱스
                        # didxs = data['arr_etc'][2].astype(np.int32) + 1
                        # didxe = data['arr_etc'][3].astype(np.int32) + 1

                        # 특정 고도각 선택
                        # Arng = np.arange(didxs[srtEA - 1], didxe[srtEA - 1] + 1)

                        # 고도각에 따른 인덱스
                        # pattern = r'D:/Data190/|D:/Data191/|D:/2022/X0810/|' + re.escape(datDRA)
                        pattern = r'Data190|Data191|2022/X0810|' + re.escape(datDRA[0])

                        if re.search(pattern, datDR, re.IGNORECASE):
                            didxs = np.array(data['arr_etc'][4], dtype=np.int32)  # arr_etc{5} -> arr_etc[4]
                            didxe = np.array(data['arr_etc'][5], dtype=np.int32)  # arr_etc{6} -> arr_etc[5]
                        else:
                            didxs = np.array(data['arr_etc'][2], dtype=np.int32)  # arr_etc{3} -> arr_etc[2]
                            didxe = np.array(data['arr_etc'][3], dtype=np.int32)  # arr_etc{4} -> arr_etc[3]

                        # 인덱스 값 변경 (0-based indexing 보정)
                        didxs = didxs + 1  # set '0' to '1'
                        didxe = didxe + 1

                        # didxs와 didxe를 합쳐서 배열 생성
                        # didX2 = np.array([didxs, didxe])
                        didX2 = np.column_stack((didxs, didxe)).ravel()

                        # elv_r 배열에서 인덱스 값 추출 (인덱스는 0-based이므로 조정 필요)
                        Fang = elv_r[didxs]  # numpy 배열 인덱싱
                        # print(Fang)  # 출력 예시

                        # dual para
                        if data['arr_prt_prm_vel'][0][0] == data['arr_prt_prm_vel'][0][-1]:
                            Tprf = 'sing'
                        else:
                            Tprf = 'dual'

                        bw = data['arr_lat_lon_alt_bwh'][3]
                        print(bw)

                        # Elev. ang
                        Arng = range(didxs[srtEA - 1], didxe[srtEA - 1] + 1)
                        print(list(Arng))

                        # Var. info
                        rVar_rf = data['arr_ref'].T
                        rVar_rk = data['arr_kdp'].T
                        rVar_rd = data['arr_zdr'].T
                        rVar_rc = data['arr_phv'].T
                        rVar_rp = data['arr_pdp'].T
                        rVar_rv = data['arr_vel'].T

                        rVar_rf[rVar_rf < 0] = 0

                        rVarf = rVar_rf[:, Arng]
                        rVark = rVar_rk[:, Arng]
                        rVard = rVar_rd[:, Arng]
                        rVarc = rVar_rc[:, Arng]
                        rVarp = rVar_rp[:, Arng]
                        rVarv = rVar_rv[:, Arng]

                        # cal. rain
                        # rainfall(mm)
                        [Rcalo, Rcas] = calRain_SN(rVarf, rVark, rVard, Rtyp, dt, 10)

                        # grid
                        gw = 1

                        azm = azm_r[Arng]
                        rng = rng_r
                        elv = elv_r[Arng]

                        # xr = rng * (np.sin(np.deg2rad(azm.)) * np.cos(np.deg2rad(elv))) / 1000
                        # yr = rng * (np.cos(np.deg2rad(azm)) * np.cos(np.deg2rad(elv))) / 1000

                        # 1196 x 1080 -> 960 x 360
                        xr = rng[:, None] * (np.sin(np.deg2rad(azm.T)) * np.cos(np.deg2rad(elv.T))) / 1000

                        # 1196 x 1080 -> 960 x 360
                        yr = rng[:, None] * (np.cos(np.deg2rad(azm.T)) * np.cos(np.deg2rad(elv.T))) / 1000
                        dxr = xr.flatten()
                        dyr = yr.flatten()

                        # 격자 설정
                        xi, yi = np.meshgrid(np.arange(-300, 301, gw), np.arange(-300, 301, gw))

                        # zh.linear unit in mm6 m-3
                        zh = np.power(10.0, rVarf / 10.0)
                        zhh = griddata((dxr, dyr), zh.flatten(), (xi, yi), method='linear')

                        # refl.
                        rVarf[np.isnan(rVarf)] = 0
                        ziR = griddata((dxr, dyr), rVarf.flatten(), (xi, yi), method='linear')

                        # rain
                        dzr = Rcalo[Ralg]
                        dzr[np.isnan(dzr)] = 0
                        Rcal = griddata((dxr, dyr), dzr.flatten(),(xi, yi), method='linear')

                        # xy->lonlat
                        lat0 = data['arr_lat_lon_alt_bwh'][0][0]
                        lon0 = data['arr_lat_lon_alt_bwh'][1][0]
                        elv0 = data['arr_lat_lon_alt_bwh'][2][0]

                        enu_proj = Proj(proj='tmerc', lat_0=lat0, lon_0=lon0, ellps='WGS84', units='km')
                        wgs84_proj = Proj(proj='latlong', datum='WGS84')
                        transformer = Transformer.from_proj(enu_proj, wgs84_proj)
                        xlong, ylatg = transformer.transform(xi, yi)
                        h0 = np.zeros_like(xi)



                        # 누적 계산 반사도 팩터
                        ZcaloA = ZcaloA + zhh

                        # 누적 계산 반사도
                        FcaloA = FcaloA + ziR

                        # 누적 계산 강우강도, mm/hr
                        RcaloA = RcaloA + Rcal

                        # 1time당 sumFcalA 반사도
                        sFcalA[j] = np.nansum(ziR)

                        # 1time당 sumFcalA 강우강도
                        sRcalA[j] = np.nansum(Rcal)

                        # 1time당 maxFcalA
                        xFcalA[j] = np.nanmax(ziR)

                        # 1time당 sumFcalA
                        xRcalA[j] = np.nanmax(Rcal)

                        import xarray as xr

                        lon2D = xlong
                        lat2D = ylatg

                        xdim = lon2D.shape[0]
                        ydim = lon2D.shape[1]

                        # 202112241240
                        dtDateInfo = pd.to_datetime('202112241240', format = '%Y%m%d%H%M')

                        dsData = xr.Dataset(
                            {
                                'zhh': (('time', 'row', 'col'), (zhh).reshape(1, xdim, ydim))
                                , 'ziR': (('time', 'row', 'col'), (ziR).reshape(1, xdim, ydim))
                                , 'Rcal': (('time', 'row', 'col'), (Rcal).reshape(1, xdim, ydim))
                            }
                            , coords={
                                'row': np.arange(xdim)
                                , 'col': np.arange(ydim)
                                , 'lon': (('row', 'col'), lon2D)
                                , 'lat': (('row', 'col'), lat2D)
                                , 'time': pd.date_range(dtDateInfo, periods=1)
                            }
                        )

                        # NetCDF 저장
                        saveNcFile = '{}/{}/{}_{}.nc'.format(globalVar['outPath'], serviceName, 'RDR_GDK_FQC', dtDateInfo.strftime('%Y%m%d%H%M'))
                        os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
                        dsData.to_netcdf(saveNcFile)
                        log.info(f'[CHECK] saveNcFile : {saveNcFile}')

                        # dsData.isel(time = 0)['zhh'].plot()
                        # plt.show()

                        target_lat = 37.0
                        target_lon = 127.0

                        # dsData['lat'].values

                        result = dsData.sel(x=1, y=2)
                        # result = dsData.sel(lat=target_lat, lon=target_lon, method='nearest')



                        # 융합 ASOS/AWS 지상 관측소
                        sysOpt['stnList'] = [90, 104, 105, 106, 520, 523, 661, 670, 671]

                        inpAllStnFile = '{}/{}'.format(globalVar['cfgPath'], 'stnInfo/ALL_STN_INFO.csv')
                        allStnData = pd.read_csv(inpAllStnFile)
                        allStnDataL1 = allStnData[['STN', 'STN_KO', 'LON', 'LAT']]
                        allStnDataL2 = allStnDataL1[allStnDataL1['STN'].isin(sysOpt['stnList'])]


                        from sklearn.neighbors import BallTree
                        # 최근접 화소
                        cfgData = dsData
                        cfgDataL1 = cfgData.to_dataframe().reset_index(drop=False)
                        baTree = BallTree(np.deg2rad(cfgDataL1[['lat', 'lon']].values), metric='haversine')


                        for i, posInfo in allStnDataL2.iterrows():
                            if (pd.isna(posInfo['LAT']) or pd.isna(posInfo['LON'])): continue

                            closest = baTree.query(np.deg2rad(np.c_[posInfo['LAT'], posInfo['LON']]), k=1)
                            cloDist = closest[0][0][0] * 1000.0
                            cloIdx = closest[1][0][0]
                            cfgInfo = cfgDataL1.loc[cloIdx]

                            allStnDataL2.loc[i, 'posRow'] = cfgInfo['row']
                            allStnDataL2.loc[i, 'posCol'] = cfgInfo['col']
                            allStnDataL2.loc[i, 'posLat'] = cfgInfo['lon']
                            allStnDataL2.loc[i, 'posLon'] = cfgInfo['lat']
                            allStnDataL2.loc[i, 'posDistKm'] = cloDist


                        for i, posInfo in allStnDataL2.iterrows():
                            if (pd.isna(posInfo['posRow']) or pd.isna(posInfo['posCol'])): continue

                            posData = dsData.interp({'row': posInfo['posRow'], 'col': posInfo['posCol']}, method='nearest')
                            # posData = dsData.interp({'row': posInfo['posRow'], 'col': posInfo['posCol']}, method='linear')
                            # log.info(f'[CHECK] posInfo : {posInfo}')

                        # ZcaloA += np.nan_to_num(zhh)
                        #
                        # # 레이더 데이터를 저장
                        # sio.savemat(os.path.join(fwDir, f"zhh{j + 1:03d}.mat"), {'zhh': zhh})
                        #
                        # # 강우량 계산 및 누적
                        # Rcal = zhh  # 간단한 예시, 실제 강우량 계산 로직 대체 필요
                        # RcaloA += np.nan_to_num(Rcal)
                        # sio.savemat(os.path.join(fwDir, f"Rcal{j + 1:03d}.mat"), {'Rcal': Rcal})
                        #
                        # # 특정 위치 값 저장 (임의의 위치로 설정, 예시)
                        # aws_data[j, :, 0] = zhh[300, 300]  # 반사도 값 예시
                        # aws_data[j, :, 1] = zhh[349, 275]  # 다른 위치 값 예시
                        # aws_data[j, :, 2] = Rcal[349, 275]  # 강우량 값 예시

                    #         aws90(j,3)=Rcal(349,275);
                    #     aws104(j,3)=Rcal(300,300);
                    #     aws105(j,3)=Rcal(294,303);
                    #     aws106(j,3)=Rcal(267,324);
                    #     aws520(j,3)=Rcal(340,271);
                    #     aws523(j,3)=Rcal(310,297);
                    #     aws661(j,3)=Rcal(382,261);
                    #     aws670(j,3)=Rcal(331,280);
                    #     aws671(j,3)=Rcal(342,277);

                    # 누적된 데이터를 최종 저장
                    sio.savemat(os.path.join(fwDir, 'dat.mat'), {
                        'ZcaloA': ZcaloA, 'RcaloA': RcaloA, 'FcaloA': FcaloA, 'aws_data': aws_data
                    })

                    # 결과 시각화
                    plt.figure(figsize=(10, 8))
                    plt.pcolor(xi, yi, ZcaloA, shading='flat', cmap='jet')
                    plt.colorbar()
                    plt.title("Cumulative Reflectivity")
                    plt.savefig(os.path.join(fwDir, 'cf.png'), dpi=300)
                    plt.close()

                    plt.figure(figsize=(10, 8))
                    plt.pcolor(xi, yi, RcaloA, shading='flat', cmap='jet')
                    plt.colorbar()
                    plt.title("Cumulative Rainfall")
                    plt.savefig(os.path.join(fwDir, 'cr.png'), dpi=300)
                    plt.close()


        except Exception as e:
            log.error(f"Exception : {str(e)}")
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