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
    # Python을 이용한 대한민국 기상청 레이더 자료처리 및 다양한 자료 저장

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
    serviceName = 'LSH0485'

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

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': '2023-02-09'
                , 'endDate': '2023-02-10'
                , 'invDate': '5T'

                # 수행 목록
                , 'nameList': ['radar']

                # 모델 정보 : 파일 경로, 파일명
                , 'nameInfo': {
                    'radar': {
                        'filePath': '/DATA/INPUT/LSH0485/GDK_230209-10'
                        , 'fileName': 'RDR_GDK_FQC_%Y%m%d%H%M.uf'
                    }
                }
            }

            # ======================================================================================
            # 테스트 파일
            # ======================================================================================
            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

            for dtDateIdx, dtDateInfo in enumerate(dtDateList):
                log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                for nameIdx, nameType in enumerate(sysOpt['nameList']):
                    # log.info(f'[CHECK] nameType : {nameType}')

                    modelInfo = sysOpt['nameInfo'].get(nameType)
                    if modelInfo is None: continue

                    inpFile = '{}/{}'.format(modelInfo['filePath'], modelInfo['fileName'])
                    inpFileDate = dtDateInfo.strftime(inpFile)
                    fileList = sorted(glob.glob(inpFileDate))

                    if fileList is None or len(fileList) < 1:
                        # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                        continue

                    for j, fileInfo in enumerate(fileList):
                        data = pyart.io.read(fileInfo)
                        log.info(f'[CHECK] fileInfo : {fileInfo}')

                        fileNameNoExt = os.path.basename(fileInfo).split('.uf')[0]

                        dataL1 = {}
                        for field in data.fields.keys():
                            log.info(f'[CHECK] field : {field}')
                            dataL1[field] = data.fields[field]

                        dataL1.keys()

                        # data.latitude
                        # data.longitude
                        # data.nrays

                        # data.fields['reflectivity']

                        display = pyart.graph.RadarDisplay(data)
                        display.plot('reflectivity', 0, title='NEXRAD Reflectivity')
                        plt.show()

                        # NetCDf 저장
                        saveNcFile = '{}/{}/{}.nc'.format(globalVar['outPath'], serviceName, fileNameNoExt)
                        os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
                        pyart.io.write_cfradial(saveNcFile, data)
                        log.info('[CHECK] saveNcFile : {}'.format(saveNcFile))

                        # CSV 저장



            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'zdr_dom_all.mat')
            # fileList = sorted(glob.glob(inpFile))
            #
            # if fileList is None or len(fileList) < 1:
            #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #
            # fileInfo = fileList[0]
            # matData = io.loadmat(fileInfo)
            # matData.keys()

            # # ======================================================================================
            # # 파일 검색
            # # ======================================================================================
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*.*')
            # fileList = sorted(glob.glob(inpFile))
            #
            # if fileList is None or len(fileList) < 1:
            #     log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
            #     exit(0)
            #
            # # ======================================================================================
            # # (차등반사도의 방위각 종속성) 특정 사상에 대하여 방위각 방향의 차등반사도 변화를 모니터링
            # # ======================================================================================
            # # ***************************************************
            # # 단일 파일에서 주요 변수 (vnamB) 자료 처리
            # # ***************************************************
            # # 3차원 배열 초기화
            # mrVarT_btA = np.nan * np.ones((len(fileList) * 4, len(fileList) * 1, 360))
            # # fileInfo = fileList[0]
            # for fileIdx, fileInfo in enumerate(fileList):
            #     log.info(f'[CHECK] fileInfo: {fileInfo}')
            #
            #     fileNameNoExt = os.path.basename(fileInfo).split('.')[0]
            #
            #     # uf 레이더 파일 읽기
            #     data = readUfRadarData(fileInfo)
            #
            #     # Reinitialize variables
            #     refFlt = 'no'  # ref filter
            #     refThz = 0  # 10 dBZ
            #     phvFlt = 'no'  # phv filter
            #     phvThz = 0.95  # 0.95 0.65
            #     appPDPelim = 'no'
            #     pdpThz = 15  # 15
            #     ntex = 7  # odd, OPERA 9
            #     mtex = 7
            #     spwFlt = 'no'  # ((yes))
            #     spwThz = 0.1  # 0.4m/s
            #     vnamB = ['ZDR', 'PDP', 'PHV', 'REF']
            #     pco = 0.9925
            #     srtEA = 3
            #     endEA = srtEA
            #
            #     # azm_r = np.transpose(data['arr_azm_rng_elv'][0])
            #     # rng_r = np.transpose(data['arr_azm_rng_elv'][1])
            #     # elv_r = np.transpose(data['arr_azm_rng_elv'][2])
            #     azm_r = data['arr_azm_rng_elv'][0].T
            #     rng_r = data['arr_azm_rng_elv'][1].T
            #     elv_r = data['arr_azm_rng_elv'][2].T
            #
            #     Tang = data['fix_ang']
            #     Tang[Tang > 180] -= 360
            #
            #     # if datDR == ':\\Data190\\' or datDR == ':\\Data191\\':
            #     #     didxs = data['arr_etc'][4].astype(int)
            #     #     didxe = data['arr_etc'][5].astype(int)
            #     # else:
            #     didxs = data['arr_etc'][2].astype(int)
            #     didxe = data['arr_etc'][3].astype(int)
            #
            #     # set '0' to '1'
            #     didxs += 1
            #     didxe += 1
            #     didX2 = np.vstack((didxs, didxe))
            #
            #     Fang = elv_r[didxs].T
            #     log.info(f'[CHECK] Fang : {Fang}')
            #
            #     if data['arr_prt_prm_vel'][0][0] == data['arr_prt_prm_vel'][0][-1]:
            #         Tprf = 'sing'
            #     else:
            #         Tprf = 'dual'
            #
            #     bw = data['arr_lat_lon_alt_bwh'][3]
            #
            #     rVarI_bA = {}
            #     # mrVarT_btA = (ip_max, j_max, ta_len)
            #     for ip, vnam_b in enumerate(vnamB):
            #         vnam_b = vnam_b.lower()
            #         log.info(f'[CHECK] vnam_b : {vnam_b}')
            #
            #         # for i in range(srtEA - 1, endEA):
            #         for i in range(srtEA, endEA + 1):
            #             # Arng = np.arange(didxs[i], didxe[i] + 1)
            #             Arng = range(didxs[i - 1], didxe[i - 1])
            #             # log.info(f'[CHECK] Arng : {Arng}')
            #
            #             # 아래에 있는 getVarInfo191013()는 MATLAB 코드에 정의된 함수로,
            #             # 해당 파이썬 버전이 필요하며 이는 상황에 맞게 정의해야 합니다.
            #             rVar_b, rVarI_b = getVarInfo191013(data, vnam_b)
            #             rVarI_bA[ip] = rVarI_b
            #
            #             # rVarf_R = np.transpose(data['arr_ref'])
            #             # rVarc_R = np.transpose(data['arr_phv'])
            #             # rVarp_R = np.transpose(data['arr_pdp'])
            #             rVarf_R = data['arr_ref'].T
            #             rVarc_R = data['arr_phv'].T
            #             rVarp_R = data['arr_pdp'].T
            #
            #             rVar_bT = rVar_b[:, Arng]
            #             rVarf = rVarf_R[:, Arng]
            #             rVarc = rVarc_R[:, Arng]
            #             rVarp = rVarp_R[:, Arng]
            #
            #             rVarT_b, vidx_b = prepcss_new(refFlt, phvFlt, appPDPelim, refThz, phvThz, pdpThz, ntex, mtex, rVarf, rVarc, rVarp, rVar_bT)
            #
            #             # xr.DataArray(rVarT_b).plot()
            #             # # plt.show()
            #             #
            #             # xr.DataArray(vidx_b).plot()
            #             # # plt.show()
            #
            #             rVarT_bt = rVarT_b
            #
            #             if vnam_b == 'zdr':
            #                 rVarT_bt[np.logical_or(rVarT_bt < -10, rVarT_bt > 10)] = np.nan
            #             elif vnam_b == 'pdp':
            #                 rVarT_bt[np.logical_or(rVarT_bt < -300, rVarT_bt > 300)] = np.nan
            #             elif vnam_b == 'phv':
            #                 rVarT_bt[np.logical_or(rVarT_bt < 0, rVarT_bt > 1)] = np.nan
            #             elif vnam_b == 'ref':
            #                 rVarT_bt[np.logical_or(rVarT_bt < -100, rVarT_bt > 100)] = np.nan
            #
            #             # xr.DataArray(rVarT_bt).plot()
            #             # # plt.show()
            #
            #             # mrVarT_bt = np.nanmean(rVarT_bt)
            #             mrVarT_bt = np.nanmean(rVarT_bt, axis=0)
            #             # mrVarT_bt = np.convolve(mrVarT_bt, np.ones((3,)) / 3, mode='same')
            #             mrVarT_bt = pd.Series(mrVarT_bt).rolling(window=3).mean()
            #
            #             ta = np.nan * np.ones((360,))
            #             # ta = np.nan * np.ones((,360))
            #             ta[:len(mrVarT_bt)] = mrVarT_bt
            #
            #             # mrVarT_btA[ip, i, :] = ta
            #             # mrVarT_btA = np.vstack((mrVarT_btA, ta))
            #             # mrVarT_btA = np.vstack((mrVarT_btA, ta))
            #             mrVarT_btA[ip, i, :] = ta
            #
            #             # xr.DataArray(ta).plot()
            #             # xr.DataArray(mrVarT_btA).plot()
            #             # # plt.show()
            #
            #             # fwDir = '{}/{}/{}/'.format(globalVar['figPath'], serviceName, fileNameNoExt)
            #             fwDir = '{}/{}/'.format(globalVar['figPath'], serviceName)
            #             os.makedirs(os.path.dirname(fwDir), exist_ok=True)
            #             if vnam_b in ['zdr', 'pdp']:
            #                 dspCycl(fwDir, vnam_b, None, ta, 'fix', 'each', fileIdx, rVarI_bA[ip])
            #
            # # ***************************************************
            # # 전체 파일 목록에서 주요 변수 (vnamB) 자료 처리
            # # ***************************************************
            # for ipi, vnam_b in enumerate(vnamB):
            #     log.info(f'[CHECK] vnam_b : {vnam_b}')
            #
            #     # mrVarT_btAs = np.squeeze(mrVarT_btA[ipi, :])
            #     mrVarT_btAs = np.squeeze(mrVarT_btA[ipi, :, :])
            #
            #     # fwDir = '{}/{}/{}/'.format(globalVar['figPath'], serviceName, fileNameNoExt)
            #     fwDir = '{}/{}/'.format(globalVar['figPath'], serviceName)
            #     dspCycl(fwDir, vnam_b, mrVarT_btA, mrVarT_btAs, 'fix', 'total', None, rVarI_bA[ipi])
            #
            # # ======================================================================================
            # # 편파 매개변수의 측정 오류 추정치를 이용하여 레이더 하드웨어 및 데이터 수집 시스템의 품질 평가
            # # ======================================================================================
            # # ***************************************************
            # # 단일 파일에서 주요 변수 (vnamA) 자료 처리
            # # ***************************************************
            # mrVarT_btA = np.empty((0, 360))
            # # fileInfo = fileList[0]
            # for fileIdx, fileInfo in enumerate(fileList):
            #     log.info(f'[CHECK] fileInfo: {fileInfo}')
            #
            #     # if (fileIdx > 10): continue
            #
            #     # uf 레이더 파일 읽기
            #     dictData = readUfRadarData(fileInfo)
            #
            #     # low (ref)
            #     refFlt = 'no'  # ref filter
            #     refThz = 0  # 10 dBZ
            #
            #     # low (phv)
            #     phvFlt = 'no'  # phv filter
            #     phvThz = 0.95  # 0.95 0.65
            #
            #     # large Tex(pdp)
            #     appPDPelim = 'no'
            #     pdpThz = 15  # 15
            #
            #     ntex = 7  # odd, OPERA 9
            #     mtex = 7
            #
            #     # low (spw)
            #     spwFlt = 'no'  # ((yes))
            #     spwThz = 0.1  # 0.4m/s
            #
            #     vnamA = ['SPW', 'SPW']  # m/s
            #     vnamB = ['ZDR', 'PDP']  # dB
            #     # vnamA=['SPW']   # m/s
            #     # vnamB=['ZDR']   # dB
            #     pco = 0.9925
            #     srtEA = 2
            #     endEA = 2
            #
            #     data = dictData
            #
            #     azm_r = data['arr_azm_rng_elv'][0]  # 1080x1 (360x3)
            #     rng_r = data['arr_azm_rng_elv'][1]  # 1196x1
            #     elv_r = data['arr_azm_rng_elv'][2]  # 1080x1 (360x3)
            #     Tang = data['fix_ang']
            #     Tang[Tang > 180] = Tang[Tang > 180] - 360
            #     didxs = data['arr_etc'][2] + 1
            #     didxe = data['arr_etc'][3] + 1
            #     didX2 = [didxs, didxe]
            #     Fang = elv_r[didxs]
            #
            #     if data['arr_prt_prm_vel'][0][0] == data['arr_prt_prm_vel'][0][-1]:
            #         Tprf = 'sing'
            #     else:
            #         Tprf = 'dual'
            #
            #     bw = data['arr_lat_lon_alt_bwh'][3]
            #
            #     # << dual para >>
            #     for ip in range(len(vnamA)):
            #         vnam_a = vnamA[ip]
            #         vnam_b = vnamB[ip]
            #
            #         for i in range(srtEA, endEA + 1):
            #             Arng = np.arange(didxs[i] - 1, didxe[i])
            #
            #             rVar_a, rVarI_a = getVarInfo191013(data, vnam_a)  # need translation of function getVarInfo191013
            #             rVar_b, rVarI_b = getVarInfo191013(data, vnam_b)  # need translation of function getVarInfo191013
            #
            #             rVarf_R = np.transpose(data['arr_ref'])
            #             rVarc_R = np.transpose(data['arr_phv'])
            #             rVarp_R = np.transpose(data['arr_pdp'])
            #
            #             rVar_aT = rVar_a[:, Arng]
            #             rVar_bT = rVar_b[:, Arng]
            #
            #             rVarf = rVarf_R[:, Arng]
            #             rVarc = rVarc_R[:, Arng]
            #             rVarp = rVarp_R[:, Arng]
            #
            #             if ip == 1:
            #                 rVarct0 = rVarc
            #                 rVarct = np.where((rVarc < 0) | (rVarc > 1), np.nan, rVarc)
            #
            #                 rVarft0 = rVarf
            #                 rVarft = np.where((rVarf < -100) | (rVarf > 100), np.nan, rVarf)
            #
            #                 rVarpt0 = rVarp
            #                 rVarpt = np.where((rVarp < -300) | (rVarp > 300), np.nan, rVarp)
            #
            #                 rVardt0 = rVar_bT
            #                 rVardt = np.where((rVar_bT < -10) | (rVar_bT > 10), np.nan, rVar_bT)
            #
            #             rVarT_a, vidx_a = prepcss_new(refFlt, phvFlt, appPDPelim, refThz, phvThz, pdpThz, ntex, mtex, rVarf, rVarc, rVarp, rVar_aT)
            #             rVarT_b, vidx_b = prepcss_new(refFlt, phvFlt, appPDPelim, refThz, phvThz, pdpThz, ntex, mtex, rVarf, rVarc, rVarp, rVar_bT)
            #
            #             if vnam_b == 'ZDR':
            #                 rVarT_b = np.where(rVarc < pco, np.nan, rVarT_b)
            #
            #             if ip == 1:
            #                 rVarT_bt = rVarT_b
            #                 rVarT_bt = np.where((rVarT_bt < -10) | (rVarT_bt > 10), np.nan, rVarT_bt)
            #
            #                 # mrVarT_bt = np.nanmean(rVarT_bt)
            #                 mrVarT_bt = np.nanmean(rVarT_bt, axis=0)
            #
            #                 # mrVarT_bt = np.convolve(mrVarT_bt, np.ones(3), 'valid') / 3
            #                 mrVarT_bt = pd.Series(mrVarT_bt).rolling(window=3).mean()
            #
            #                 ta = np.full(360, np.nan)
            #                 ta[:len(mrVarT_bt)] = mrVarT_bt
            #
            #                 mrVarT_btA = np.vstack((mrVarT_btA, ta))
            #
            #                 # std
            #                 texRng = np.ones((ntex, mtex))
            #                 rVarT_b[np.isnan(rVarT_b)] = 0
            #                 rVarT_b = generic_filter(rVarT_b, np.nanstd, footprint=texRng)
            #
            #                 atyp = data['str_typ']
            #
            #                 # if RdrNam in ['BSL', 'SBS']:
            #                 #     TLE_fname = fname[8:-4]
            #                 # else:
            #                 #     TLE_fname = fname[:-4]
            #                 TLE_fname = os.path.basename(fileInfo)[8:-4]
            #
            #                 if Tang[i] < 0:
            #                     TLE_Tang = '-' + "{:.1f}".format(abs(Tang[i]))
            #                 else:
            #                     TLE_Tang = '+' + "{:.1f}".format(Tang[i])
            #
            #                 TLE = [atyp, '(' + TLE_Tang[0:2] + ',' + TLE_Tang[3] + 'deg' + ')_' + Tprf + '_' + vnam_a + '-' + vnam_b + '_' + TLE_fname]
            #
            #                 vidxAll = np.logical_or(vidx_a, vidx_b)
            #
            #                 # Apar = {rVarT_a, rVarT_b, 0, 0}
            #                 Apar = [rVarT_a, rVarT_b, 0, 0]
            #
            #                 fwDir = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, TLE[0][0] + TLE[1])
            #                 fPar, fPar2, mPar = ccpltXY_191013_n(Apar[0], Apar[1], Apar[2], Apar[3],
            #                                                      vnam_a, vnam_b,
            #                                                      rVarI_a, rVarI_b, Tang[i],
            #                                                      fwDir, TLE,
            #                                                      vidxAll,
            #                                                      refFlt, phvFlt, appPDPelim,
            #                                                      refThz, phvThz, pdpThz, ntex, mtex)
            #
            # # ***************************************************
            # # 전체 파일 목록에서 주요 변수 (vnamA) 자료 처리
            # # ***************************************************
            # mvStep = 90
            # fwDir = '{}/{}/'.format(globalVar['figPath'], serviceName)
            #
            # # Compute mean along the specified axis
            # mrVarT_btAm = np.nanmean(mrVarT_btA, axis=0)
            #
            # # Compute moving average
            # mrVarT_btAmMV = pd.Series(mrVarT_btAm).rolling(window=mvStep).mean().values
            #
            # plt.figure()
            # plt.plot(mrVarT_btA.T)
            # plt.plot(mrVarT_btAm, 'k-', linewidth=3)
            # plt.plot(mrVarT_btAmMV, 'w-', linewidth=2)
            # plt.xlim([0, 360])
            # plt.xticks(np.arange(0, 361, 30))
            # plt.grid(True)
            # saveImg = f"{fwDir}zdr_dom_all.png"
            # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            # plt.savefig(saveImg, dpi=600)
            # log.info(f'[CHECK] saveImg : {saveImg}')
            # # plt.show()
            # plt.close()
            #
            # plt.figure()
            # plt.plot(mrVarT_btAm, 'k-', linewidth=3)
            # plt.plot(mrVarT_btAmMV, 'b-', linewidth=2)
            # plt.xlim([0, 360])
            # plt.xticks(np.arange(0, 361, 30))
            # plt.grid(True)
            # saveImg = f"{fwDir}zdr_dom_all2.png"
            # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            # plt.savefig(saveImg, dpi=600)
            # log.info(f'[CHECK] saveImg : {saveImg}')
            # # plt.show()
            # plt.close()
            #
            # mrVarT_btAmDT = mrVarT_btAm - mrVarT_btAmMV
            #
            # plt.figure()
            # plt.plot(mrVarT_btAmDT, 'r-', linewidth=2)
            # plt.xlim([0, 360])
            # # plt.ylim([-0.2, 0.2])
            # plt.xticks(np.arange(0, 361, 30))
            # plt.grid(True)
            # saveImg = f"{fwDir}zdr_dom_all3.png"
            # os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            # plt.savefig(saveImg, dpi=600)
            # log.info(f'[CHECK] saveImg : {saveImg}')
            # # plt.show()
            # plt.close()

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
