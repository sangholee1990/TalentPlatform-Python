# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import subprocess
import sys
import traceback
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib import font_manager

# 필수 라이브러리
# pip install climate-indices

# 명령 프롬프트 (cmd) 상에서 "process_climate_indices" 수행 테스트 필요
# [root bdwide-dev@/usr/local/anaconda3/envs/py38/bin]# ./process_climate_indices
# 2023-02-06  05:30:07 INFO Start time:    2023-02-06 05:30:07.256184
# usage: process_climate_indices [-h] --index {spi,spei,pnp,scaled,pet,palmers,all} --periodicity
#                                {monthly,daily} [--scales [SCALES [SCALES ...]]]
#                                [--calibration_start_year CALIBRATION_START_YEAR]
#                                [--calibration_end_year CALIBRATION_END_YEAR]
#                                [--netcdf_precip NETCDF_PRECIP] [--var_name_precip VAR_NAME_PRECIP]
#                                [--netcdf_temp NETCDF_TEMP] [--var_name_temp VAR_NAME_TEMP]
#                                [--netcdf_pet NETCDF_PET] [--var_name_pet VAR_NAME_PET]
#                                [--netcdf_awc NETCDF_AWC] [--var_name_awc VAR_NAME_AWC]
#                                --output_file_base OUTPUT_FILE_BASE
#                                [--multiprocessing {single,all_but_one,all}] [--chunksizes {none,input}]
# process_climate_indices: error: the following arguments are required: --index, --periodicity, --output_file_base



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

        # 글꼴 설정
        fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        plt.rcParams['font.family'] = fontName

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

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 SPEI 패키지에서 가뭄 산출물 생산

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
    serviceName = 'LSH0395'

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
                    'srtDate' : '2015-01'
                    , 'endDate' : '2020-12'
                    , 'keyList': ['ACCESS-CM2']

                    # SPEI 생산 시 주기성 설정
                    , 'periodicity': '1 2 3 6 9'
                }

            else:
                # 옵션 설정
                sysOpt = {
                    'srtDate' : '2015-01'
                    , 'endDate' : '2020-12'
                    , 'keyList' : ['ACCESS-CM2']
                    # SPEI 생산 시 주기성 설정
                    , 'periodicity' : '1 2 3 6 9'
                }

            globalVar['inpPath'] = '/DATA/INPUT'
            globalVar['outPath'] = '/DATA/OUTPUT'
            globalVar['figPath'] = '/DATA/FIG'

            # 테스트
            # process_climate_indices --index spei  --periodicity monthly --netcdf_precip '/DATA/OUTPUT/LSH0395/ACCESS-CM4.nc' --var_name_precip 'pr' --netcdf_pet '/DATA/OUTPUT/LSH0395/ACCESS-CM4.nc' --var_name_pet 'eto' --output_file_base '/DATA//OUTPUT/LSH0395/SPEI-TEST' --calibration_start_year 2015 --calibration_end_year 2020 --scales 1 2 3 6 9 --multiprocessing all

            # 샘플 자료
            # process_climate_indices --index spei --periodicity monthly --netcdf_precip /DATA/INPUT/LSH0395/nclimgrid_lowres_prcp.nc --var_name_precip prcp --netcdf_pet /DATA/INPUT/LSH0395/nclimgrid_lowres_pet.nc --var_name_pet pet --output_file_base /DATA/OUTPUT/LSH0395/nclimgrid_lowres --scales 9 18 --calibration_start_year 1951 --calibration_end_year 2010 --multiprocessing all

            # ********************************************************************
            # NetCDF 파일 읽기
            # ********************************************************************
            for i, keyInfo in enumerate(sysOpt['keyList']):
                log.info(f'[CHECK] keyInfo : {keyInfo}')

                # 강수량 파일 검사
                inpFile = '{}/{}/*{}*ssp585_201501-210012_pr.nc'.format(globalVar['inpPath'], serviceName, keyInfo)
                fileList = glob.glob(inpFile)
                if fileList is None or len(fileList) < 1:
                    log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

                # 강수량 파일 읽기
                prData = xr.open_dataset(fileList[0])

                # 증발산 파일 검사
                inpFile = '{}/{}/prevPenData*{}*ssp585_eto.nc'.format(globalVar['inpPath'], serviceName, keyInfo)
                fileList = glob.glob(inpFile)
                if fileList is None or len(fileList) < 1:
                    log.error('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))

                # 증발산 읽기
                etoData = xr.open_dataset(fileList[0])
                etoData = etoData.rename({list(etoData.data_vars)[0] : 'eto'})

                # prDataL1 = prData
                # etoDataL1 = etoData

                prDataL1 = prData.sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
                etoDataL1 = etoData.sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))

                # 시간 일치
                prDataL1['time'] = pd.to_datetime(pd.to_datetime(prDataL1['time']).strftime("%Y-%m-01"), format='%Y-%m-%d')
                etoDataL1['time'] = pd.to_datetime(pd.to_datetime(etoDataL1['time']).strftime("%Y-%m-01"), format='%Y-%m-%d')

                # 위도 기준으로 선형 내삽
                latList = prData['lat'].values
                etoDataL1 = etoDataL1.interp(lat=latList, method='linear')

                # 데이터 병합
                data = xr.merge([prDataL1, etoDataL1], join='inner')

                # 기준 차원 변경
                dataL2 = data[['pr', 'eto']].to_dataframe().reset_index().set_index(['lat', 'lon', 'time']).to_xarray()
                dataL2['pr'].attrs = {'units': 'millimeter'}
                dataL2['eto'].attrs = {'units': 'millimeter'}

                saveFile = '{}/{}/AUX-{}.nc'.format(globalVar['outPath'], serviceName, keyInfo)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL2.to_netcdf(saveFile)
                log.info(f'[CHECK] saveFile : {saveFile}')

                # *****************************************************************************
                # cmd 명령어 수행
                # *****************************************************************************
                # process_climate_indices --index spei  --periodicity monthly --netcdf_precip '/DATA/OUTPUT/LSH0395/ACCESS-CM4.nc' --var_name_precip 'pr' --netcdf_pet '/DATA/OUTPUT/LSH0395/ACCESS-CM4.nc' --var_name_pet 'eto' --output_file_base '/DATA//OUTPUT/LSH0395/SPEI-TEST' --calibration_start_year 2015 --calibration_end_year 2020 --scales 1 2 3 6 9 --multiprocessing all
                timeList = dataL2['time'].values

                # 보정 시작/종료연도
                calibSrtYear = pd.to_datetime(timeList.min()).strftime('%Y')
                calibEndYear = pd.to_datetime(timeList.max()).strftime('%Y')

                # 주기성
                # periodicity = '1 2 3 6 9'
                periodicity = sysOpt['periodicity']

                # SPEI 산출
                cmdProcSpei = f"process_climate_indices --index spei  --periodicity monthly --netcdf_precip '{saveFile}' --var_name_precip 'pr' --netcdf_pet '{saveFile}' --var_name_pet 'eto' --output_file_base '{os.path.dirname(saveFile)}/{calibSrtYear}-{calibEndYear}-{keyInfo}' --calibration_start_year {calibSrtYear} --calibration_end_year {calibEndYear} --scales {periodicity} --multiprocessing all"
                cmd = f"source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate py38 && {cmdProcSpei}"
                log.info(f'[CHECK] cmd : {cmd}')

                res = subprocess.run(cmd, shell=True, executable='/bin/bash')
                if res.returncode != 0: print('[ERROR] cmd : {}'.format(cmd))

                # cmdProcSpei = f"process_climate_indices --index spei  --periodicity monthly --netcdf_precip '{saveFile}' --var_name_precip 'pr' --netcdf_pet '{saveFile}' --var_name_pet 'eto' --output_file_base '{os.path.dirname(saveFile)}/{calibSrtYear}-{calibEndYear}-{keyInfo}' --calibration_start_year {calibSrtYear} --calibration_end_year {calibEndYear} --scales {periodicity} --multiprocessing all"
                # cmd = f"conda activate base & {cmdProcSpei}"
                # log.info(f'[CHECK] cmd : {cmd}')
                #
                # res = subprocess.run(cmd, shell=True, executable='cmd.exe')
                # if res.returncode != 0: print('[ERROR] cmd : {}'.format(cmd))

                # *****************************************************************************
                # SPEI 결과 파일
                # *****************************************************************************
                inpFile = '{}/{}/{}-{}*.nc'.format(globalVar['outPath'], serviceName, 'SPEI', keyInfo)
                fileList = glob.glob(inpFile)

                for fileInfo in fileList:
                    log.info(f'[CHECK] fileInfo : {fileInfo}')

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