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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
# from global_land_mask import globe
import cftime
import pyeto
import xclim
import calendar

import xclim.core.units
from pandas.tseries.offsets import MonthEnd
# from pyeto import fao

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

def sol_dec(day_of_year):
    return 0.409 * np.sin(((2.0 * np.pi / 365.0) * day_of_year - 1.39))

def sunset_hour_angle(latitude, sol_dec):
    cos_sha = -np.tan(latitude) * np.tan(sol_dec)
    return np.arccos(np.minimum(np.maximum(cos_sha, -1.0), 1.0))

def inv_rel_dist_earth_sun(day_of_year):
    return 1 + (0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year))

def et_rad(latitude, sol_dec, sha, ird):
    #: Solar constant [ MJ m-2 min-1]
    SOLAR_CONSTANT = 0.0820

    tmp1 = (24.0 * 60.0) / np.pi
    tmp2 = sha * np.sin(latitude) * np.sin(sol_dec)
    tmp3 = np.cos(latitude) * np.cos(sol_dec) * np.sin(sha)
    return tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)

def daylight_hours(sha):
    return (24.0 / np.pi) * sha

def svp_from_t(t):
    return 0.6108 * np.exp((17.27 * t) / (t + 237.3))


def delta_svp(t):
    tmp = 4098 * (0.6108 * np.exp((17.27 * t) / (t + 237.3)))
    return tmp / np.power((t + 237.3), 2)


def net_out_lw_rad(tmin, tmax, sol_rad, cs_rad, avp):
    # Stefan Boltzmann constant [MJ K-4 m-2 day-1]
    STEFAN_BOLTZMANN_CONSTANT = 0.000000004903

    tmp1 = (STEFAN_BOLTZMANN_CONSTANT *
            ((np.power(tmax, 4) + np.power(tmin, 4)) / 2))
    tmp2 = (0.34 - (0.14 * np.sqrt(avp)))
    tmp3 = 1.35 * (sol_rad / cs_rad) - 0.35
    return tmp1 * tmp2 * tmp3


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python 이용한 NetCDF 파일 처리 및 3종 증발산량 (Penman, Hargreaves, Thornthwaite) 계산

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
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PV'

    prjName = 'test'
    serviceName = 'LSH0336'

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
                    'srtDate': '2020-09-01'
                    , 'endDate': '2020-09-03'

                    # 경도 최소/최대/간격
                    , 'lonMin': 0
                    , 'lonMax': 360
                    , 'lonInv': 0.5

                    # 위도 최소/최대/간격
                    , 'latMin': -90
                    , 'latMax': 90
                    , 'latInv': 0.5
                }
            else:
                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': globalVar['srtDate']
                    # , 'endDate': globalVar['endDate']
                }

            # globalVar['outPath'] = 'F:/Global Temp/aski'

            modelList = ['MRI-ESM2-0']
            for i, modelInfo in enumerate(modelList):
                log.info("[CHECK] modelInfo : {}".format(modelInfo))

                inpFile = '{}/{}/{} ssp585 2015-2100_*.nc'.format(globalVar['inpPath'], serviceName, modelInfo)
                fileList = sorted(glob.glob(inpFile))
                log.info("[CHECK] fileList : {}".format(fileList))

                dsData = xr.open_mfdataset(fileList)
                dsData = dsData.sel(lon = slice(120, 150), time = slice('2015-01', '2015-12'))
                # dsData = dsData.sel(time = slice('2015-01', '2020-12'))

                # 월별 시간 변환
                dsData['time'] = pd.to_datetime(pd.to_datetime(dsData['time'].values).strftime("%Y-%m"), format='%Y-%m')

                # 단위 설정
                # 켈빈 to 섭씨
                dsData['tasCel'] = dsData['tas'] - 273.15
                dsData['tasminCel'] = dsData['tasmin'] - 273.15
                dsData['tasmaxCel'] = dsData['tasmax'] - 273.15

                dsData['tasminCel'].attrs['units'] = 'degC'
                dsData['tasmaxCel'].attrs['units'] = 'degC'
                dsData['tasCel'].attrs['units'] = 'degC'

                # 단위 환산을 위한 매월 마지막 날 계산
                lon1D = dsData['lon'].values
                lat1D = dsData['lat'].values
                time1D = dsData['time'].values

                timeEndMonth = []
                timeYear = dsData['time.year'].values
                timeMonth = dsData['time.month'].values

                for i in range(0, len(timeYear)):
                    timeEndMonth.append(calendar.monthrange(timeYear[i], timeMonth[i])[1])

                latRad1D =  pyeto.deg2rad(dsData['lat'])
                dayOfYear1D = dsData['time.dayofyear']

                latRad3D = np.tile(np.transpose(np.tile(latRad1D, (len(lon1D), 1))), (len(time1D), 1, 1))
                dayOfYear3D = np.transpose(np.tile(dayOfYear1D, (len(lon1D), len(lat1D), 1)))

                timeEndMonth3D = np.transpose(np.tile(timeEndMonth, (len(lon1D), len(lat1D), 1)))

                tmpData = xr.Dataset(
                    {
                        'timeEndMonth': (('time', 'lat', 'lon'), (timeEndMonth3D).reshape(len(time1D), len(lat1D), len(lon1D)))
                        , 'latRad': (('time', 'lat', 'lon'), (latRad3D).reshape(len(time1D), len(lat1D), len(lon1D)))
                        , 'dayOfYear': (('time', 'lat', 'lon'), (dayOfYear3D).reshape(len(time1D), len(lat1D), len(lon1D)))
                    }
                    , coords={
                        'lat': lat1D
                        , 'lon': lon1D
                        , 'time': time1D
                    }
                )

                # ********************************************************************************************
                # FAO-56 Penman-Monteith 방법
                # ********************************************************************************************
                # https://pyeto.readthedocs.io/en/latest/fao56_penman_monteith.html 매뉴얼 참조
                # 1 W/m2 = 1 J/m2를 기준으로 MJ/day 변환
                dsData['rsdsMJ'] = dsData['rsds'] * 86400 / (10 ** 6)

                dsData['tasKel'] = dsData['tas']
                dsData['tasminKel'] = dsData['tasmin']
                dsData['tasmaxKel'] = dsData['tasmax']
                dsData['tasKel'].attrs['units'] = 'degK'
                dsData['tasminKel'].attrs['units'] = 'degK'
                dsData['tasmaxKel'].attrs['units'] = 'degK'

                # 섭씨 to 켈빈
                # dsData['tasKel'] = dsData['tas'] + 273.15
                # dsData['tasminKel'] = dsData['tasmin'] + 273.15
                # dsData['tasmaxKel'] = dsData['tasmax'] + 273.15

                dsData['svp'] = svp_from_t(dsData['tasCel'])
                dsData['svpMax'] = svp_from_t(dsData['tasmaxCel'])
                dsData['svpMin'] = svp_from_t(dsData['tasminCel'])

                tmpData['solDec'] = sol_dec(tmpData['dayOfYear'])
                tmpData['sha'] = sunset_hour_angle(tmpData['latRad'], tmpData['solDec'])
                tmpData['dayLightHour'] = daylight_hours(tmpData['latRad'])
                tmpData['ird'] = inv_rel_dist_earth_sun(tmpData['dayOfYear'])
                tmpData['etRad'] = et_rad(tmpData['latRad'], tmpData['solDec'], tmpData['sha'], tmpData['ird'])
                tmpData['csRad'] = pyeto.cs_rad(altitude=1.5, et_rad=tmpData['etRad'])
                dsData['deltaSvp'] = delta_svp(dsData['tasCel'])

                # 대기 온도 1.5 m 가정
                psy = pyeto.psy_const(atmos_pres=pyeto.atm_pressure(altitude=15))

                dsData['avp'] = pyeto.avp_from_rhmin_rhmax(dsData['svpMax'], dsData['svpMin'], dsData['hurs'].min(), dsData['hurs'].max())
                niSwRad = pyeto.net_in_sol_rad(dsData['rsdsMJ'], albedo=0.23)
                niLwRad = net_out_lw_rad(dsData['tasminKel'], dsData['tasmaxKel'], dsData['rsdsMJ'], tmpData['csRad'], dsData['avp'])
                dsData['net_rad'] = pyeto.net_rad(ni_sw_rad=niSwRad, no_lw_rad=niLwRad)

                faoRes = pyeto.fao56_penman_monteith(dsData['net_rad'], dsData['tasKel'], dsData['sfcWind'], dsData['svp'], dsData['avp'], dsData['deltaSvp'], psy, shf=0)

                # ********************************************************************************************
                # Hargreaves 방법
                # ********************************************************************************************
                # https://xclim.readthedocs.io/en/stable/indicators_api.html 매뉴얼 참조
                harRes = xclim.indices.potential_evapotranspiration(dsData['tasminCel'], dsData['tasmaxCel'], dsData['tasCel'], dsData['lat'], method='hargreaves85')

                # 1 kg/m2/s = 86400 mm/day를 기준으로 mm/month 변환
                harResL1 = harRes * 86400.0 * tmpData['timeEndMonth']

                # https://pyeto.readthedocs.io/en/latest/thornthwaite.html 매뉴얼 참조
                # harRes = pyeto.hargreaves(dsData['tasminCel'], dsData['tasmaxCel'], dsData['tasCel'], tmpData['etRad'])
                # harResL1 = harRes

                # ********************************************************************************************
                # Thornthwaite 방법
                # ********************************************************************************************
                # https://xclim.readthedocs.io/en/stable/indicators_api.html 매뉴얼 참조
                thwRes = xclim.indices.potential_evapotranspiration(dsData['tasminCel'], dsData['tasmaxCel'], dsData['tasCel'], dsData['lat'], method ='thornthwaite48')

                # 1 kg/m2/s = 86400 mm/day를 기준으로 mm/month 변환
                thwResL1 = thwRes * 86400.0 * tmpData['timeEndMonth']

                # https://pyeto.readthedocs.io/en/latest/thornthwaite.html 매뉴얼 참조
                # thwRes = pyeto.thornthwaite(dsData['tasCel'], tmpData['dayLightHour'])
                # thwResL1 = thwResL1

                # ********************************************************************************************
                # 데이터 병합
                # ********************************************************************************************
                data = xr.Dataset(
                    {
                        'hargreaves': (('time', 'lat', 'lon'), (harResL1.values).reshape(len(time1D), len(lat1D), len(lon1D)))
                        , 'thornthwaite': (('time', 'lat', 'lon'), (thwResL1.values).reshape(len(time1D), len(lat1D), len(lon1D)))
                        , 'penman-monteith': (('time', 'lat', 'lon'), (faoRes.values).reshape(len(time1D), len(lat1D), len(lon1D)))
                    }
                    , coords={
                        'lat': lat1D
                        , 'lon': lon1D
                        , 'time': time1D
                    }
                )

                dataL1 = xr.merge( [ data, dsData ] )

                # # NetCDF 파일 저장
                saveFile = '{}/{}/{}_eto.nc'.format(globalVar['outPath'], serviceName, modelInfo)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL1.to_netcdf(saveFile)
                log.info('[CHECK] saveFile : {}'.format(saveFile))

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
