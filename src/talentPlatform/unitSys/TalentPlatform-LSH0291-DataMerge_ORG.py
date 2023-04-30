# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import logging.handlers
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
from mpl_toolkits.basemap import Basemap

import numpy as np
import math
from scipy.stats import t
import numpy as np, scipy.stats as st

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

    saveLogFile = "{}/{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
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

            log.info("[CHECK] {} / val : {}".format(key, val))

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
    # Python을 이용한 한국 월별 수출액과 코스피 종간 간의 회귀분석

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
        contextPath = os.getcwd() if env in 'local' else '/home/dxinyu/TEST'

    prjName = 'test'
    serviceName = 'LSH0291'

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

            # bash RunShell-Python.sh "TalentPlatform-LSH0291-Analy.py" "2018-01-01" "2021-01-01"
            # nohup bash RunShell-Python.sh "TalentPlatform-LSH0291-Analy.py" "2018-01-01" "2021-01-01" &
            # /home/dxinyu/TEST/OUTPUT
            # /home/dxinyu/TEST/OUTPUT

            # python3 "/home/dxinyu/TEST/TalentPlatform-LSH0291-DataMerge.py" --inpPath "/home/dxinyu/TEST/OUTPUT" --outPath "/home/dxinyu/TEST/OUTPUT"

            if (platform.system() == 'Windows'):

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    'srtDate': '2018-01-01'
                    , 'endDate': '2021-01-01'
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작/종료 시간
                    # 'srtDate': globalVar['srtDate']
                    # , 'endDate': globalVar['endDate']
                }

            # inpPosFile = '{}/{}'.format(globalVar['cfgPath'], 'nationInfo/GEO_NATION_CONT_INFO.csv')
            # posData = pd.read_csv(inpPosFile)
            # posDataL1 = posData[['lon', 'lat', 'landSea', 'cont']]

            # *******************************************************
            # 육/해상, 대륙별 배출량
            # *******************************************************
            keyList = ['total', 'Power', 'Industry', 'Residential', 'GroundTransportation', 'InternationalAviation', 'InternationalShipping', 'DomesticAviation']

            for i, keyInfo in enumerate(keyList):

                # dtYear = 2019
                for dtYear in range(2018, 2022):
                    log.info("[CHECK] keyInfo : {}".format(keyInfo))
                    log.info("[CHECK] dtYear : {}".format(dtYear))

                    inpFilePattern = '{}_{}_{}*.nc'.format(serviceName, keyInfo, dtYear)
                    inpFile = '{}/{}'.format(globalVar['outPath'], inpFilePattern)
                    log.info("[CHECK] inpFile : {}".format(inpFile))

                    fileList = sorted(glob.glob(inpFile))
                    log.info('[CHECK] fileList : {}'.format(fileList))
                    if (len(fileList) < 1): continue

                    dsData = xr.open_mfdataset(fileList)
                    log.info('[CHECK] dsData : {}'.format(dsData))

                    # saveFile = '{}/{}_{}_{}-{}.nc'.format(globalVar['outPath'], serviceName, keyInfo, min(dsData['time']).dt.strftime('%Y%m%d').values, max(dsData['time']).dt.strftime('%Y%m%d').values)
                    # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    # dsData.to_netcdf(saveFile)
                    # log.info('[CHECK] saveFile : {}'.format(saveFile))

                    # dsDataL1 = dsData.groupby('time.year').sum('time').sel(year=dtYear)

                    time1D = dsData['time'].values
                    lon1D = dsData['lon'].values
                    lat1D = dsData['lat'].values
                    lon2D, lat2D = np.meshgrid(lon1D, lat1D)

                    # dsDataL2 = dsData.stack(geo=['lon', "lat"])

                    dsData = xr.where((dsData == 0), np.nan, dsData)

                    # 위도/경도
                    cnt2D = dsData.count(['time'])
                    mean2D = dsData.mean(['time'])
                    # sd2D = dsData.std(['time'], ddof=1)
                    sd2D = dsData.std(['time'])
                    sum2D = dsData.sum(['time'])




                    # 2022-02-26 18:15:40,474 [test | 5 | <ipython-input-134-d61e27224a5e>] [INFO ] [CHECK] cntVal : 22.176065277777777
                    # 2022-02-26 18:15:40,850 [test | 6 | <ipython-input-134-d61e27224a5e>] [INFO ] [CHECK] sumVal : 11404.05326278668
                    # 2022-02-26 18:15:40,850 [test | 7 | <ipython-input-134-d61e27224a5e>] [INFO ] [CHECK] meanVal : 514.2477934804217
                    # 2022-02-26 18:15:40,850 [test | 8 | <ipython-input-134-d61e27224a5e>] [INFO ] [CHECK] sdVal : 78.64721413830199
                    cntVal = np.nanmean(cnt2D['emission'])
                    sumVal = np.nanmean(sum2D['emission'])
                    meanVal = np.nanmean(mean2D['emission'])
                    sdVal = np.nanmean(sd2D['emission'])
                    log.info('[CHECK] cntVal : {}'.format(cntVal))
                    log.info('[CHECK] sumVal : {}'.format(sumVal))
                    log.info('[CHECK] meanVal : {}'.format(meanVal))
                    log.info('[CHECK] sdVal : {}'.format(sdVal))

                    # dd = dsData['emission'].values


                    import numpy as np
                    import xarray as xr
                    from scipy import stats

                    # def func(x, axis):
                    #     mode, count = np.apply_along_axis(stats.mode, axis, x)
                    #     return mode.squeeze()


                    # df = cnt2D
                    df = len(time1D)
                    tVal = t(df)
                    # tVal = t( cnt2D['emission'].values)
                    t025 = tVal.ppf(0.975)

                    # 신뢰구간 95%  불확실성 범위 (2)
                    # leftConf = mean2D - t025 * (sd2D / np.sqrt(df))
                    # rightConf = mean2D + t025 * (sd2D / np.sqrt(df))

                    # 확장 불확도
                    up = t025 * (sd2D / np.sqrt(df))

                    # 상대 불확도 (%)
                    urp = (up * 100) / mean2D

                    dsDataL2 = xr.Dataset(
                        {
                            'mean': (('lat', 'lon'), (mean2D['emission'].values).reshape(len(lat1D), len(lon1D)))
                            , 'cnt': (('lat', 'lon'), (cnt2D['emission'].values).reshape(len(lat1D), len(lon1D)))
                            , 'sd': (('lat', 'lon'), (sd2D['emission'].values).reshape(len(lat1D), len(lon1D)))
                            , 'sum': (('lat', 'lon'), (sum2D['emission'].values).reshape(len(lat1D), len(lon1D)))
                            , 'up': (('lat', 'lon'), (up['emission'].values).reshape(len(lat1D), len(lon1D)))
                            , 'urp': (('lat', 'lon'), (urp['emission'].values).reshape(len(lat1D), len(lon1D)))
                        }
                        , coords={
                            'lat': lat1D
                            , 'lon': lon1D
                        }
                    )

                    saveFile = '{}/{}_{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, keyInfo, 'uncertainty', dtYear)
                    # if (os.path.exists(saveFile)): continue

                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)

                    dsDataL2.to_netcdf(saveFile)
                    log.info('[CHECK] saveFile : {}'.format(saveFile))


                    #
                    #
                    #
                    #
                    # # meanVal = np.nanmean(val1d)
                    # # sdVal = np.nanstd(val1d, ddof=1)
                    #
                    #
                    #
                    #
                    #
                    # a = dsData
                    # dim = 'time'
                    # n = a[dim].shape[0]
                    # df = n - 1
                    # a_mean = a.mean(dim)
                    # d = a_mean - popmean
                    # v = a.var(dim, ddof=1)
                    # denom = xrf.sqrt(v / float(n))
                    #
                    # t = d / denom
                    # prob = stats.distributions.t.sf(xrf.fabs(t), df) * 2
                    # prob_xa = xr.DataArray(prob, coords=a_mean.coords)
                    #
                    #
                    # def remove_time_mean(x):
                    #     return x - x.mean(dim='time')
                    #
                    # gg =  dsData.stack(lonlat=('lat', 'lon'))
                    # gg =  dsData.stack(lonlat=('lat', 'lon'))
                    # dd = gg.groupby('lonlat')
                    # print(dd)
                    # ss = dd.apply(remove_time_mean)
                    #
                    #
                    #
                    #
                    # gb = dsData.groupby('time')
                    # gb
                    #
                    # ss = gb['lon'].apply(np.mean)

                    # gg4 = dsData.stack(lonlat=('lat', 'lon')).groupby('lonlat').sum()
                    # log.info('[CHECK] gg4 : {}'.format(gg4))

                    #
                    # dd = dsData.groupby('')
                    #
                    # # log.info('[CHECK] sum2D : {}'.format(sum2D))
                    #
                    # res = xr.apply_ufunc(special_mean, dsData)


                    # def fn(x):
                    #     return np.mean(x)
                    #
                    # sum2D = dsData.fn(['time'])
                    # sum2D = xr.apply_ufunc(fn)
                    # log.info('[CHECK] sum2D : {}'.format(sum2D))
                    #
                    #
                    #
                    #
                    # def standardize(x):
                    #     return (x - x.mean()) / x.std()
                    #
                    # # sum2D = dsData.len(['time'])
                    #
                    # # gg = dsDataL2.groupby('ge').apply(sum)
                    # # gg2 = dsDataL2.groupby('geo').apply(standardize)
                    # # gg3 = dsDataL2.groupby(['lon', 'lat']).apply(fn)
                    # # gg3 = dsData.stack(lonlat=('lat', 'lon')).groupby('lonlat').apply(fn)
                    #
                    # gg4 = dsData.stack(lonlat=('lat', 'lon')).groupby('lonlat').sum()
                    # log.info('[CHECK] gg4 : {}'.format(gg4))
                    # gg3 = dsDataL2.groupby(['lon', 'lat']).size()

                    # def uncertane(val1d):
                    #
                    #     # t 분포를 통한 신뢰구간 추정 (신뢰구간 95%)
                    #     # 자유도
                    #     df = len(val1d[~np.isnan(val1d)])
                    #     tVal = t(df)
                    #     # t025 = ci95(val1d)
                    #     # t025 = t025[1]
                    #     t025 = tVal.ppf(0.975)
                    #     meanVal = np.nanmean(val1d)
                    #     sdVal = np.nanstd(val1d, ddof=1)
                    #
                    #     # 신뢰구간 95%  불확실성 범위 (2)
                    #     leftConf = round(meanVal - t025 * (sdVal / np.sqrt(df)), 2)
                    #     rightConf = round(meanVal + t025 * (sdVal / np.sqrt(df)), 2)
                    #
                    #     # 확장 불확도
                    #     up = t025 * (sdVal / np.sqrt(df))
                    #
                    #     # 상대 불확도 (%)
                    #     urp = (up * 100) / meanVal
                    #
                    #     return up, urp
                    #
                    # def uncertanetotal(urps1d):
                    #     tt = np.sqrt(np.sum(urps1d ** 2))
                    #     return tt
                    #
                    # cc = dsData.reduce(np.sum, axis=1)



                            # def standardize(x):
                    #     return (x - x.mean()) / x.std()
                    #
                    # dd = dsData2.groupby('geo').map(standardize)
                    # mean2D = dsData2.groupby('geo').mean()
                    # sd2D = dsData2.groupby('geo').std()
                    # dsDataL1 = dsData.groupby([lon1D, lat1D]).mean()
                    # log.info("[CHECK] dsDataL1 : {}".format(dsDataL1))

                    # dsDataL1['emission'].plot()
                    # plt.show()

                    # lon1D = dsDataL1['lon'].values
                    # lat1D = dsDataL1['lat'].values
                    # lon2D, lat2D = np.meshgrid(lon1D, lat1D)

                    # val2D = dsDataL1['emission'].values
                    # log.info("[CHECK] dsDataL1 : {}".format(dsDataL1))

                    # val2D = np.where((val2D == 0), np.nan, val2D)

                    # val2D.mean()
                    # previous_time.blue.plot();

                    # aa = dsDataL1['emission'].values
                    #
                    # dsDataL1['emission'].plot()
                    # plt.show()

                    # lon = dsDataL1['lon'].data
                    # lon1D = np.reshape(lon, (1,np.product(lon.shape)))[0]
                    # lon.shape
                    # lat1D = dsDataL1['lat'].data

                    # minVal = 0
                    # maxVal = np.nanmax(np.where((val2D == 0), np.nan, val2D))



                    # 시각화
                    # val2D = sum2D['emission'].values

                    val2D = urp['emission'].values
                    # val2D = np.where((val2D == 0), np.nan, val2D)

                    meanVal = np.nanmean(val2D)
                    log.info('[CHECK] val2D : {}'.format(meanVal))

                    from matplotlib import colors

                    plt.figure(figsize=(10, 8))
                    map = Basemap(projection="cyl", lon_0=0.0, lat_0=0.0, resolution="c")
                    # cs = map.scatter(lon2D, lat2D, c=val2D, s=0.05, norm=colors.LogNorm(), cmap=plt.cm.get_cmap('jet'))
                    cs = map.scatter(lon2D, lat2D, c=val2D, s=0.05, cmap=plt.cm.get_cmap('Spectral_r'), vmin=0, vmax=80)


                    map.drawcoastlines()
                    map.drawmapboundary()
                    map.drawcountries(linewidth=1, linestyle='solid', color='k')
                    map.drawmeridians(range(-180, 180, 30), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 0, 1])
                    map.drawparallels(range(-90, 90, 30), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 0, 0, 0])

                    plt.ylabel(None, fontsize=15, labelpad=35)
                    plt.xlabel(None, fontsize=15, labelpad=20)
                    cbar = map.colorbar(cs, location='right')
                    cbar.set_label(None, fontsize=13)
                    plt.title('', fontsize=15)
                    plt.show()

                    # saveImg = '{}/{}_{}_{}-{}.png'.format(globalVar['figPath'], serviceName, keyInfo, min(dsData['time']).dt.strftime('%Y%m%d').values,  max(dsData['time']).dt.strftime('%Y%m%d').values)
                    # plt.savefig(saveImg, dpi=600, bbox_inches='tight')


                    # plt.show()
                    #
                    # val1d = np.where((val2D == 0), np.nan, val2D)
                    #
                    # # 신뢰구간 95%  불확실성 (1)
                    # # st.t.interval(0.95, len(val2d) - 1, loc=np.mean(val2d), scale=st.sem(val2d))
                    # # st.t.interval(0.90, len(val1d) - 1, loc=np.mean(val1d), scale=st.sem(val1d))
                    #
                    ## t-분포를 통한 신뢰구간 추정(신뢰구간 90%)
                    # df = len(val1d) - 1
                    # df = len(val1d[~np.isnan(val1d)])
                    # t_ = t(df)
                    # t_05 = t_.ppf(0.975)
                    #
                    # sample_mean = np.nanmean(val1d)
                    # sample_std = np.nanstd(val1d, ddof=1)
                    #
                    # # 신뢰구간 95%  불확실성 범위 (2)
                    # l_ = round(sample_mean - t_05 * (sample_std / np.sqrt(df)), 2)
                    # u_ = round(sample_mean + t_05 * (sample_std / np.sqrt(df)), 2)
                    #
                    # # 확장불확도
                    # up = t_05 * (sample_std / np.sqrt(df))
                    # # 상대불확도 (%)
                    # urp = up * 100 / sample_mean
                    #
                    # # 총 불확도
                    # urb = np.sqrt(up**2)
                    #
                    # dict = {
                    #     'df': [df]
                    #     , 't_05': [t_05]
                    #     , 'sample_mean': [sample_mean]
                    #     , 'sample_std': [sample_std]
                    #     , 'l_': [l_]
                    #     , 'u_': [u_]
                    #     , 'up': [up]
                    #     , 'urp': [urp]
                    #     , 'urb': [urb]
                    # }
                    #
                    # resDict = pd.DataFrame.from_dict(dict)
                    #
                    # log.info("[CHECK] resDict : {}".format(resDict))

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
