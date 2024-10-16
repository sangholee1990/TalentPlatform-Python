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
from datetime import timedelta
from urllib.parse import quote_plus

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import xarray as xr
from pandas.tseries.offsets import Hour
import yaml
from multiprocessing import Pool
import multiprocessing as mp
# import avl
# import harp
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
import matplotlib.colors as colors
from netCDF4 import Dataset, num2date

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
        # , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        # , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        # , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        # , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        # , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        # , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        # , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        # , 'sysPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.cfg')
        # , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        # , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
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

    return globalVar

def format_date(x, pos=None):
    dt_obj = num2date(x, units="s since 2000-01-01", only_use_cftime_datetimes=False)
    return dt_obj.strftime("%H:%M:%S")

def plot_parameter_2D(
        parameter="wind_result_wind_velocity",
        channel="rayleigh",
        obs_type="clear",
        QC_filter=True,
        error_estimate_threshold=800,
        start_bin=0,
        end_bin=-1,
        ds=None
):
    # if channel == "rayleigh":
    #     ds = ds_rayleigh
    # elif channel == "mie":
    #     ds = ds_mie

    # define necessary parameters for plotting
    X0 = ds[channel + "_wind_result_start_time"].values
    X1 = ds[channel + "_wind_result_stop_time"].values

    Y0 = ds[channel + "_wind_result_bottom_altitude"].values / 1000.0
    Y1 = ds[channel + "_wind_result_top_altitude"].values / 1000.0
    Z = ds[channel + "_" + parameter].values

    # create a mask out of different filters which can be applied to the different parameters
    mask = np.zeros(len(Z), dtype=bool)

    # mask dependent on start and end bin given as parameter to the plot function
    mask[0:start_bin] = True
    mask[end_bin:-1] = True

    # mask where validity flag is 0
    if QC_filter:
        mask = mask | (ds[channel + "_wind_result_validity_flag"] == 0)

    # mask dependent on observation type
    if obs_type == "cloudy":
        mask = mask | (ds[channel + "_wind_result_observation_type"] != 1)
    elif obs_type == "clear":
        mask = mask | (ds[channel + "_wind_result_observation_type"] != 2)

    # mask where wind results have error estimates larger than a given threshold
    mask = mask | (ds[channel + "_wind_result_HLOS_error"] > error_estimate_threshold)

    # mask all necessary parameters for plotting
    # tilde before mask inverts the boolean mask array
    X0 = X0[~mask]
    X1 = X1[~mask]
    Y0 = Y0[~mask]
    Y1 = Y1[~mask]
    Z = Z[~mask]

    patches = []
    for x0, x1, y0, y1 in zip(X0, X1, Y0, Y1):
        patches.append(((x0, y0), (x0, y1), (x1, y1), (x1, y0)))

    # define min and max value for the colorbar
    if parameter == "wind_result_wind_velocity":
        Z_vmax = np.amax(np.abs(np.asarray([np.nanpercentile(Z, 2), np.nanpercentile(Z, 98)])))
        Z_vmin = -Z_vmax
    else:
        Z_vmax = np.nanpercentile(Z, 99)
        Z_vmin = np.nanpercentile(Z, 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    coll = PolyCollection(
        patches,
        array=Z,
        cmap=cm.RdBu_r,
        norm=colors.Normalize(
            vmin=Z_vmin,
            vmax=Z_vmax,
            clip=False,
        ),
    )
    ax.add_collection(coll)

    ax.scatter(
        ds[channel + "_wind_result_COG_time"][~mask],
        ds[channel + "_wind_result_alt_of_DEM_intersection"][~mask] / 1000.0,
        marker='o',
        c='r',
        s=5,
        label='DEM altitude',
    )
    # ax.set_ylim(-1, 30)
    ax.set_xlabel("Date [UTC]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("{} - {} \n {} wind results".format(channel.title(), parameter, len(Z)))
    ax.grid()
    ax.legend()

    ax.xaxis.set_major_formatter(format_date)
    ax.autoscale()
    fig.colorbar(coll, ax=ax, aspect=50, pad=0.01)
    fig.autofmt_xdate()

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 포스트SQL 연동 테스트

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    # env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        # contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/PyCharm'
        contextPath = os.getcwd() if env in 'local' else '/vol01/SYSTEMS/KIER/PROG/PYTHON'

    prjName = 'test'
    serviceName = 'INDI2024'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info(f"[START] __init__ : init")

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

        except Exception as e:
            log.error("Exception : {}".format(e))
            raise e
        finally:
            log.info(f"[END] __init__ : init")

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info(f"[START] exec")

        try:

            if platform.system() == 'Windows':
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            # 옵션 설정
            sysOpt = {
                # 시작일, 종료일, 시간 간격
                'srtDate': '2020-10-20'
                , 'endDate': '2020-10-21'
                , 'invDate': '1h'

                # , 'modelList': ['KIER-LDAPS', 'KIER-RDAPS']

                # 비동기 다중 프로세스 개수
                # , 'cpuCoreNum': 5
            }

            # **************************************************************************************************************
            # 자료 처리
            # **************************************************************************************************************
            from viresclient import AeolusRequest
            import numpy as np
            from netCDF4 import num2date
            import matplotlib.pyplot as plt
            from matplotlib.collections import PolyCollection
            import matplotlib.cm as cm
            import matplotlib.colors as colors
            import cartopy.crs as ccrs

            # https://aeolus.services 플랫폼
            # https://aeolus.services/ows
            # hAAFUwbvPvnzzgGTxaU3ttAfufVKyp9-

            # Set up connection with server
            # Set collection to use
            # request.set_collection('ALD_U_N_2B')
            #
            # request.set_fields(rayleigh_wind_fields=[
            #     "rayleigh_wind_result_start_time",
            #     "rayleigh_wind_result_stop_time",
            #     "rayleigh_wind_result_bottom_altitude",
            #     "rayleigh_wind_result_top_altitude",
            #     "rayleigh_wind_result_wind_velocity",
            # ])
            #
            # data = request.get_between(
            #     start_time="2020-04-10T06:21:58Z",
            #     end_time="2020-04-10T07:50:33Z",
            #     filetype="nc"
            # )

            # Aeolus product
            DATA_PRODUCT = "ALD_U_N_2B"

            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

            for i, dtDateInfo in enumerate(dtDateList):
                if (i + 1) == len(dtDateList): continue

                measurement_start = dtDateList[i].tz_localize('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
                measurement_stop = dtDateList[i + 1].tz_localize('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
                print(measurement_start, measurement_stop)

                # measurement period in yyyy-mm-ddTHH:MM:SS
                # measurement_start = "2020-10-20T00:00:00Z"
                # measurement_stop = "2020-10-21T00:00:00Z"
                # measurement_start = "2024-06-01T00:00:00Z"
                # measurement_stop = "2024-06-02T00:00:00Z"

                # Product parameters to retrieve
                # uncomment parameters of interest

                # Rayleigh wind fields
                parameter_rayleigh = [
                    "wind_result_start_time",
                    "wind_result_stop_time",
                    "wind_result_COG_time",
                    "wind_result_bottom_altitude",
                    "wind_result_top_altitude",
                    "wind_result_range_bin_number",
                    "wind_result_start_latitude",
                    "wind_result_start_longitude",
                    "wind_result_stop_latitude",
                    "wind_result_stop_longitude",
                    "wind_result_COG_latitude",
                    "wind_result_COG_longitude",
                    "wind_result_HLOS_error",
                    "wind_result_wind_velocity",
                    "wind_result_observation_type",
                    "wind_result_validity_flag",
                    "wind_result_alt_of_DEM_intersection",
                ]
                parameter_rayleigh = ["rayleigh_" + param for param in parameter_rayleigh]

                # Mie wind fields
                parameter_mie = [
                    "wind_result_start_time",
                    "wind_result_stop_time",
                    "wind_result_COG_time",
                    "wind_result_bottom_altitude",
                    "wind_result_top_altitude",
                    "wind_result_range_bin_number",
                    "wind_result_start_latitude",
                    "wind_result_start_longitude",
                    "wind_result_stop_latitude",
                    "wind_result_stop_longitude",
                    "wind_result_COG_latitude",
                    "wind_result_COG_longitude",
                    "wind_result_HLOS_error",
                    "wind_result_wind_velocity",
                    "wind_result_observation_type",
                    "wind_result_validity_flag",
                    "wind_result_alt_of_DEM_intersection",
                ]
                parameter_mie = ["mie_" + param for param in parameter_mie]

                # Data request for Rayleigh wind measurements
                # check if parameter list is not empty
                if len(parameter_rayleigh) > 0:
                    # request = AeolusRequest()
                    request = AeolusRequest(url='https://aeolus.services/ows', token=None)
                    
                    request.set_collection(DATA_PRODUCT)

                    # set wind fields
                    request.set_fields(
                        rayleigh_wind_fields=parameter_rayleigh,
                    )

                    # It is possible to apply a filter by different parameters of the product
                    # Here, for example, a filter by geolocation is applied
                    request.set_range_filter(parameter="rayleigh_wind_result_COG_latitude", minimum=0, maximum=90)
                    request.set_range_filter(
                        parameter="rayleigh_wind_result_COG_longitude", minimum=180, maximum=360
                    )

                    # set start and end time and request data
                    data_rayleigh = request.get_between(
                        start_time=measurement_start, end_time=measurement_stop, filetype="nc", asynchronous=True
                    )

                # Data request for Mie wind measurements
                # check if parameter list is not empty
                if len(parameter_mie) > 0:
                    # request = AeolusRequest()
                    request = AeolusRequest(url='https://aeolus.services/ows', token=None)

                    request.set_collection(DATA_PRODUCT)

                    # set measurement fields
                    request.set_fields(
                        mie_wind_fields=parameter_mie,
                    )

                    # It is possible to apply a filter by different parameters of the product
                    # Here, for example, a filter by geolocation is applied
                    request.set_range_filter(parameter="mie_wind_result_COG_latitude", minimum=0, maximum=90)
                    request.set_range_filter(parameter="mie_wind_result_COG_longitude", minimum=180, maximum=360)

                    # set start and end time and request data
                    data_mie = request.get_between(
                        start_time=measurement_start, end_time=measurement_stop, filetype="nc", asynchronous=True
                    )

                # Save data as xarray data sets
                # check if variable is assigned
                # if "data_rayleigh" in globals():
                #     ds_rayleigh = data_rayleigh.as_xarray()
                # if "data_mie" in globals():
                #     ds_mie = data_mie.as_xarray()

                ds_rayleigh = data_rayleigh.as_xarray()
                ds_mie = data_mie.as_xarray()

                # if len(ds_rayleigh) < 0: return

                # ds_rayleigh['rayleigh_wind_result_wind_velocity'].plot()
                # plt.show()

                # ds_mie['mie_wind_result_wind_velocity'].plot()
                # plt.show()

                dtSrtDate = pd.to_datetime(measurement_start)
                dtEndDate = pd.to_datetime(measurement_stop)

                srtDate = dtSrtDate.strftime('%Y%m%d%H%M')
                endDate = dtEndDate.strftime('%Y%m%d%H%M')

                procFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, 'AE_OPER_ALD_U_N_2B_rayleigh_wind-velocity', srtDate, endDate)
                os.makedirs(os.path.dirname(procFile), exist_ok=True)
                data_rayleigh.to_file(procFile, overwrite=True)
                log.info(f'[CHECK] procFile : {procFile}')

                procFile = '{}/{}/{}_{}_{}.nc'.format(globalVar['outPath'], serviceName, 'AE_OPER_ALD_U_N_2B_mie_wind-velocity', srtDate, endDate)
                os.makedirs(os.path.dirname(procFile), exist_ok=True)
                # ds_mie['mie_wind_result_wind_velocity'].to_netcdf(procFile)
                data_mie.to_file(procFile, overwrite=True)
                log.info(f'[CHECK] procFile : {procFile}')

                plot_parameter_2D(
                    parameter="wind_result_wind_velocity",
                    channel="rayleigh",
                    obs_type="clear",
                    QC_filter=True,
                    error_estimate_threshold=800,
                    start_bin=0,

                    end_bin=-1,
                    ds=ds_rayleigh
                )

                saveImg = '{}/{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, 'AE_OPER_ALD_U_N_2B_rayleigh_wind-velocity', srtDate, endDate)
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
                # plt.tight_layout()
                plt.show()
                plt.close()
                log.info(f'[CHECK] saveImg : {saveImg}')

                plot_parameter_2D(
                    parameter="wind_result_wind_velocity",
                    channel="mie",
                    obs_type="cloudy",
                    QC_filter=True,
                    error_estimate_threshold=500,
                    start_bin=0,
                    end_bin=-1,
                    ds=ds_mie
                )

                saveImg = '{}/{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, 'AE_OPER_ALD_U_N_2B_mie_wind-velocity', srtDate, endDate)
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
                # plt.tight_layout()
                plt.show()
                plt.close()
                log.info(f'[CHECK] saveImg : {saveImg}')

                # 시작-종료 맵 시각화
                # fig, ax = plt.subplots(2,1, figsize=(8, 8), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True)

                # for ds, obs_type in zip([ds_rayleigh, ds_mie], ["rayleigh", "mie"]):
                for ds, obs_type in zip([ds_rayleigh, ds_mie], ["rayleigh"]):
                    fig, axis = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True)
                    axis.stock_img()
                    axis.gridlines(draw_labels=True, linewidth=0.3, color="black", alpha=0.5, linestyle="-")
                    axis.scatter(
                        ds[obs_type + "_wind_result_COG_longitude"],
                        ds[obs_type + "_wind_result_COG_latitude"],
                        marker="o",
                        c="k",
                        s=3,
                        label='wind result COG',
                        transform=ccrs.Geodetic(),
                    )
                    axis.scatter(
                        ds[obs_type + "_wind_result_COG_longitude"][0],
                        ds[obs_type + "_wind_result_COG_latitude"][0],
                        marker="o",
                        c="g",
                        edgecolor="g",
                        s=40,
                        label="start",
                        transform=ccrs.Geodetic(),
                    )
                    axis.scatter(
                        ds[obs_type + "_wind_result_COG_longitude"][-1],
                        ds[obs_type + "_wind_result_COG_latitude"][-1],
                        marker="o",
                        c="r",
                        edgecolor="r",
                        s=40,
                        label="stop",
                        transform=ccrs.Geodetic(),
                    )
                    axis.legend()
                    axis.set_title(obs_type.title())

                    fig.suptitle("Aeolus orbit \n from {} to {} \n".format(measurement_start, measurement_stop))

                    saveImg = '{}/{}/{}_{}_{}_{}.png'.format(globalVar['figPath'], serviceName, 'AE_OPER_ALD_U_N_2B_orbit', obs_type, srtDate, endDate)
                    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
                    # plt.tight_layout()
                    plt.show()
                    plt.close()
                    log.info(f'[CHECK] saveImg : {saveImg}')

                    def makePlot(
                            parameter="wind_result_wind_velocity",
                            channel="rayleigh",
                            obs_type="clear",
                            QC_filter=True,
                            error_estimate_threshold=800,
                            start_bin=0,
                            end_bin=-1,
                            ds=None
                    ):

                        # define necessary parameters for plotting
                        X0 = ds[channel + "_wind_result_start_time"].values
                        X1 = ds[channel + "_wind_result_stop_time"].values

                        Y0 = ds[channel + "_wind_result_bottom_altitude"].values / 1000.0
                        Y1 = ds[channel + "_wind_result_top_altitude"].values / 1000.0
                        Z = ds[channel + "_" + parameter].values

                        # create a mask out of different filters which can be applied to the different parameters
                        mask = np.zeros(len(Z), dtype=bool)

                        # mask dependent on start and end bin given as parameter to the plot function
                        mask[0:start_bin] = True
                        mask[end_bin:-1] = True

                        # mask where validity flag is 0
                        if QC_filter:
                            mask = mask | (ds[channel + "_wind_result_validity_flag"] == 0)

                        # mask dependent on observation type
                        if obs_type == "cloudy":
                            mask = mask | (ds[channel + "_wind_result_observation_type"] != 1)
                        elif obs_type == "clear":
                            mask = mask | (ds[channel + "_wind_result_observation_type"] != 2)

                        # mask where wind results have error estimates larger than a given threshold
                        mask = mask | (ds[channel + "_wind_result_HLOS_error"] > error_estimate_threshold)

                        # mask all necessary parameters for plotting
                        # tilde before mask inverts the boolean mask array
                        X0 = X0[~mask]
                        X1 = X1[~mask]
                        Y0 = Y0[~mask]
                        Y1 = Y1[~mask]
                        Z = Z[~mask]

                        patches = []
                        for x0, x1, y0, y1 in zip(X0, X1, Y0, Y1):
                            patches.append(((x0, y0), (x0, y1), (x1, y1), (x1, y0)))

                        # define min and max value for the colorbar
                        if parameter == "wind_result_wind_velocity":
                            Z_vmax = np.amax(np.abs(np.asarray([np.nanpercentile(Z, 2), np.nanpercentile(Z, 98)])))
                            Z_vmin = -Z_vmax
                        else:
                            Z_vmax = np.nanpercentile(Z, 99)
                            Z_vmin = np.nanpercentile(Z, 1)

                        # fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
                        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
                        # subplot_kw = {"projection": ccrs.PlateCarree() if ax3 else None

                        coll = PolyCollection(
                            patches,
                            array=Z,
                            cmap=cm.RdBu_r,
                            norm=colors.Normalize(
                                vmin=Z_vmin,
                                vmax=Z_vmax,
                                clip=False,
                            ),
                        )
                        ax.add_collection(coll)

                        ax.scatter(
                            ds[channel + "_wind_result_COG_time"][~mask],
                            ds[channel + "_wind_result_alt_of_DEM_intersection"][~mask] / 1000.0,
                            marker='o',
                            c='r',
                            s=5,
                            label='DEM altitude',
                        )
                        # ax.set_ylim(-1, 30)
                        ax.set_xlabel("Date [UTC]")
                        ax.set_ylabel("Altitude [km]")
                        ax.set_title("{} - {} \n {} wind results".format(channel.title(), parameter, len(Z)))
                        ax.grid()
                        ax.legend()

                        ax.xaxis.set_major_formatter(format_date)
                        ax.autoscale()
                        fig.colorbar(coll, ax=ax, aspect=50, pad=0.01)
                        fig.autofmt_xdate()







        except Exception as e:
            log.error(f"Exception : {e}")
            raise e

        finally:
            log.info(f"[END] exec")


# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print(f'[START] main')

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
        print('[END] main')
