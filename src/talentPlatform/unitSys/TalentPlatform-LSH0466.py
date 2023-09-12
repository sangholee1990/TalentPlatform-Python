# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy.ext.declarative import declarative_base
from xclim import sdba

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

Base = declarative_base()

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
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'mapInfo')
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


def calcLassoScore(contIdx, fileNameNoExt, dataset, var1, var2):
    # log.info(f'[START] calcLassoScore')
    result = None

    try:
        X = dataset[var1].values.flatten()[:, np.newaxis]
        y = dataset[var2].values.flatten()

        # NaN 값을 가진 행의 인덱스를 찾습니다.
        mask = ~np.isnan(y) & ~np.isnan(X[:, 0])

        # NaN 값을 제거합니다.
        X = X[mask]
        y = y[mask]

        start_time = time.time()
        lasso_lars_ic = make_pipeline(StandardScaler(), LassoLarsIC(criterion="aic")).fit(X, y)
        fit_time = time.time() - start_time

        valData = pd.DataFrame(
            {
                "alphas": lasso_lars_ic[-1].alphas_,
                "AIC criterion": lasso_lars_ic[-1].criterion_,
            }
        )
        alpha_aic = lasso_lars_ic[-1].alpha_

        lasso_lars_ic.set_params(lassolarsic__criterion="bic").fit(X, y)
        valData["BIC criterion"] = lasso_lars_ic[-1].criterion_
        alpha_bic = lasso_lars_ic[-1].alpha_

        # CSV 자료 저장
        saveFile = '{}/{}/{}-{}_{}-{}_{}.csv'.format(globalVar['outPath'], serviceName, 'RES-ABIC', var1, var2, contIdx, fileNameNoExt)
        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
        valData.to_csv(saveFile, index=False)
        # log.info(f'[CHECK] saveFile : {saveFile}')

        # 검증 스코어 저장
        plt.figure(dpi=600)
        saveImg = '{}/{}/{}-{}_{}-{}_{}.png'.format(globalVar['figPath'], serviceName, 'RES-ABIC', var1, var2, contIdx, fileNameNoExt)
        os.makedirs(os.path.dirname(saveImg), exist_ok=True)
        ax = valData.plot()
        ax.vlines(
            alpha_aic,
            valData["AIC criterion"].min(),
            valData["AIC criterion"].max(),
            label="alpha: AIC estimate",
            linestyles="--",
            color="tab:blue",
        )

        ax.vlines(
            alpha_bic,
            valData["BIC criterion"].min(),
            valData["BIC criterion"].max(),
            label="alpha: BIC estimate",
            linestyle="--",
            color="tab:orange",
        )
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("criterion")
        ax.set_xscale("log")
        ax.legend()
        _ = ax.set_title(f"Information-criterion for model selection (training time {fit_time:.2f}s)")
        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
        plt.tight_layout()
        # plt.show()
        plt.close()

        result = {
            'msg': 'succ'
            , 'saveFile': saveFile
            , 'isFileExist': os.path.exists(saveFile)
            , 'saveImg': saveImg
            , 'isImgExist': os.path.exists(saveImg)
        }

        return result

    except Exception as e:
        log.error(f'Exception : {e}')

        return result

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 전 지구 규모의 일 단위 강수량 편의보정 및 성능평가

    # 편향 조정 관련 API 명세
    # https://xclim.readthedocs.io/en/v0.30.1/sdba_api.html#bias-adjustment-algos
    # https://xclim.readthedocs.io/en/stable/notebooks/sdba.html
    # https://xclim.readthedocs.io/en/stable/apidoc/xclim.sdba.html#xclim.sdba.processing.jitter_under_thresh
    # https://xclim.readthedocs.io/en/stable/notebooks/sdba.html

    # cd /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/unitSys
    # conda activate py38
    # nohup python TalentPlatform-LSH0466.py &

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
    serviceName = 'LSH0466'

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
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'
                globalVar['updPath'] = '/DATA/CSV'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                # 'srtDate': '1990-01-01'
                # , 'endDate': '1993-01-01'
                'srtDate': '1979-01-01'
                , 'endDate': '1989-01-01'

                # 경도 최소/최대/간격
                , 'lonMin': 0
                , 'lonMax': 360
                , 'lonInv': 1

                # 위도 최소/최대/간격
                , 'latMin': -90
                , 'latMax': 90
                , 'latInv': 1
            }

            # 날짜 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDayList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1D')
            dtMonthList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1M')

            # 위경도 설정
            lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
            latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])
            log.info(f'[CHECK] len(lonList) : {len(lonList)}')
            log.info(f'[CHECK] len(latList) : {len(latList)}')

            # ********************************************************************
            # 대륙별 분류 전처리
            # ********************************************************************
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'TT4.csv')
            fileList = glob.glob(inpFile)
            if fileList is None or len(fileList) < 1:
                log.error('[ERROR] inpFile : {} / {}'.format(fileList, '입력 자료를 확인해주세요.'))
                raise Exception('[ERROR] inpFile : {} / {}'.format(fileList, '입력 자료를 확인해주세요.'))

            contData = pd.read_csv(fileList[0]).rename(columns={'type': 'contIdx'})
            contDataL1 = contData[['lon', 'lat', 'isLand', 'contIdx']]
            contDataL2 = contDataL1.set_index(['lat', 'lon'])
            contDataL3 = contDataL2.to_xarray()
            contDataL4 = contDataL3.interp({'lon': lonList, 'lat': latList}, method='nearest')

            # contDataL3['contIdx'].plot()
            # contDataL4['contIdx'].plot()
            # plt.show()

            # ********************************************************************
            # 강수량 파일 전처리
            # ********************************************************************
            # 실측 자료
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ERA5_1979_2020.nc')
            fileList = sorted(glob.glob(inpFile))
            obsData = xr.open_dataset(fileList[0]).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))

            # 경도 변환 (-180~180 to 0~360)
            obsDataL1 = obsData
            obsDataL1.coords['lon'] = (obsDataL1.coords['lon']) % 360
            obsDataL1 = obsDataL1.sortby(obsDataL1.lon)

            obsDataL2 = obsDataL1.interp({'lon': lonList, 'lat': latList}, method='linear')

            # obsDataL2.attrs
            # obsDataL2['rain'].attrs

            # 모델 자료
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'pr_day_MRI-ESM2-0_historical_r1i1p1f1_gn_19500101-19991231-003.nc')
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'pr_*.nc')
            fileList = sorted(glob.glob(inpFile))

            # fileInfo = fileList[0]
            for fileInfo in fileList:
                log.info(f"[CHECK] fileInfo : {fileInfo}")

                fileNameNoExt = os.path.basename(fileInfo).split('.')[0]

                # modData = xr.open_dataset(fileInfo).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
                modData = xr.open_dataset(fileInfo).sel(time=slice(sysOpt['srtDate'], sysOpt['endDate']))
                if (len(modData['time']) < 1): continue

                # 필요없는 변수 삭제
                selList = ['lat_bnds', 'lon_bnds', 'time_bnds']

                for i, selInfo in enumerate(selList):
                    try:
                        modData = modData.drop([selInfo])
                    except Exception as e:
                        log.error("Exception : {}".format(e))


                modDataL1 = modData.interp({'lon': lonList, 'lat': latList}, method='linear')

                # 일 강수량 단위 환산 : 60 * 60 * 24
                modDataL1['pr'] = modDataL1['pr'] * 86400
                modDataL1['pr'].attrs["units"] = "mm d-1"

                modDataL2 = modDataL1
                # modDataL2.attrs
                # modDataL2['rain'].attrs
                # modDataL2.isel(time = 0).plot

                mrgData = xr.merge([obsDataL2, modDataL2])

                # ***********************************************************************************
                # Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
                # ***********************************************************************************
                # 일수 단위로 가중치 조정
                # qdm = sdba.QuantileDeltaMapping.train(mrgData['rain'], mrgData['pr'], group='time.dayofyear')
                # 연 단위로 가중치 조정
                # qdm = sdba.QuantileDeltaMapping.train(mrgData['rain'], mrgData['pr'], group='time.year')
                # 계절 단위로 가중치 조정
                # qdm = sdba.QuantileDeltaMapping.train(mrgData['rain'], mrgData['pr'], group='time.season')
                # 월 단위로 가중치 조정
                qdm = sdba.QuantileDeltaMapping.train(mrgData['rain'], mrgData['pr'], group='time.month')
                # 일 단위로 가중치 조정
                # qdm = sdba.QuantileDeltaMapping.train(mrgData['rain'], mrgData['pr'], group='time')
                qdmData = qdm.adjust(mrgData['pr'], extrapolation="nan", interp="linear")

                # qdm.ds.af.plot()
                # plt.show()

                # ***********************************************************************************
                # Dequé, M. (2007). Frequency of precipitation and temperature extremes over France in an anthropogenic scenario: Model results and statistical correction according to observed values. Global and Planetary Change, 57(1–2), 16–26. https://doi.org/10.1016/j.gloplacha.2006.11.030
                # ***********************************************************************************
                # eqm =  sdba.EmpiricalQuantileMapping.train(mrgData['rain'], mrgData['pr'], group='time.dayofyear')
                eqm =  sdba.EmpiricalQuantileMapping.train(mrgData['rain'], mrgData['pr'], group='time.month')
                # eqm =  sdba.EmpiricalQuantileMapping.train(mrgData['rain'], mrgData['pr'], group='time')
                eqmData = eqm.adjust(mrgData['pr'], extrapolation="nan", interp="linear")

                # eqm.ds.af.plot()
                # plt.show()

                # ***********************************************************************************
                # Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? Journal of Climate, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
                # ***********************************************************************************
                # dqm = sdba.DetrendedQuantileMapping.train(mrgData['rain'], mrgData['pr'], group='time.dayofyear')
                dqm = sdba.DetrendedQuantileMapping.train(mrgData['rain'], mrgData['pr'], group='time.month')
                # dqm = sdba.DetrendedQuantileMapping.train(mrgData['rain'], mrgData['pr'], group='time')
                dqmData = dqm.adjust(mrgData['pr'], extrapolation="nan", interp="linear")

                # ***********************************************************************************
                # Cannon, A. J. (2018). Multivariate quantile mapping bias correction: An N-dimensional probability density function transform for climate model simulations of multiple variables. Climate Dynamics, 50(1), 31–49. https://doi.org/10.1007/s00382-017-3580-6
                # Pitie, F., Kokaram, A. C., & Dahyot, R. (2005). N-dimensional probability density function transfer and its application to color transfer. Tenth IEEE International Conference on Computer Vision (ICCV’05) Volume 1, 2, 1434-1439 Vol. 2. https://doi.org/10.1109/ICCV.2005.166
                # Szekely, G. J. and Rizzo, M. L. (2004) Testing for Equal Distributions in High Dimension, InterStat, November (5)
                # ***********************************************************************************
                # dref = mrgData['rain']
                # dhist = mrgData['pr']
                # dsim = mrgData['pr']
                #
                # # additive for tasmax
                # # QDMtx = sdba.QuantileDeltaMapping.train(
                # #     dref, dhist, nquantiles=20, kind="+", group="time"
                # # )
                # # # Adjust both hist and sim, we'll feed both to the Npdf transform.
                # # scenh_tx = QDMtx.adjust(dhist)
                # # scens_tx = QDMtx.adjust(dsim)
                #
                # # remove == 0 values in pr:
                # dref = sdba.processing.jitter_under_thresh(dref, "0.01 mm d-1")
                # dhist = sdba.processing.jitter_under_thresh(dhist, "0.01 mm d-1")
                # dsim = sdba.processing.jitter_under_thresh(dsim, "0.01 mm d-1")
                #
                # # multiplicative for pr
                # QDMpr = sdba.QuantileDeltaMapping.train(
                #     dref, dhist, nquantiles=20, kind="*", group="time"
                # )
                # # Adjust both hist and sim, we'll feed both to the Npdf transform.
                # scenh_pr = QDMpr.adjust(dhist)
                # scens_pr = QDMpr.adjust(dsim)
                #
                # dref =  xr.Dataset(dict(tasmax=dref, pr=dref))
                # scenh = xr.Dataset(dict(tasmax=scenh_pr, pr=scenh_pr))
                # scens = xr.Dataset(dict(tasmax=scens_pr, pr=scens_pr))
                #
                # # Stack the variables (tasmax and pr)
                # ref = sdba.processing.stack_variables(dref)
                # scenh = sdba.processing.stack_variables(scenh)
                # scens = sdba.processing.stack_variables(scens)
                #
                # # Standardize
                # ref, _, _ = sdba.processing.standardize(ref)
                #
                # allsim_std, _, _ = sdba.processing.standardize(xr.concat((scenh, scens), "time"))
                # scenh_std = allsim_std
                # scens_std = allsim_std
                #
                # from xclim import set_options
                #
                # # See the advanced notebook for details on how this option work
                # with set_options(sdba_extra_output=True):
                #     out = sdba.adjustment.NpdfTransform.adjust(
                #         ref,
                #         scenh_std,
                #         scens_std,
                #         base=sdba.QuantileDeltaMapping,  # Use QDM as the univariate adjustment.
                #         base_kws={"nquantiles": 20, "group": "time"},
                #         n_iter=20,  # perform 20 iteration
                #         n_escore=1000,  # only send 1000 points to the escore metric (it is realy slow)
                #     )
                #
                # scenh_npdft = out.scenh.rename(time_hist="time")  # Bias-adjusted historical period
                # scens_npdft = out.scen  # Bias-adjusted future period
                # extra = out.drop_vars(["scenh", "scen"])
                #
                # scenh = sdba.processing.reordering(scenh_npdft, scenh, group="time")
                # scens = sdba.processing.reordering(scens_npdft, scens, group="time")
                #
                # scenh = sdba.processing.unstack_variables(scenh)
                # scens = sdba.processing.unstack_variables(scens)

                # ***********************************************************************************
                # 요약 통계량
                # ***********************************************************************************
                timeIdx = 0
                print(f"[CHECK] min : {np.nanmin(mrgData['rain'].isel(time=timeIdx))} / max : {np.nanmax(mrgData['rain'].isel(time=timeIdx))}")
                print(f"[CHECK] min : {np.nanmin(mrgData['pr'].isel(time=timeIdx))} / max : {np.nanmax(mrgData['pr'].isel(time=timeIdx))}")
                print(f"[CHECK] min : {np.nanmin(qdmData.isel(time=timeIdx))} / max : {np.nanmax(qdmData.isel(time=timeIdx))}")
                print(f"[CHECK] min : {np.nanmin(eqmData.isel(time=timeIdx))} / max : {np.nanmax(eqmData.isel(time=timeIdx))}")
                print(f"[CHECK] min : {np.nanmin(dqmData.isel(time=timeIdx))} / max : {np.nanmax(dqmData.isel(time=timeIdx))}")

                # mrgData['rain'].isel(time=2).plot(vmin=0, vmax=100)
                # mrgData['pr'].isel(time=2).plot(vmin=0, vmax=100)
                # qdmData.isel(time=2).plot(x='lon', y='lat', vmin=0, vmax=100, cmap='viridis')
                # eqmData.isel(time=2).plot(vmin=0, vmax=100, cmap='viridis')
                # plt.show()

                # 동적으로 생성
                lat1D = mrgData['lat'].values
                lon1D = mrgData['lon'].values
                time1D = mrgData['time'].values

                # (time: 91, lat: 180, lon: 360)
                mrgDataL1 = xr.Dataset(
                    {
                        'OBS': (('time', 'lat', 'lon'), (mrgData['rain'].values).reshape(len(time1D), len(lat1D), len(lon1D)))
                        , 'MOD': (('time', 'lat', 'lon'), (mrgData['pr'].values).reshape(len(time1D), len(lat1D), len(lon1D)))
                        , 'QDM': (('time', 'lat', 'lon'), (qdmData.transpose('time', 'lat', 'lon').values).reshape(len(time1D), len(lat1D), len(lon1D)))
                        , 'EQM': (('time', 'lat', 'lon'), (eqmData.transpose('time', 'lat', 'lon').values).reshape(len(time1D), len(lat1D), len(lon1D)))
                        , 'DQM': (('time', 'lat', 'lon'), (dqmData.transpose('time', 'lat', 'lon').values).reshape(len(time1D), len(lat1D), len(lon1D)))
                        , 'isLand': (('time', 'lat', 'lon'), np.tile(contDataL4['isLand'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(len(time1D), len(lat1D), len(lon1D)))
                        , 'contIdx': (('time', 'lat', 'lon'), np.tile(contDataL4['contIdx'].values[np.newaxis, :, :], (len(time1D), 1, 1)).reshape(len(time1D), len(lat1D), len(lon1D)))

                    }
                    , coords={
                        'time': time1D
                        , 'lat': lat1D
                        , 'lon': lon1D
                    }
                )

                # NetCDF 자료 저장
                saveFile = '{}/{}/{}_{}.nc'.format(globalVar['outPath'], serviceName, 'RES-MBC', fileNameNoExt)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                mrgDataL1.to_netcdf(saveFile)
                log.info(f'[CHECK] saveFile : {saveFile}')

                # CSV 자료 저장
                saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, 'RES-MBC', fileNameNoExt)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                mrgDataL1.to_dataframe().reset_index(drop=False).to_csv(saveFile, index=False)
                log.info(f'[CHECK] saveFile : {saveFile}')

                # mrgDataL1.isel(time = 0)['OBS'].plot()
                # mrgDataL1.isel(time = 0)['MOD'].plot()
                # mrgDataL1.isel(time = 0)['QDM'].plot()
                # mrgDataL1.isel(time = 0)['EQM'].plot()
                # mrgDataL1.isel(time = 0)['isLand'].plot()
                # mrgDataL1.isel(time = 1)['contIdx'].plot()
                # plt.show()

                getContIdx = np.unique(mrgDataL1['contIdx'].values)
                contIdxList = getContIdx[~np.isnan(getContIdx)].astype(int)

                # contIdx = 100
                for contIdxInfo in contIdxList:
                    if np.isnan(contIdxInfo): continue
                    selData = mrgDataL1.where(mrgDataL1['contIdx'] == contIdxInfo, drop=True)

                    result = calcLassoScore(contIdxInfo, fileNameNoExt, selData, 'OBS', 'QDM')
                    print(f'[CHECK] result : {result}')

                    result = calcLassoScore(contIdxInfo, fileNameNoExt, selData, 'OBS', 'EQM')
                    print(f'[CHECK] result : {result}')

                    result = calcLassoScore(contIdxInfo, fileNameNoExt, selData, 'OBS', 'DQM')
                    print(f'[CHECK] result : {result}')

                # 95% 이상 분위수 계산
                mrgDataL2 = mrgDataL1.quantile(0.95, dim='time')

                # NetCDF 자료 저장
                saveFile = '{}/{}/{}_{}.nc'.format(globalVar['outPath'], serviceName, 'RES-95', fileNameNoExt)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                mrgDataL2.to_netcdf(saveFile)
                log.info(f'[CHECK] saveFile : {saveFile}')

                # CSV 자료 저장
                saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, 'RES-95', fileNameNoExt)
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                mrgDataL2.to_dataframe().reset_index(drop=False).to_csv(saveFile, index=False)
                log.info(f'[CHECK] saveFile : {saveFile}')

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
