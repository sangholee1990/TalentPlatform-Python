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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


class DistributedROC(object):
    """
    ROC sparse representation that can be aggregated and can generate ROC curves and performance diagrams.
    A DistributedROC object is given a specified set of thresholds (could be probability or real-valued) and then
    stores a pandas DataFrame of contingency tables for each threshold. The contingency tables are updated with a
    set of forecasts and observations, but the original forecast and observation values are not kept. DistributedROC
    objects can be combined by adding them together or by storing them in an iterable and summing the contents of the
    iterable together. This is especially useful when verifying large numbers of cases in parallel.
    Attributes:
        thresholds (numpy.ndarray): List of probability thresholds in increasing order.
        obs_threshold (float):  Observation values >= obs_threshold are positive events.
        contingency_tables (pandas.DataFrame): Stores contingency table counts for each probability threshold
    Examples:
    #     >>> import numpy as np
    #     >>> forecasts = np.random.random(size=1000)
    #     >>> obs = np.random.random_integers(0, 1, size=1000)
    #     >>> roc = DistributedROC(thresholds=np.arange(0, 1.1, 0.1), obs_threshold=1)
    #     >>> roc.update(forecasts, obs)
    #     >>> print(roc.auc())
    # """

    def __init__(self, thresholds=np.arange(0, 1.1, 0.1), obs_threshold=1.0, input_str=None):
        """
        Initializes the DistributedROC object. If input_str is not None, then the DistributedROC object is
         initialized with the contents of input_str. Otherwise an empty contingency table is created.
        Args:
            thresholds (numpy.array): Array of thresholds in increasing order.
            obs_threshold (float): Split threshold (>= is positive event) (< is negative event)
            input_str (None or str): String containing information for DistributedROC
        """
        self.thresholds = thresholds
        self.obs_threshold = obs_threshold
        self.contingency_tables = pd.DataFrame(np.zeros((thresholds.size, 4), dtype=int),
                                               columns=["TP", "FP", "FN", "TN"])
        if input_str is not None:
            self.from_str(input_str)

    def update(self, forecasts, observations):
        """
        Update the ROC curve with a set of forecasts and observations
        Args:
            forecasts: 1D array of forecast values
            observations: 1D array of observation values.
        """
        for t, threshold in enumerate(self.thresholds):
            tp = np.count_nonzero((forecasts >= threshold) & (observations >= self.obs_threshold))
            fp = np.count_nonzero((forecasts >= threshold) &
                                  (observations < self.obs_threshold))
            fn = np.count_nonzero((forecasts < threshold) &
                                  (observations >= self.obs_threshold))
            tn = np.count_nonzero((forecasts < threshold) &
                                  (observations < self.obs_threshold))
            self.contingency_tables.iloc[t] += [tp, fp, fn, tn]

    def clear(self):
        self.contingency_tables.loc[:, :] = 0

    def __add__(self, other):
        """
        Add two DistributedROC objects together and combine their contingency table values.
        Args:
            other: Another DistributedROC object.
        """
        sum_roc = DistributedROC(self.thresholds, self.obs_threshold)
        sum_roc.contingency_tables = self.contingency_tables + other.contingency_tables
        return sum_roc

    def merge(self, other_roc):
        """
        Ingest the values of another DistributedROC object into this one and update the statistics inplace.
        Args:
            other_roc: another DistributedROC object.
        """
        if other_roc.thresholds.size == self.thresholds.size and np.all(other_roc.thresholds == self.thresholds):
            self.contingency_tables += other_roc.contingency_tables
        else:
            print("Input table thresholds do not match.")

    def roc_curve(self):
        """
        Generate a ROC curve from the contingency table by calculating the probability of detection (TP/(TP+FN)) and the
        probability of false detection (FP/(FP+TN)).
        Returns:
            A pandas.DataFrame containing the POD, POFD, and the corresponding probability thresholds.
        """
        pod = self.contingency_tables["TP"].astype(float) / (self.contingency_tables["TP"] +
                                                             self.contingency_tables["FN"])
        pofd = self.contingency_tables["FP"].astype(float) / (self.contingency_tables["FP"] +
                                                              self.contingency_tables["TN"])
        return pd.DataFrame({"POD": pod, "POFD": pofd, "Thresholds": self.thresholds},
                            columns=["POD", "POFD", "Thresholds"])

    def performance_curve(self):
        """
        Calculate the Probability of Detection and False Alarm Ratio in order to output a performance diagram.
        Returns:
            pandas.DataFrame containing POD, FAR, and probability thresholds.
        """
        pod = self.contingency_tables["TP"] / (self.contingency_tables["TP"] + self.contingency_tables["FN"])
        far = self.contingency_tables["FP"] / (self.contingency_tables["FP"] + self.contingency_tables["TP"])
        far[(self.contingency_tables["FP"] + self.contingency_tables["TP"]) == 0] = np.nan
        return pd.DataFrame({"POD": pod, "FAR": far, "Thresholds": self.thresholds},
                            columns=["POD", "FAR", "Thresholds"])

    def auc(self):
        """
        Calculate the Area Under the ROC Curve (AUC).
        """
        roc_curve = self.roc_curve()
        return np.abs(np.trapz(roc_curve['POD'], x=roc_curve['POFD']))

    def max_csi(self):
        """
        Calculate the maximum Critical Success Index across all probability thresholds
        Returns:
            The maximum CSI as a float
        """
        csi = self.contingency_tables["TP"] / (self.contingency_tables["TP"] + self.contingency_tables["FN"] +
                                               self.contingency_tables["FP"])
        return csi.max()

    def max_threshold_score(self, score="ets"):
        cts = self.get_contingency_tables()
        scores = np.array([getattr(ct, score)() for ct in cts])
        return self.thresholds[scores.argmax()], scores.max()

    def get_contingency_tables(self):
        """
        Create an Array of ContingencyTable objects for each probability threshold.
        Returns:
            Array of ContingencyTable objects
        """
        return np.array([ContingencyTable(*ct) for ct in self.contingency_tables.values])

    def __str__(self):
        """
        Output the information within the DistributedROC object to a string.
        """
        out_str = "Obs_Threshold:{0:0.2f}".format(self.obs_threshold) + ";"
        out_str += "Thresholds:" + " ".join(["{0:0.2f}".format(t) for t in self.thresholds]) + ";"
        for col in self.contingency_tables.columns:
            out_str += col + ":" + " ".join(["{0:d}".format(t) for t in self.contingency_tables[col]]) + ";"
        out_str = out_str.rstrip(";")
        return out_str

    def __repr__(self):
        return self.__str__()

    def from_str(self, in_str):
        """
        Read the DistributedROC string and parse the contingency table values from it.
        Args:
            in_str (str): The string output from the __str__ method
        """
        parts = in_str.split(";")
        for part in parts:
            var_name, value = part.split(":")
            if var_name == "Obs_Threshold":
                self.obs_threshold = float(value)
            elif var_name == "Thresholds":
                self.thresholds = np.array(value.split(), dtype=float)
                self.contingency_tables = pd.DataFrame(columns=self.contingency_tables.columns,
                                                       data=np.zeros((self.thresholds.size,
                                                                      self.contingency_tables.columns.size)))
            elif var_name in self.contingency_tables.columns:
                self.contingency_tables[var_name] = np.array(value.split(), dtype=int)


def performance_diagram(roc_objs, obj_labels, colors, markers, filename, figsize=(8, 8),
                        xlabel="Success Ratio (1-FAR)",
                        ylabel="Probability of Detection", ticks=np.arange(0, 1.1, 0.1),
                        dpi=300, csi_cmap="Blues",
                        csi_label="Critical Success Index", title="Performance Diagram",
                        legend_params=None, bootstrap_sets=None, ci=(2.5, 97.5), label_fontsize=14,
                        title_fontsize=16, tick_fontsize=12):

    if legend_params is None:
        legend_params = dict(loc=4, fontsize=10, framealpha=1, frameon=True)
    plt.figure(figsize=figsize)
    grid_ticks = np.arange(0, 1.01, 0.01)
    sr_g, pod_g = np.meshgrid(grid_ticks, grid_ticks)
    bias = pod_g / sr_g
    csi = 1.0 / (1.0 / sr_g + 1.0 / pod_g - 1.0)
    # csi_contour = plt.contourf(sr_g, pod_g, csi, np.arange(0.1, 1.1, 0.1), extend="max", cmap=csi_cmap)
    csi_contour = plt.contour(sr_g, pod_g, csi, np.arange(0.1, 1.1, 0.1), extend="max", colors='green')
    plt.clabel(csi_contour, fmt="%1.1f")

    # b_contour = plt.contour(sr_g, pod_g, bias, [0.5, 1.0, 1.5, 2.0, 4.0], colors="k", linestyles="dashed")
    # plt.clabel(b_contour, fmt="%1.1f", manual=[(0.2, 0.9), (0.4, 0.9), (0.6, 0.9), (0.7, 0.7), (0.9, 0.2)])
    b_contour = plt.contour(sr_g, pod_g, bias, [0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0], colors="k", linestyles="dashed")
    plt.clabel(b_contour, fmt="%1.1f", manual=[(0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.45, 0.55), (0.4, 0.6), (0.3, 0.7), (0.25, 0.775), (0.15, 0.85), (0.1, 0.9)])

    if bootstrap_sets is not None:
        for b, b_set in enumerate(bootstrap_sets):
            perf_curves = np.dstack([b_roc.performance_curve().values for b_roc in b_set])
            pod_range = np.nanpercentile(perf_curves[:, 0], ci, axis=1)
            sr_range = np.nanpercentile(1 - perf_curves[:, 1], ci, axis=1)
            pod_poly = np.concatenate((pod_range[1], pod_range[0, ::-1]))
            sr_poly = np.concatenate((sr_range[1], sr_range[0, ::-1]))
            pod_poly[np.isnan(pod_poly)] = 0
            sr_poly[np.isnan(sr_poly)] = 1
            plt.fill(sr_poly, pod_poly, alpha=0.5, color=colors[b])

    for r, roc_obj in enumerate(roc_objs):
        perf_data = roc_obj.performance_curve()
        plt.plot(1 - perf_data["FAR"], perf_data["POD"], marker=markers[r], color=colors[r], label=obj_labels[r])

    # cbar = plt.colorbar(csi_contour)
    # cbar.set_label(csi_label)

    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xticks(ticks, fontsize=tick_fontsize)
    plt.yticks(ticks, fontsize=tick_fontsize)
    plt.title(title, fontsize=title_fontsize)
    # plt.xlim(0, 1.02)
    # plt.ylim(0, 1.02)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(**legend_params)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    #  Python을 이용한 성능 다이어그램 시각화

    # 1개의 라벨값, 5개의 샘플데이터 npy 파일입니다.
    # 하나의 npy 파일에는 (60624, 100) 으로 2차원 파일이며 60,624는 시계열로 1시간부터 60,624시간까지 연속성 데이터이며, 100은 해당 시간대에 추출한 포인트 데이터입니다.
    # POD, CSI, SR, bias 계산시 [0~60624, :] 로 시간 별로 평가를 한 뒤 중앙값으로 그래프를 만들어 주셨으면 합니다.
    #
    # ex)
    # for i in range(60624):
    # true[i. :]
    # sample1[i,:]
    #
    # threshhold 값은 강수자료 이므로 0.1 로 부탁드리겠습니다.
    # 0.1 이상은 강수(1)
    # 0.1미만은 무강수(0)

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
    serviceName = 'LSH0412'

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
                }

            else:

                # 옵션 설정
                sysOpt = {
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # ********************************************************************************************
            # 자료 읽기
            # ********************************************************************************************
            # 참조 파일
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'true.npy')
            fileList = sorted(glob.glob(inpFile))
            if (len(fileList) < 1):
                raise Exception("[ERROR] inpFile : {} : {}".format("입력 자료를 확인해주세요.", inpFile))

            fileInfo = fileList[0]
            refData = np.load(fileInfo)
            refDataL1 = pd.DataFrame(refData).median(axis=1)
            refDataL2 = np.where(refDataL1 >= 0.1, 1, 0)

            # 샘플 파일
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'sample_*.npy')
            fileList = sorted(glob.glob(inpFile))
            if (len(fileList) < 1):
                raise Exception("[ERROR] inpFile : {} : {}".format("입력 자료를 확인해주세요.", inpFile))

            rocList = []
            labelList = []
            for i, fileInfo in enumerate(fileList):
                log.info(f'[CHECK] fileInfo : {fileInfo}')
                fileNameNoExt = os.path.basename(fileInfo).split('.')[0]

                simData = np.load(fileInfo)
                simDataL1 = pd.DataFrame(simData).median(axis=1)
                simDataL2 = np.where(simDataL1 >= 0.1, 1, 0)

                # rocInfo = DistributedROC(thresholds=np.arange(0, 1.0, 0.1), obs_threshold=0.1)
                rocInfo = DistributedROC(thresholds=np.array([0.5]), obs_threshold=0.5)

                # rocInfo.update(simDataL1, refDataL1)
                rocInfo.update(simDataL2, refDataL2)

                rocList.append(rocInfo)
                labelList.append(fileNameNoExt)

            cbarList = cm.rainbow(np.linspace(0, 1, len(rocList)))

            # 시각화
            mainTitle = 'Performance Diagram'
            saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
            os.makedirs(os.path.dirname(saveImg), exist_ok=True)

            performance_diagram(
                roc_objs=rocList
                , obj_labels=labelList
                , colors=cbarList
                , markers=['o'] * 5
                , figsize=(10, 10)
                , filename=saveImg
                , title=mainTitle
                , dpi=600
            )

            log.info('[CHECK] saveImg : {}'.format(saveImg))

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
