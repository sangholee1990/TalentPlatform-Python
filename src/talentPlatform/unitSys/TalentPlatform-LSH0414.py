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

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):
    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 뇌파 측정 자료 처리 및 스펙트럼 및 푸리에 변환 시각화

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
    serviceName = 'LSH0415'

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
            import rhd
            from mne.io import read_epochs_eeglab as loadeeg

            # 참조 파일
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '3_230325_175822.rhd')
            fileList = sorted(glob.glob(inpFile))
            if (len(fileList) < 1):
                raise Exception(f'[ERROR] inpFile : {inpFile} : 입력 자료를 확인해주세요.')

            data = rhd.read_data(fileList[0], no_floats=True)

            data.keys()
            # 증폭기 채널 정보
            # data['amplifier_channels']
            chLabel = []
            for i, chInfo in enumerate(data['amplifier_channels']):
                chLabel.append(chInfo['custom_channel_name'])
            # 증폭기 데이터
            data['amplifier_data'][11]

            # 증폭기 시간
            data['t_amplifier']

            # plt.plot(data['t_supply_voltage'], data['supply_voltage_data'][0])
            # plt.show()




            # plt.plot( data['t_amplifier'], data['amplifier_data'][11])
            # plt.show()

            x =  data['t_amplifier']
            # y =  data['amplifier_data'][0]
            y =  data['amplifier_data']
            # spacing = 3000
            spacing = 300000

            plot_multichan(x, y, spacing = spacing, ch_names = chLabel)

            def plot_multichan(x, y, spacing=3000, figsize=(10, 10), ch_names = chLabel):
                # Set color theme
                color_template = np.array([[1, .09, .15], [1, .75, .28], [.4, .2, 0], [.6, .7, .3], [.55, .55, .08]])
                color_space = np.tile(color_template,
                                      (int(np.ceil([float(y.shape[0]) / color_template.shape[0]])[0]), 1))
                # Open figure and plot
                # plt.figure(figsize=figsize)
                y_center = np.linspace(-spacing, spacing, int(y.shape[0]))
                for chanIdx in range(y.shape[0]):
                    shift = y_center[chanIdx] + np.nanmean(y[chanIdx, :])
                    plt.plot(x, y[chanIdx, :] - shift, color=color_space[chanIdx,], linewidth=1)
                plt.xlabel('Time (sec)')
                plt.ylim((-1.1 * spacing, 1.1 * spacing))
                plt.yticks(y_center, ch_names[::-1])
                plt.gca().set_facecolor((1, 1, 1))
                plt.show()

                return y_center

            # Demo 2-3. Visualization of ERP time trace
            targetCondition = 6  # <- Try changing this
            trialIdx = np.where((EEG.events[:, 2]) == targetCondition)[0]
            erp = np.nanmean(EEG.data[:, :, trialIdx], 2)
            c = plot_multichan(EEG.times, erp, spacing=300)
            plt.title('ERP sample. Condition: %s' % (condNames[targetCondition - 1]));
            plt.gcf().savefig(dir_fig + 'fig2-3.png', format='png', dpi=300);



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
