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
import re
import numpy as np
import matplotlib.pyplot as plt
from pydantic.schema import datetime
from scipy.io import loadmat
from datetime import timedelta
import seaborn as sns
from pandas.tseries.offsets import Day

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
    # Python을 이용한 레이더 및 우량계 품질검사 및 시각화

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
    serviceName = 'LSH0464'

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
                globalVar['updPath'] = '/DATA/CSV'

            # 옵션 설정
            sysOpt = {
                # 시작일, 종료일, 시간 간격
                'srtDate': '2023-06-01'
                , 'endDate': '2023-07-01'
                # 'srtDate': globalVar['srtDate']
                # , 'endDate': globalVar['endDate']
                , 'invDate': 1

                # 수행 목록
                , 'nameList': ['XLSX']
                # , 'nameList': ['CSV']

                # 모델 정보 : 파일 경로, 파일명, 시간 간격
                , 'nameInfo': {
                    'XLSX': {
                        'filePath': '/DATA/INPUT/LSH0464/PRG_err/dat'
                        , 'fileName': 'DATA_GvsR_SBS_실시간_*월_TEST.xlsx'
                        , 'searchKey': 'SBS'
                    }
                    , 'CSV': {
                        'filePath': '/DATA/INPUT/LSH0464/%Y/%Y%m/%d'
                        , 'fileName': 'RDR_SBS_GvsR_%Y%m%d.txt'
                        , 'fileName2': 'RDR_SBS_DPV_BIAS_%Y%m%d.txt'
                        , 'searchKey': 'SBS'
                    }
                }
            }

            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Day(sysOpt['invDate']))

            for nameType in sysOpt['nameList']:
                log.info(f'[CHECK] nameType : {nameType}')

                namelInfo = sysOpt['nameInfo'].get(nameType)
                if namelInfo is None: continue

                # ********************************************************************
                # 전처리
                # ********************************************************************
                if re.search('CSV', nameType, re.IGNORECASE):

                    dataL1 = pd.DataFrame()
                    dtDateInfo = dtDateList[0]
                    for i, dtDateInfo in enumerate(dtDateList):

                        log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')

                        inpFile = '{}/{}'.format(namelInfo['filePath'], namelInfo['fileName'])
                        inpFileDate = dtDateInfo.strftime(inpFile)

                        inpFile2 = '{}/{}'.format(namelInfo['filePath'], namelInfo['fileName2'])
                        inpFileDate2 = dtDateInfo.strftime(inpFile2)
                        fileList = glob.glob(inpFileDate) + glob.glob(inpFileDate2)

                        if fileList is None or len(fileList) < 2:
                            # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                            continue

                        fileInfo = fileList[0]
                        fileInfo2 = fileList[1]
                        log.info(f'[CHECK] fileInfo : {fileInfo}')
                        log.info(f'[CHECK] fileInfo2 : {fileInfo2}')

                        if not re.search(namelInfo['searchKey'], fileInfo, re.IGNORECASE): continue

                        fileNameNoExt = os.path.basename(fileInfo).split('.')[0]
                        data = pd.read_csv(fileInfo, delimiter='\s+', header=None, names=['date', 'site', 'AWS', 'RDRorg', 'RDRnew'])

                        sheetList = pd.ExcelFile(fileInfo).sheet_names
                        # sheetInfo = sheetList[1]
                        for sheetInfo in sheetList:
                            log.info(f'[CHECK] sheetInfo : {sheetInfo}')

                            data = pd.read_excel(fileInfo, sheet_name=sheetInfo, usecols="I:M", nrows=25000 - 4,
                                                 skiprows=3, header=None,
                                                 names=['date', 'site', 'AWS', 'RDRorg', 'RDRnew'])

                            if len(data) < 1: continue

                            # total(daily) rainfall, hourly max rainfall
                            statData = pd.read_excel(fileInfo, sheet_name=sheetInfo, nrows=2, skiprows=1, header=None).iloc[1,]
                            chk = statData[25]
                            chk2 = statData[10]
                            log.info(f'[CHECK] total(daily) rainfall : {chk}')
                            log.info(f'[CHECK] hourly max rainfall : {chk2}')

                            if not (chk > 100 and chk2 > 20): continue

                            # site 마다 24개(시간) 분포
                            datS = data.sort_values(by=['site', 'date']).dropna().reset_index(drop=True)
                            dataL1 = pd.concat([dataL1, datS], ignore_index=True)

                        if len(dataL1) < 1: continue

                        # 자료 형변환
                        dataL1['site'] = dataL1['site'].astype(int).astype(str)
                        dataL1['date'] = dataL1['date'].astype(int).astype(str)

                        # 0보다 작은 경우 0으로 대체
                        dataL1['AWS'] = np.where(dataL1['AWS'] < 0, 0, dataL1['AWS'])
                        dataL1['RDRorg'] = np.where(dataL1['RDRorg'] < 0, 0, dataL1['RDRorg'])
                        dataL1['RDRnew'] = np.where(dataL1['RDRnew'] < 0, 0, dataL1['RDRnew'])

                        # CSV 저장
                        saveFile = '{}/{}/{}{}.csv'.format(globalVar['outPath'], serviceName, namelInfo['searchKey'],
                                                           'errRstSite')
                        os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                        # dataL1.to_csv(saveFile, index=False)
                        dataL1 = pd.read_csv(saveFile)
                        log.info(f'[CHECK] saveFile : {saveFile}')

                        # # date, site를 기준으로 AWS, RDRorg, RDRnew 생성
                        # datSTAa = dataL1.pivot_table(index='date', columns=['site'], values='AWS')
                        # datSTOa = dataL1.pivot_table(index='date', columns=['site'], values='RDRorg')
                        # datSTNa = dataL1.pivot_table(index='date', columns=['site'], values='RDRnew')
                        #
                        # # 엑셀 저장
                        # saveFile = '{}/{}/{}{}.xlsx'.format(globalVar['outPath'], serviceName, namelInfo['searchKey'], 'errRstSite')
                        # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                        # with pd.ExcelWriter(saveFile, engine='openpyxl') as writer:
                        #     datSTAa.to_excel(writer, sheet_name='RSTA', startcol=1, startrow=1, index=True)
                        #     datSTOa.to_excel(writer, sheet_name='RSTO', startcol=1, startrow=1, index=True)
                        #     datSTNa.to_excel(writer, sheet_name='RSTN', startcol=1, startrow=1, index=True)
                        # log.info(f'[CHECK] saveFile : {saveFile}')

                elif re.search('XLSX', nameType, re.IGNORECASE):
                    # procCSV(namelInfo)
                    inpFile = '{}/{}'.format(namelInfo['filePath'], namelInfo['fileName'])
                    # inpFileDate = dtDateInfo.strftime(inpFile)
                    fileList = sorted(glob.glob(inpFile))

                    if fileList is None or len(fileList) < 1:
                        log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                        continue

                    # fileInfo = fileList[0]
                    dataL1 = pd.DataFrame()
                    for fileInfo in fileList:
                        log.info(f'[CHECK] fileInfo : {fileInfo}')

                        if not re.search(namelInfo['searchKey'], fileInfo, re.IGNORECASE): continue

                        sheetList = pd.ExcelFile(fileInfo).sheet_names
                        # sheetInfo = sheetList[1]
                        for sheetInfo in sheetList:
                            log.info(f'[CHECK] sheetInfo : {sheetInfo}')

                            data = pd.read_excel(fileInfo, sheet_name=sheetInfo, usecols="I:M", nrows=25000 - 4,
                                                 skiprows=3, header=None,
                                                 names=['date', 'site', 'AWS', 'RDRorg', 'RDRnew'])

                            data['AWSflag'] = np.where((0 <= data['AWS']) & (data['AWS'] <= 1), data['AWS'], -99.9)

                            if len(data) < 1: continue

                            # total(daily) rainfall, hourly max rainfall
                            statData = pd.read_excel(fileInfo, sheet_name=sheetInfo, nrows=2, skiprows=1, header=None).iloc[1,]
                            chk = statData[25]

                            np.nansum(data['RDRorg'] > 0.1)
                            np.nansum(data['RDRnew'] > 0.1)

                            chk2 = statData[10]
                            np.nanmax(data['AWS'])
                            log.info(f'[CHECK] total(daily) rainfall : {chk}')
                            log.info(f'[CHECK] hourly max rainfall : {chk2}')

                            if not (chk > 100 and chk2 > 20): continue

                            # site 마다 24개(시간) 분포
                            datS = data.sort_values(by=['site', 'date']).dropna().reset_index(drop=True)
                            dataL1 = pd.concat([dataL1, datS], ignore_index=True)

                    if len(dataL1) < 1: continue

                    # 자료 형변환
                    dataL1['site'] = dataL1['site'].astype(int).astype(str)
                    dataL1['date'] = dataL1['date'].astype(int).astype(str)

                    # 0보다 작은 경우 0으로 대체
                    dataL1['AWS'] = np.where(dataL1['AWS'] < 0, 0, dataL1['AWS'])
                    dataL1['RDRorg'] = np.where(dataL1['RDRorg'] < 0, 0, dataL1['RDRorg'])
                    dataL1['RDRnew'] = np.where(dataL1['RDRnew'] < 0, 0, dataL1['RDRnew'])

                    # CSV 저장
                    saveFile = '{}/{}/{}{}.csv'.format(globalVar['outPath'], serviceName, namelInfo['searchKey'],
                                                       'errRstSite')
                    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    # dataL1.to_csv(saveFile, index=False)
                    dataL1 = pd.read_csv(saveFile)
                    log.info(f'[CHECK] saveFile : {saveFile}')

                    # # date, site를 기준으로 AWS, RDRorg, RDRnew 생성
                    # datSTAa = dataL1.pivot_table(index='date', columns=['site'], values='AWS')
                    # datSTOa = dataL1.pivot_table(index='date', columns=['site'], values='RDRorg')
                    # datSTNa = dataL1.pivot_table(index='date', columns=['site'], values='RDRnew')
                    #
                    # # 엑셀 저장
                    # saveFile = '{}/{}/{}{}.xlsx'.format(globalVar['outPath'], serviceName, namelInfo['searchKey'], 'errRstSite')
                    # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                    # with pd.ExcelWriter(saveFile, engine='openpyxl') as writer:
                    #     datSTAa.to_excel(writer, sheet_name='RSTA', startcol=1, startrow=1, index=True)
                    #     datSTOa.to_excel(writer, sheet_name='RSTO', startcol=1, startrow=1, index=True)
                    #     datSTNa.to_excel(writer, sheet_name='RSTN', startcol=1, startrow=1, index=True)
                    # log.info(f'[CHECK] saveFile : {saveFile}')

                else:
                    log.error(f'모델 종류 ({nameType})를 확인해주세요.')
                    sys.exit(1)

                # ********************************************************************
                # 시각화
                # ******************************************************************
                dataL1['dtDateTime'] = pd.to_datetime(dataL1['date'], format='%Y%m%d%H%M')
                dataL1['dtDate'] = dataL1['dtDateTime'].dt.date

                # site 별로 출력 (각 사이트 별로 시간 단위로 일기간 만큼 출력)
                siteList = sorted(set(dataL1['site']))
                dtDateList = sorted(set(dataL1['dtDate']))
                # siteInfo = siteList[0]
                # dtDateInfo = dtDateList[0]
                for siteInfo in siteList:
                    for dtDateInfo in dtDateList:

                        minDate = dtDateInfo
                        maxDate = dtDateInfo + timedelta(days=1)

                        dataL2 = dataL1.loc[(dataL1['site'] == siteInfo) & (minDate <= dataL1['dtDate']) & (dataL1['dtDate'] <= maxDate)].reset_index(drop=True)
                        if len(dataL2) < 1: continue

                        log.info(f'[CHECK] siteInfo : {siteInfo} / dtDateInfo : {dtDateInfo}')

                        # 시간
                        dataL2['cumHour'] = ((dataL2['dtDateTime'] - pd.to_datetime(minDate)).dt.total_seconds() / 3600).astype(int)

                        sumAWS = np.nansum(dataL2['AWS'])
                        maxAWS = np.nanmax(dataL2['AWS'])

                        log.info(f'[CHECK] sumAWS : {sumAWS}')
                        log.info(f'[CHECK] maxAWS : {maxAWS}')
                        if not (sumAWS > 10): continue

                        rRat = 0.65

                        # 우량계 기준치 (상한)
                        dataL2['dRUbnd'] = 6.4 * (dataL2['RDRorg'] ** 0.725)
                        # 레이더 기준치 (하한)
                        dataL2['dGLbnd'] = dataL2['AWS'] * rRat
                        dataL2['dRLbnd'] = 0.04 * (dataL2['RDRorg'] ** 1.45)

                        dataL3 = dataL2.melt(id_vars=['cumHour'], value_vars=['AWS', 'RDRorg', 'RDRnew'], var_name='key', value_name='val')
                        dataL4 = dataL3.pivot(index='cumHour', columns='key', values='val').reset_index()

                        # 시각화
                        mainTitle = f'SBS_{siteInfo}_{minDate.strftime("%Y%m%d%H%M")}-{maxDate.strftime("%Y%m%d%H%M")}'
                        saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                        os.makedirs(os.path.dirname(saveImg), exist_ok=True)

                        fig, ax = plt.subplots(figsize=(10, 6), dpi=600)

                        # 바 그래프
                        ax = dataL4.plot(x='cumHour', y = ['AWS', 'RDRorg', 'RDRnew'], kind='bar', width=0.8, color=['black', 'red', 'green'], label=['우량계', '레이더', '레이더 (보정후)'])

                        # 선 그래프
                        ax.plot(dataL2['cumHour'], dataL2['dRUbnd'], 'k-.', linewidth=1.0, label='우량계 기준치 (상한)')
                        ax.plot(dataL2['cumHour'], dataL2['dGLbnd'], 'm-.', linewidth=1.0, label='레이더 기준치 (하한)')
                        ax.plot(dataL2['cumHour'], dataL2['dRLbnd'], 'c-.', linewidth=1.0, label=None)

                        # 점 그래프
                        for i, row in dataL2.iterrows():
                            if row['AWS'] > row['dRUbnd']:
                                plt.plot(row['cumHour'], 34, 'rv', markerfacecolor='y', markersize=10)

                            if row['RDRorg'] > row['dGLbnd']:
                                plt.plot(row['cumHour'], 30, 'r^', markerfacecolor='r', markersize=5)

                            if row['RDRnew'] > row['dGLbnd']:
                                plt.plot(row['cumHour'], 32, 'go', markerfacecolor='g', markersize=5)

                        # xmag = 0.0
                        xmag = 11
                        plt.text(xmag, 30, '레이더 기준치 미만족', fontsize=8)
                        plt.text(xmag, 32, '레이더 보정 후 만족', fontsize=8)
                        plt.text(xmag, 34, '우량계 이상치 추정', fontsize=8)

                        # 범례 표시
                        ax.legend(loc='upper right', fontsize=8)
                        plt.xlabel('Time (hour)')
                        plt.ylabel('Hourly rainfall (mm)')
                        plt.title(mainTitle)
                        plt.grid(linewidth=0.2)

                        # if maxAWS > 0:
                        #     plt.ylim([0, maxAWS * 1.2])
                        # else:
                        #     plt.ylim([0, 30 * 1.2])
                        plt.ylim([0, 30 * 1.2])

                        plt.setp(ax.get_xticklabels(), rotation=0)
                        plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
                        plt.tight_layout()
                        # plt.show()
                        plt.close()

                        log.info(f'[CHECK] saveImg : {saveImg}')

        except Exception as e:
            log.error(f'Exception : {e}')

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
