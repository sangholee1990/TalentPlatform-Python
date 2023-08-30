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
from scipy.io import loadmat

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
                'srtDate': '2022-06-01'
                , 'endDate': '2022-06-02'
                # 'srtDate': globalVar['srtDate']
                # , 'endDate': globalVar['endDate']
                , 'invHour': 1

                # 수행 목록
                , 'nameList': ['XLSX']

                # 모델 정보 : 파일 경로, 파일명, 시간 간격
                , 'nameInfo': {
                    'XLSX': {
                        'filePath': '/DATA/INPUT/LSH0464/PRG_err/dat'
                        , 'fileName': 'DATA_GvsR_SBS_실시간_*월_TEST.xlsx'
                        , 'searchKey': 'SBS'
                    }
                    , 'CSV': {
                        'filePath': '/DATA/INPUT/LSH0464/PRG_err/dat'
                        , 'fileName': 'DATA_GvsR_SBS_실시간_*월_TEST.xlsx'
                        , 'searchKey': 'SBS'
                    }
                }
            }

            for nameType in sysOpt['nameList']:
                log.info(f'[CHECK] nameType : {nameType}')

                namelInfo = sysOpt['nameInfo'].get(nameType)
                if namelInfo is None: continue

                # # ********************************************************************
                # # 전처리
                # # ********************************************************************
                # inpFile = '{}/{}'.format(namelInfo['filePath'], namelInfo['fileName'])
                # # inpFileDate = dtDateInfo.strftime(inpFile)
                # fileList = sorted(glob.glob(inpFile))
                #
                # if fileList is None or len(fileList) < 1:
                #     log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
                #     continue
                #
                # # fileInfo = fileList[0]
                # dataL1 = pd.DataFrame()
                # for fileInfo in fileList:
                #     log.info(f'[CHECK] fileInfo : {fileInfo}')
                #
                #     if not re.search(namelInfo['searchKey'], fileInfo, re.IGNORECASE): continue
                #
                #     sheetList = pd.ExcelFile(fileInfo).sheet_names
                #     # sheetInfo = sheetList[1]
                #     for sheetInfo in sheetList:
                #         log.info(f'[CHECK] sheetInfo : {sheetInfo}')
                #
                #         data = pd.read_excel(fileInfo, sheet_name=sheetInfo, usecols="I:M", nrows=25000 - 4, skiprows=3, header=None, names=['date', 'site', 'AWS', 'RDRorg', 'RDRnew'])
                #
                #         if len(data) < 1: continue
                #
                #         # total(daily) rainfall, hourly max rainfall
                #         statData = pd.read_excel(fileInfo, sheet_name=sheetInfo, nrows=2, skiprows=1, header=None).iloc[1,]
                #         chk = statData[25]
                #         chk2 = statData[10]
                #         log.info(f'[CHECK] total(daily) rainfall : {chk}')
                #         log.info(f'[CHECK] hourly max rainfall : {chk2}')
                #
                #         if not (chk > 100 and chk2 > 20): continue
                #
                #         # site 마다 24개(시간) 분포
                #         datS = data.sort_values(by=['site', 'date']).dropna().reset_index(drop=True)
                #         dataL1 = pd.concat([dataL1, datS], ignore_index=True)
                #
                # if len(dataL1) < 1: continue
                #
                # # 자료 형변환
                # dataL1['site'] = dataL1['site'].astype(int).astype(str)
                # dataL1['date'] = dataL1['date'].astype(int).astype(str)
                #
                # # 0보다 작은 경우 0으로 대체
                # dataL1['AWS'] = np.where(dataL1['AWS'] < 0, 0, dataL1['AWS'])
                # dataL1['RDRorg'] = np.where(dataL1['RDRorg'] < 0, 0, dataL1['RDRorg'])
                # dataL1['RDRnew'] = np.where(dataL1['RDRnew'] < 0, 0, dataL1['RDRnew'])

                # CSV 저장
                saveFile = '{}/{}/{}{}.csv'.format(globalVar['outPath'], serviceName, namelInfo['searchKey'], 'errRstSite')
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

                        # start_date = datetime.date(2022, 6, 24)
                        # end_date = start_date + datetime.timedelta(days=1)

                        dtDateInfo + datetime.timedelta(days=1)


                        dataL2 = dataL1.loc[(dataL1['site'] == siteInfo) & (dtDateInfo <= dataL1['dtDate']) & (dataL1['dtDate'] <= end_date)].reset_index(drop=True)
                        if len(dataL2) < 1: continue

                        log.info(f'[CHECK] siteInfo : {siteInfo} / dtDateInfo : {dtDateInfo}')

                        sumAWS = np.nansum(dataL2['AWS'])
                        maxAWS = np.nanmax(dataL2['AWS'])

                        log.info(f'[CHECK] sumAWS : {sumAWS}')
                        log.info(f'[CHECK] maxAWS : {maxAWS}')
                        if not (sumAWS > 10): continue

                        rRat = 0.65
                        dataL2['dGLbnd'] = dataL2['AWS'] * rRat
                        dataL2['dRLbnd'] = 0.04 * (dataL2['RDRorg'] ** 1.45)
                        dataL2['dRUbnd'] = 6.4 * (dataL2['RDRorg'] ** 0.725)

                        # dt = dataL2['dtDateTime']
                        # dAWS = dataL2['AWS']
                        # dRDR = dataL2['RDRorg']
                        # dRDRa = dataL2['RDRnew']

                        # dGLbnd = dAWS * rRat
                        # dRLbnd = 0.04 * (dRDR ** 1.45)
                        # dRUbnd = 6.4 * (dRDR ** 0.725)


                        fig, ax = plt.subplots()
                        # xmag = dt[0]
                        # ax.set_xlim([0, len(dt) + 1])
                        if maxAWS > 0:
                            ax.set_ylim([0, maxAWS * 1.2])
                        else:
                            ax.set_ylim([0, 30 * 1.2])

                        # mainTitle = '{}'.format('담배 인상 전후에 따른 성별별 흡연자 비중')
                        # saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, mainTitle)
                        # os.makedirs(os.path.dirname(saveImg), exist_ok=True)


                        melted = dataL2.melt(id_vars=['dtDateTime'], value_vars=['AWS', 'RDRorg', 'RDRnew'], var_name='key', value_name='val')

                        custom_colors = {
                            'AWS': 'black',
                            'RDRorg': 'red',
                            'RDRnew': 'green'
                        }

                        # 월별 강수량 합계 계산
                        # grouped = df.groupby('Month').sum()

                        # 바 차트 그리기
                        # melted.plot(kind='bar', color='blue', legend=False)
                        # plt.show()

                        import matplotlib.dates as mdates
                        fig, ax = plt.subplots()

                        # 미리 'dtDateTime'을 날짜/시간 형식으로 변환
                        melted['dtDateTime'] = pd.to_datetime(melted['dtDateTime'])

                        # fig, ax = plt.subplots()

                        pivot_df = melted.pivot(index='dtDateTime', columns='key', values='val').reset_index()
                        pivot_df['dtDateTime'] = pd.to_datetime(pivot_df['dtDateTime'])

                        ax = pivot_df.plot(x='dtDateTime', kind='bar', width=0.8)

                        plt.xlabel('Date Time')
                        plt.ylabel('Value')
                        plt.title('Bar Chart per Key and Date Time')
                        plt.tight_layout()
                        plt.grid(axis='y')

                        # x축의 눈금 간격 설정: 여기서는 2시간 간격으로 설정
                        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        # plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                        # ax.set_xticks(range(len(pivot_df)))
                        ax.set_xticklabels(pivot_df['dtDateTime'].dt.strftime('%H'))

                        # plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                        plt.setp(ax.get_xticklabels(), rotation=0)
                        plt.show()





                        plt.plot(dataL2['dtDateTime'], dataL2['AWS'], linestyle="-", label="cosine")
                        plt.plot(dataL2['dtDateTime'], dataL2['RDRorg'])
                        plt.plot(dataL2['dtDateTime'], dataL2['RDRnew'])
                        plt.show()


                        ax = sns.barplot(data=melted, x='dtDateTime', y='val', hue='key', ci=None, palette=custom_colors)
                        handles, _ = ax.get_legend_handles_labels()
                        new_labels = ['우량계', '레이더', '레이더(보정후)']
                        ax.legend(title=None, handles=handles, loc='upper left', labels=new_labels)

                        start_date = melted['dtDateTime'].min()
                        end_date = melted['dtDateTime'][100]
                        date_ticks = pd.date_range(start=start_date, end=end_date, freq='10')
                        tick_positions = melted['dtDateTime'].isin(date_ticks).to_numpy().nonzero()[0]

                        # Ensure tick_positions and date_ticks have the same length
                        min_length = min(len(tick_positions), len(date_ticks))
                        tick_positions = tick_positions[:min_length]
                        date_ticks = date_ticks[:min_length]

                        plt.xticks(ticks=tick_positions, labels=date_ticks.strftime('%Y-%m-%d'), rotation=45, ha='right')
                        plt.tight_layout()
                        plt.show()


                    # for i in ax.containers:
                    #     ax.bar_label(i, )
                    # plt.ylabel('흡연 비중')
                    # plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                    # plt.tight_layout()
                    # plt.show()
                    # plt.close()

                    # ax.plot(dt, dAWS, '-', color=[0.8, 0.8, 0.8], linewidth=0.5)
                    # ax.text(xmag + 0.2, 30, '레이더 기준치 미만족', fontsize=8)
                    # ax.text(xmag + 0.2, 32, '레이더 보정 후 만족', fontsize=8)
                    # ax.text(xmag + 0.2, 34, '우량계 이상치 추정', fontsize=8)
                    #
                    # ax.text(xmag + 2, 23.5, '-\cdot- 우량계 기준치(상한)', fontsize=9, color='k')
                    # ax.text(xmag + 2, 22, '-\cdot- 레이더 기준치(하한)', fontsize=9, color='m')
                    # ax.plot(xmag + 2, 20, 's', color='k', markerfacecolor='k', markersize=9)
                    # ax.text(xmag + 2.5, 20, '우량계', fontsize=9, color='k')
                    # ax.plot(xmag + 2, 18.5, 's', color='r', markerfacecolor='r', markersize=9)
                    # ax.text(xmag + 2.5, 18.5, '레이더', fontsize=9, color='r')
                    # ax.plot(xmag + 2, 17, 's', color='g', markerfacecolor='g', markersize=9)
                    # ax.text(xmag + 2.5, 17, '레이더(보정후)', fontsize=9, color='g')

                    fig, ax = plt.subplots()
                    ax.set_xlabel('Time(hour)')
                    ax.set_ylabel('Hourly rainfall(mm)')

                    bars = dataL2[['AWS']].plot(kind='bar', color=['k', 'r', 'g'], ax=ax, linewidth=0.5)
                    plt.show()
                    # for t in range(len(x)):
                    #     bars = ax.bar(range(1, t + 2), [dAWS[:t + 1, 0], dRDR[:t + 1, 0], dRDRa[:t + 1, 0]], color=['k', 'r', 'g'], linewidth=0.5)
                    #     if dAWS[t, 0] > dRUbnd[t, 0]:
                    #         ax.plot(t + 0.75, 34, 'rv', markerfacecolor='y', markersize=10)
                    #         tck = t
                    #     if dRDR[t, 0] > dGLbnd[t, 0]:
                    #         ax.plot(t + 1, 30, 'r^', markerfacecolor='r', markersize=5)
                    #     if dRDRa[t, 0] > dGLbnd[t, 0]:
                    #         ax.plot(t + 1, 32, 'go', markerfacecolor='g', markersize=5)
                    #     ax.plot(range(1, t + 2), dGLbnd[:t + 1, 0], 'm-.', linewidth=1.0)
                    #     ax.plot(range(1, t + 2), dRLbnd[:t + 1, 0], 'c-.', linewidth=1.0)
                    #     ax.plot(range(1, t + 2), dRUbnd[:t + 1, 0], 'k-.', linewidth=1.0)
                    #     ax.set_xlim([0, len(x) + 1])
                    #     ax.set_ylim([0, 30 * 1.2])
                    #     plt.draw()
                    #     plt.grid(True)
                    #     plt.box(True)

                    # plt.show()

                    # plt.figure()
                    # plt.xlim([0, len(x) + 1])
                    # if dAWS.any() > 0:
                    #     plt.ylim([0, max(dAWS) * 1.2])
                    # else:
                    #     plt.ylim([0, 30 * 1.2])
                    #
                    # plt.plot(range(1, len(x) + 1), dAWS, '-', color=[0.8, 0.8, 0.8], linewidth=0.5)

                    # Plotting
                    # fig, ax = plt.subplots()
                    #
                    # # for spine in ax.spines.values():
                    # #     spine.set_visible(True)
                    #
                    # # Setting xlim, ylim
                    # # ax.set_xlim([dt[0], dt[len(dataL2) - 1]])
                    # if dAWS.max() > 0:
                    #     ax.set_ylim([0, dAWS.max() * 1.2])
                    # else:
                    #     ax.set_ylim([0, 30 * 1.2])
                    #
                    # # Adding text to the plot
                    # ax.text(dt[0], 30, '레이더 기준치 미만족', fontsize=8)
                    # # ... (other texts)
                    #
                    # # Plotting data
                    # ax.plot(dt, dAWS, '-', color=[0.8, 0.8, 0.8], linewidth=0.5)
                    #
                    # bar_width = 0.3
                    # indices = np.arange(len(dt))
                    #
                    # ax.bar(indices - bar_width, dAWS, bar_width, color='black', label='dAWS')
                    # ax.bar(indices, dRDR, bar_width, color='red', label='dRDR', bottom=dAWS)
                    # ax.bar(indices + bar_width, dRDRa, bar_width, color='green', label='dRDRa', bottom=dAWS + dRDR)
                    #
                    # # More plotting commands (plot arrows, other lines, etc.)
                    #
                    # ax.set_xlabel('Time(hour)')
                    # ax.set_ylabel('Hourly rainfall(mm)')
                    # ax.grid(True)
                    # # ax.box(True)
                    #
                    # plt.show()

                    #
                    # ax.grid(True)
                    # plt.show()
                    #
                    # # plt.savefig(f"{RDRnam}_{datSTa[y]}.png", dpi=300)
                    # # plt.close(fig)
                    #

            #     # 24개(시간)씩 site개수 만큼 반복
            #     aa = datS.iloc[:, 1].values
            #     dv = (aa != aa[0]).argmax()
            #
            #     # site, date, AWS, RDR org, RDR new
            #     datST = datS.iloc[:, 1].values.reshape(dv, -1)
            #     datSTD = datS.iloc[:, 0].values.reshape(dv, -1)
            #     datSTA = datS.iloc[:, 2].values.reshape(dv, -1)
            #     datSTO = datS.iloc[:, 3].values.reshape(dv, -1)
            #     datSTN = datS.iloc[:, 4].values.reshape(dv, -1)
            #
            #     if icn == 1:
            #         # site, date, AWS, RDR org, RDR new
            #         datSTa = datST[0, :]
            #         datSTDa = datSTD[:, 0]
            #         datSTAa = datSTA
            #         datSTOa = datSTO
            #         datSTNa = datSTN
            #     else:
            #         # site x day, sites
            #         datSTDa = np.hstack([datSTDa, datSTD[:, 0]])
            #         datSTAa = np.hstack((datSTAa, datSTA))
            #         datSTOa = np.hstack([datSTOa, datSTO])
            #         datSTNa = np.hstack([datSTNa, datSTN])
            #
            #                 # datSTAa.shape
            #                 # datSTA.shape
            #
            #     datSTAa[datSTAa < 0] = 0
            #     datSTOa[datSTOa < 0] = 0
            #     datSTNa[datSTNa < 0] = 0
            #
            #     saveFile = '{}/{}/{}{}.xlsx'.format(globalVar['outPath'], serviceName, modelInfo['searchKey'], 'errRstSite')
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     with pd.ExcelWriter(saveFile, engine='openpyxl') as writer:
            #         pd.DataFrame({'site': datSTa}).to_excel(writer, sheet_name='RSTA', startcol=2, startrow=1, index=False)
            #         pd.DataFrame({'date': datSTDa}).to_excel(writer, sheet_name='RSTA', startcol=1, startrow=2, index=False)
            #         pd.DataFrame(datSTNa).to_excel(writer, sheet_name='RSTA', startcol=2, startrow=2, index=False)
            #     log.info(f'[CHECK] saveFile : {saveFile}')

            #
            # datST = data_sorted[data_sorted.columns[1]].values.reshape(dv, -1)
            # datSTD = data_sorted[data_sorted.columns[0]].values.reshape(dv, -1)
            # datSTA = data_sorted[data_sorted.columns[2]].values.reshape(dv, -1)
            # datSTO = data_sorted[data_sorted.columns[3]].values.reshape(dv, -1)
            # datSTN = data_sorted[data_sorted.columns[4]].values.reshape(dv, -1)
            #
            # chk = pd.read_excel(xls, sheet, usecols="Z", nrows=1, skiprows=2).values[0][0]
            # chk2 = pd.read_excel(xls, sheet, usecols="K", nrows=1, skiprows=2).values[0][0]
            #
            # if chk > 100 and chk2 > 20:
            #     if icn == 1:
            #         datSTa = datST[0, :]
            #         datSTDa = datSTD[:, 0]
            #         datSTAa = datSTA
            #         datSTOa = datSTO
            #         datSTNa = datSTN
            #     else:
            #         datSTDa = np.concatenate([datSTDa, datSTD[:, 0]])
            #         datSTAa = np.concatenate([datSTAa, datSTA])
            #         datSTOa = np.concatenate([datSTOa, datSTO])
            #         datSTNa = np.concatenate([datSTNa, datSTN])
            #
            #
            #
            #
            #
            # datS = dat[np.argsort(dat[:, 1]), :]
            # aa = datS[:, 1]
            # aat = aa - aa[0]
            # dv = np.where(aat)[0][0]
            #
            # datST = np.reshape(datS[:, 1], (dv, datS.shape[0] // dv))
            # datSTD = np.reshape(datS[:, 0], (dv, datS.shape[0] // dv))
            # datSTA = np.reshape(datS[:, 2], (dv, datS.shape[0] // dv))
            # datSTO = np.reshape(datS[:, 3], (dv, datS.shape[0] // dv))
            # datSTN = np.reshape(datS[:, 4], (dv, datS.shape[0] // dv))
            #
            # chk = pd.read_excel(xls_path, sheet_name=sheet, usecols="Z", nrows=1).values
            # chk2 = pd.read_excel(xls_path, sheet_name=sheet, usecols="K", nrows=1).values
            #
            # if chk > 100 and chk2 > 20:
            #     if icn == 1:
            #         datSTa_list.append(datST[0, :])
            #         datSTDa_list.extend(datSTD[:, 0])
            #         datSTAa_list.extend(datSTA.ravel())
            #         datSTOa_list.extend(datSTO.ravel())
            #         datSTNa_list.extend(datSTN.ravel())
            #     else:
            #         datSTDa_list.extend(datSTD[:, 0])
            #         datSTAa_list.extend(datSTA.ravel())
            #         datSTOa_list.extend(datSTO.ravel())
            #         datSTNa_list.extend(datSTN.ravel())

            #
            # # NetCDF 파일 읽기
            # for j, fileInfo in enumerate(fileList):
            #     data = xr.open_dataset(fileInfo, engine='pynio')
            #     log.info(f'[CHECK] fileInfo : {fileInfo}')

            # Get list of xls files
            # xlsList = [f for f in os.listdir(xlsFolderR) if f.startswith('DATA_GvsR_' + RDRnam) and f.endswith('.xlsx')]

            # icn = 0
            # datSTa_list, datSTDa_list, datSTAa_list, datSTOa_list, datSTNa_list = [], [], [], [], []
            #
            # for file in xlsList:
            #     filename = os.path.join(xlsFolderR, file)
            #     if file[10:13] == RDRnam:
            #         print(file)
            #         icn += 1
            #             xls = pd.ExcelFile(fileInfo)
            #             sheets = xls.sheet_names
            #             for sheet in sheets:
            #                 dat = pd.read_excel(filename, sheet_name=sheet, usecols="I:M", skiprows=3, nrows=24997)
            #                 chk = pd.read_excel(filename, sheet_name=sheet, usecols="Z", skiprows=2, nrows=1).values[0][0]
            #                 chk2 = pd.read_excel(filename, sheet_name=sheet, usecols="K", skiprows=2, nrows=1).values[0][0]
            #
            #                 if chk > 100 and chk2 > 20:
            #                     if not dat.empty:
            #                         datS = dat.sort_values(by=dat.columns[1])
            #                         aat = datS[datS.columns[1]] - datS[datS.columns[1]].iloc[0]
            #                         dv = aat.ne(0).idxmax()
            #
            #                         datST = datS[datS.columns[1]].values.reshape(dv, -1)
            #                         datSTD = datS[datS.columns[0]].values.reshape(dv, -1)
            #                         datSTA = datS[datS.columns[2]].values.reshape(dv, -1)
            #                         datSTO = datS[datS.columns[3]].values.reshape(dv, -1)
            #                         datSTN = datS[datS.columns[4]].values.reshape(dv, -1)
            #
            #                         if icn == 1:
            #                             datSTa_list, datSTDa_list, datSTAa_list, datSTOa_list, datSTNa_list = \
            #                                 [datST[0, :]], [datSTD[:, 0]], [datSTA], [datSTO], [datSTN]
            #                         else:
            #                             datSTa_list.append(datST[0, :])
            #                             datSTDa_list.append(datSTD[:, 0])
            #                             datSTAa_list.append(datSTA)
            #                             datSTOa_list.append(datSTO)
            #                             datSTNa_list.append(datSTN)
            #
            #     datSTAa = pd.DataFrame(datSTAa_list).applymap(lambda x: max(0, x))
            #     datSTOa = pd.DataFrame(datSTOa_list).applymap(lambda x: max(0, x))
            #     datSTNa = pd.DataFrame(datSTNa_list).applymap(lambda x: max(0, x))
            #
            #     with pd.ExcelWriter(os.path.join(xlsFolderW, RDRnam + 'errRstSite.xlsx'), engine='openpyxl') as writer:
            #         pd.DataFrame(datSTa_list).to_excel(writer, sheet_name='RSTA', startcol=2, startrow=1, header=False, index=False)
            #         pd.DataFrame(datSTDa_list).to_excel(writer, sheet_name='RSTA', startcol=1, startrow=2, header=False, index=False)
            #         datSTAa.to_excel(writer, sheet_name='RSTA', startcol=2, startrow=2, header=False, index=False)
            #
            #         pd.DataFrame(datSTa_list).to_excel(writer, sheet_name='RSTO', startcol=2, startrow=1, header=False, index=False)
            #         pd.DataFrame(datSTDa_list).to_excel(writer, sheet_name='RSTO', startcol=1, startrow=2, header=False, index=False)
            #         datSTOa.to_excel(writer, sheet_name='RSTO', startcol=2, startrow=2, header=False, index=False)
            #
            #         pd.DataFrame(datSTa_list).to_excel(writer, sheet_name='RSTN', startcol=2, startrow=1, header=False, index=False)
            #         pd.DataFrame(datSTDa_list).to_excel(writer, sheet_name='RSTN', startcol=1, startrow=2, header=False, index=False)
            #         datSTNa.to_excel(writer, sheet_name='RSTN', startcol=2, startrow=2, header=False, index=False)
            #
            #     # Save data as .mat equivalent (using .pkl)
            #     pd.to_pickle({
            #         'datSTDa': datSTDa_list,
            #         'datSTa': datSTa_list,
            #         'datSTAa': datSTAa,
            #         'datSTOa': datSTOa,
            #         'datSTNa': datSTNa
            #     }, os.path.join(xlsFolderW, RDRnam + 'errRstSite.pkl'))
            # #
            #
            #
            #
            #
            #
            # # 시작일/종료일 설정
            # dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            # dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            # dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=Hour(sysOpt['invHour']))
            #
            # # 기준 위도, 경도, 기압 설정
            # lonList = np.arange(sysOpt['lonMin'], sysOpt['lonMax'], sysOpt['lonInv'])
            # latList = np.arange(sysOpt['latMin'], sysOpt['latMax'], sysOpt['latInv'])
            # levList = np.array(sysOpt['levList'])
            #
            # log.info(f'[CHECK] len(lonList) : {len(lonList)}')
            # log.info(f'[CHECK] len(latList) : {len(latList)}')
            # log.info(f'[CHECK] len(levList) : {len(levList)}')
            #
            # for dtDateIdx, dtDateInfo in enumerate(dtDateList):
            #     log.info(f'[CHECK] dtDateInfo : {dtDateInfo}')
            #
            #     dataL1 = xr.Dataset()
            #     for modelIdx, modelType in enumerate(sysOpt['modelList']):
            #         log.info(f'[CHECK] modelType : {modelType}')
            #
            #         for i, modelKey in enumerate(sysOpt[modelType]):
            #             log.info(f'[CHECK] modelKey : {modelKey}')
            #
            #             modelInfo = sysOpt[modelType].get(modelKey)
            #             if modelInfo is None: continue
            #
            #             inpFile = '{}/{}'.format(modelInfo['filePath'], modelInfo['fileName'])
            #             inpFileDate = dtDateInfo.strftime(inpFile)
            #             fileList = sorted(glob.glob(inpFileDate))
            #
            #             if fileList is None or len(fileList) < 1:
            #                 # log.error(f'inpFile : {inpFile} / 입력 자료를 확인해주세요')
            #                 continue
            #
            #             # NetCDF 파일 읽기
            #             for j, fileInfo in enumerate(fileList):
            #
            #                 data = xr.open_dataset(fileInfo, engine='pynio')
            #                 log.info(f'[CHECK] fileInfo : {fileInfo}')
            #
            #                 # pygrib에서 분석/예보 시간 추출
            #                 gribData = pygrib.open(fileInfo).select()[0]
            #                 anaDt = gribData.analDate
            #                 fotDt = gribData.validDate
            #
            #                 log.info(f'[CHECK] anaDt : {anaDt} / fotDt : {fotDt}')
            #
            #                 # 파일명에서 분석/예보 시간 추출
            #                 # isMatch = re.search(r'f(\d+)', fileInfo)
            #                 # if not isMatch: continue
            #                 # int(isMatch.group(1))
            #
            #                 # anaDt = dtDateInfo
            #                 # fotDt = anaDt + pd.Timedelta(hours = int(isMatch.group(1)))
            #
            #                 for level, orgVar, newVar in zip(modelInfo['level'], modelInfo['orgVar'], modelInfo['newVar']):
            #                     if data.get(orgVar) is None: continue
            #
            #                     try:
            #                         if level == -1:
            #                             selData = data[orgVar].interp({modelInfo['comVar']['lon']: lonList, modelInfo['comVar']['lat']: latList}, method='linear')
            #                             selDataL1 = selData
            #                         else:
            #                             selData = data[orgVar].interp({modelInfo['comVar']['lon']: lonList, modelInfo['comVar']['lat']: latList, modelInfo['comVar']['lev']: levList}, method='linear')
            #                             selDataL1 = selData.sel({modelInfo['comVar']['lev']: level})
            #
            #                         selDataL2 = xr.Dataset(
            #                             {
            #                                 f'{modelType}_{newVar}': (('anaDt', 'fotDt', 'lat', 'lon'), (selDataL1.values).reshape(1, 1, len(latList), len(lonList)))
            #                             }
            #                             , coords={
            #                                 'anaDt': pd.date_range(anaDt, periods=1)
            #                                 , 'fotDt': pd.date_range(fotDt, periods=1)
            #                                 , 'lat': latList
            #                                 , 'lon': lonList
            #                             }
            #                         )
            #
            #                         dataL1 = xr.merge([dataL1, selDataL2])
            #                     except Exception as e:
            #                         log.error(f'Exception : {e}')
            #
            #     if len(dataL1) < 1: continue
            #
            #     # NetCDF 자료 저장
            #     saveFile = '{}/{}/{}_{}.nc'.format(globalVar['outPath'], serviceName, 'ecmwf-gfs_model', dtDateInfo.strftime('%Y%m%d%H%M'))
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     dataL1.to_netcdf(saveFile)
            #     log.info(f'[CHECK] saveFile : {saveFile}')
            #
            #     # 비교
            #     dataL1['DIFF_T2'] = dataL1['ECMWF_T2'] - dataL1['GFS_T2']
            #
            #     # 시각화
            #     saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, 'ecmwf_t2', dtDateInfo.strftime('%Y%m%d%H%M'))
            #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #     dataL1['ECMWF_T2'].isel(anaDt=0, fotDt=0).plot()
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.show()
            #     log.info(f'[CHECK] saveImg : {saveImg}')
            #
            #     saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, 'gfs_t2', dtDateInfo.strftime('%Y%m%d%H%M'))
            #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #     dataL1['GFS_T2'].isel(anaDt=0, fotDt=0).plot()
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.show()
            #     log.info(f'[CHECK] saveImg : {saveImg}')
            #
            #     saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, 'diff_t2', dtDateInfo.strftime('%Y%m%d%H%M'))
            #     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
            #     dataL1['DIFF_T2'].isel(anaDt=0, fotDt=0).plot()
            #     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
            #     plt.show()
            #     log.info(f'[CHECK] saveImg : {saveImg}')

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
