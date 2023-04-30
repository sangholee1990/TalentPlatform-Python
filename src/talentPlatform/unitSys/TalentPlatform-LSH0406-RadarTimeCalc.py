# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import multiprocessing as mp
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime
from multiprocessing import Pool

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

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


def subRadarCalc(dtMinDateInfo, i, data, infoData, refColList):

    result = None

    try:
        procInfo = mp.current_process()

        # log.info(f'[START] subRadarCalc : {procInfo.name} / pid : {procInfo.pid}')

        selCol = f'V{i + 1}'
        if (data.filter(regex=selCol).shape[1] < 1): return

        dataL2 = data[['RRRCCC', selCol]]
        dataL2.columns = ['RRRCCC', 'RDR', 'U1', 'N2']

        if (len(dataL2) < 1): return

        dataL3 = pd.merge(infoData, dataL2.rename(columns={'RRRCCC': 'No'}), how='left', left_on=['No'], right_on=['No'])
        dataL4 = dataL3

        colList = dataL4.filter(regex='Adj_No_*').columns
        for k, colInfo in enumerate(colList):
            dataL4 = pd.merge(dataL4, dataL2[['RRRCCC', 'U1', 'N2']].rename(columns={'RRRCCC': colInfo}), how='left', left_on=[colInfo], right_on=[colInfo], suffixes=('', '_' + str(k + 1)))

        # 컬럼 재 정렬
        dataL4 = dataL4[refColList[0:dataL4.shape[1]]]

        for k in range(1, 10):
            dataL4[54 + k] = ((dataL4.iloc[:, 14 + k - 1] * (dataL4.iloc[:, 36 + k - 1] * 0.7)) + (dataL4.iloc[:, 23 + k - 1] * (dataL4.iloc[:, 45 + k - 1] * 0.3))) / 64.0
            dataL4[63 + k] = ((dataL4.iloc[:, 23 + k - 1] * dataL4.iloc[:, 45 + k - 1])) / 64.0

        # 컬럼 재 정렬
        dataL4 = dataL4[refColList[0:dataL4.shape[1]]]

        dataL4[73] = dataL4.iloc[:, 3 - 1:4].sum(axis=1, skipna=True)
        dataL4[74] = dataL4.iloc[:, 55 - 1:72].sum(axis=1, skipna=True)
        dataL4[75] = dataL4.iloc[:, 74 - 1] / dataL4.iloc[:, 73 - 1] * 3600 * 1000 / (64 * 64) * dataL4.iloc[:, 33 - 1] / 6

        dataL5 = dataL4.groupby(by=['No']).max()[['RDR', 75]].reset_index(drop=False).rename(columns={75: 'val'})
        dataL5['time'] = dtMinDateInfo
        dataL5['row'] = dataL5['No'].astype('str').str.slice(0, 3).astype('int')
        dataL5['col'] = dataL5['No'].astype('str').str.slice(3, 6).astype('int')

        result = dataL5

        return result

    except Exception as e:
        log.error("Exception : {}".format(e))
        return result

    finally:
        log.info(f'[END] subRadarCalc : {dtMinDateInfo} / pid : {procInfo.pid}')


# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 레이더 유출량 산정 프로그램 변환 (+고도화)

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
        # contextPath = os.getcwd() if env in 'local' else '/home/dxinyu/SYSTEMS/PROG/PYTHON'

    prjName = 'test'
    serviceName = 'LSH0406'

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
                    # 시작일 및 종료일 설정
                    'srtDate': '2022-09-06'
                    , 'endDate': '2022-09-07'

                    # 4종 및 DB 입력자료 패턴
                    , 'inpFilePattern': 'Test_Info_xy_1km_30.txt'
                    , 'inp2FilePattern': 'Test_RDR_1km_%dt2Ymd%.txt'
                    , 'inp3FilePattern': 'Test_D_U1_1km_%dt2Ymd%.txt'
                    , 'inp4FilePattern': 'Test_D_N2_1km_%dt2Ymd%.txt'
                    , 'infoFilePattern': 'Test_DB_30.csv'

                    # 비동기 다중 프로세스 개수
                    # , 'cpuCoreNum': 4
                    , 'cpuCoreNum': 8

                    # 최종 컬럼 변수명 지정
                    , 'refColList': ['No', 'No4', 'sum_ub', 'sum_nub', 'slopeAVG', 'Adj_No_1',
                                     'Adj_No_2', 'Adj_No_3', 'Adj_No_4', 'Adj_No_5', 'Adj_No_6', 'Adj_No_7',
                                     'Adj_No_8', 'Adj_No_9', 'cnt_U1_1', 'cnt_U1_2', 'cnt_U1_3', 'cnt_U1_4',
                                     'cnt_U1_5', 'cnt_U1_6', 'cnt_U1_7', 'cnt_U1_8', 'cnt_U1_9', 'cnt_N2_1',
                                     'cnt_N2_2', 'cnt_N2_3', 'cnt_N2_4', 'cnt_N2_5', 'cnt_N2_6', 'cnt_N2_7',
                                     'cnt_N2_8', 'cnt_N2_9', 'ratio', 'RDR', 'U1', 'N2',
                                     'U1_1', 'U1_2', 'U1_3', 'U1_4', 'U1_5', 'U1_6',
                                     'U1_7', 'U1_8', 'U1_9', 'N2_1', 'N2_2', 'N2_3',
                                     'N2_4', 'N2_5', 'N2_6', 'N2_7', 'N2_8', 'N2_9',
                                     55, 56, 57, 58, 59, 60,
                                     61, 62, 63, 64, 65, 66,
                                     67, 68, 69, 70, 71, 72,
                                     73, 74, 75]
                }

            else:

                # 옵션 설정
                sysOpt = {
                    # 시작일 및 종료일 설정
                    'srtDate': '2022-09-06'
                    , 'endDate': '2022-09-07'

                    # 4종 및 DB 입력자료 패턴
                    , 'inpFilePattern': 'Test_Info_xy_1km_30.txt'
                    , 'inp2FilePattern': 'Test_RDR_1km_%dt2Ymd%.txt'
                    , 'inp3FilePattern': 'Test_D_U1_1km_%dt2Ymd%.txt'
                    , 'inp4FilePattern': 'Test_D_N2_1km_%dt2Ymd%.txt'
                    , 'infoFilePattern': 'Test_DB_30.csv'

                    # 비동기 다중 프로세스 개수
                    # , 'cpuCoreNum' : 4
                    , 'cpuCoreNum' : 8

                    # 최종 컬럼 변수명 지정
                    , 'refColList': ['No', 'No4', 'sum_ub', 'sum_nub', 'slopeAVG', 'Adj_No_1',
                                     'Adj_No_2', 'Adj_No_3', 'Adj_No_4', 'Adj_No_5', 'Adj_No_6', 'Adj_No_7',
                                     'Adj_No_8', 'Adj_No_9', 'cnt_U1_1', 'cnt_U1_2', 'cnt_U1_3', 'cnt_U1_4',
                                     'cnt_U1_5', 'cnt_U1_6', 'cnt_U1_7', 'cnt_U1_8', 'cnt_U1_9', 'cnt_N2_1',
                                     'cnt_N2_2', 'cnt_N2_3', 'cnt_N2_4', 'cnt_N2_5', 'cnt_N2_6', 'cnt_N2_7',
                                     'cnt_N2_8', 'cnt_N2_9', 'ratio', 'RDR', 'U1', 'N2',
                                     'U1_1', 'U1_2', 'U1_3', 'U1_4', 'U1_5', 'U1_6',
                                     'U1_7', 'U1_8', 'U1_9', 'N2_1', 'N2_2', 'N2_3',
                                     'N2_4', 'N2_5', 'N2_6', 'N2_7', 'N2_8', 'N2_9',
                                     55, 56, 57, 58, 59, 60,
                                     61, 62, 63, 64, 65, 66,
                                     67, 68, 69, 70, 71, 72,
                                     73, 74, 75]
                }

                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # ********************************************************************
            # 입력자료
            # ********************************************************************
            log.info(f'[CHECK] sysOpt : {sysOpt}')

            # 시작일 및 종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1D')

            # dtIncDateInfo = dtIncDateList[0]
            for i, dtIncDateInfo in enumerate(dtIncDateList):
                log.info(f'[CHECK] dtIncDateInfo : {dtIncDateInfo}')

                dt2Ymd = dtIncDateInfo.strftime('%y%m%d')
                dtMinDateList = pd.date_range(start=dtIncDateInfo, end=dtIncDateInfo + pd.DateOffset(days=1), freq='10T')

                # 4종 입력 자료 찾기
                inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, sysOpt['inpFilePattern'])
                fileList = sorted(glob.glob(inpFile))
                if fileList is None or len(fileList) < 1:
                    log.error(f'[ERROR] inpFile : {inpFile} / 입력 자료를 확인해주세요.')
                    continue

                inp2File = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, sysOpt['inp2FilePattern']).replace('%dt2Ymd%', dt2Ymd)
                fileList = sorted(glob.glob(inp2File))
                if fileList is None or len(fileList) < 1:
                    log.error(f'[ERROR] inpFile : {inp2File} / 입력 자료를 확인해주세요.')
                    continue

                inp3File = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, sysOpt['inp3FilePattern']).replace('%dt2Ymd%', dt2Ymd)
                fileList = sorted(glob.glob(inp3File))
                if fileList is None or len(fileList) < 1:
                    log.error(f'[ERROR] inpFile : {inp3File} / 입력 자료를 확인해주세요.')
                    continue

                inp4File = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, sysOpt['inp4FilePattern']).replace('%dt2Ymd%', dt2Ymd)
                fileList = sorted(glob.glob(inp4File))
                if fileList is None or len(fileList) < 1:
                    log.error(f'[ERROR] inpFile : {inp4File} / 입력 자료를 확인해주세요.')
                    continue

                infoFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, sysOpt['infoFilePattern'])
                fileList = sorted(glob.glob(infoFile))
                if fileList is None or len(fileList) < 1:
                    log.error(f'[ERROR] inpFile : {infoFile} / 입력 자료를 확인해주세요.')
                    continue

                # 입력 자료 읽기
                inpData = pd.read_csv(inpFile, sep='\t')
                log.info(f'[CHECK] inpFile : {inpFile}')

                inp2Data = pd.read_csv(inp2File, sep='\t')
                log.info(f'[CHECK] inp2File : {inp2File}')

                inp3Data = pd.read_csv(inp3File, sep='\t')
                log.info(f'[CHECK] inp3File : {inp3File}')

                inp4Data = pd.read_csv(inp4File, sep='\t')
                log.info(f'[CHECK] inp4File : {inp4File}')

                infoData = pd.read_csv(infoFile)
                log.info(f'[CHECK] infoFile : {infoFile}')

                # 입력 자료 가공
                data = pd.concat([inpData['RRRCCC'], inp2Data, inp3Data, inp4Data], axis=1)

                # **************************************************************************************************************
                # 비동기 다중 프로세스 수행
                # **************************************************************************************************************
                # 비동기 다중 프로세스 개수
                pool = Pool(sysOpt['cpuCoreNum'])

                rtnList = []
                for i, dtMinDateInfo in enumerate(dtMinDateList):
                    rtnInfo = pool.apply_async(subRadarCalc, args=(dtMinDateInfo, i, data, infoData, sysOpt['refColList']))
                    rtnList.append(rtnInfo)
                pool.close()
                pool.join()

                dataL6 = pd.DataFrame()
                for rtnInfo in rtnList:
                   dataL6 = pd.concat([dataL6, rtnInfo.get()], ignore_index=True)

                # **************************************************************************************************************
                # 동기 단일 프로세스 수행
                # **************************************************************************************************************
                # rtnList = []
                # for i, dtMinDateInfo in enumerate(dtMinDateList):
                #     rtnInfo = subRadarCalc(dtMinDateInfo, i, data, infoData, sysOpt['refColList'])
                #     rtnList.append(rtnInfo)
                #
                # dataL6 = pd.DataFrame()
                # for rtnInfo in rtnList:
                #    dataL6 = pd.concat([dataL6, rtnInfo], ignore_index=True)

                # **************************************************************************************************************
                # 최종 결과 생성
                # **************************************************************************************************************
                # CSV to NetCDF 변환
                dataL7 = dataL6.set_index(['time', 'col', 'row']).to_xarray()
                saveFile = '{}/{}/{}_{}.nc'.format(globalVar['outPath'], serviceName, 'Result', dtIncDateInfo.strftime('%Y%m%d'))
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL7.to_netcdf(saveFile)
                log.info(f'[CHECK] saveFile : {saveFile}')

                # RDR 그림 생성
                saveImg = '{}/{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, 'Result', 'RDR', dtIncDateInfo.strftime('%Y%m%d'))
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                dataL7.isel(time=36)['RDR'].plot()
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                plt.tight_layout()
                # plt.show()
                plt.close()
                log.info(f'[CHECK] saveImg : {saveImg}')

                # val 그림 생성
                saveImg = '{}/{}/{}_{}_{}.png'.format(globalVar['figPath'], serviceName, 'Result', 'val', dtIncDateInfo.strftime('%Y%m%d'))
                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                dataL7.isel(time=36)['val'].plot(vmin=0, vmax=20)
                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                plt.tight_layout()
                # plt.show()
                plt.close()
                log.info(f'[CHECK] saveImg : {saveImg}')

                # CSV 생성
                dataL8 = dataL7.to_dataframe().sort_values(by=['row', 'col']).reset_index().dropna()
                saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, 'Result', dtIncDateInfo.strftime('%Y%m%d'))
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                dataL8.to_csv(saveFile, index=False)
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
