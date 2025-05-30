# ================================================
# 요구사항
# ================================================
# Python을 이용한 미국 전역 관측소를 기준으로 매칭 자동화

# 프로그램 시작
# cd /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
# nohup /SYSTEMS/LIB/anaconda3/envs/py38/bin/python3.8 TalentPlatform-LSH0589-DaemonFramework.py &
# tail -f nohup.out

# 프로그램 종료
# ps -ef | grep "TalentPlatform-LSH0589-DaemonFramework" | awk '{print $2}' | xargs kill -9

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
# from _ast import expr
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import gzip
import shutil
from io import StringIO
import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# from pyproj import Proj, Transformer
import re
import matplotlib.cm as cm
from multiprocessing import Pool
import multiprocessing as mp
from retrying import retry
# from pymap3d import enu2geodetic
import requests
from datetime import datetime
import time
import pandas as pd
from tqdm import tqdm
import time
import pytz
import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor
import requests
import gzip
import shutil
import dask.dataframe as dd

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

kst = pytz.timezone("Asia/Seoul")

# plt.rc('font', family='Malgun Gothic')
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
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'output', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , 'sysCfg': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.json')
        , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    return globalVar


#  초기 전달인자 설정
def initArgument(globalVar):
    parser = argparse.ArgumentParser()

    for i, argv in enumerate(sys.argv[1:]):
        if not argv.__contains__('--'): continue
        parser.add_argument(argv)

    inParInfo = vars(parser.parse_args())
    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        if env not in 'local' and key.__contains__('Path'):
            os.makedirs(val, exist_ok=True)
        globalVar[key] = val

    return globalVar

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'  # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'LSH0589'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self):

        log.info('[START] {}'.format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar)

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("init"))

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info('[START] {}'.format("exec"))

        try:

            if (platform.system() == 'Windows'):
                pass
            else:
                pass

            globalVar['inpPath'] = '/DATA/INPUT'
            globalVar['outPath'] = '/DATA/OUTPUT'
            globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작일, 종료일, 시간 간격 (연 1y, 월 1h, 일 1d, 시간 1h, 분 1t)
                'srtDate': '1996-01-01'
                , 'endDate': '2011-12-31'
                , 'invDate': '1d'
            }

            # 시작일/종료일 설정
            dtSrtDate = pd.to_datetime(sysOpt['srtDate'], format='%Y-%m-%d')
            dtEndDate = pd.to_datetime(sysOpt['endDate'], format='%Y-%m-%d')
            dtDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq=sysOpt['invDate'])

            # =========================================================
            # 데이터 수집
            # =========================================================
            # dtYearList = pd.date_range(start=pd.to_datetime('1750', format='%Y'), end=pd.to_datetime('2024', format='%Y'), freq='1y')
            # for dtYear in dtYearList:
            #     log.info(f'[CHECK] dtYear : {dtYear}')
            #
            #     url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/{dtYear.strftime('%Y')}.csv.gz"
            #     gzFile = f"{globalVar['inpPath']}/{serviceName}/noaa/{dtYear.strftime('%Y')}.csv.gz"
            #     csvFile = f"{globalVar['inpPath']}/{serviceName}/noaa/{dtYear.strftime('%Y')}.csv"
            #     os.makedirs(os.path.dirname(csvFile), exist_ok=True)
            #
            #     response = requests.get(url, stream=True)
            #
            #     if not response.status_code == 200:
            #         log.info(f'Failed to download {gzFile}. Status code: {response.status_code}')
            #         continue
            #
            #     with open(gzFile, 'wb') as f:
            #         f.write(response.raw.read())
            #     log.info(f'Downloaded {gzFile} successfully.')
            #
            #     # Extract the gzipped file
            #     with gzip.open(gzFile, 'rb') as f_in:
            #         with open(csvFile, 'wb') as f_out:
            #             shutil.copyfileobj(f_in, f_out)
            #     log.info(f'Extracted to {csvFile} successfully.')
            #
            #     if os.path.exists(gzFile):
            #         os.remove(gzFile)

            # =========================================================
            # 데이터 처리
            # =========================================================
            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ghcnd-states.txt')
            fileList = sorted(glob.glob(inpFile))
            stateData = pd.read_fwf(fileList[0], header=None, names=["abbr", "state"])

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'ghcnd-stations.txt')
            fileList = sorted(glob.glob(inpFile))

            colSpec = [
                (0, 11),  # STATION ID
                (12, 20),  # LATITUDE
                (21, 30),  # LONGITUDE
                (31, 37),  # ELEVATION
                (38, 40),  # STATE
                (41, 71),  # NAME
                (72, 75),  # GSNFLAG
                (76, 79),  # HCNFLAG
                (80, 85)  # WMOID
            ]

            stationData = pd.read_fwf(fileList[0], colspecs=colSpec, names= ["STATION", "LATITUDE", "LONGITUDE", "ELEVATION", "STATE", "NAME", "GSNFLAG", "HCNFLAG", "WMOID"])

            inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'rsv.csv')
            fileList = sorted(glob.glob(inpFile))
            rsvData = pd.read_csv(fileList[0])

            for i, stateInfo in stateData.iterrows():

                # stateInfo['abbr'] = 'MN'
                # if not stateInfo['abbr'] == 'MN': continue

                xlsxFilePattern = f"{globalVar['outPath']}/{serviceName}/{stateInfo['abbr']}-{stateInfo['state']}_Average_Values_Across_Stations_*.xlsx"
                if len(glob.glob(xlsxFilePattern)) > 0: continue

                rsvDataL1 = rsvData.loc[(rsvData['HOSPST'] == stateInfo['abbr'])].reset_index(drop=False)
                if len(rsvDataL1) < 1: continue

                set(rsvDataL1['year'])

                log.info(f"[CHECK] abbr : {stateInfo['abbr']}")

                # ******************************************************************
                # 문턱값 설정
                # ******************************************************************
                columns_needed = ['AWEEK1', 'HOSPSTCO', 'rsv', 'COUNTYPOP', 'mbirth_rate', 'year', 'weekyear']

                # # Minnesota.ipynb 참조
                # # Count unique values in the "AWEEK1" column and sort by index
                # aweek1_counts = rsvDataL1['AWEEK1'].value_counts().sort_index()
                #
                # # 최대값
                # maxVal = aweek1_counts.max()
                #
                # # 문턱값 계산 (최대값의 80%)
                # maxThres = maxVal * 0.8
                #
                # # 기준값 이상인 최소값 찾기
                # minIdxAweek1 = aweek1_counts[aweek1_counts >= maxThres].index.min()
                # log.info(f"[CHECK] maxVal: {maxVal}, maxThres: {maxThres:.1f}, minIdxAweek1: {minIdxAweek1}")
                #
                # # Filter for necessary columns and 'AWEEK1' > 680

                # # filtered_data = data_main[columns_needed]
                # filtered_data = rsvDataL1[columns_needed]
                #
                # # filtered_data = filtered_data[filtered_data['AWEEK1'] > 680]
                # filtered_data = filtered_data[filtered_data['AWEEK1'] > minIdxAweek1]
                #
                # # Remove duplicates and drop missing values
                # filtered_data = filtered_data.drop_duplicates().dropna()

                # ******************************************************************
                # 문턱값 미설정
                # ******************************************************************
                filtered_data = rsvDataL1[columns_needed].drop_duplicates().dropna()

                # Fill in missing weeks with averaged RSV values
                filled_data = []
                for _, group in filtered_data.groupby('HOSPSTCO'):
                    group = group.sort_values('AWEEK1').reset_index(drop=True)
                    for i in range(len(group) - 1):
                        current_week = group.loc[i, 'AWEEK1']
                        next_week = group.loc[i + 1, 'AWEEK1']
                        rsv_current = group.loc[i, 'rsv']

                        if next_week - current_week > 1:
                            avg_rsv_per_week = rsv_current / (next_week - current_week)
                            for week in range(current_week + 1, next_week):

                                try:
                                    interpolated_row = group.loc[i].copy()
                                    interpolated_row['AWEEK1'] = week
                                    interpolated_row['rsv'] = avg_rsv_per_week
                                    filled_data.append(interpolated_row)
                                except Exception as e:
                                    pass

                    filled_data.append(group)

                if len(filled_data) < 1: continue

                # Concatenate all rows and convert back to DataFrame
                interpolated_data = pd.concat(filled_data).reset_index(drop=True)

                # Group by 'AWEEK1', 'year', 'weekyear' and aggregate data
                grouped_data = interpolated_data.groupby(['AWEEK1', 'year', 'weekyear']).agg({
                    'rsv': 'sum',
                    'COUNTYPOP': 'sum',
                    'mbirth_rate': lambda x: (x * interpolated_data.loc[x.index, 'COUNTYPOP']).sum() / interpolated_data.loc[
                        x.index, 'COUNTYPOP'].sum()
                }).reset_index()
                if len(grouped_data) < 1: continue

                # Plotting RSV cases over AWEEK1 values
                # plt.figure(figsize=(10, 6))
                # plt.plot(grouped_data['AWEEK1'], grouped_data['rsv'], marker='o', linestyle='-')
                # plt.xlabel('AWEEK1')
                # plt.ylabel('RSV Cases')
                # plt.title('RSV Cases over AWEEK1')
                # plt.grid(True)
                # plt.show()

                # Define the path template for each year's file
                # path_template = r"C:\Users\hongz\Downloads\{}.csv\{}.csv"
                path_template = f"{globalVar['inpPath']}/{serviceName}/noaa/{{}}.csv"

                # Load the station list from the "station_florida.xlsx" file
                # station_florida_path = r"C:\Users\hongz\Downloads\station minnesota.xlsx"
                # station_florida_data = pd.read_excel(station_florida_path)
                station_florida_data = stationData.loc[(stationData['STATE'] == stateInfo['abbr'])].reset_index(drop=True)
                if len(station_florida_data) < 1: continue

                # Assume the stations are in the first column
                stations_of_interest = station_florida_data.iloc[:, 0].unique()
                if len(stations_of_interest) < 1: continue

                minYear = int(grouped_data['year'].min())
                maxYear = int(grouped_data['year'].max())
                log.info(f"[CHECK] minYear : {minYear} / maxYear : {maxYear}")

                # Initialize an empty DataFrame to hold all the filtered and reshaped data
                combined_data = pd.DataFrame()
                # for year in range(2000, 2011):
                for year in range(minYear, maxYear):
                    log.info(f"[CHECK] year : {year}")

                    # Create the file path for the current year
                    # file_path = path_template.format(year, year)
                    file_path = path_template.format(year)

                    # Check if the file exists
                    if not os.path.exists(file_path):
                        log.info(f"[CHECK] File for year {year} not found at path: {file_path}")
                        continue  # Skip to the next year if file is missing

                    # Load only the first four columns for the current year without headers
                    # data = pd.read_csv(file_path, header=None, usecols=[0, 1, 2, 3], names=['Station', 'Date', 'Variable', 'Value'])
                    data = dd.read_csv(file_path, header=None, usecols=[0, 1, 2, 3], names=['Station', 'Date', 'Variable', 'Value'])
                    if len(data) < 1: continue

                    # Filter the data to only include stations of interest
                    filtered_data = data[data['Station'].isin(stations_of_interest)]
                    if len(filtered_data) < 1: continue

                    # Pivot the data so that each variable has its own column
                    # reshaped_data = filtered_data.pivot_table(
                    #     index=['Station', 'Date'],
                    #     columns='Variable',
                    #     values='Value',
                    #     aggfunc='first'
                    # ).reset_index()

                    reshaped_data = filtered_data.compute().pivot_table(
                        index=['Station', 'Date'],
                        columns='Variable',
                        values='Value',
                        aggfunc='first'
                    ).reset_index()

                    variable_columns = reshaped_data.columns.difference(['Station', 'Date'])
                    average_by_date = reshaped_data.groupby('Date')[variable_columns].mean().reset_index()
                    resDataL1 = average_by_date.loc[:, ~average_by_date.isna().any()]

                    # Append the reshaped data for the current year to the combined DataFrame
                    # combined_data = pd.concat([combined_data, reshaped_data], ignore_index=True)
                    combined_data = pd.concat([combined_data, resDataL1], ignore_index=True)

                # Define the output path for the Excel file
                # output_file_path = r"C:\Users\hongz\Downloads\MinnesotaCombined_Station_Data_Pivoted_2001_2010.xlsx"

                # Save the combined reshaped data to an Excel file
                # combined_data.to_excel(output_file_path, index=False)
                # (f"All data from 2001 to 2010 saved with variables as columns in {output_file_path}")


                # Load the provided Excel file
                # file_path = r"C:\Users\hongz\Downloads\MinnesotaCombined_Station_Data_Pivoted_2001_2010.xlsx"
                # data = pd.read_excel(file_path)

                # Group by 'Date' and calculate the mean for each variable column, ignoring NaN values
                if len(combined_data) < 1: continue
                # comData = combined_data.reset_index()
                comData = combined_data.reset_index(drop=True)

                # variable_columns = data.columns.difference(['Station', 'Date'])
                # average_by_date = data.groupby('Date')[variable_columns].mean().reset_index()
                # variable_columns = combined_data.columns.difference(['Station', 'Date'])
                # average_by_date = combined_data.groupby('Date')[variable_columns].mean().reset_index()

                # 결측값 개수
                # nanCnt = average_by_date.isna().sum()

                # avgDataL1 = average_by_date.dropna(axis=1, how='all').reset_index(drop=True)
                # avgDataL1 = average_by_date.loc[:, ~average_by_date.isna().any()]

                # 시작일/종료일을 기준으로 데이터 병합
                dtDateData = pd.DataFrame(dtDateList.strftime('%Y%m%d').astype(int), columns=['Date'])
                comDataL1 = pd.merge(dtDateData, comData, how='left', left_on=['Date'], right_on=['Date'])

                # Define the output path for the file with averages by date
                # output_file_path = r"C:\Users\hongz\Downloads\MinnesotaAverage_Values_Across_Stations.xlsx"
                xlsxFile = f"{globalVar['outPath']}/{serviceName}/{stateInfo['abbr']}-{stateInfo['state']}_Average_Values_Across_Stations_{minYear}-{maxYear}.xlsx"
                os.makedirs(os.path.dirname(xlsxFile), exist_ok=True)

                # Save the averages by date to an Excel file
                # average_by_date.to_excel(xlsxFile, index=False)
                # avgDataL1.to_excel(xlsxFile, index=False)
                comDataL1.to_excel(xlsxFile, index=False)

                # print(f"Averages by date saved to {output_file_path}")
                log.info(f"[CHECK] xlsxFile : {xlsxFile}")

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("exec"))

# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print('[START] {}'.format("main"))

    try:
        # 부 프로그램 호출
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))