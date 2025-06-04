# ================================================
# 요구사항
# ================================================
# Python을 이용한 한방부동산 매물 수집체계 구축

# /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0615-DaemonFramework-colctKarhanbang.py

# cd /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
# nohup /SYSTEMS/LIB/anaconda3/envs/py38/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0615-DaemonFramework-colctKarhanbang.py &
# tail -f nohup.out

# pkill -f TalentPlatform-LSH0615-DaemonFramework-colctKarhanbang.py

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
from matplotlib import font_manager, rc
import csv
import os
import pandas as pd
import re
import subprocess
import requests
from bs4 import BeautifulSoup
import re
from lxml import etree
from multiprocessing import Pool
import multiprocessing as mp

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

def getTagText(tag, xpath):
    try:
        ele = tag.find(xpath)
        return ele.text
    except AttributeError:
        return None

def colctProc(sysOpt, addrInfo):
    try:
        procInfo = mp.current_process()
        pageList = np.arange(1, 9999, 1)
        data = pd.DataFrame()
        maxPage = None

        for pageInfo in pageList:
            # log.info(f'[CHECK] pageInfo : {pageInfo}')

            addrIdx = sysOpt['addrList'].index(addrInfo)
            url = sysOpt['url'].format(sido=addrInfo, sido_no=(addrIdx + 1), page=pageInfo)
            res = requests.get(url)
            if res.status_code != 200: continue

            soup = BeautifulSoup(res.text, 'html.parser')
            lxml = etree.HTML(str(soup))

            if maxPage is None:
                jsFun = lxml.xpath('/html/body/div/div[3]/a[6]')[0].get('onclick')
                maxPage = int(re.findall(r"'\s*(\d+)\s*'\)", jsFun)[0])

            if maxPage and pageInfo > maxPage:
                break

            # tagInfo = tagList[0]
            tagList = lxml.xpath('/html/body/div/div[1]/div[2]/ul/li[*]')
            for tagInfo in tagList:
                jsFun = tagInfo.find('a').get('href')
                mmNo, topM = re.findall(r"'(.*?)'", jsFun)
                urlDtl = sysOpt['urlDtl'].format(mmNo=mmNo, topM=topM)

                type = getTagText(tagInfo, 'a/dl/dt/em[1]')
                price = getTagText(tagInfo, 'a/dl/dt/strong[1]')
                aptName = getTagText(tagInfo, 'a/dl/dt/span[1]')
                aptType = getTagText(tagInfo, 'a/dl/dd[2]/div/ul/li[1]')
                buildNum = getTagText(tagInfo, 'a/dl/dd[2]/div/ul/li[2]')
                area = getTagText(tagInfo, 'a/dl/dd[2]/div/ul/li[3]')
                floor = getTagText(tagInfo, 'a/dl/dd[2]/div/ul/li[4]')
                fee = getTagText(tagInfo, 'a/dl/dd[2]/div/ul/li[5]')

                lastDate = getTagText(tagInfo, 'div/span[1]')
                agencyName = getTagText(tagInfo, 'div/button[1]')
                agencyOfficePhone = getTagText(tagInfo, 'div/em[1]')
                agencyMobilePhone = getTagText(tagInfo, 'div/em[2]')

                dict = {
                    'addrInfo': [addrInfo],
                    'page': [pageInfo],
                    'url': [url],
                    'urlDtl': [urlDtl],
                    'type': [type],
                    'price': [price],
                    'aptName': [aptName],
                    'aptType': [aptType],
                    'buildNum': [buildNum],
                    'area': [area],
                    'floor': [floor],
                    'fee': [fee],
                    'lastDate': [lastDate],
                    'agencyName': [agencyName],
                    'agencyOfficePhone': [agencyOfficePhone],
                    'agencyMobilePhone': [agencyMobilePhone],
                }

                data = pd.concat([data, pd.DataFrame.from_dict(dict)], ignore_index=True)

            log.info(f'[CHECK] addrInfo : {addrInfo} / per : {round((pageInfo / maxPage) * 100, 1)}  / cnt : {len(data)} / pid : {procInfo.pid}')

        if len(data) > 0:
            saveFile = sysOpt['preDt'].strftime(sysOpt['saveFile']).format(addrInfo=addrInfo)
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            data.to_csv(saveFile, index=False)
            log.info(f"[CHECK] saveFile : {saveFile}")

    except Exception as e:
        log.error(f'Exception : {e}')
        # raise e

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'LSH0615'

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
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                'addrList': ['서울', '경기', '인천', '부산', '대구', '광주', '대전', '울산', '강원', '상남', '경북', '전남', '전북', '충남', '충북', '세종', '제주'],
                'url': 'https://karhanbang.com/mamulCommon/mamul_list.asp?gure_cd=0&mamulGubun=01&cate_cd=01&sido={sido}&gugun=&dong=&sido_no={sido_no}&gugun_no=&dong_no=&hDong_no=&danji_no=&schType=1&tab_gubun=mamul&gugun_chk=&danji_name=&schGongMeter=&schdanjiDongNm=&schdanjiCurrFloor=&stateCheck=&trade_yn=&txt_amt_sell_s=&txt_amt_sell_e=&txt_amt_guar_s=&txt_amt_guar_e=&txt_amt_month_s=&txt_amt_month_e=&txt_amt_month2_s=&txt_amt_month2_e=&sel_area=&txt_area_s=&txt_area_e=&txt_room_cnt_s=&txt_room_cnt_e=&won_room_cnt=&txt_const_year=&txt_estimate_meter_s=&txt_estimate_meter_e=&sel_area3=&txt_area3_s=&txt_area3_e=&sel_building_use_cd=&sel_gunrak_cd=&sel_area5=&txt_area5_s=&txt_area5_e=&sel_area6=&txt_area6_s=&txt_area6_e=&txt_floor_high_s=&txt_floor_high_e=&officetel_use_cd=&sel_option_cd=&txt_land_s=&txt_land_e=&sel_jimok_cd=&txt_road_meter_s=&txt_road_meter_e=&sel_store_use_cd=&sel_sangga_cd=&sangga_cd=&sangga_chk=&sel_sangga_ipji_cd=&sel_office_use_cd=&orderByGubun=&regOrderBy=&confirmOrderBy=&meterOrderBy=&priceOrderBy=&currFloorBy=&chk_rentalhouse_yn=NN&chk_soon_move_yn=NN&chk_kyungmae_yn=NN&txt_yong_jiyuk2_nm=&txt_amt_dang_s=&txt_amt_dang_e=&gong_meter_s=&gong_meter_e=&gun_meter_s=&gun_meter_e=&toji_meter_s=&toji_meter_e=&txt_const_year_s=&txt_const_year_e=&txt_curr_floor_s=&txt_curr_floor_e=&page={page}&flag=S&theme=&',
                'urlDtl': 'https://karhanbang.com/detail/?topM={topM}&schType=3&mm_no={mmNo}&mapGubun=N',
                'saveFile': '/DATA/OUTPUT/LSH0615/%Y%m%d_매물_{addrInfo}.csv',
                'preDt': datetime.now(),
                'cpuCoreNum': '5',
            }

            # **************************************************************************************************************
            # 비동기 다중 프로세스 수행
            # **************************************************************************************************************
            # 비동기 다중 프로세스 개수
            pool = Pool(int(sysOpt['cpuCoreNum']))
            for addrInfo in sysOpt['addrList']:
                pool.apply_async(colctProc, args=(sysOpt, addrInfo))
            pool.close()
            pool.join()

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