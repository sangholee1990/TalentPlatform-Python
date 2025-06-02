# ================================================
# 요구사항
# ================================================
# Python, Looker Studio을 이용한 19년간 공개 생산문서 대시보드

# rm -f 20250320_ydg2007-2025.csv
# cat *.csv > 20250320_ydg2007-2025.csv

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
                'url': 'https://karhanbang.com/mamulCommon/mamul_list.asp?gure_cd=0&mamulGubun=01&cate_cd=01&sido={sido}&gugun=&dong=&sido_no={sido_no}&gugun_no=&dong_no=&hDong_no=&danji_no=&schType=1&tab_gubun=mamul&gugun_chk=&danji_name=&schGongMeter=&schdanjiDongNm=&schdanjiCurrFloor=&stateCheck=&trade_yn=&txt_amt_sell_s=&txt_amt_sell_e=&txt_amt_guar_s=&txt_amt_guar_e=&txt_amt_month_s=&txt_amt_month_e=&txt_amt_month2_s=&txt_amt_month2_e=&sel_area=&txt_area_s=&txt_area_e=&txt_room_cnt_s=&txt_room_cnt_e=&won_room_cnt=&txt_const_year=&txt_estimate_meter_s=&txt_estimate_meter_e=&sel_area3=&txt_area3_s=&txt_area3_e=&sel_building_use_cd=&sel_gunrak_cd=&sel_area5=&txt_area5_s=&txt_area5_e=&sel_area6=&txt_area6_s=&txt_area6_e=&txt_floor_high_s=&txt_floor_high_e=&officetel_use_cd=&sel_option_cd=&txt_land_s=&txt_land_e=&sel_jimok_cd=&txt_road_meter_s=&txt_road_meter_e=&sel_store_use_cd=&sel_sangga_cd=&sangga_cd=&sangga_chk=&sel_sangga_ipji_cd=&sel_office_use_cd=&orderByGubun=&regOrderBy=&confirmOrderBy=&meterOrderBy=&priceOrderBy=&currFloorBy=&chk_rentalhouse_yn=NN&chk_soon_move_yn=NN&chk_kyungmae_yn=NN&txt_yong_jiyuk2_nm=&txt_amt_dang_s=&txt_amt_dang_e=&gong_meter_s=&gong_meter_e=&gun_meter_s=&gun_meter_e=&toji_meter_s=&toji_meter_e=&txt_const_year_s=&txt_const_year_e=&txt_curr_floor_s=&txt_curr_floor_e=&page=10&flag=S&theme=&',
                'urlDtl': 'https://karhanbang.com/detail/?topM={topM}&schType=3&mm_no={mmNo}&mapGubun=N',
                'saveFile': '/DATA/OUTPUT/LSH0615/{addrInfo}/매물_{addrInfo}_{d2}.csv',
            }

            # {sido}, {sido_no}
            # =================================================================
            # dfg 파일을 이용한 csv 파일 변환
            # =================================================================
            for i, addrInfo in enumerate(sysOpt['addrList']):
                log.info(f'[CHECK] addrInfo : {addrInfo}')

                try:
                    url = sysOpt['url'].format(sido=addrInfo, sido_no=(i+1))
                    response = requests.get(url)
                    response.raise_for_status()

                    # 2. HTML 파싱하기
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # soup.prettify()


                    # //*[@id="print_no"]
                    import re
                    from lxml import etree
                    soup = BeautifulSoup(response.text, 'html.parser')
                    lxml = etree.HTML(str(soup))

                    try:
                        tagList = lxml.xpath('/html/body/div/div[1]/div[2]/ul/li[*]')
                        # tagInfo = tagList[0]
                        for tagInfo in tagList:

                            jsFun = tagInfo.find('a').get('href')
                            mmNo, topM  = re.findall(r"'(.*?)'", jsFun)

                            # = "/detail/?topM="+var2+"&schType=3&mm_no=" + var1 + "&mapGubun=N";
                            # https://karhanbang.com/detail/?topM=11&schType=3&mm_no=49818964&mapGubun=N
                            urlDtl = sysOpt['urlDtl'].format(mmNo=mmNo, topM=topM)

                            tagInfo.find('a/dl/dt/em[1]').text
                            tagInfo.find('a/dl/dt/strong[1]').text
                            tagInfo.find('a/dl/dt/span[1]').text
                            tagInfo.find('a/dl/dd[2]/div/ul/li[1]').text
                            tagInfo.find('a/dl/dd[2]/div/ul/li[2]').text
                            tagInfo.find('a/dl/dd[2]/div/ul/li[3]').text
                            tagInfo.find('a/dl/dd[2]/div/ul/li[4]').text
                            tagInfo.find('a/dl/dd[2]/div/ul/li[5]').text

                            tagInfo.find('div/span[1]').text
                            tagInfo.find('div/button[1]').text
                            tagInfo.find('div/span[1]').text
                            tagInfo.find('div/em[1]').text
                            tagInfo.find('div/em[2]').text

                    except Exception:
                        dtDateTime = None



                    # 3. 데이터 추출하기 (예시: 특정 태그와 클래스를 가진 요소 찾기)
                    # 실제로는 웹사이트의 HTML 구조를 분석하여 정확한 선택자를 사용해야 합니다.
                    # 예를 들어, 매물 정보를 담고 있는 태그가 <div class="item"> 이라면:
                    # items = soup.find_all('div', class_='item')

                    # for item in items:
                    #     # 각 매물에서 필요한 정보 추출 (예: 가격, 위치 등)
                    #     price = item.find('span', class_='price').text
                    #     location = item.find('p', class_='location').text
                    #     print(f"가격: {price}, 위치: {location}")

                    # print("HTML 내용을 성공적으로 가져왔습니다. 이제 원하는 데이터를 추출하세요.")
                    # print(soup.prettify()) # 전체 HTML 내용 출력 (구조 파악용)


                except Exception as e:
                    print(f"오류 발생: {e}")

            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '20250316_ydgDBF/ydg2013.dbf')
            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '20250316_ydgDBF/ydg*.dbf')
            # inpFile = sysOpt['ydgFilePattern']
            # fileList = sorted(glob.glob(inpFile))

            # fileInfo = fileList[0]
            # dataL3 = pd.DataFrame()
            # # for fileInfo in fileList:
            # for i, fileInfo in enumerate(fileList):
            #     log.info(f"[CHECK] fileInfo : {fileInfo}")
            #
            #     data = DBF(fileInfo, encoding='euc-kr', char_decode_errors='ignore', ignore_missing_memofile=True)
            #     dataL1 = pd.DataFrame(data)
            #     # dataL2 =  dataL1.drop(['_NullFlags'], axis=1, errors='ignore')
            #     dataL2 =  dataL1.drop(['_NullFlags', 'MEMO'], axis=1, errors='ignore')
            #     # dataL3 = pd.concat([dataL3, dataL2], ignore_index=True)
            #
            #     fileName = os.path.basename(fileInfo)
            #     fileNameNotExt = fileName.split(".")[0]
            #     isHeader = True if i == 0 else False
            #
            #     dataL2['YEAR_DATE'] = pd.to_datetime(dataL2['RC_DATE'], format='%Y-%m-%d').dt.strftime('%Y')
            #     dataL2['DEPART'] = dataL2['DEPART'].str.replace(r'[\n\r\x0f\x06\x14\x0e\|]', '', regex=True).str.strip()
            #
            #     # dataL2['DEPART'].unique()
            #
            #     saveFile = '{}/{}/{}.csv'.format(globalVar['outPath'], serviceName, fileNameNotExt)
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     dataL2.to_csv(saveFile, index=False, header=isHeader)
            #     log.info(f"[CHECK] saveFile : {saveFile}")
            #
            # # saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, datetime.now().strftime("%Y%m%d"), 'ydg_2007_2025')
            # # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # # dataL3.to_csv(saveFile, index=False)
            # # log.info(f"[CHECK] saveFile : {saveFile}")
            #
            # # =================================================================
            # # csv 파일 병합
            # # =================================================================
            # cmd = sysOpt['cmd'].format(exe='cat', tmpFilePattern=sysOpt['tmpFilePattern'], sep='>', tmpFile=sysOpt['tmpFile'])
            # log.info(f'[CHECK] cmd : {cmd}')
            #
            # try:
            #     res = subprocess.run(cmd, shell=True, executable='/bin/bash')
            #     log.info(f'returncode : {res.returncode} / args : {res.args}')
            #
            #     if res.returncode != 0: log.error(f'[ERROR] cmd : {cmd}')
            # except subprocess.CalledProcessError as e:
            #     raise ValueError(f'[ERROR] 실행 프로그램 실패 : {str(e)}')
            #
            # # =================================================================
            # # 빅쿼리 업로드
            # # =================================================================
            # jsonFile = sysOpt['jsonFile']
            # jsonList = sorted(glob.glob(jsonFile))
            # if jsonList is None or len(jsonList) < 1:
            #     log.error(f'jsonFile : {jsonFile} / 설정 파일 검색 실패')
            #     exit(1)
            #
            # jsonInfo = jsonList[0]
            #
            # try:
            #     credentials = service_account.Credentials.from_service_account_file(jsonInfo)
            #     client = bigquery.Client(credentials=credentials, project=credentials.project_id)
            # except Exception as e:
            #     log.error(f'Exception : {e} / 빅쿼리 연결 실패')
            #     exit(1)
            #
            # # inpFile = '{}/{}/{}'.format(globalVar['outPath'], serviceName, '20250320_ydg2007-2025.csv')
            # inpFile = sysOpt['tmpFile']
            # fileList = sorted(glob.glob(inpFile))
            # if fileList is None or len(fileList) < 1:
            #     log.error(f'inpFile : {inpFile} / 파일 검색 실패')
            #     exit(1)
            #
            # fileInfo = fileList[0]
            # data = pd.read_csv(fileInfo)
            #
            # # 데이터 병합
            # sectGrpData = pd.read_csv(sysOpt['sectGrpFile'])
            # mrgData = pd.merge(left=data, right=sectGrpData, how='left', left_on=['SECTION'], right_on=['SECTION'])
            # mrgData.loc[mrgData['GROUP'].isna(), 'GROUP'] = '미분류'
            #
            # saveFile = sysOpt['saveFile']
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # mrgData.to_csv(saveFile, index=False)
            # log.info(f'[CHECK] saveFile : {saveFile}')

            # mrgDataL1 = mrgData[mrgData['GROUP'].isna()]
            # mrgData['SECTION'].unique()
            # mrgData['GROUP'].unique()
            # data['DEPART'].unique()
            # data['YEAR_DATE'].unique()

            # 단위업무 중복제거
            # sectionList = data['SECTION'].unique()
            # sectionData = pd.DataFrame(sectionList, columns=["SECTION"])

            # saveFile = sysOpt['saveFile']
            # os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            # sectionData.to_csv(saveFile, index=False)
            # log.info(f'[CHECK] saveFile : {saveFile}')
            # sys.exit(1)
            #
            # jobCfg = bigquery.LoadJobConfig(
            #     source_format=bigquery.SourceFormat.CSV,
            #     skip_leading_rows=1,
            #     autodetect=True,
            #     # autodetect=False,
                # schema=[  # BigQuery 테이블 스키마 정의 (열 이름, 데이터 타입)
                #     bigquery.SchemaField("Y_NO", "INTEGER"),
                #     bigquery.SchemaField("DEPART", "STRING"),
                #     bigquery.SchemaField("DEPART_NO", "STRING"),
                #     bigquery.SchemaField("SECTION", "STRING"),
                #     bigquery.SchemaField("SUBJECT", "STRING"),
                #     bigquery.SchemaField("NAME", "STRING"),
                #     bigquery.SchemaField("YEAR", "STRING"),
                #     bigquery.SchemaField("YEAR_DATE", "STRING"),
                #     bigquery.SchemaField("PUBLIC", "STRING"),
                #     bigquery.SchemaField("RC_DATE", "DATE"),
                #     bigquery.SchemaField("REG_DATE", "DATE"),
                #     bigquery.SchemaField("SIZE", "INTEGER"),
                # ],
            #     write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            #     max_bad_records=1000,
            # )
            #
            # tableId = f"{credentials.project_id}.DMS01.TB_YDG"
            # # with open(fileInfo, "rb") as file:
            # with open(saveFile, "rb") as file:
            #     job = client.load_table_from_file(file, tableId, job_config=jobCfg)
            # job.result()
            # log.info(f"[CHECK] tableId : {tableId}")

            # dataL1 = data.astype(str)
            # dataL1.dtypes

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