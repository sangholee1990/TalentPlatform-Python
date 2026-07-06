# ================================================
# 요구사항
# ================================================
# Python을 이용한 한국공인중개사 협회 수집

# cd /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys
# /SYSTEMS/LIB/anaconda3/envs/py39/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0612-DaemonFramework-analy-naverSearchApi.py
# nohup /SYSTEMS/LIB/anaconda3/envs/py39/bin/python /SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/unitSys/TalentPlatform-LSH0612-DaemonFramework-analy-naverSearchApi.py &
# tail -f nohup.out

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
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import pandas as pd
import re
import json
from datetime import datetime, timedelta
# from konlpy.tag import Okt
from collections import Counter
import pytz
import os
import sys
import urllib.request
import os
import sys
import requests
import json
from konlpy.tag import Okt
from newspaper import Article
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import time

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

# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
    )

    os.makedirs(os.path.dirname(saveLogFile), exist_ok=True)

    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    streamHandler = logging.StreamHandler()
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1, backupCount=30, encoding='utf-8')

    streamHandler.setFormatter(format)
    fileHandler.setFormatter(format)

    log.addHandler(streamHandler)
    log.addHandler(fileHandler)

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


def propOrgData(baseUrl, page):

    result = pd.DataFrame()

    try:
        url = f"{baseUrl.rstrip('/')}/ptemplate/construction.asp"
        page.goto(url, timeout=10000)

        # 탭 메뉴 리스트 추출
        tabList = page.locator(".organized_tab_wrap_sn ul li a")
        tabCnt = tabList.count()

        if tabCnt == 0:
            log.error(f"탭 메뉴 없음, {url}")
            return result

        for i in range(tabCnt):
            tab = tabList.nth(i)
            tabName = tab.get_attribute("title")
            page.evaluate(f"fnChangeGrade('11', '', '{tabName}');")
            log.info(f"tabName : {tabName}")

            try:
                page.wait_for_selector(".name_card", timeout=3000)
                time.sleep(0.5)

                # --- [병합된 파싱 로직 시작] ---
                html = page.content()
                soup = BeautifulSoup(html, 'html.parser')
                cards = soup.find_all('div', class_='name_card')
                parsed_results = []

                for card in cards:
                    if not card.text.strip(): continue

                    name = ""
                    position = ""
                    office_name = ""
                    address = ""
                    phone = ""
                    fax = ""
                    email = ""
                    img_url = ""

                    img_tag = card.find('img')
                    if img_tag and img_tag.get('src'):
                        img_url = img_tag.get('src').strip()

                    name_tag = card.select_one('.lc01')
                    if name_tag:
                        name = name_tag.text.strip()

                    pos_tag = card.select_one('.lc03')
                    if pos_tag:
                        position = pos_tag.text.strip()

                    trs = card.find_all('tr')
                    for tr in trs:
                        tds = tr.find_all('td')
                        if not tds: continue

                        label = tds[0].text.strip().replace(' ', '')

                        if label == '직위':
                            position = tds[1].text.strip() if len(tds) > 1 else position
                        elif label == '사무소명칭':
                            office_name = tds[1].text.strip() if len(tds) > 1 else ""
                        elif label == '사무소소재지':
                            address = tds[1].text.strip() if len(tds) > 1 else ""
                        elif label == '일반전화':
                            phone = tds[1].text.strip() if len(tds) > 1 else ""
                            if len(tds) > 2:
                                fax = tds[2].text.replace('FAX', '').strip()
                        elif label == 'E-mail' or label == '이메일':
                            email = tds[1].text.strip() if len(tds) > 1 else ""

                    if not name and not office_name:
                        continue

                    parsed_results.append({
                        "조직명": tabName,  # 두 함수를 합쳤기 때문에 tabName을 바로 넣을 수 있습니다.
                        "이름": name,
                        "직위": position,
                        "사무소명칭": office_name,
                        "사무소 소재지": address,
                        "일반전화": phone,
                        "팩스번호": fax,
                        "이메일": email,
                        "이미지URL": img_url
                    })

                cardData = pd.DataFrame(parsed_results)
                result = pd.concat([result, cardData], ignore_index=True)
            except Exception as e:
                log.error(f'Exception : {e}')
    except Exception as e:
        log.error(f'Exception : {e}')

    return result

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

    prjName = 'colctKar'
    serviceName = 'VERSE2026'

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
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                },
                'url': 'https://www.kar.or.kr/pabout/branch.asp',
                'saveFile': '/DATA/OUTPUT/VERSE2026/한국공인중개사협회 시도지회.csv',
            }

            # ==========================================================================================================
            # 기본정보 수집
            # ==========================================================================================================
            url = sysOpt['url']
            headers = sysOpt['headers']

            res = requests.get(url, headers=headers)
            if not res.status_code == 200:
                log.error(f"자료 없음")
                return

            res.encoding = res.apparent_encoding
            soup = BeautifulSoup(res.text, 'html.parser')
            urlList = soup.find_all('a', class_='loc')

            urlItem = {}
            for tag in urlList:
                href = tag.get('href')
                name = tag.get_text(strip=True)
                if href and href.startswith('http'):
                    urlItem[name] = href
            log.info(f"urlItem : {urlItem}")

            data = pd.DataFrame()
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                for name, href in urlItem.items():
                    propData = propOrgData(href, page)
                    data = pd.concat([data, propData], ignore_index=True)
                browser.close()

            saveFile = sysOpt['saveFile']
            os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            data.to_csv(saveFile, index=False)
            log.info(f'saveFile : {saveFile}')



            # =================================================================
            # 네이버뉴스 API 전처리
            # =================================================================
            # okt = Okt()
            # for modelType in sysOpt['modelList']:
            #     log.info(f'[CHECK] modelType : {modelType}')
            #
            #     modelInfo = sysOpt.get(modelType)
            #     if modelInfo is None: continue
            #
            #     inpFile = modelInfo['inpFile']
            #     fileList = sorted(glob.glob(inpFile), reverse=True)
            #     fileInfo = fileList[0]
            #
            #     data = pd.read_csv(fileInfo)
            #     data['titleDesc'] = data['title'].fillna('') + ' ' + data['description'].fillna('')
            #
            #     # key = 'title'
            #     # key = 'description'
            #     key = 'titleDesc'
            #
            #     # for i, row in data.iterrows():
            #     #     if i > 20: break
            #     #     per = round(i / len(data) * 100, 1)
            #     #     log.info(f'[CHECK] i : {i} / {per}%')
            #     #
            #     #     try:
            #     #         articleInfo = Article(row['link'], language='ko')
            #     #
            #     #         # 뉴스 다운로드/파싱/자연어 처리
            #     #         articleInfo.download()
            #     #         articleInfo.parse()
            #     #         articleInfo.nlp()
            #     #
            #     #         # 명사/동사/형용사 추출
            #     #         text = articleInfo.text
            #     #         data.loc[i, f'text'] = None if text is None or len(text) < 1 else str(text)
            #     #         data.loc[i, f'summary'] = None if articleInfo.summary is None or len(articleInfo.summary) < 1 else str(articleInfo.summary)
            #     #         data.loc[i, f'authors'] = None if articleInfo.authors is None or len(articleInfo.authors) < 1 else str(articleInfo.authors)
            #     #     except Exception as e:
            #     #         log.error(f"Exception : {e}")
            #
            #     dataL1 = data.groupby(['sgg', 'search'])[key].apply(
            #         lambda x: ' '.join([str(text).strip() for text in x.dropna() if str(text).strip() != ''])
            #     ).reset_index()
            #     dataL1 = dataL1[dataL1[key].str.strip() != ''].reset_index(drop=True)
            #
            #     keywordDataL2 = pd.DataFrame()
            #     for i, row in dataL1.iterrows():
            #         textList = row[key]
            #         if textList is None or len(textList) < 1: continue
            #         posTagList = okt.pos(textList, stem=True)
            #
            #         keyList = ['Noun']
            #         for keyInfo in keyList:
            #             keywordList = [word for word, pos in posTagList if pos in keyInfo]
            #
            #             # 불용어 제거
            #             keywordList = [word for word in keywordList if word not in sysOpt['stopWordList'] and len(word) > 1]
            #
            #             # 빈도수 계산
            #             keywordCnt = Counter(keywordList).most_common(100)
            #             keywordData = pd.DataFrame(keywordCnt, columns=['keyword', 'cnt']).sort_values(by='cnt', ascending=False)
            #             keywordDataL1 = keywordData[keywordData['keyword'].str.len() >= 2].reset_index(drop=True)
            #
            #             keywordDataL1['sgg'] = row['sgg']
            #             keywordDataL1['search'] = row['search']
            #
            #             keywordDataL2 = pd.concat([keywordDataL2, keywordDataL1], ignore_index=True)
            #
            #     # =================================================================
            #     # CSV 통합파일
            #     # =================================================================
            #     saveFile = sysOpt['키워드']['saveFile']
            #     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
            #     keywordDataL2.to_csv(saveFile, index=False)
            #     log.info(f'[CHECK] saveFile : {saveFile}')
            #
            #     # =================================================================
            #     # 빅쿼리 업로드
            #     # =================================================================
            #     jsonFile = sysOpt['jsonFile']
            #     jsonList = sorted(glob.glob(jsonFile))
            #     if jsonList is None or len(jsonList) < 1:
            #         log.error(f'설정 파일 없음 : {jsonFile}')
            #         raise Exception(f'설정 파일 없음 : {jsonFile}')
            #         # exit(1)
            #
            #     jsonInfo = jsonList[0]
            #
            #     try:
            #         credentials = service_account.Credentials.from_service_account_file(jsonInfo)
            #         client = bigquery.Client(credentials=credentials, project=credentials.project_id)
            #     except Exception as e:
            #         log.error(f'빅쿼리 연결 실패 : {e}')
            #         raise Exception(f'빅쿼리 연결 실패 : {e}')
            #         # exit(1)
            #
            #     jobCfg = bigquery.LoadJobConfig(
            #         source_format=bigquery.SourceFormat.CSV,
            #         skip_leading_rows=1,
            #         autodetect=True,
            #         write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            #         max_bad_records=1000,
            #     )
            #
            #     tableId = f"{credentials.project_id}.DMS01.TB_KEYWORD"
            #     with open(saveFile, "rb") as file:
            #         job = client.load_table_from_file(file, tableId, job_config=jobCfg)
            #     job.result()
            #     log.info(f"[CHECK] tableId : {tableId}")

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