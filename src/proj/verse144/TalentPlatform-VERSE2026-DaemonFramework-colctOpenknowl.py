# ================================================
# 요구사항
# ================================================
# Python을 이용한 오픈놀 청창사 동문수첩 수집

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
import re
import json
from datetime import datetime, timedelta
from collections import Counter
import pytz
import urllib.request
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# =================================================
# 1. 초기 설정
# =================================================
warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus'] = False

tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')

# =================================================
# 2. 유틸리티 함수
# =================================================
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

def initGlobalVar(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

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

def initArgument(globalVar):
    parser = argparse.ArgumentParser()

    for i, argv in enumerate(sys.argv[1:]):
        if not argv.__contains__('--'): continue
        parser.add_argument(argv)

    inParInfo = vars(parser.parse_args())
    log.info(f"[CHECK] inParInfo : {inParInfo}")

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

    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'
    env = 'dev'

    if platform.system() == 'Windows':
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'colctOpenknowl'
    serviceName = 'VERSE2026'

    log = initLog(env, contextPath, prjName)
    globalVar = initGlobalVar(env, contextPath, prjName)

    def __init__(self):
        log.info('[START] {}'.format("init"))
        try:
            initArgument(globalVar)
        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("init"))

    def exec(self):
        log.info('[START] {}'.format("exec"))

        try:
            if (platform.system() == 'Windows'):
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            sysOpt = {
                'url': 'https://portal.nbox365.com/openknowl',
                'saveFile': f"{globalVar['outPath']}/VERSE2026/오픈놀_청창사_동문수첩.csv" if platform.system() == 'Windows' else '/DATA/OUTPUT/VERSE2026/오픈놀_청창사_동문수첩.csv',
            }

            url = sysOpt['url']
            dataList = []

            with sync_playwright() as p:
                # browser = p.chromium.launch(headless=True)
                browser = p.chromium.launch(headless=False)
                page = browser.new_page()
                page.goto(url, wait_until='networkidle')
                
                log.info("로그인 시작")
                
                # 상위 이메일 탭 클릭
                try:
                    page.locator('#pnl1_f20680693_1').click()
                    page.wait_for_timeout(1000)
                except Exception as e:
                    log.warning(f"이메일 탭 클릭 실패 (무시됨): {e}")
                    
                page.fill('input[type="email"], input[placeholder*="이메일"], input[placeholder*="Email"], input[name="loginId"], input[name="email"], input#email', 'topbdscokr@gmail.com')
                page.fill('input[type="password"], input[placeholder*="비밀번호"], input[name="password"], input#password', 'topbds9367!')
                page.click('button[type="submit"], button:has-text("로그인"), a:has-text("로그인")')
                
                page.wait_for_timeout(3000)
                log.info("로그인 완료, 청창사 클릭")
                

                try:
                    page.locator('text="청창사"').first.click()
                    page.wait_for_timeout(3000)
                except Exception as e:
                    log.error(f"청창사 메뉴 찾기 실패: {e}")
                
                # 그룹 목록 확인
                group_loc = page.locator('a:has-text("명")')
                group_count = group_loc.count()
                log.info(f"그룹 개수: {group_count}")
                
                for i in range(group_count):
                    try:
                        group_loc = page.locator('a:has-text("명")')
                        group_name_text = group_loc.nth(i).inner_text().replace('\n', ' ')
                        log.info(f"그룹 클릭: {group_name_text}")
                        group_loc.nth(i).click()
                        page.wait_for_timeout(3000)
                        
                        # 스크롤 내리기 (총 회원수 확인 후)
                        try:
                            total_text = page.locator('text="총"').first.inner_text()
                            total_count = int(re.sub(r'[^0-9]', '', total_text))
                        except:
                            total_count = 1000
                            
                        loaded_count = 0
                        prev_loaded = -1
                        while loaded_count < total_count and loaded_count != prev_loaded:
                            prev_loaded = loaded_count
                            page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                            page.wait_for_timeout(1000)
                            loaded_count = page.locator('.clickable, .card, div:has(> div > h5)').count()
                            log.info(f"스크롤 진행: {loaded_count} / {total_count}")
                            
                        # 카드 수집
                        mobile_labels = page.locator('text="Mobile :"').all()
                        log.info(f"해당 그룹의 연락처(Mobile) 개수: {len(mobile_labels)}")
                        
                        for mobile_label in mobile_labels:
                            try:
                                card = mobile_label.locator('xpath=ancestor::div[contains(@class, "card") or contains(@class, "clickable") or contains(@class, "row")][1]')
                                
                                try:
                                    name = card.locator('h5').first.inner_text().strip()
                                except:
                                    name = ""
                                    
                                company = ""
                                group_name = ""
                                mobile = ""
                                email = ""
                                
                                try:
                                    company_text = card.locator('text="회사 :"').first.inner_text()
                                    company = company_text.split('회사 :')[-1].strip()
                                except:
                                    pass
                                try:
                                    group_text = card.locator('text="그룹 :"').first.inner_text()
                                    group_name = group_text.split('그룹 :')[-1].strip()
                                except:
                                    pass
                                try:
                                    mobile_text = mobile_label.inner_text()
                                    mobile = mobile_text.split('Mobile :')[-1].strip()
                                except:
                                    pass
                                try:
                                    email_text = card.locator('text="Email :"').first.inner_text()
                                    email = email_text.split('Email :')[-1].strip()
                                except:
                                    pass
                                    
                                if name or company or mobile:
                                    dataList.append({
                                        "이름": name,
                                        "회사": company,
                                        "그룹": group_name,
                                        "Mobile": mobile,
                                        "Email": email
                                    })
                            except Exception as e:
                                pass
                                
                    except Exception as e:
                        log.error(f"그룹 수집 실패: {e}")
                        
                    # 목록으로 돌아가기 (뒤로 가기)
                    page.go_back()
                    page.wait_for_timeout(3000)
                    if page.locator('a:has-text("명")').count() == 0:
                        try:
                            page.locator('text="청창사"').first.click()
                            page.wait_for_timeout(3000)
                        except:
                            pass
                            
                browser.close()
                
            data = pd.DataFrame(dataList)
            if not data.empty:
                data = data.drop_duplicates()
                saveFile = sysOpt['saveFile']
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                data.to_csv(saveFile, index=False, encoding='utf-8-sig')
                log.info(f'수집 완료, 저장 파일: {saveFile}')
            else:
                log.info('수집된 데이터가 없습니다.')

        except Exception as e:
            log.error(f"Exception : {e}")
            raise e
        finally:
            log.info('[END] {}'.format("exec"))

# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':
    print('[START] {}'.format("main"))
    try:
        subDtaProcess = DtaProcess()
        subDtaProcess.exec()
    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)
    finally:
        print('[END] {}'.format("main"))