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

def get_txt(parent, sel):
    try:
        loc = parent.locator(sel)
        if loc.count() > 0:
            return loc.first.inner_text(timeout=500).strip()
    except:
        pass
    return ""

def get_txt_by_label(parent, label):
    try:
        loc = parent.locator(f'text="{label}"')
        if loc.count() > 0:
            # 가져온 요소의 부모 텍스트를 통해 값을 포함하여 가져오기
            parent_text = loc.first.locator('xpath=..').inner_text(timeout=500)
            parent_text = parent_text.replace('\xa0', ' ') # 줄바꿈/공백 정규화
            # 원본 라벨 대신 콜론(:) 기준으로 분리하여 '회사  : 메디프리터' 같은 찌꺼기 문자 제거
            val = parent_text.split(':')[-1].strip()
            return val.split('\n')[0].strip()
    except:
        pass
    return ""

def parse_modal_text(text):
    details = {}
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    current_section = None
    known_labels = [
        "성명", "전화번호", "Email", "업체명", "업체명(영문)", "회사대표",
        "사업자등록번호", "법인등록번호", "업태", "종목", "사업장(국가)",
        "우편번호", "주소", "창립일", "Fax", "홈페이지", "매출액(백만원)",
        "근로자수", "주생산품", "주요시장(국가)", "품질인증", "그룹",
        "담당자", "회사 전경", "회사 소개"
    ]
    for k, line in enumerate(lines):
        if line == "담당자": current_section = "담당자"; continue
        elif line in ["회사 소개", "회사 전경", "회사 상세"]: current_section = "회사"; continue
        target_key = None
        if line == "성명" and current_section == "담당자": target_key = "담당자_성명"
        elif line == "전화번호": target_key = "담당자_전화번호" if current_section == "담당자" else "회사소개_전화번호"
        elif line == "Email" and current_section == "담당자": target_key = "담당자_Email"
        elif line in known_labels:
            if line not in ["성명", "전화번호", "Email", "담당자", "회사 전경", "회사 소개", "그룹"]:
                target_key = "회사소개_" + line
            else:
                target_key = line
        if target_key:
            if k + 1 < len(lines):
                next_line = lines[k+1]
                if next_line not in known_labels and not next_line.startswith("회사") and not next_line.startswith("그룹") and not next_line.startswith("Mobile") and ":" not in next_line:
                    details[target_key] = next_line
                else:
                    details[target_key] = ""
    return details

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
                'userId': None,
                'userPw': None,
                'saveFile': '/DATA/OUTPUT/VERSE2026/오픈놀_청창사_동문수첩.csv',
            }

            url = sysOpt['url']
            dataList = []

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                # browser = p.chromium.launch(headless=False)
                page = browser.new_page()
                page.goto(url, wait_until='networkidle')
                
                log.info("로그인 시작")
                
                # 상위 이메일 탭 클릭
                try:
                    page.locator('#pnl1_f20680693_1').click()
                    page.wait_for_timeout(1000)
                except Exception as e:
                    log.warning(f"이메일 탭 클릭 실패 (무시됨): {e}")
                    
                # 아이디(이메일), 비밀번호 입력 (동적 생성되는 ID 반영)
                page.fill('#pnl1_f20098890', sysOpt['userId'])
                page.fill('#pnl1_f20098891', sysOpt['userPw'])
                page.click('#pnl1_tbtn4')
                
                page.wait_for_timeout(3000)
                log.info("로그인 완료, 청창사 클릭")
                

                try:
                    page.locator('text="청창사"').first.click()
                    page.wait_for_timeout(3000)
                except Exception as e:
                    log.error(f"청창사 메뉴 찾기 실패: {e}")
                
                # 그룹 목록 URL 저장
                dashboard_url = page.url
                # log.info(f"대시보드 기준 URL 저장: {dashboard_url}")
                
                # 그룹 목록 확인
                group_loc = page.locator('a:has-text("명")')
                group_count = group_loc.count()
                log.info(f"그룹 개수: {group_count}")
                
                for i in range(group_count):
                    try:
                        # 대시보드 렌더링 확인 및 대기
                        try:
                            # 그룹 리스트 항목이 이미 나타나 있는지 확인
                            page.locator('a:has-text("명")').first.wait_for(state='visible', timeout=5000)
                        except:
                            log.info("대시보드를 찾지 못해 완전히 재로그인하여 복구합니다.")
                            try:
                                page.goto(url, wait_until='networkidle')
                                page.wait_for_timeout(3000)
                                
                                try:
                                    page.locator('#pnl1_f20680693_1').click(timeout=3000)
                                    page.wait_for_timeout(1000)
                                except:
                                    pass
                                    
                                page.fill('#pnl1_f20098890', sysOpt['userId'])
                                page.fill('#pnl1_f20098891', sysOpt['userPw'])
                                page.click('#pnl1_tbtn4')
                                page.wait_for_timeout(3000)
                                
                                menu = page.locator('text="청창사"').first
                                menu.click(timeout=5000)
                                page.wait_for_timeout(2000)
                                
                                page.locator('a:has-text("명")').first.wait_for(state='visible', timeout=10000)
                            except Exception as e:
                                log.warning(f"재로그인 복구 실패: {e}")
                                
                        # 렌더링 후 새로 요소를 찾기 위해 locator 다시 가져오기
                        group_loc = page.locator('a:has-text("명")')
                        if group_loc.count() <= i:
                            log.warning(f"인덱스 {i}의 그룹 항목을 찾을 수 없습니다. (현재 렌더링된 수: {group_loc.count()})")
                            continue
                            
                        # 해당 링크의 부모 요소를 참조하여 "청창사 14기" 등 실제 그룹 제목 추출
                        group_total_count = None
                        try:
                            parent_text = group_loc.nth(i).locator('xpath=..').inner_text(timeout=3000)
                            parent_text = parent_text.replace('\xa0', ' ')
                            # 보통 "청창사 14기 \n 총 104명" 형태로 존재하므로 첫 줄(제목)만 사용
                            group_name_text = parent_text.split('\n')[0].strip()
                            
                            # 인원수 추출 시도 (예: 총 104명)
                            match = re.search(r'총\s*([0-9,]+)\s*명', parent_text)
                            if match:
                                group_total_count = int(match.group(1).replace(',', ''))
                        except:
                            group_name_text = group_loc.nth(i).inner_text(timeout=3000).replace('\xa0', ' ').replace('\n', ' ')
                            
                        log.info(f"그룹 클릭: {group_name_text} (예상 인원수: {group_total_count})")
                        group_loc.nth(i).click(timeout=3000)
                        page.wait_for_timeout(3000)
                        
                        # 스크롤 내리기 (총 회원수 확인 후)
                        try:
                            total_text = page.locator('text="총"').first.inner_text(timeout=3000)
                            total_count = int(re.sub(r'[^0-9]', '', total_text))
                        except:
                            if group_total_count is not None:
                                total_count = group_total_count
                            else:
                                total_count = 1000
                            log.warning(f"총 인원수를 명시적으로 찾지 못해 {total_count}명으로 가정합니다.")
                            
                        total_processed = 0
                        current_page_num = 1
                        
                        while total_processed < total_count:
                            loaded_count = 0
                            prev_loaded = -1
                            
                            # 스크롤해서 한 페이지 분량 렌더링 (최대 100개)
                            while loaded_count < total_count and loaded_count != prev_loaded:
                                prev_loaded = loaded_count
                                page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                                
                                # 마지막 요소 기준으로 스크롤 (특정 div 내부에 스크롤바가 있을 경우 대비)
                                items = page.locator('text="Mobile :"')
                                if items.count() > 0:
                                    try:
                                        items.last.scroll_into_view_if_needed(timeout=1000)
                                    except:
                                        pass
                                        
                                page.wait_for_timeout(1500)
                                loaded_count = items.count()
                                log.info(f"{current_page_num}페이지 스크롤 진행: {loaded_count} 요소")
    
                                if loaded_count >= 100:
                                    break
                                    
                            # 현재 페이지의 카드 수집
                            mobile_labels = page.locator('text="Mobile :"').all()
                            labels_count = len(mobile_labels)
                            if labels_count == 0:
                                log.warning(f"{current_page_num}페이지에서 수집할 연락처 항목을 찾지 못했습니다.")
                                break

                            # labels_count = 5
                            log.info(f"{current_page_num}페이지 내 수집 가능 항목 수: {labels_count}")
                            
                            for idx in range(labels_count):
                                try:
                                    # 리스트가 초기화되었을 수 있으므로 필요한 만큼 스크롤 복구
                                    current_labels = page.locator('text="Mobile :"').count()
                                
                                    if current_labels == 0 and idx > 0:
                                        log.warning("화면에서 카드 목록을 찾을 수 없습니다. (이전 카드의 상세 화면에서 복귀 실패)")
                                        break
                                    
                                    scroll_attempts = 0
                                    while current_labels <= idx and scroll_attempts < 15:
                                        page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                                        page.wait_for_timeout(1000)
                                        current_labels = page.locator('text="Mobile :"').count()
                                        scroll_attempts += 1
                                    
                                    mobile_labels = page.locator('text="Mobile :"').all()
                                    if idx >= len(mobile_labels):
                                        log.warning(f"인덱스 {idx}에 해당하는 카드를 찾을 수 없어 건너뜁니다.")
                                        continue
                                    
                                    mobile_label = mobile_labels[idx]
                                    card = mobile_label.locator('xpath=ancestor::div[contains(@class, "card") or contains(@class, "clickable") or contains(@class, "row")][1]')
                                
                                    name = get_txt(card, '#name, h5')
                                    company = get_txt(card, '#company') or get_txt_by_label(card, '회사 :')
                                    group_name = get_txt(card, '#group_name') or get_txt_by_label(card, '그룹 :')
                                    mobile = get_txt(card, '#mobile') or get_txt_by_label(card, 'Mobile :')
                                    email = get_txt(card, '#email') or get_txt_by_label(card, 'Email :')
                                    parsed_details = {}
                                
                                    current_url = page.url
                                    try:
                                        try:
                                            card.locator('h5').first.click(timeout=2000)
                                        except:
                                            card.click(timeout=2000)
                                        page.wait_for_timeout(1500)
                                        
                                        modal = page.locator('.modal.show, [role="dialog"], .offcanvas.show, .modal-content, .el-dialog, .v-dialog, .ant-modal, .MuiDialog-root').locator('visible=true').first
                                    
                                        if modal.count() > 0:
                                            modal_text = modal.inner_text()
                                            context_node = modal
                                        else:
                                            modal_text = page.locator('body').inner_text()
                                            context_node = page
                                        
                                        parsed_details = parse_modal_text(modal_text)
                                    
                                        id_keys = {
                                            "회사소개_업체명": '#업체명', "회사소개_업체명(영문)": '#업체명_영문', "회사소개_회사대표": '#회사대표', 
                                            "회사소개_사업자등록번호": '#사업자등록번호', "회사소개_법인등록번호": '#법인등록번호',
                                            "회사소개_업태": '#업태', "회사소개_종목": '#종목', "회사소개_사업장(국가)": '#사업장_국가', "회사소개_우편번호": '#우편번호', 
                                            "회사소개_주소": '#주소', "회사소개_창립일": '#창립일', "회사소개_전화번호": '#회사_전화번호', "회사소개_Fax": '#Fax', 
                                            "회사소개_홈페이지": '#홈페이지', "회사소개_매출액(백만원)": '#매출액', "회사소개_근로자수": '#근로자수', 
                                            "회사소개_주생산품": '#주생산품', "회사소개_주요시장(국가)": '#주요시장_국가', "회사소개_품질인증": '#품질인증',
                                            "담당자_성명": '#담당자_성명', "담당자_전화번호": '#담당자_전화번호', "담당자_Email": '#담당자_Email'
                                        }
                                        for key, sel in id_keys.items():
                                            val = get_txt(context_node, sel)
                                            if val: parsed_details[key] = val
                                        
                                        try:
                                            if page.url != current_url:
                                                page.go_back(wait_until='domcontentloaded')
                                                page.wait_for_timeout(2000)
                                            else:
                                                back_btn = page.locator('text="뒤로가기", text="목록", text="목록으로", .el-icon-back, .mdi-arrow-left, button:has-text("목록")')
                                                if back_btn.count() > 0:
                                                    back_btn.first.click(timeout=1500)
                                                    page.wait_for_timeout(2000)
                                                else:
                                                    page.keyboard.press('Escape')
                                                    page.wait_for_timeout(500)
                                                
                                                    close_btn = page.locator('button:has-text("닫기"), button[aria-label="Close"], .btn-close, .el-dialog__close, .el-icon-close')
                                                    if close_btn.count() > 0:
                                                        close_btn.first.click(timeout=1000)
                                                        page.wait_for_timeout(500)
                                        except Exception as nav_e:
                                            log.warning(f"화면 복귀 실패: {nav_e}")
                                    except Exception as inner_e:
                                        log.warning(f"상세 정보 수집 실패: {inner_e}")
                                    
                                    if name or company or mobile:
                                        row_data = {
                                            "이름": name,
                                            "회사": company,
                                            "그룹": group_name,
                                            "Mobile": mobile,
                                            "Email": email
                                        }
                                        row_data.update(parsed_details)
                                        log.info(row_data)
                                        dataList.append(row_data)
                                    
                                    total_processed += 1
                                except Exception as e:
                                    log.error(f"카드 {idx} 처리 중 에러 발생: {e}")
                                
                            # 다음 페이지 체크
                            if total_processed < total_count:
                                current_page_num += 1
                                try:
                                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                                    page.wait_for_timeout(2000)
                                    
                                    selectors = [
                                        f'li.number:has-text("{current_page_num}")',
                                        f'a.number:has-text("{current_page_num}")',
                                        f'button:has-text("{current_page_num}")',
                                        f'a:text-is("{current_page_num}")',
                                        f'button:text-is("{current_page_num}")',
                                        f'li:text-is("{current_page_num}")',
                                        f'span:text-is("{current_page_num}")',
                                        '.btn-next'
                                    ]
                                    
                                    next_page_btn = None
                                    for sel in selectors:
                                        btn = page.locator(sel).locator('visible=true').first
                                        if btn.count() > 0:
                                            next_page_btn = btn
                                            break
                                            
                                    if next_page_btn is None:
                                        log.info("페이지 버튼을 한 번에 찾지 못했습니다. 새로고침(UI 갱신 유도) 후 다시 확인합니다.")
                                        page.evaluate("window.scrollBy(0, -500)")
                                        page.wait_for_timeout(1000)
                                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                                        page.wait_for_timeout(2000)
                                        
                                        for sel in selectors:
                                            btn = page.locator(sel).locator('visible=true').first
                                            if btn.count() > 0:
                                                next_page_btn = btn
                                                break

                                    if next_page_btn is not None and next_page_btn.count() > 0:
                                        log.info(f"{current_page_num}페이지로 자동 새로고침(이동)합니다.")
                                        next_page_btn.click(timeout=5000)
                                        page.wait_for_timeout(3000)
                                    else:
                                        log.warning(f"추가 항목({total_processed}/{total_count})이 있으나 {current_page_num}페이지 버튼을 찾을 수 없습니다.")
                                        break
                                except Exception as page_e:
                                    log.error(f"페이지 이동 중 오류 발생: {page_e}")
                                    break
                                
                    except Exception as e:
                        log.error(f"그룹 수집 실패: {e}")
                        
                    try:
                        page.keyboard.press('Escape')
                        page.wait_for_timeout(500)
                        
                        page.goto(dashboard_url, wait_until='domcontentloaded')
                        page.wait_for_timeout(2000)
                    except Exception as back_err:
                        log.warning(f"대시보드 복귀 중 예외 발생 (무시하고 다음 루프에서 복구 시도): {back_err}")
                            
                browser.close()
                
            data = pd.DataFrame(dataList)
            if not data.empty:
                data = data.drop_duplicates()
                saveFile = sysOpt['saveFile']
                os.makedirs(os.path.dirname(saveFile), exist_ok=True)
                data.to_csv(saveFile, index=False, encoding='utf-8-sig')
                log.info(f'saveFile : {saveFile}')
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