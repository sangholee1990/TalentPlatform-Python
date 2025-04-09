# ================================================
# 요구사항
# ================================================
# Python을 이용한 청소년 인터넷 게임 중독 관련 소셜데이터 수집과 분석을 위한 한국형 온톨로지 개발 및 평가

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
# from matplotlib import font_manager, rc
# from dbfread import DBF, FieldParser
import csv
import os
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import re
# from GoogleNews import GoogleNews
from gnews import GNews
from newspaper import Article

import json
from datetime import datetime, timedelta
from googlenewsdecoder import gnewsdecoder
from konlpy.tag import Okt
from collections import Counter
import pytz

import io
import re
import os
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL

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

    log.propagate = False

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
        ,
        'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
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
    serviceName = 'LSH0612'

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
                # 시작/종료 시간
                'srtDate': '2025-01-01',
                'endDate': '2025-01-05',
                'invDate': '1d',
                'searchMaxPage': 99,

                # 언어 설정
                # , 'language' : 'en'
                'language': 'ko',

                # 국가 설정
                # , 'country' : 'US'
                'country': 'KR',

                # 키워드 설정
                'keywordList': ['청소년 게임 중독'],

                # 저장 경로
                'saveCsvFile': '/DATA/OUTPUT/LSH0612/gnews_%Y%m%d.csv',
                'saveXlsxFile': '/DATA/OUTPUT/LSH0612/gnews_%Y%m%d.xlsx',
            }

            # =================================================================
            # 구글 Gemini 온톨로지 생성
            # =================================================================

            import io
            import re
            from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL

            # Provided ontology map data
            ontology_map = {
                "정의 및 진단": {
                    "WHO/의학계 정의": (
                    ["게임 이용 장애", "Gaming Disorder", "중독성 장애", "질병으로 인정", "의학적 도움", "정신 질환"], "WHO/의학계 공식 정의 및 필요성"),
                    "WHO 질병코드 부여": (["질병코드", "질병 코드", "ICD-11", "국제질병분류"], "WHO 질병코드(ICD-11) 지정"),
                    "국내 도입 논의": (["한국표준질병사인분류", "KCD", "국내 도입", "도입 예정"], "국내 질병코드(KCD) 도입 논의"),
                    "질병코드 논쟁": (["질병코드 도입", "반대", "찬성", "게임업계", "의료계", "문체부", "논쟁", "질병인가 증상인가"], "질병코드 도입 관련 찬반 논쟁"),
                    "진단 기준": (["진단 기준", "통제", "우선시", "우선 순위", "부정적인 결과", "지속", "중단하지 못하는", "자가진단"], "게임중독 진단 기준 (WHO 등)"),
                    "유병률/실태": (["중독률", "과몰입군", "위험군", "실태조사", "%가 게임", "증가 추세"], "국내외 게임중독 유병률 및 실태")
                },
                "원인": {
                    "뇌과학적 요인": (["도파민", "뇌 기능", "전두엽", "신경 가소성", "보상회로", "해마", "편도체", "가지치기"], "뇌과학적 원인 (도파민, 전두엽 발달 등)"),
                    "환경/사회적 요인": (
                    ["디지털 기기 보편화", "PC", "스마트폰 사양 향상", "가정 환경", "부모", "학교", "지역사회", "놀이 문화 부족", "학업 스트레스", "입시", "경쟁 사회",
                     "또래", "공공 공간 감소"], "환경 및 사회적 원인"),
                    "심리/정서적 요인": (
                    ["스트레스", "현실 도피", "자존감", "외로움", "우울", "불안", "ADHD", "공존 장애", "공백", "충동성", "주의력", "정서적 교류 부재"],
                    "심리/정서적 원인 (스트레스, 도피, 공존질환 등)"),
                    "게임 자체 요인": (
                    ["중독성 설계", "보상 구조", "알고리즘", "접근성", "온라인게임", "RPG", "FPS", "MMORPG"], "게임 자체의 중독 유발 요인 및 종류")
                },
                "증상 및 영향": {
                    "행동적 증상": (
                    ["게임 시간", "통제하지 못하고", "통제 실패", "중단하지 못하는", "금단", "내성", "밤샘 게임", "집착"], "행동 변화 (시간 통제 실패, 금단 등)"),
                    "심리/정신적 영향": (
                    ["뇌 기능 저하", "인지 기능", "감정 조절", "우울증", "불안", "충동 조절 장애", "공격성", "과몰입", "ADHD", "현실-가상 혼동", "정신병리학적 문제"],
                    "심리/정신 건강 문제"),
                    "신체적 영향": (
                    ["수면 부족", "수면 장애", "건강 문제", "시력", "디스크", "영양실조", "뇌 구조", "뇌 기능 변화", "뇌파 기능", "안구건조증", "근골격계"],
                    "신체 건강 문제"),
                    "사회/학업적 영향": (
                    ["학업 부진", "성적 하락", "결석", "학교 부적응", "사회적 고립", "대인관계", "가족 갈등", "일상생활 파괴", "범죄", "금품탈취", "폭력", "살해",
                     "자살"], "사회생활 및 학업 영향")
                },
                "대책 및 해결방안": {
                    "치료/상담": (
                    ["치료", "상담", "캠프", "정신건강의학과", "약물", "가족 치료", "정신 치료", "인지행동치료", "동기 강화 치료"], "전문 치료 및 상담, 캠프"),
                    "예방/교육": (["예방", "교육", "리터러시", "대안 활동", "취미", "운동", "캠핑"], "예방 교육 및 대안 활동 제공"),
                    "부모/가족 역할": (["부모", "가족", "소통", "이해", "규칙 설정", "햇볕 정책", "관심", "양육 태도", "애착", "친밀감", "기다려주"],
                                 "부모 및 가족의 역할 (소통, 규칙, 지지)"),
                    "정책/규제": (
                    ["정책", "규제", "셧다운제", "게임시간 선택제", "시간 제한", "컨트롤 타워", "정부", "법", "등급제", "실명제", "쿨링오프제", "치유부담금"],
                    "정부 정책 및 규제 (셧다운제 등)"),
                    "사회적 지원": (
                    ["지역사회", "지지 시스템", "사회적 지원", "상담센터", "치료재활", "네트워크", "협력체계", "Wee센터", "돌봄"], "지역사회 및 사회적 지원체계")
                },
                "사회적 맥락 및 논쟁": {
                    "게임 산업과의 관계": (
                    ["게임 산업", "게임업계", "산업 위축", "게임사", "책임", "부담금", "미래 먹거리", "매출", "수출"], "게임 산업계 입장 및 경제적 영향"),
                    "질병코드 논쟁 심화": (["질병코드 도입", "반대", "찬성", "프레임", "논쟁", "과학적 근거", "사회적 합의"], "질병코드 도입 관련 사회적 논쟁 심화"),
                    "다른 중독과의 비교": (
                    ["SNS", "유튜브", "스마트폰", "도박", "알코올", "마약", "다른 중독", "행위 중독", "디지털 미디어"], "타 중독(SNS, 도박, 알코올 등)과의 비교"),
                    "규제 실효성 논란": (["규제 실효성", "셧다운제 효과", "실효성", "VPN", "명의 도용", "우회", "규제 회피"], "규제 정책(셧다운제 등) 실효성 논란"),
                    "청소년 인권/문화": (["인권", "자율권", "행복추구권", "자기결정권", "문화 향유권", "청소년 문화", "여가", "소통 수단", "긍정적 측면", "놀이", "낙인"],
                                  "청소년 인권 및 게임 문화적 측면")
                }
            }

            # Create RDF graph
            g = Graph()

            # Define namespaces
            namespace_str = "http://example.org/adolescent_game_addiction#"
            ns = Namespace(namespace_str)
            g.bind('aga', ns)  # Bind prefix 'aga' to the namespace
            g.bind('owl', OWL)
            g.bind('rdfs', RDFS)

            # Function to create valid URIs from Korean terms
            def create_valid_uri(term):
                # Replace spaces and slashes with underscores
                term = term.replace(" ", "_").replace("/", "_")
                # Basic removal of some problematic characters for URIs
                term = re.sub(r'[()\[\]{}<>"\']', '', term)
                # More robust handling could involve URL encoding, but underscore often works
                return term

            # Define Classes
            Category = ns.Category
            Subcategory = ns.Subcategory
            SpecificConcept = ns.SpecificConcept
            g.add((Category, RDF.type, OWL.Class))
            g.add((Subcategory, RDF.type, OWL.Class))
            g.add((SpecificConcept, RDF.type, OWL.Class))

            # Define Properties
            hasSubcategory = ns.hasSubcategory
            hasSpecificConcept = ns.hasSpecificConcept
            hasKeyword = ns.hasKeyword
            g.add((hasSubcategory, RDF.type, OWL.ObjectProperty))
            g.add((hasSpecificConcept, RDF.type, OWL.ObjectProperty))
            g.add((hasKeyword, RDF.type, OWL.DatatypeProperty))  # Keywords are literals

            # Keep track of added URIs to avoid duplicate definitions
            added_uris = set()

            # Add data to the graph
            for large_cat, medium_cats in ontology_map.items():
                category_uri = ns[create_valid_uri(large_cat)]
                if category_uri not in added_uris:
                    g.add((category_uri, RDF.type, Category))
                    g.add((category_uri, RDFS.label, Literal(large_cat, lang='ko')))
                    added_uris.add(category_uri)

                for medium_cat, (keywords, small_cat_desc) in medium_cats.items():
                    subcategory_uri = ns[create_valid_uri(medium_cat)]
                    if subcategory_uri not in added_uris:
                        g.add((subcategory_uri, RDF.type, Subcategory))
                        g.add((subcategory_uri, RDFS.label, Literal(medium_cat, lang='ko')))
                        added_uris.add(subcategory_uri)

                    # Link Category to Subcategory
                    g.add((category_uri, hasSubcategory, subcategory_uri))

                    concept_uri = ns[create_valid_uri(small_cat_desc)]
                    if concept_uri not in added_uris:
                        g.add((concept_uri, RDF.type, SpecificConcept))
                        g.add((concept_uri, RDFS.label, Literal(small_cat_desc, lang='ko')))
                        # Add keywords as literals linked to the specific concept
                        for keyword in keywords:
                            g.add((concept_uri, hasKeyword, Literal(keyword, lang='ko')))
                        added_uris.add(concept_uri)

                    # Link Subcategory to SpecificConcept
                    g.add((subcategory_uri, hasSpecificConcept, concept_uri))

            # # Save the RDF graph in RDF/XML format (OWL) to a BytesIO object
            # rdf_output = io.BytesIO()
            # # Explicitly using encoding='utf-8' is important for Korean characters
            # g.serialize(destination=rdf_output, format="xml", encoding='utf-8')
            # rdf_output.seek(0)
            #
            # # Define the filename
            # rdf_file_name = 'ontology_map_output.owl'
            #
            # print(f"'{rdf_file_name}' 파일이 RDF/XML(OWL) 형식으로 생성되었습니다.")

            # --- 파일 저장 부분 ---
            # 1. 원하는 파일명 지정 (여기서는 예시로 'my_ontology.owl' 사용)
            file_name_to_save = "my_ontology.owl"

            # 2. 파일 경로 지정 (코드 실행 환경 내 임시 경로)
            #    주의: 이 경로는 사용자 로컬 컴퓨터의 경로가 아닙니다.
            internal_file_path = os.path.join("/tmp", file_name_to_save)  # /tmp 디렉토리 사용 예시

            try:
                # 3. 지정된 파일명(경로)으로 그래프 직렬화 (파일 쓰기)
                g.serialize(destination=internal_file_path, format="xml", encoding='utf-8')
                print(f"파일 '{file_name_to_save}'(이)가 내부 경로 '{internal_file_path}'에 성공적으로 저장되었습니다.")

                # 4. 내부적으로 저장된 파일을 다시 읽어서 BytesIO 객체로 만듦 (사용자에게 전달하기 위해)
                with open(internal_file_path, "rb") as f:
                    file_content_bytes = f.read()
                rdf_output_for_download = io.BytesIO(file_content_bytes)
                rdf_output_for_download.seek(0)

                # 5. 사용자에게 파일 전달 준비 (실제 전달은 인터페이스가 처리)
                # print(f"'{file_name_to_save}' 파일을 다운로드할 수 있도록 준비했습니다.")

            except Exception as e:
                print(f"파일 저장 중 오류 발생: {e}")
                rdf_output_for_download = None  # 오류 발생 시 None으로 설정

            # --- 파일 전달 부분 (실제 인터페이스에서 처리) ---
            # 이 코드 블록 다음에는 rdf_output_for_download 와 file_name_to_save 를 사용하여
            # 사용자에게 파일 다운로드를 제공하는 로직이 실행됩니다. (아래 <ctrl97>file 부분)

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
