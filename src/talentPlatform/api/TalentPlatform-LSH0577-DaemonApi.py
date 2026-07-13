# ============================================
# 요구사항
# ============================================
# LSH0577. Python을 이용한 빅쿼리 기반으로 API 자료서비스

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api
# conda activate py39
# uvicorn TalentPlatform-LSH0577-DaemonApi:app --reload --host=0.0.0.0 --port=9000
# /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api/TalentPlatform-LSH0577-DaemonApi:app --reload --host=0.0.0.0 --port=9000
# nohup uvicorn TalentPlatform-LSH0577-DaemonApi:app --host=0.0.0.0 --port=9000 > nohup.log 2>&1 &
# tail -f nohup.out

# 빠른 재시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api
# conda activate py39
# pkill -f "TalentPlatform-LSH0577-DaemonApi"
# nohup uvicorn TalentPlatform-LSH0577-DaemonApi:app --host=0.0.0.0 --port=9000 > nohup.out 2>&1 &
# tail -f nohup.out

# 프로그램 종료
# ps -ef | grep "TalentPlatform-LSH0577-DaemonApi" | awk '{print $2}' | xargs kill -9

# 테스트 서버
# uvicorn TalentPlatform-LSH0577-DaemonApi:app --reload --host=0.0.0.0 --port=9400
# nohup uvicorn TalentPlatform-LSH0597-DaemonApi:app --reload --host=0.0.0.0 --port=9400 &
# lsof -i :9400 | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9000
# lsof -i :9000 | awk '{print $2}' | xargs kill -9

# pkill -f "TalentPlatform-LSH0577-DaemonApi"
# bash /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/shell/RunShell-ProcAgentCheck.sh

# "[TOP BDS] [통합] 아파트 보고서 (데이터 분석, 가격 예측)" 및 빅쿼리 기반으로 API 배포체계를 전달하오니 확인 부탁드립니다.
# 명세1) http://49.247.41.71:9000/docs
# 명세2) http://49.247.41.71:9000/redoc

# 2024.11.08 아실 플랫폼과 같은 시군구/읍면동/계약년월을 기준으로 최고상승/최고하락 아파트 목록
# https://docs.google.com/document/d/1jUxkICwo2WqHACLc_dEziGl7_Y4f5MMClHW7yyKn_XQ/edit?tab=t.0#heading=h.ebgsmeszn577
# 실거래가 기준 전월 대비 상승 비율이 높은 아파트 리스트 구성
# 금액의 비율이 가장 높은 순으로만 아파트 구성
# 전월의 평균금액과 이번월의 평균금액의 비율 변화
# 전월에 거래가 없으면 해당 아파트는 미표시
# 읍면동 단위로 구분하여 아파트 랭킹 (상위 10개, 더보기 통해 상위 30개)
# 초기 화면은 시 단위, 메뉴바 통해 시군구까지 조회 가능
# 아파트의 평형별로 구분하여 아파트 랭킹 구성
# 레퍼런스 : 아실 데이터 https://asil.kr/asil/index.jsp

# 1. 현재 실거래 리스트에서 연도는 선택만으로 가능한 상황인데 여러 연도로 해야할까요? 그렇게 하려면 API에서 여러 연도를 추가해주시면 (시작연도, 끝연도) 작업 가능합니다.
# 2. 아파트 상세 내용에서 전체 세대수 API가 대부분의 아파트에서 정보가 없습니다. 혹시 세대수를 따로 받을 수 있는 API가 있으면 알려주세요.
# 3. 시군구데이터에서 현재 최대거래량을 전체 데이트를 가져와서 하나하나 세야 하는 방식입니다. 그래서 시군구별 한개만 선택해서 데이터 산출이 가능한데요. 여러개 선택할 경우 서버 응답 시간이 초과되는 상황입니다. 혹시 API에서 시군구별로 거래량을 산출한 후에 보내주면 여러 시군구별이나 전체 데이터로 가능할 것으로 보이는데 어떠신지요?

# ============================================
# 라이브러리
# ============================================
import glob
import os
import platform
import warnings
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
# from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Any, Dict
from typing import Literal
import configparser
import os
from urllib.parse import quote_plus
import requests
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
import os
import shutil
from pydantic import BaseModel, Field, constr
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
import pytz
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import asyncio
from fastapi import FastAPI
import socket
import json
from google.cloud import bigquery
from google.oauth2 import service_account
import db_dtypes
# from src.api.guest.router import router as guest_router
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import requests
import time
from concurrent.futures import ProcessPoolExecutor
import configparser
import uuid
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Optional
from fastapi import File, UploadFile, Form
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import re
from email.utils import formataddr
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from google import genai

# ============================================
# 유틸리티 함수
# ============================================
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

# 인증키 검사
def chkApiKey(api_key: str = Depends(APIKeyHeader(name="api", auto_error=False))):
    if api_key is None:
        raise HTTPException(status_code=401, detail="API 인증키 없음")

    if api_key != '20240922-topbds':
        raise HTTPException(status_code=400, detail="API 인증 실패")

def resResponse(status: str, code: int, message: str, allCnt: int = 0, rowCnt: int = 0, data: Any = None) -> dict:
    return {
        "status": status
        , "code": code
        , "message": message
        , "allCnt": allCnt
        , "rowCnt": rowCnt
        , "data": data
    }

def fetchApi(apiUrl, payload, apiType, recAptData):

    # 공통 반환 컬럼
    result = pd.DataFrame()

    try:
        response = requests.post(apiUrl, data=payload, verify=False, timeout=30)
        response.raise_for_status()

        resJson = response.json().get('recommends', {})
        rawList = resJson.get(apiType, [])

        if rawList is None or len(rawList) < 1:
            return result

        # 추천 서버 응답: [[idx, score], ...] 형태 가정
        resData = pd.DataFrame(rawList, columns=['rec_key', 'score'])

        if len(resData) < 1:
            return result

        aptData = recAptData.copy()

        # ------------------------------------
        # 키 호환 처리
        # 1) apt_idx 우선
        # 2) 없으면 idx 사용
        # 3) 둘 다 없으면 recAptData index를 임시 키로 사용
        # ------------------------------------
        if 'apt_idx' in aptData.columns:
            merge_key = 'apt_idx'
        elif 'idx' in aptData.columns:
            merge_key = 'idx'
        else:
            aptData = aptData.reset_index().rename(columns={'index': 'apt_idx'})
            merge_key = 'apt_idx'

        # 타입 맞춤
        resData['rec_key'] = pd.to_numeric(resData['rec_key'], errors='coerce')
        aptData[merge_key] = pd.to_numeric(aptData[merge_key], errors='coerce')

        resData = resData.dropna(subset=['rec_key']).copy()
        aptData = aptData.dropna(subset=[merge_key]).copy()

        if len(resData) < 1 or len(aptData) < 1:
            return pd.DataFrame(columns=[merge_key, 'score'])

        # 추천 key -> 실제 apt 정보 매핑
        result = pd.merge(
            resData,
            aptData,
            how='left',
            left_on='rec_key',
            right_on=merge_key
        )

        # 자기 자신 추천 제외
        target_apt_idx = pd.to_numeric(pd.Series([payload.get('apt_idx')]), errors='coerce').iloc[0]
        if pd.notna(target_apt_idx) and merge_key in result.columns:
            result = result[result[merge_key] != target_apt_idx]

        # 중복 제거 / 점수순 정렬
        if merge_key in result.columns:
            result = (
                result
                .sort_values(by='score', ascending=False)
                .drop_duplicates(subset=[merge_key], keep='first')
                .reset_index(drop=True)
            )
        else:
            result = result.sort_values(by='score', ascending=False).reset_index(drop=True)

    except Exception as e:
        log.error(f"Exception : {e} / apiType={apiType} / payload={payload}")

    return result

def get_valid_geo_condition(col='geo'):
    geoCol = f"TRIM(CAST({col} AS STRING))"
    return f"""
        {col} IS NOT NULL
        AND {geoCol} != ''
        AND NOT REGEXP_CONTAINS(LOWER({geoCol}), r'^(nan|null|none)?\\s*,\\s*(nan|null|none)?$')
        AND REGEXP_CONTAINS({geoCol}, r'^[-+]?\\d+(\\.\\d+)?\\s*,\\s*[-+]?\\d+(\\.\\d+)?$')
    """

# ============================================
# 주요 설정
# ============================================
# env = 'local'
env = 'dev'
serviceName = 'LSH0577'
prjName = 'test'

ctxPath = os.getcwd()
# ctxPath = f"/SYSTEMS/PROG/PYTHON/IDE"

log = initLog(env, ctxPath, prjName)

# 작업 경로 설정
# os.chdir(f"{ctxPath}")
# log.info(f"[CHECK] getcwd : {os.getcwd()}")

# 옵션 설정
sysOpt = {
    # 빅쿼리 설정 정보 (상호)
    # 'jsonFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/iconic-ruler-239806-7f6de5759012.json',
    # 'dbHostName': 'iconic-ruler-239806.DMS01',

    # 빅쿼리 설정 정보 (verse144)
    'jsonFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/project-p-32424-f1fe6277556d.json',
    'dbHostName': 'project-p-32424.DMS01',

    # 설정 정보
    'cfgFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
    'cfgKey': 'gemini-api-key',
    'cfgVal': 'oper',

    # CORS 설정
    'oriList': ['*'],

    'rcmd': {
        # 'apiCfUrl': 'http://125.251.52.42:9010/recommends_cf',
        # 'apiSimUrl': 'http://125.251.52.42:9010/recommends_simil',
        'apiCfUrl': 'http://localhost:9010/recommends_cf',
        'apiSimUrl': 'http://localhost:9010/recommends_simil',
        'propAptFile': '/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/dataset/tbl_apts.xlsx',
        'propUserFile': '/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/rcmdapt/dataset/tbl_users.xlsx',
        # 'propAptFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/xlsx/20250526_tbl_apts.xlsx',
        # 'propUserFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/xlsx/20250526_tbl_users.xlsx',
    },

    # 설정 정보
    'cfgFile': '/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
    'cfgKey': 'gemini-api-key',
    'cfgVal': 'oper',
    # 'cfgVal': 'local',
}

app = FastAPI(
    openapi_url='/api'
    , docs_url='/docs'
    , redoc_url='/redoc'
)

# 공유 설정
# app.mount('/UPLOAD', StaticFiles(directory='/DATA/UPLOAD'), name='/DATA/UPLOAD')

app.add_middleware(
    CORSMiddleware
    # , allow_origins=["*"]
    , allow_origins=sysOpt['oriList']
    , allow_credentials=True
    , allow_methods=["*"]
    , allow_headers=["*"]
)

# 빅쿼리 설정 정보
jsonFile = sysOpt['jsonFile']
jsonList = sorted(glob.glob(jsonFile))
if jsonList is None or len(jsonList) < 1:
    log.error(f'jsonFile : {jsonFile} / 설정 파일 검색 실패')
    exit(1)

jsonInfo = jsonList[0]

try:
    credentials = service_account.Credentials.from_service_account_file(jsonInfo)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
except Exception as e:
    log.error(f'Exception : {e} / 빅쿼리 연결 실패')
    exit(1)

# 사용자 설정 정보
inpFile = sysOpt['rcmd']['propUserFile']
fileList = sorted(glob.glob(inpFile), reverse=True)
if fileList is None or len(fileList) < 1:
    log.error(f'파일 없음 : {inpFile}')
    sys.exit(1)

recUserData = pd.read_excel(fileList[0])

# 아파트 설정 정보
inpFile = sysOpt['rcmd']['propAptFile']
fileList = sorted(glob.glob(inpFile), reverse=True)
if fileList is None or len(fileList) < 1:
    log.error(f'파일 없음 : {inpFile}')
    sys.exit(1)

recAptData = pd.read_excel(fileList[0])

# 컬럼명 호환 처리
if 'gu' not in recAptData.columns:
    if 'gu_name' in recAptData.columns:
        recAptData['gu'] = recAptData['gu_name']
    elif 'sgg' in recAptData.columns:
        recAptData['gu'] = recAptData['sgg']
    else:
        recAptData['gu'] = ''

# 좌표 컬럼 호환 처리
if 'lat' not in recAptData.columns and 'latitude' in recAptData.columns:
    recAptData['lat'] = recAptData['latitude']

if 'lon' not in recAptData.columns and 'longitude' in recAptData.columns:
    recAptData['lon'] = recAptData['longitude']

drop_cols = [col for col in ['idx'] if col in recAptData.columns]

group_cols = [col for col in ['gu', 'apt', 'area'] if col in recAptData.columns]

if len(group_cols) >= 2:
    recAptData2 = (
        recAptData
        .drop(columns=drop_cols)
        .groupby(group_cols, as_index=False)
        .mean(numeric_only=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
else:
    recAptData2 = recAptData.copy()


def add_apt_age_condition(condList, aptAge):
    if not aptAge:
        return

    curYear = datetime.now().year
    ageExpr = f"({curYear} - SAFE_CAST(conYear AS INT64))"

    if aptAge == '0-1':
        condList.append(
            f"{ageExpr} <= 1"
        )

    elif aptAge == '0-3':
        condList.append(
            f"{ageExpr} >= 0 AND {ageExpr} <= 3"
        )

    elif aptAge == '0-5':
        condList.append(
            f"{ageExpr} >= 0 AND {ageExpr} <= 5"
        )

    elif aptAge == '0-10':
        condList.append(
            f"{ageExpr} >= 0 AND {ageExpr} <= 10"
        )

    elif aptAge == '10+':
        condList.append(
            f"{ageExpr} >= 10"
        )

def add_capacity_range_condition(condList, area, col='capacity'):
    """
    면적 조건 추가
    - 프론트 표시값: 20평대, 30평대 등
    - API 입력값: 51-83, 84-100 등 m² 범위
    - 과거 입력값: 24평 같은 단일 평수도 방어적으로 지원
    """
    if not area:
        return

    conds = []
    pyeong_to_m2 = 3.305785

    for raw in str(area).split(','):
        val = raw.strip()
        if not val:
            continue

        # 51-83, 84~100 형태: capacity(m²) 범위 조건
        m = re.match(r'^(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)$', val)
        if m:
            min_val = float(m.group(1))
            max_val = float(m.group(2))
            conds.append(f"SAFE_CAST({col} AS FLOAT64) BETWEEN {min_val} AND {max_val}")
            continue

        # 20평대, 30평대 형태: 평대 -> m² 범위로 변환
        m = re.match(r'^(\d+)\s*평대$', val)
        if m:
            pyeong_from = int(m.group(1))
            pyeong_to = pyeong_from + 9.9999
            min_val = round(pyeong_from * pyeong_to_m2, 4)
            max_val = round(pyeong_to * pyeong_to_m2, 4)
            conds.append(f"SAFE_CAST({col} AS FLOAT64) BETWEEN {min_val} AND {max_val}")
            continue

        # 24평 형태: 단일 평수 -> ±0.5평 범위로 변환
        m = re.match(r'^(\d+(?:\.\d+)?)\s*평$', val)
        if m:
            pyeong = float(m.group(1))
            min_val = round((pyeong - 0.5) * pyeong_to_m2, 4)
            max_val = round((pyeong + 0.5) * pyeong_to_m2, 4)
            conds.append(f"SAFE_CAST({col} AS FLOAT64) BETWEEN {min_val} AND {max_val}")
            continue

        # 숫자만 들어온 경우: capacity(m²) 단일값 근사 검색
        m = re.match(r'^(\d+(?:\.\d+)?)$', val)
        if m:
            num_val = float(m.group(1))
            conds.append(f"SAFE_CAST({col} AS FLOAT64) BETWEEN {num_val - 0.5} AND {num_val + 0.5}")

    if conds:
        condList.append(f"({' OR '.join(conds)})")


# ============================================
# 비즈니스 로직
# ============================================
# Gemini API키
config = configparser.ConfigParser()
config.read(sysOpt['cfgFile'], encoding='utf-8')
apiKey = config.get(sysOpt['cfgKey'], sysOpt['cfgVal'])
genaiClient = genai.Client(api_key=apiKey)

class cfgRcmd(BaseModel):
    gender: str = Field(..., description='성별 (최소-최대, 1-1 남성, 2-2 여성, 1-2 전체)', examples=['1-1'])
    age: str = Field(..., description='나이 (최소-최대, 20-39)', examples=['20-39'])
    price: str = Field(..., description='가격 억원 (최소-최대, 3-6)', examples=['3-6'])
    area: str = Field(..., description='면적 m² (최소-최대, 58-100)', examples=['58-100'])
    # debtRat: str = Field(..., description='부채 비율', examples=['0.25'])
    apt: str = Field(..., description='아파트 도로명주소, 두산(가산로 99)', examples=['두산'])
    cnt: str = Field(..., alias='cnt', description='추천 개수', examples=['10'])

# ============================================
# 주소 목록
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post(f"/api/sel-aptBaseInfo", dependencies=[Depends(chkApiKey)])
def selAptBaseInfo(
        sgg: str = Query(None, description="시군구", examples=["서울특별시 금천구"])
        , dong: str = Query(None, description="법정동", examples=["가산동"])
        , apt: str = Query(None, description="TB_REAL 아파트명/도로명주소", examples=["두산(가산로)"])
        , aptDtl: str = Query(None, description="TB_REAL 도로명주소 상세", examples=["서울특별시 금천구 가산로 99 두산"])
        , key: str = Query(None, description="TB_REAL 지번주소", examples=["두산(769)"])
        , keyDtl: str = Query(None, description="TB_REAL 지번주소 상세", examples=["서울특별시 금천구 가산동 769 두산"])
        , minScore: float = Query(0.815, description="기본정보 노출 최소 매핑 점수")
    ):
    """
    기능\n
        아파트 상세화면 기본정보 조회 API\n
        TB_REAL 값(apt/aptDtl/key/keyDtl/sgg/dong) → TB_REAL_APT_MAPPING → TB_APT_DETAIL 조회\n
        match_score >= 0.815 이상인 경우에만 기본정보 반환\n
    """
    try:
        if not any([sgg, dong, apt, aptDtl, key, keyDtl]):
            return resResponse("fail", 400, "검색조건 없음. sgg/dong/apt/aptDtl/key/keyDtl 중 1개 이상 필요", 0, 0, None)

        sgg = sgg.strip() if sgg and len(str(sgg).strip()) > 0 else None
        dong = dong.strip() if dong and len(str(dong).strip()) > 0 else None
        apt = apt.strip() if apt and len(str(apt).strip()) > 0 else None
        aptDtl = aptDtl.strip() if aptDtl and len(str(aptDtl).strip()) > 0 else None
        key = key.strip() if key and len(str(key).strip()) > 0 else None
        keyDtl = keyDtl.strip() if keyDtl and len(str(keyDtl).strip()) > 0 else None

        baseSql = f"""
            SELECT
                m.`key` AS mapping_key,
                m.keyDtl AS mapping_keyDtl,
                m.apt AS mapping_apt,
                m.aptDtl AS mapping_aptDtl,
                m.sgg AS mapping_sgg,
                m.dong AS mapping_dong,
                m.geo AS mapping_geo,
                m.kaptCode AS mapping_kaptCode,
                m.kaptName AS mapping_kaptName,
                m.doroJuso AS mapping_doroJuso,
                m.match_score,
                m.match_type,

                d.kaptCode,
                d.kaptName,
                d.kaptAddr,
                d.doroJuso,
                d.region,
                d.bjdCode,
                d.sggCd,
                d.umdCd,
                d.codeAptNm,
                d.codeSaleNm,
                d.codeHeatNm,
                d.codeMgrNm,
                d.codeHallNm,
                d.kaptUsedate,
                d.kaptTarea,
                d.kaptDongCnt,
                d.kaptdaCnt,
                d.hoCnt,
                d.kaptMarea,
                d.kaptMparea60,
                d.kaptMparea85,
                d.kaptMparea135,
                d.kaptMparea136,
                d.privArea,
                d.kaptTopFloor,
                d.ktownFlrNo,
                d.kaptBaseFloor,
                d.kaptdEcntp,
                d.kaptBcompany,
                d.kaptAcompany,
                d.kaptTel,
                d.kaptUrl,
                d.kaptFax,
                d.zipcode
            FROM `{sysOpt['dbHostName']}.TB_REAL_APT_MAPPING` AS m
            INNER JOIN `{sysOpt['dbHostName']}.TB_APT_DETAIL` AS d
                ON m.kaptCode = d.kaptCode
            WHERE m.kaptCode IS NOT NULL
              AND m.match_score >= @minScore
        """

        query_params = [bigquery.ScalarQueryParameter("minScore", "FLOAT64", float(minScore))]

        # -------------------------------------------------
        # 시군구 조건
        # - 프론트에서는 sido/sgg가 분리되어 '강북구'만 넘어오는 경우가 많음
        # - TB_REAL_APT_MAPPING.m.sgg 값은 '서울특별시 강북구' 형태일 수 있음
        # - 따라서 exact(=)가 아니라 LIKE로 처리
        #   예) 입력: 강북구 → m.sgg='서울특별시 강북구' 매칭 가능
        # -------------------------------------------------
        if sgg:
            baseSql += " AND m.sgg LIKE @sggLike"
            query_params.append(bigquery.ScalarQueryParameter("sggLike", "STRING", f"%{sgg}%"))
        # -------------------------------------------------
        # 법정동 조건
        # - dong은 보통 '미아동', '가산동'처럼 그대로 들어오므로 exact 유지
        # - 필요 시 LIKE로 완화 가능하지만, 현재는 오매칭 방지를 위해 exact 권장
        # -------------------------------------------------
        if dong:
            baseSql += " AND m.dong = @dong"
            query_params.append(bigquery.ScalarQueryParameter("dong", "STRING", dong))
        # -------------------------------------------------
        # 아파트명/주소 조건
        # - 프론트에서 TB_REAL에 있는 apt 값을 그대로 넘기는 구조라 exact 유지
        # - 예: '미아동부센트레빌(숭인로7가길 37)'
        # - '미아동부' 같은 일부 검색어까지 허용하면 오매칭 위험이 커짐
        # -------------------------------------------------
        if apt:
            baseSql += " AND m.apt = @apt"
            query_params.append(bigquery.ScalarQueryParameter("apt", "STRING", apt))

        if aptDtl:
            baseSql += " AND m.aptDtl = @aptDtl"
            query_params.append(bigquery.ScalarQueryParameter("aptDtl", "STRING", aptDtl))

        if key:
            baseSql += " AND m.`key` = @key"
            query_params.append(bigquery.ScalarQueryParameter("key", "STRING", key))

        if keyDtl:
            baseSql += " AND m.keyDtl = @keyDtl"
            query_params.append(bigquery.ScalarQueryParameter("keyDtl", "STRING", keyDtl))

        baseSql += """
            ORDER BY
                m.match_score DESC,
                CASE m.match_type
                    WHEN 'road_address' THEN 1
                    WHEN 'jibun_address' THEN 2
                    WHEN 'geo_coordinate' THEN 3
                    WHEN 'name_fuzzy' THEN 4
                    ELSE 9
                END ASC
            LIMIT 1
        """

        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        rows = [dict(row) for row in client.query(baseSql, job_config=job_config).result()]

        if rows is None or len(rows) < 1:
            return resResponse("succ", 200, "매핑된 기본정보 없음", 0, 0, {
                "mapping": None,
                "detail": None,
                "minScore": minScore,
            })

        row = rows[0]

        def to_jsonable(v):
            if hasattr(v, "isoformat"):
                return v.isoformat()
            return v

        row = {k: to_jsonable(v) for k, v in row.items()}

        mapping = {
            "key": row.get("mapping_key"),
            "keyDtl": row.get("mapping_keyDtl"),
            "apt": row.get("mapping_apt"),
            "aptDtl": row.get("mapping_aptDtl"),
            "sgg": row.get("mapping_sgg"),
            "dong": row.get("mapping_dong"),
            "geo": row.get("mapping_geo"),
            "kaptCode": row.get("mapping_kaptCode"),
            "kaptName": row.get("mapping_kaptName"),
            "doroJuso": row.get("mapping_doroJuso"),
            "match_score": row.get("match_score"),
            "match_type": row.get("match_type"),
            "is_show_detail": row.get("match_score") is not None and float(row.get("match_score")) >= float(minScore),
        }

        detail = {
            "kaptCode": row.get("kaptCode"),
            "kaptName": row.get("kaptName"),
            "kaptAddr": row.get("kaptAddr"),
            "doroJuso": row.get("doroJuso"),
            "region": row.get("region"),
            "bjdCode": row.get("bjdCode"),
            "sggCd": row.get("sggCd"),
            "umdCd": row.get("umdCd"),
            "codeAptNm": row.get("codeAptNm"),
            "codeSaleNm": row.get("codeSaleNm"),
            "codeHeatNm": row.get("codeHeatNm"),
            "codeMgrNm": row.get("codeMgrNm"),
            "codeHallNm": row.get("codeHallNm"),
            "kaptUsedate": row.get("kaptUsedate"),
            "kaptTarea": row.get("kaptTarea"),
            "kaptDongCnt": row.get("kaptDongCnt"),
            "kaptdaCnt": row.get("kaptdaCnt"),
            "hoCnt": row.get("hoCnt"),
            "kaptMarea": row.get("kaptMarea"),
            "kaptMparea60": row.get("kaptMparea60"),
            "kaptMparea85": row.get("kaptMparea85"),
            "kaptMparea135": row.get("kaptMparea135"),
            "kaptMparea136": row.get("kaptMparea136"),
            "privArea": row.get("privArea"),
            "kaptTopFloor": row.get("kaptTopFloor"),
            "ktownFlrNo": row.get("ktownFlrNo"),
            "kaptBaseFloor": row.get("kaptBaseFloor"),
            "kaptdEcntp": row.get("kaptdEcntp"),
            "kaptBcompany": row.get("kaptBcompany"),
            "kaptAcompany": row.get("kaptAcompany"),
            "kaptTel": row.get("kaptTel"),
            "kaptUrl": row.get("kaptUrl"),
            "kaptFax": row.get("kaptFax"),
            "zipcode": row.get("zipcode"),
        }

        return resResponse("succ", 200, "처리 완료", 1, 1, {
            "mapping": mapping,
            "detail": detail,
            "minScore": minScore,
        })

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRentSearchByYearMonth", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRentSearchByYear")
def statRealSearch(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        , minMonthlyRent: str = Query(None, description="최소 월세")
        , maxMonthlyRent: str = Query(None, description="최대 월세")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 통계 거래량 그래프 AP\n
        맞집 좌측 검색조건 거래년도 참조
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"""
                  SELECT SUBSTR(CAST(date AS STRING), 1, 7)                  AS yearMonth,
                         COUNT(*)                                            AS cnt,
                         AVG(CASE WHEN deposit > 0 THEN deposit ELSE NULL END) AS real_deposit,
                         area
                  FROM `{sysOpt['dbHostName']}.TB_RENT` \
                  """

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if minMonthlyRent or maxMonthlyRent:
            condList.append(f"CAST(monthlyRent AS FLOAT64) BETWEEN {minMonthlyRent} AND {maxMonthlyRent}")

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY yearMonth, area"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-chatModelCont", dependencies=[Depends(chkApiKey)])
async def selChatModelCont(
    chatModel: str = Form(..., description='생성형 AI 종류', examples=['gemini-2.5-flash'], enum=['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite']),
    cont: str = Form(..., description='비교 리포트 (종합 성능, 상세 스펙, 종합 분석)', examples=['자전거 비교 리포트 (종합 성능, 상세 스펙, 종합 분석)']),
):
    """
    기능\n
        AI 지역/아파트 리포트 헬퍼\n
    테스트\n
        chatModel: 생성형 AI 종류 (gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite)\n
        cont: 비교 리포트 (종합 성능, 상세 스펙, 종합 분석)\n
    """
    try:
        # model = request.model
        if chatModel is None or len(chatModel) < 1:
            return resResponse("fail", 400, f"생성형 AI 모델 종류 없음, chatModel : {chatModel}")

        # cont = request.cont
        if cont is None or len(cont) < 1:
            return resResponse("fail", 400, f"요청사항 없음, cont : {cont}")

        contTemplate = '''
            %cont%
           '''
        contents = contTemplate.replace('%cont%', cont)
        log.info(f"contents : {contents}")

        response = genaiClient.models.generate_content(
            model=chatModel,
            contents=contents
        )
        result = response.text
        # log.info(f"result : {result}")

        if result is None or len(result) < 1:
            return resResponse("fail", 400, "처리 실패")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sendEmail", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sendEmail")
async def sendEmail(
    sendEmail: str = Form(..., description='송신 이메일', examples=['sangho.lee.1990@gmail.com']),
    sendAppPwd: str = Form(..., description='송신 앱 비밀번호', examples=['hlyc nnfe hbpp mtxx']),
    recvEmail: str = Form(..., description='수신 이메일', examples=['sangho.lee.1990@gmail.com']),
    subject: str = Form(..., description='수신 이메일 제목', examples=['테스트 이메일입니다.']),
    content: str = Form(..., description='수신 이메일 내용', examples=['안녕하세요. 테스트 이메일 내용입니다.']),
    file: Optional[UploadFile] = File(None, description='수신 첨부파일')
    ):
    """
    기능\n
        이메일 발송 API\n
    파라미터\n
        sendEmail: 송신 이메일\n
        sendAppPwd: 송신 앱 비밀번호\n
        recvEmail: 수신 이메일\n
        subject: 수신 이메일 제목\n
        content: 수신 이메일 내용\n
        file: 수신 첨부파일\n
    """
    try:
        if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', sendEmail):
            return resResponse("fail", 400, f"이메일 발송 실패, 송신 이메일 주소를 확인")

        if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', recvEmail):
            return resResponse("fail", 400, f"이메일 발송 실패, 수신 이메일 주소를 확인")

        msg = MIMEMultipart()
        # msg['From'] = sendEmail
        msg['From'] = formataddr(("벌스일사사 고객지원", sendEmail))

        msg['To'] = recvEmail
        msg['Subject'] = subject
        msg.attach(MIMEText(content, 'plain'))

        # 첨부파일 처리
        if file:
            file_content = await file.read()
            attachment = MIMEApplication(file_content)
            attachment.add_header('Content-Disposition', 'attachment', filename=file.filename)
            msg.attach(attachment)

        # SMTP 서버 연결 및 발송
        server = 'smtp.gmail.com'
        if sendEmail.endswith("@naver.com"):
            server = 'smtp.naver.com'

        with smtplib.SMTP(server, 587) as server:
            server.starttls()
            server.login(sendEmail, sendAppPwd)
            result = server.send_message(msg)
            if result:
                return resResponse("fail", 400, f"이메일 발송 실패, 수신 이메일을 확인해주세요.")

        return resResponse("succ", 200, f"이메일 발송 완료")
    except Exception as e:
        log.error(f'Exception : {e}')
        return resResponse("fail", 400, f"이메일 발송 실패, {e}")

@app.post(f"/api/sel-rcmd", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-rcmd")
def selRcmd(request: cfgRcmd = Form(...)):
    """
    기능\n
        CF/유사도 기반 아파트 추천 서비스 API\n
        검색조건 사용자 설정 (gender, age, price, area), 아파트 설정 (apt, cnt)\n
    """
    try:
        gender = request.gender
        age = request.age
        price = request.price
        area = request.area
        apt = request.apt
        cnt = request.cnt

        minGender, maxGender = gender.split('-')
        if gender is None or len(gender) < 1 or minGender is None or maxGender is None:
            return resResponse("fail", 400, f"성별을 확인해주세요 ({gender}).", None)

        minAge, maxAge = age.split('-')
        if age is None or len(age) < 1 or minAge is None or maxAge is None:
            return resResponse("fail", 400, f"나이를 확인해주세요 ({age}).", None)

        minPrice, maxPrice = price.split('-')
        buyLimit = float(maxPrice)
        meanPrice = np.mean([float(minPrice), float(maxPrice)])
        if price is None or len(price) < 1 or minPrice is None or maxPrice is None:
            return resResponse("fail", 400, f"가격을 확인해주세요 ({price}).", None)

        minArea, maxArea = area.split('-')
        targetArea = np.mean([float(minArea), float(maxArea)])
        if area is None or len(area) < 1 or minArea is None or maxArea is None:
            return resResponse("fail", 400, f"면적을 확인해주세요 ({area}).", None)

        # -------------------------------------------------
        # 1) 사용자 후보
        # - 기존 필터는 최대한 유지
        # - 단, 선택은 "가장 싼 사용자"가 아니라
        #   요청 중심값(가격/면적/연령)에 가장 가까운 사용자 1명으로 변경
        # -------------------------------------------------
        recUserDataL1 = recUserData.loc[
            (recUserData['gender'] >= int(minGender)) & (recUserData['gender'] <= int(maxGender))
            & (recUserData['age'] >= float(minAge)) & (recUserData['age'] <= float(maxAge))
            & (recUserData['price_from'] >= float(minPrice) * 0.5) & (recUserData['price_to'] <= float(maxPrice) * 1.5)
            & (recUserData['area_from'] >= float(minArea)) & (recUserData['area_to'] <= float(maxArea))
        ].copy()

        if len(recUserDataL1) < 1:
            return resResponse("fail", 400, "사용자 설정 정보를 확인해주세요.", None)

        targetAge = np.mean([float(minAge), float(maxAge)])
        recUserDataL1['age_mid'] = (recUserDataL1['age']).astype(float)
        recUserDataL1['price_mid'] = (
            recUserDataL1['price_from'].astype(float) + recUserDataL1['price_to'].astype(float)
        ) / 2.0
        recUserDataL1['area_mid'] = (
            recUserDataL1['area_from'].astype(float) + recUserDataL1['area_to'].astype(float)
        ) / 2.0

        recUserDataL1['age_dist'] = abs(recUserDataL1['age_mid'] - targetAge) / max(targetAge, 1.0)
        recUserDataL1['price_dist'] = abs(recUserDataL1['price_mid'] - meanPrice) / max(meanPrice, 0.1)
        recUserDataL1['area_dist'] = abs(recUserDataL1['area_mid'] - targetArea) / max(targetArea, 1.0)

        # 가격/면적 중심, 나이는 보조
        recUserDataL1['dist'] = (
            recUserDataL1['price_dist'] * 0.45
            + recUserDataL1['area_dist'] * 0.45
            + recUserDataL1['age_dist'] * 0.10
        )

        recUserDataL2 = recUserDataL1.nsmallest(1, 'dist').iloc[0]

        # -------------------------------------------------
        # 2) 아파트 후보
        # -------------------------------------------------
        apt_norm = str(apt).replace(' ', '').strip()
        recAptDataTmp = recAptData.copy()
        recAptDataTmp['apt_norm'] = recAptDataTmp['apt'].astype(str).str.replace(r'\s+', '', regex=True)

        recAptCand = recAptDataTmp.loc[
            recAptDataTmp['apt_norm'].str.contains(apt_norm, na=False)
        ].copy()

        if len(recAptCand) < 1:
            return resResponse("fail", 400, "아파트 설정 정보를 확인해주세요.", None)

        if 'price' in recAptCand.columns:
            recAptCand['price'] = pd.to_numeric(recAptCand['price'], errors='coerce')
            recAptCand = recAptCand[recAptCand['price'] <= buyLimit].copy()

        if len(recAptCand) < 1:
            return resResponse("fail", 400, "아파트 설정 정보를 확인해주세요.", None)

        recAptDataL1 = recAptCand.loc[
            (recAptCand['area'] >= float(minArea)) & (recAptCand['area'] <= float(maxArea))
        ].copy()

        if len(recAptDataL1) < 1:
            recAptDataL1 = recAptCand.copy()

        recAptDataL1['area_dist'] = abs(recAptDataL1['area'] - targetArea) / max(targetArea, 1)

        if meanPrice and meanPrice > 0:
            recAptDataL1['price_dist'] = abs(recAptDataL1['price'] - meanPrice) / meanPrice
        else:
            recAptDataL1['price_dist'] = 0.0

        recAptDataL1['dist'] = recAptDataL1['area_dist'] * 0.7 + recAptDataL1['price_dist'] * 0.3
        recAptDataL2 = recAptDataL1.nsmallest(1, 'dist').iloc[0]

        payload = {
            'user_id': recUserDataL2['user_id'] if 'user_id' in recUserDataL2.index else recUserDataL2['idx'],
            'apt_idx': recAptDataL2['apt_idx'] if 'apt_idx' in recAptDataL2.index else recAptDataL2['idx'],
            'rcmd_count': cnt,
        }

        def postprocess_recommendations(df, base_apt_row, limit_cnt):
            """
            품질 우선 후처리
            1) 기준 아파트와 완전히 같은 단지명 + 같은 좌표는 제거
            2) 결과 내 중복되는 같은 단지명 + 같은 좌표는 1건만 유지
            3) 같은 단지의 '다른 면적'은 1건 정도는 허용
            """

            if df is None or len(df) < 1:
                return pd.DataFrame()

            data = df.copy()

            if 'price' in data.columns:
                data['price'] = pd.to_numeric(data['price'], errors='coerce')
                data = data[data['price'] <= buyLimit].copy()

            if len(data) < 1:
                return pd.DataFrame(columns=df.columns)

            # ---------------------------------------
            # 컬럼 호환
            # ---------------------------------------
            if 'apt_norm' not in data.columns and 'apt' in data.columns:
                data['apt_norm'] = data['apt'].astype(str).str.replace(r'\s+', '', regex=True)

            if 'lat' not in data.columns and 'latitude' in data.columns:
                data['lat'] = data['latitude']

            if 'lon' not in data.columns and 'longitude' in data.columns:
                data['lon'] = data['longitude']

            base_apt_norm = str(base_apt_row.get('apt_norm', '')).replace(' ', '').strip()

            try:
                base_lat = round(float(base_apt_row.get('lat', base_apt_row.get('latitude'))), 6)
                base_lon = round(float(base_apt_row.get('lon', base_apt_row.get('longitude'))), 6)
            except:
                base_lat, base_lon = None, None

            base_area = None
            try:
                base_area = float(base_apt_row.get('area'))
            except:
                pass

            # ---------------------------------------
            # 좌표 정규화
            # ---------------------------------------
            def _round_or_none(v):
                try:
                    return round(float(v), 6)
                except:
                    return None

            data['lat_round'] = data['lat'].apply(_round_or_none) if 'lat' in data.columns else None
            data['lon_round'] = data['lon'].apply(_round_or_none) if 'lon' in data.columns else None

            # ---------------------------------------
            # 1) 기준 아파트와 완전히 같은 단지+같은좌표 제거
            #    단, 면적이 다르면 1개 정도는 살릴 수 있게 완전 동일성만 제거
            # ---------------------------------------
            if base_apt_norm and base_lat is not None and base_lon is not None:
                same_base_mask = (
                    (data['apt_norm'] == base_apt_norm)
                    & (data['lat_round'] == base_lat)
                    & (data['lon_round'] == base_lon)
                )

                # 같은 단지/같은 좌표 중에서도
                # 기준 면적과 사실상 동일한 것만 제거
                if 'area' in data.columns and base_area is not None:
                    same_area_mask = (data['area'].astype(float) - base_area).abs() < 0.01
                    data = data.loc[~(same_base_mask & same_area_mask)].copy()
                else:
                    data = data.loc[~same_base_mask].copy()

            if len(data) < 1:
                return data

            # ---------------------------------------
            # 2) 결과 내부 중복 제거
            #    같은 단지명 + 같은 좌표는 대표 1건만 유지
            #    단, 기준 단지와 동일한 경우에는 "다른 면적 1건" 허용
            # ---------------------------------------
            kept_rows = []
            seen_keys = set()
            same_base_kept = False

            for _, row in data.sort_values(by='score', ascending=False).iterrows():
                apt_norm = str(row.get('apt_norm', '')).replace(' ', '').strip()
                lat_r = row.get('lat_round')
                lon_r = row.get('lon_round')
                key = (apt_norm, lat_r, lon_r)

                is_same_base_complex = (
                    base_apt_norm
                    and base_lat is not None
                    and base_lon is not None
                    and apt_norm == base_apt_norm
                    and lat_r == base_lat
                    and lon_r == base_lon
                )

                # 기준 단지/좌표와 같은 후보는 1건만 허용
                if is_same_base_complex:
                    if same_base_kept:
                        continue
                    same_base_kept = True
                    kept_rows.append(row)
                    continue

                # 그 외 동일 단지+동일 좌표는 1건만 유지
                if key in seen_keys:
                    continue

                seen_keys.add(key)
                kept_rows.append(row)

                if len(kept_rows) >= int(limit_cnt):
                    break

            if len(kept_rows) < 1:
                return pd.DataFrame(columns=data.columns)

            result_df = pd.DataFrame(kept_rows).reset_index(drop=True)

            # 보조 컬럼 제거
            drop_cols = [c for c in ['lat_round', 'lon_round'] if c in result_df.columns]
            if drop_cols:
                result_df = result_df.drop(columns=drop_cols)

            return result_df

        with ThreadPoolExecutor(max_workers=2) as executor:
            futureCf = executor.submit(fetchApi, sysOpt['rcmd']['apiCfUrl'], payload, 'cf', recAptData)
            futureSim = executor.submit(fetchApi, sysOpt['rcmd']['apiSimUrl'], payload, 'simil', recAptData)

            cfData = futureCf.result()
            simData = futureSim.result()

        # 후처리 적용
        cfData = postprocess_recommendations(cfData, recAptDataL2, cnt)
        simData = postprocess_recommendations(simData, recAptDataL2, cnt)

        result = {
            'user': recUserDataL2.to_dict(),
            'apt': recAptDataL2.to_dict(),
            'cf': cfData.to_dict(orient='records'),
            'sim': simData.to_dict(orient='records'),
        }

        return resResponse("succ", 200, "처리 완료", len(result), len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-rcmdAptData", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-rcmdAptData")
def selRcmdAptData(
        gu: str = Form(None, description='시군구', examples=['금천구']),
        apt: str = Form(None, description='아파트 도로명주소, 두산(가산로 99)', examples=['두산(가산로 99)']),
        # price: str = Form(None, description='가격 억원 (최소-최대, 3-6)', examples=['3-6']),
        area: str = Form(None, description='면적 m² (최소-최대, 58-100)', examples=['58-100']),
        limit: int = Form(50, description='조회 개수', examples=['50']),
    ):
    """
    기능\n
        아파트 추천 서비스 API를 위한 아파트 설정 목록 API\n
        검색조건 아파트 설정 (gu, apt)\n
    """
    try:
        condition = pd.Series(True, index=recAptData2.index)

        if gu and len(str(gu).strip()) > 0:
            condition &= recAptData2['gu'].astype(str).str.strip().str.contains(gu.strip(), na=False)

        if apt and len(str(apt).strip()) > 0:
            condition &= recAptData2['apt'].astype(str).str.strip().str.contains(apt.strip(), na=False)

        if area and len(str(area).strip()) > 0:
            minArea, maxArea = area.split('-')
            minArea = float(minArea)
            maxArea = float(maxArea)
            condition &= (recAptData2['area'] >= minArea) & (recAptData2['area'] <= maxArea)

        coord_cols = []
        if 'lat' in recAptData2.columns and 'lon' in recAptData2.columns:
            coord_cols = ['lat', 'lon']
        elif 'latitude' in recAptData2.columns and 'longitude' in recAptData2.columns:
            coord_cols = ['latitude', 'longitude']
        else:
            raise HTTPException(status_code=400, detail="좌표 컬럼 없음 (lat/lon 또는 latitude/longitude)")

        recAptDataL1 = (
            recAptData2.loc[condition]
            .dropna(subset=coord_cols)
            .reset_index(drop=False)
            .copy()
        )

        # 응답 호환용
        if 'lat' not in recAptDataL1.columns and 'latitude' in recAptDataL1.columns:
            recAptDataL1['lat'] = recAptDataL1['latitude']
        if 'lon' not in recAptDataL1.columns and 'longitude' in recAptDataL1.columns:
            recAptDataL1['lon'] = recAptDataL1['longitude']

        if len(recAptDataL1) < 1:
            return resResponse("succ", 200, "조회 결과 없음", 0, 0, {"apt": []})

        # 목록 API이므로 정렬/개수 제한
        sort_cols = [c for c in ['gu', 'apt', 'area'] if c in recAptDataL1.columns]
        if sort_cols:
            recAptDataL1 = recAptDataL1.sort_values(sort_cols).reset_index(drop=True)

        recAptDataL1 = recAptDataL1.head(int(limit))

        result = {
            'apt': recAptDataL1.to_dict(orient='records'),
        }

        return resResponse("succ", 200, "처리 완료", len(recAptDataL1), len(recAptDataL1), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# ============================================
# AI 아파트 추천 미리보기
# ============================================
@app.post(f"/api/sel-rcmdPreview", dependencies=[Depends(chkApiKey)])
def selRcmdPreview(
    previewType: str = Form(..., description="미리보기 타입", examples=["2030"]),
    cnt: int = Form(20, description="호환용 파라미터, 일별 반환 개수는 20개 고정", examples=["20"]),
):
    """
    기능\n
        AI 매물 추천 미리보기 데이터 조회 API\n
        - 2030 / 3040 / 5060 대표 추천
        - 월 단위 seed 기반으로 추천 아파트 500개 고정
        - 일 단위 seed 기반으로 월간 추천 목록 중 20개 반환
    """
    try:
        previewType = str(previewType).strip()

        preview_map = {
            '2030': [(20, 29, 0.5), (30, 39, 0.5)],
            '3040': [(30, 39, 0.5), (40, 49, 0.5)],
            '5060': [(50, 59, 0.5), (60, 69, 0.5)],
        }

        if previewType not in preview_map:
            return resResponse("fail", 400, f"previewType을 확인해주세요. ({previewType})", None)

        month_rcmd_cnt = 500
        day_rcmd_cnt = 20

        # -------------------------------------------------
        # 1) 대표 사용자 프로필 계산
        # -------------------------------------------------
        band_defs = preview_map[previewType]
        band_profiles = []

        for min_age, max_age, weight in band_defs:
            band_df = recUserData.loc[
                (recUserData['age'] >= min_age) & (recUserData['age'] <= max_age)
            ].copy()

            if len(band_df) < 1:
                continue

            # 중앙값 기반 대표값
            age_mid = float(band_df['age'].median())
            price_mid = float(((band_df['price_from'] + band_df['price_to']) / 2.0).median())
            area_mid = float(((band_df['area_from'] + band_df['area_to']) / 2.0).median())

            pref_mode = None
            if 'prefer' in band_df.columns and len(band_df['prefer'].dropna()) > 0:
                pref_mode = band_df['prefer'].mode().iloc[0]

            gu_mode = None
            if 'gu_name' in band_df.columns and len(band_df['gu_name'].dropna()) > 0:
                gu_mode = band_df['gu_name'].mode().iloc[0]

            band_profiles.append({
                'min_age': min_age,
                'max_age': max_age,
                'weight': weight,
                'age_mid': age_mid,
                'price_mid': price_mid,
                'area_mid': area_mid,
                'prefer': pref_mode,
                'gu_name': gu_mode,
            })

        if len(band_profiles) < 1:
            return resResponse("fail", 400, "미리보기용 사용자 대표값 계산에 실패했습니다.", None)

        # 혼합 대표값
        sum_w = sum([x['weight'] for x in band_profiles])
        rep_age = sum([x['age_mid'] * x['weight'] for x in band_profiles]) / sum_w
        rep_price = sum([x['price_mid'] * x['weight'] for x in band_profiles]) / sum_w
        rep_area = sum([x['area_mid'] * x['weight'] for x in band_profiles]) / sum_w

        # 대표 선호카테고리: 가중치 합 최대값
        pref_score = {}
        gu_score = {}
        for x in band_profiles:
            if x['prefer']:
                pref_score[x['prefer']] = pref_score.get(x['prefer'], 0) + x['weight']
            if x['gu_name']:
                gu_score[x['gu_name']] = gu_score.get(x['gu_name'], 0) + x['weight']

        rep_prefer = max(pref_score, key=pref_score.get) if len(pref_score) > 0 else None
        rep_gu = max(gu_score, key=gu_score.get) if len(gu_score) > 0 else None

        # -------------------------------------------------
        # 2) 아파트 후보 점수 계산
        # -------------------------------------------------
        apt_df = recAptData.copy()

        # 컬럼 방어
        if 'apt_idx' not in apt_df.columns:
            apt_df = apt_df.reset_index().rename(columns={'index': 'apt_idx'})

        if 'gu' not in apt_df.columns:
            if 'gu_name' in apt_df.columns:
                apt_df['gu'] = apt_df['gu_name']
            else:
                apt_df['gu'] = ''

        for col in ['price', 'area']:
            apt_df[col] = pd.to_numeric(apt_df[col], errors='coerce')

        # 주변시설 점수 컬럼 방어
        for col in ['교통_rel_total', '교육_rel_total', '주거환경_rel_total', '편의시설_rel_total']:
            if col not in apt_df.columns:
                apt_df[col] = 0.0
            apt_df[col] = pd.to_numeric(apt_df[col], errors='coerce').fillna(0.0)

        apt_df = apt_df.dropna(subset=['price', 'area']).copy()

        if len(apt_df) < 1:
            return resResponse("fail", 400, "추천 가능한 아파트 데이터가 없습니다.", None)

        # 거리 점수
        apt_df['price_dist'] = (apt_df['price'] - rep_price).abs() / max(rep_price, 0.1)
        apt_df['area_dist'] = (apt_df['area'] - rep_area).abs() / max(rep_area, 1.0)

        # 선호 카테고리 점수
        pref_col_map = {
            '교통': '교통_rel_total',
            '교육': '교육_rel_total',
            '주거환경': '주거환경_rel_total',
            '편의시설': '편의시설_rel_total',
        }

        if rep_prefer in pref_col_map:
            apt_df['pref_score'] = apt_df[pref_col_map[rep_prefer]]
        else:
            apt_df['pref_score'] = 0.0

        # 지역 약한 가중치
        apt_df['gu_bonus'] = 0.0
        if rep_gu:
            apt_df.loc[apt_df['gu'].astype(str).str.contains(str(rep_gu), na=False), 'gu_bonus'] = 0.08

        # 과도한 고가/저가 치우침 방지용 price gate
        apt_df['price_gate'] = 1.0
        apt_df.loc[apt_df['price'] > rep_price * 1.8, 'price_gate'] = 0.85
        apt_df.loc[apt_df['price'] < rep_price * 0.4, 'price_gate'] = 0.90

        # 최종 점수
        # 거리 작을수록 좋고, 선호 점수/지역 보너스 클수록 좋음
        apt_df['score_preview'] = (
            (1.0 / (1.0 + apt_df['price_dist'])) * 0.35
            + (1.0 / (1.0 + apt_df['area_dist'])) * 0.35
            + apt_df['pref_score'] * 0.22
            + apt_df['gu_bonus'] * 0.08
        ) * apt_df['price_gate']

        # 중복 제거: 같은 단지+좌표는 대표 1건만
        apt_df['apt_norm'] = apt_df['apt'].astype(str).str.replace(r'\s+', '', regex=True)
        if 'lat' in apt_df.columns:
            apt_df['lat_round'] = pd.to_numeric(apt_df['lat'], errors='coerce').round(6)
        else:
            apt_df['lat_round'] = np.nan

        if 'lon' in apt_df.columns:
            apt_df['lon_round'] = pd.to_numeric(apt_df['lon'], errors='coerce').round(6)
        else:
            apt_df['lon_round'] = np.nan

        apt_df = (
            apt_df
            .sort_values(by='score_preview', ascending=False)
            .drop_duplicates(subset=['apt_norm', 'lat_round', 'lon_round'], keep='first')
            .reset_index(drop=True)
        )

        # preview_list = apt_df.head(cnt).copy()

        # if len(preview_list) < 1:
        #     return resResponse("fail", 400, "미리보기 추천 결과가 없습니다.", None)

        # # -------------------------------------------------
        # # 3) 월 단위 대표 아파트(best) 고정 선택
        # # -------------------------------------------------
        # month_key = datetime.now().strftime('%Y%m')
        # seed_val = sum([ord(c) for c in f"{previewType}_{month_key}"])
        # rnd = np.random.RandomState(seed_val)

        # # 상위 20개 중 1개 선택
        # best_pool = preview_list.head(min(20, len(preview_list))).copy()
        # best_idx = rnd.randint(len(best_pool))
        # best_row = best_pool.iloc[best_idx].copy()

        # -------------------------------------------------
        # 3) 월 단위 추천 리스트 500개 생성 후 일 단위 20개 추출
        # -------------------------------------------------
        today = datetime.now()
        month_key = today.strftime('%Y%m')
        day_key = today.strftime('%Y%m%d')
        seed_val = sum([ord(c) for c in f"{previewType}_{month_key}"])
        day_seed_val = sum([ord(c) for c in f"{previewType}_{day_key}"])

        # 상위 후보군 안에서만 랜덤 추출
        # - 기본적으로 상위 2,500개 후보 중 500개를 월별 고정 랜덤 추출
        # - 후보군을 너무 크게 잡으면 추천 품질이 떨어질 수 있으므로 cnt * 5 정도가 적당
        pool_size = min(len(apt_df), max(month_rcmd_cnt * 5, month_rcmd_cnt))

        candidate_pool = apt_df.head(pool_size).copy()

        if len(candidate_pool) < 1:
            return resResponse("fail", 400, "미리보기 추천 후보가 없습니다.", None)

        # 후보군 수가 500개보다 많으면 월 단위 seed로 월간 추천 500개 추출
        if len(candidate_pool) > month_rcmd_cnt:
            month_list = candidate_pool.sample(
                n=month_rcmd_cnt,
                random_state=seed_val
            ).copy()
        else:
            month_list = candidate_pool.copy()

        month_list = month_list.sample(frac=1, random_state=seed_val + 1).reset_index(drop=True)

        if len(month_list) < 1:
            return resResponse("fail", 400, "미리보기 추천 결과가 없습니다.", None)

        # 월간 추천 500개 중 매일 20개 추출
        # - 일별 seed를 사용하므로 같은 날짜에는 같은 결과가 반환됨
        # - 날짜가 달라지면 월간 목록 안에서 다시 추출되어 중복/반복 노출 가능
        if len(month_list) > day_rcmd_cnt:
            preview_list = month_list.sample(
                n=day_rcmd_cnt,
                random_state=day_seed_val
            ).copy()
        else:
            preview_list = month_list.copy()

        preview_list = preview_list.sample(frac=1, random_state=day_seed_val + 1).reset_index(drop=True)

        # best는 선택 기능이 필요 없다면 일별 리스트 첫 번째 항목으로 처리
        # 프론트에서 best를 안 쓰면 그냥 무시해도 됨
        best_row = preview_list.iloc[0].copy()

        # 응답용 컬럼 정리
        keep_cols = [
            'apt_idx', 'gu', 'apt', 'area', 'price',
            'lat', 'lon',
            '교통', '교육', '주거환경', '편의시설',
            '교통_rel_total', '교육_rel_total', '주거환경_rel_total', '편의시설_rel_total',
            'score_preview'
        ]
        keep_cols = [c for c in keep_cols if c in preview_list.columns]

        preview_list = preview_list[keep_cols].copy()
        best_data = best_row[[c for c in keep_cols if c in best_row.index]].to_dict()

        result = {
            'previewType': previewType,
            'monthKey': month_key,
            'dayKey': day_key,
            'monthRcmdCnt': len(month_list),
            'dayRcmdCnt': len(preview_list),
            'profile': {
                'age_mid': round(rep_age, 2),
                'price_mid': round(rep_price, 2),
                'area_mid': round(rep_area, 2),
                'prefer': rep_prefer,
                'gu_name': rep_gu,
            },
            'best': best_data,
            'list': preview_list.to_dict(orient='records'),
        }

        return resResponse("succ", 200, "처리 완료", len(month_list), len(preview_list), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealMeanBySggDong", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealMeanBySggDong")
def selStatRealSggApt(
        sgg: str = Query(None, description="시군구")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트 시군구 법정동 별로 평균가 API 구성\n
        검색조건 시군구, 지역, 거래년도, 평수\n
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,mean_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"""
                    SELECT sgg, dong, COUNT(*) AS cnt, 
                    AVG(CASE WHEN amount > 0 THEN amount ELSE NULL END) AS mean_amount, 
                     MAX(
                        CASE
                            WHEN geo IS NOT NULL AND geo != 'nan, nan' AND TRIM(geo) != ''
                            THEN geo
                            ELSE NULL
                        END
                    ) AS geo 
                    FROM `{sysOpt['dbHostName']}.TB_REAL`
                """

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY sgg, dong HAVING geo IS NOT NULL"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")

        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealSearchBySgg", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealSearchBySgg")
def statRealSearch(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="면적 m² 범위 또는 평대 (예: 51-83, 20평대)")
        , aptAge: str = Query(None, description="연식구간 (1,1-3,1-5,1-10,10+)", examples=['3-5'])
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트 지역 시도와 시군구로 구분한 API 구성\n
        맞집 좌측 검색조건 지역 참조
        앱 홈 화면 TOP10 지역데이터에 사용됨
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        면적 m² 범위 또는 평대 (면적 m², 평수, ...): 51-83, 20평대\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """
    try:
        selCol = """
                    sgg, 
                    COUNT(*) AS cnt, 
                    AVG(CASE WHEN amount > 0 THEN amount ELSE NULL END) AS mean_real_amount
                """

        # 하나라도 없으면 기본값 적용
        if not srtDate or not endDate:
            end = datetime.now()
            start = end - relativedelta(months=6)

            srtDate = start.strftime('%Y-%m-01')
            endDate = end.strftime('%Y-%m-%d')

        srt_dt = datetime.strptime(srtDate, "%Y-%m-%d")
        end_dt = datetime.strptime(endDate, "%Y-%m-%d")
        mid_dt = srt_dt + ((end_dt - srt_dt) / 2)
        midDate = mid_dt.strftime("%Y-%m-%d")

        selCol += f""",
                SAFE_DIVIDE(
                    AVG(CASE 
                        WHEN amount > 0 
                        AND date > DATE('{midDate}') 
                        AND date <= DATE('{endDate}') 
                        THEN amount ELSE NULL 
                    END)
                    -
                    AVG(CASE 
                        WHEN amount > 0 
                        AND date >= DATE('{srtDate}') 
                        AND date <= DATE('{midDate}') 
                        THEN amount ELSE NULL 
                    END),
                    AVG(CASE 
                        WHEN amount > 0 
                        AND date >= DATE('{srtDate}') 
                        AND date <= DATE('{midDate}') 
                        THEN amount ELSE NULL 
                    END)
                ) * 100 AS real_rate
            """

        # 기본 SQL
        baseSql = f"SELECT {selCol} FROM `{sysOpt['dbHostName']}.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        add_capacity_range_condition(condList, area)

        add_apt_age_condition(condList, aptAge)

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        # if srtYear and endYear:
        #     condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        # 조건 적용
        condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY sgg"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealSearchByYear", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealSearchByYear")
def statRealSearch(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트 건축년도 표기 API 구성\n
        맞집 좌측 검색조건 거래년도 참조
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT year, COUNT(*) AS cnt, AVG(CASE WHEN amount > 0 THEN amount ELSE NULL END) AS real_amount FROM `{sysOpt['dbHostName']}.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY year"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealSearchByYearMonth", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealSearchByYear")
def statRealSearch(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트 건축년도 표기 API 구성\n
        맞집 좌측 검색조건 거래년도 참조
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"""
                  SELECT SUBSTR(CAST(date AS STRING), 1, 7)                  AS yearMonth,
                         COUNT(*)                                            AS cnt,
                         AVG(CASE WHEN amount > 0 THEN amount ELSE NULL END) AS real_amount,
                         area
                  FROM `{sysOpt['dbHostName']}.TB_REAL` \
                  """

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY yearMonth, area"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealSearchByArea", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealSearchByArea")
def statRealSearch(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트 평수 리스트 API 구성\n
        맞집 좌측 검색조건 평수 참조
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT area, COUNT(*) AS cnt FROM `{sysOpt['dbHostName']}.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY area"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealSearchByApt", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealSearchByApt")
def statRealSearch(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트명 리스트 API 구성\n
        맞집 좌측 검색조건 아파트명 참조
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT apt, COUNT(*) AS cnt FROM `{sysOpt['dbHostName']}.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")


        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY apt"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealMaxBySgg", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealMaxBySgg")
def selStatRealMaxBySgg(
        sgg: str = Query(None, description="시군구")
        , area: str = Query(None, description="평수")
        , aptAge: str = Query(None, description="연식구간 (1,1-3,3-5,5-10,10+)", examples=['3-5'])
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 아파트 시군구 별로 최대값 API 구성\n
        검색조건 시군구, 지역, 거래년도, 평수\n
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        평수 (평수, 평수, ...): 5평,6평\n
        연식 구간 (1,3,5,10,10+)\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT sgg, COUNT(*) AS cnt, MAX(amount) AS max_amount FROM `{sysOpt['dbHostName']}.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        add_apt_age_condition(condList, aptAge)

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql

        # 그룹핑
        grpList = []
        grpSql = " GROUP BY sgg"
        baseSql += grpSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRealMaxBySggApt", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statRealMaxBySggApt")
def selStatRealMaxBySggApt(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="면적 m² 범위 또는 평대 (예: 51-83, 20평대)")
        , aptAge: str = Query(None, description="연식구간 (0-1,0-3,0-5,0-10,10+)", examples=['0-5'])
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
        # TB_REAL 아파트 시군구 아파트명 별로 최대값 API 구성\n
    """
    기능\n
        TB_REAL 아파트 시군구, 아파트명 별 최근 6개월 기준 거래가 상승률, 평균 실거래가, 전체 거래량 조회 API 구성\n
        검색조건 : 시군구, 아파트명, 지역, 면적, 거래일자\n
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        면적 (m² 범위/평대): 51-83,84-100 또는 20평대,30평대\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): cnt|desc,max_amount|desc\n
    """

    try:
        # 날짜 기본값: 최근 6개월
        if not srtDate or not endDate:
            end = datetime.now()
            start = end - relativedelta(months=6)

            srtDate = start.strftime('%Y-%m-01')
            endDate = end.strftime('%Y-%m-%d')

        srt_dt = datetime.strptime(srtDate, "%Y-%m-%d")
        prev_srt_dt = srt_dt - relativedelta(months=6)
        prevSrtDate = prev_srt_dt.strftime("%Y-%m-%d")

        # 기본 SQL
        baseSql = f"""
        WITH BASE AS (
            SELECT
                sgg,
                apt,
                date,
                amount
            FROM `{sysOpt['dbHostName']}.TB_REAL`
        """

        # 동적 SQL 파라미터
        condList = []

        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"`key` LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        add_capacity_range_condition(condList, area)

        add_apt_age_condition(condList, aptAge)

        condList.append(get_valid_geo_condition('geo'))

        # 상승률 계산을 위해 최근 6개월 + 이전 6개월까지는 BASE에 포함
        condList.append(f"date BETWEEN DATE('{prevSrtDate}') AND DATE('{endDate}')")

        if condList:
            baseSql += " WHERE " + " AND ".join(condList)

        baseSql += f"""
        ),
        PRICE_STAT AS (
            SELECT
                sgg,
                apt,
                AVG(CASE
                    WHEN date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')
                    AND amount > 0
                    THEN amount ELSE NULL
                END) AS recent_6m_avg,
                AVG(CASE
                    WHEN date >= DATE('{prevSrtDate}')
                    AND date < DATE('{srtDate}')
                    AND amount > 0
                    THEN amount ELSE NULL
                END) AS prev_6m_avg
            FROM BASE
            GROUP BY sgg, apt
        ),
        TOTAL_CNT AS (
            SELECT
                sgg,
                apt,
                COUNT(*) AS cnt
            FROM BASE
            WHERE date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')
            GROUP BY sgg, apt
        )
        SELECT
            A.sgg,
            A.apt,
            IFNULL(B.cnt, 0) AS cnt,
            ROUND(
                SAFE_DIVIDE(
                    A.recent_6m_avg - A.prev_6m_avg,
                    A.prev_6m_avg
                ) * 100,
                2
            ) AS rate,
            ROUND(A.recent_6m_avg, 0) AS amount
        FROM PRICE_STAT A
        LEFT JOIN TOTAL_CNT B
            ON A.sgg = B.sgg
        AND A.apt = B.apt
        WHERE A.recent_6m_avg IS NOT NULL
        """

        # 정렬 'rate|desc,amount|desc'
        sortList = []
        if sort:
            allowedSortCols = ['sgg', 'apt', 'cnt', 'rate', 'amount']
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2:
                    continue

                sortCol = sortPart[0]
                if sortCol not in allowedSortCols:
                    continue

                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")

        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql

        cntSql = baseSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statReal", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-statReal")
def selStatReal(
        sgg: str = Query(None, description="시군구")
        , dong: str = Query(None, description="법정동")
        , area: str = Query(None, description="평수")
        , yyyymm: str = Query(None, description="연월")
    ):
    """
    기능\n
        TB_REAL 통계 목록 조회\n
    테스트\n
        시군구: 서울특별시 금천구\n
        법정동: 독산동\n
        평수: 5평\n
        연월: 202201\n

        또는 \n

        시군구: \n
        법정동: \n
        평수: \n
        연월: 202201\n
    """

    try:
        # 기본 SQL
        baseSql = """
            WITH MONTHLYAVGPRICES AS (
                SELECT
                    APT,
                    SGG,
                    DONG,
                    AREA,
                    DATE_TRUNC(DATE, MONTH) AS MONTH,
                    AVG(AMOUNT) AS AVG_AMOUNT
                FROM
                    `DMS01.TB_REAL`
                WHERE AMOUNT > 0
                GROUP BY 1, 2, 3, 4, 5
            ),

            LAGGEDPRICES AS (
                SELECT
                    APT,
                    SGG,
                    DONG,
                    AREA,
                    MONTH,
                    AVG_AMOUNT,
                    LAG(AVG_AMOUNT, 1, 0) OVER (PARTITION BY APT, SGG, DONG, AREA ORDER BY MONTH) AS PREV_AVG_AMOUNT,
                    SAFE_DIVIDE(AVG_AMOUNT - LAG(AVG_AMOUNT, 1, 0) OVER (PARTITION BY APT, SGG, DONG, AREA ORDER BY MONTH), LAG(AVG_AMOUNT, 1, 0) OVER (PARTITION BY APT, SGG, DONG, AREA ORDER BY MONTH)) * 100 AS RATE
                FROM MONTHLYAVGPRICES
                WHERE AVG_AMOUNT > 0
            )

            SELECT * FROM (
                (SELECT
                    '최고하락' AS GRP,
                    SGG,
                    DONG,
                    DATE_ADD(MONTH, INTERVAL -1 MONTH) AS PREV_MONTH,
                    MONTH AS CURR_MONTH,
                    APT,
                    AREA,
                    PREV_AVG_AMOUNT,
                    AVG_AMOUNT AS CURR_AVG_AMOUNT,
                    RATE
                FROM LAGGEDPRICES
                WHERE
                    (SGG LIKE '%' || @sgg || '%' OR @sgg IS NULL)
                    AND (DONG LIKE '%' || @dong || '%' OR @dong IS NULL)
                    AND (AREA LIKE '%' || @area || '%' OR @area IS NULL)
                    AND (MONTH = DATE_TRUNC(PARSE_DATE('%Y%m', @yyyymm), MONTH) OR @yyyymm IS NULL)
                    AND PREV_AVG_AMOUNT > 0
                ORDER BY RATE ASC
                LIMIT 30)

                UNION ALL

                (SELECT
                    '최고상승' AS GRP,
                    SGG,
                    DONG,
                    DATE_ADD(MONTH, INTERVAL -1 MONTH) AS PREV_MONTH,
                    MONTH AS CURR_MONTH,
                    APT,
                    AREA,
                    PREV_AVG_AMOUNT,
                    AVG_AMOUNT AS CURR_AVG_AMOUNT,
                    RATE
                FROM LAGGEDPRICES
                WHERE
                    (SGG LIKE '%' || @sgg || '%' OR @sgg IS NULL)
                    AND (DONG LIKE '%' || @dong || '%' OR @dong IS NULL)
                    AND (AREA LIKE '%' || @area || '%' OR @area IS NULL)
                    AND (MONTH = DATE_TRUNC(PARSE_DATE('%Y%m', @yyyymm), MONTH) OR @yyyymm IS NULL)
                    AND PREV_AVG_AMOUNT > 0
                ORDER BY RATE DESC
                LIMIT 30)
            );
        """

        # 동적 파라미터
        queryParam = [
            bigquery.ScalarQueryParameter("sgg", "STRING", sgg if sgg else None)
            , bigquery.ScalarQueryParameter("dong", "STRING", dong if dong else None)
            , bigquery.ScalarQueryParameter("area", "STRING", area if area else None)
            , bigquery.ScalarQueryParameter("yyyymm", "STRING", yyyymm if yyyymm else None)
        ]
        log.info(f"[CHECK] queryParam : {queryParam}")

        # 쿼리 실행
        queryJobCfg = bigquery.QueryJobConfig(query_parameters=queryParam)
        queryJob = client.query(baseSql, job_config=queryJobCfg)
        results = queryJob.result()

        fileList = [dict(row) for row in results]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-real", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-real")
def selReal(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        # , year: int = Query(None, description="연도")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_REAL 목록 조회\n
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): year|asc,area|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT * FROM `{sysOpt['dbHostName']}.TB_REAL`"
        cntSql = f"SELECT COUNT(*) AS CNT FROM `{sysOpt['dbHostName']}.TB_REAL`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            # condList.append(f"sgg LIKE '%{sgg}%'")
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if area:
            # condList.append(f"area LIKE '%{area}%'")
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        # if year:
        #     condList.append(f"year = {year}")
        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")
        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql
            cntSql += condSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cnt = next(cntRes)['CNT']

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-prd", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-prd")
def selPrd(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        # , year: int = Query(None, description="연도")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_PRD 목록 조회\n
    테스트\n
        시군구 (시군구, 시군구, ...): 서울특별시 강남구,서울특별시 금천구\n
        도로명주소: 두산(가산로)\n
        도로명주소 상세: 서울특별시 금천구 가산로 99.0 두산\n
        지번주소: 두산(769)\n
        지번주소 상세: 서울특별시 금천구 가산동 769 두산\n
        평수 (평수, 평수, ...): 5평,6평\n
        시작 연도: \n
        종료 연도: \n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): year|asc,area|desc\n
    """

    try:
        # 기본 SQL
        baseSql = f"SELECT * FROM `{sysOpt['dbHostName']}.TB_PRD`"
        cntSql = f"SELECT COUNT(*) AS CNT FROM `{sysOpt['dbHostName']}.TB_PRD`"

        # 동적 SQL 파라미터
        condList = []
        if sgg:
            # condList.append(f"sgg LIKE '%{sgg}%'")
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if area:
            # condList.append(f"area LIKE '%{area}%'")
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        # if year:
        #     condList.append(f"year = {year}")
        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")
        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql
            cntSql += condSql

        # 정렬 'year|desc,price|desc'
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = f" ORDER BY " + ", ".join(sortList)
            baseSql += sortSql

        # 페이징
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # log.info(f"[CHECK] baseSql : {baseSql}")
        # log.info(f"[CHECK] cntSql : {cntSql}")

        # 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cnt = next(cntRes)['CNT']

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))


@app.post(f"/api/sel-infra", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-infra")
def selInfra(
        lon: float = Query(None, description="특정 경도")
        , lat: float = Query(None, description="특정 위도")
        , distKm: float = Query(None, description="특정 경위도를 기준으로 거리 km")
        , limit: int = Query(100, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능\n
        TB_INFRA 목록 조회\n
        즉 특정 경도/위도 (아파트 위치 또는 관심지역 위치 등)를 기준으로 반경 km 주변 인프라 (편의시설, 교육, 주거환경, 교통) 조회
    테스트\n
        특정 경도: 127.10\n
        특정 위도: 37.46\n
        특정 경위도를 기준으로 거리 km: 1 km = 1000 m\n
        1쪽당 개수: 100\n
        현재 쪽: 1\n
        정렬 (컬럼|차순, 컬럼|차순, ...): distKm|desc\n
    """

    try:
        # -------------------------------------------------
        # 1) 허용 정렬 컬럼 제한
        # -------------------------------------------------
        allowedSortCols = [
            'category', 'query', 'place_name', 'distance',
            'p_x', 'p_y', 'distKm'
        ]

        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2:
                    continue

                sortCol = sortPart[0].strip()
                sortOrd = sortPart[1].strip().upper()
                if sortCol not in allowedSortCols:
                    continue
                if sortOrd not in ["ASC", "DESC"]:
                    sortOrd = "ASC"

                sortList.append(f"{sortCol} {sortOrd}")

        # 기본값: 가까운 순
        if len(sortList) < 1:
            sortList = ["distKm ASC"]

        orderSql = " ORDER BY " + ", ".join(sortList)

        # -------------------------------------------------
        # 2) 중복 제거 포함 기본 SQL
        #    - 같은 시설명 + 같은 좌표를 동일 시설로 간주
        #    - 대표값은 ANY_VALUE 사용
        #    - 거리값은 MIN(distKm) 사용
        # -------------------------------------------------
        baseSql = f"""
            SELECT
                ANY_VALUE(category) AS category,
                ANY_VALUE(query) AS query,
                ANY_VALUE(c_g_code) AS c_g_code,
                ANY_VALUE(c_g_name) AS c_g_name,
                place_name,
                ANY_VALUE(distance) AS distance,
                p_x,
                p_y,
                ANY_VALUE(p_addr) AS p_addr,
                ANY_VALUE(p_road_addr) AS p_road_addr,
                ANY_VALUE(p_phone) AS p_phone,
                ANY_VALUE(p_url) AS p_url,
                MIN(
                    ST_DISTANCE(
                        ST_GEOGPOINT(p_x, p_y),
                        ST_GEOGPOINT({lon}, {lat})
                    ) / 1000
                ) AS distKm
            FROM `{sysOpt['dbHostName']}.TB_INFRA`
            WHERE ST_DISTANCE(
                    ST_GEOGPOINT(p_x, p_y),
                    ST_GEOGPOINT({lon}, {lat})
                  ) / 1000 <= {distKm}
            GROUP BY place_name, p_x, p_y
        """

        # -------------------------------------------------
        # 3) 전체 건수 SQL
        #    - pageSql 붙이지 않음
        # -------------------------------------------------
        cntSql = f"""
            SELECT COUNT(*) AS cnt
            FROM (
                {baseSql}
            )
        """

        # -------------------------------------------------
        # 4) 정렬 + 페이징
        # -------------------------------------------------
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql = baseSql + orderSql + pageSql

        log.info(f"[CHECK] baseSql : {baseSql}")
        log.info(f"[CHECK] cntSql : {cntSql}")

        # -------------------------------------------------
        # 5) 쿼리 실행
        # -------------------------------------------------
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]

        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, "검색 결과가 없습니다.", None)

        # -------------------------------------------------
        # 6) 카테고리별 분리
        # -------------------------------------------------
        data = pd.DataFrame(fileList)
        fileDict = {}
        cntDict = {}

        for cate in ['편의시설', '교육', '주거환경', '교통']:
            dataL1 = data.loc[data['category'] == cate].copy()

            # 혹시라도 category 대표값 흔들리는 경우 방어
            if len(dataL1) < 1:
                fileDict[cate] = []
                cntDict[cate] = 0
                continue

            dataL1 = dataL1.sort_values(by='distKm', ascending=True).reset_index(drop=True)
            fileDict[cate] = dataL1.to_dict(orient='records')
            cntDict[cate] = len(fileDict[cate])

        # -------------------------------------------------
        # 7) 전체 건수 조회
        # -------------------------------------------------
        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = cntList[0]['cnt'] if cntList else 0

        return resResponse("succ", 200, "처리 완료", cnt, cntDict, fileDict)

        # # 기본 SQL
        # baseSql = f"""
        #     SELECT
        #         *,
        #         ST_DISTANCE(ST_GEOGPOINT(p_x, p_y), ST_GEOGPOINT({lon}, {lat})) / 1000 AS distKm
        #     FROM
        #         `{sysOpt['dbHostName']}.TB_INFRA`
        # """

        # # 동적 SQL 파라미터
        # condList = []
        # # if sgg:
        # #     sggList = [s.strip() for s in sgg.split(',')]
        # #     sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
        # #     condList.append(f"({' OR '.join(sggCond)})")

        # if lon and lat and distKm:
        #     condList.append(f"ST_DISTANCE(ST_GEOGPOINT(p_x, p_y), ST_GEOGPOINT({lon}, {lat})) / 1000 <= {distKm}")

        # if condList:
        #     condSql = " WHERE " + " AND ".join(condList)
        #     baseSql += condSql

        # # 그룹핑
        # # grpList = []
        # # grpSql = " GROUP BY sgg"
        # # baseSql += grpSql

        # # 정렬 'year|desc,price|desc'
        # sortList = []
        # if sort:
        #     for sortInfo in sort.split(','):
        #         sortPart = sortInfo.split('|')
        #         if sortPart is None or len(sortPart) != 2: continue
        #         sortCol = sortPart[0]
        #         sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
        #         sortList.append(f"{sortCol} {sortOrd}")

        # if sortList:
        #     sortSql = " ORDER BY " + ", ".join(sortList)
        #     baseSql += sortSql

        # cntSql = f"SELECT COUNT(*) AS cnt FROM ({baseSql})"

        # # 페이징
        # pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        # baseSql += pageSql
        # cntSql += pageSql

        # # log.info(f"[CHECK] baseSql : {baseSql}")
        # # log.info(f"[CHECK] cntSql : {cntSql}")

        # # 쿼리 실행
        # baseQuery = client.query(baseSql)
        # baseRes = baseQuery.result()
        # fileList = [dict(row) for row in baseRes]
        # if fileList is None or len(fileList) < 1:
        #     return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        # data = pd.DataFrame(fileList)
        # fileDict = {}
        # cntDict = {}
        # for cate in ['편의시설', '교육', '주거환경', '교통']:
        #     dataL1 = data.loc[data['category'] == cate]
        #     fileDict[cate] = dataL1.to_dict(orient='records')
        #     cntDict[cate] = len(fileDict[cate])

        # cntQuery = client.query(cntSql)
        # cntRes = cntQuery.result()
        # cntList = [dict(row) for row in cntRes]
        # cnt = cntList[0]['cnt']

        # return resResponse("succ", 200, "처리 완료", cnt, cntDict, fileDict)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-keyword", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-keyword")
def selKeyword(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_KEYWORD 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `{sysOpt['dbHostName']}.TB_KEYWORD`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-yearPopTrend", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-yearPopTrend")
def selYearPopTrend(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_YEAR-POP-TREND 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `{sysOpt['dbHostName']}.TB_YEAR-POP-TREND`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-upComSupply", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-upComSupply")
def selUpComSupply(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_UP-COM-SUPPLY 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `{sysOpt['dbHostName']}.TB_UP-COM-SUPPLY`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-monthPopTrend", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-monthPopTrend")
def selMonthPopTrend(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_MONTH-POP-TREND 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `{sysOpt['dbHostName']}.TB_MONTH-POP-TREND`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-monthSggPopTrend", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-monthSggPopTrend")
def selMonthPopTrend(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
         TB_MONTH-SGG-POP-TREND 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `{sysOpt['dbHostName']}.TB_MONTH-SGG-POP-TREND`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-unSoldTrend", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-unSoldTrend")
def selUnSoldTrend(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_UN-SOLD-TREND 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        # 기본 SQL
        sql = f"SELECT * FROM `{sysOpt['dbHostName']}.TB_UN-SOLD-TREND`"

        # 동적 SQL 파라미터
        condList = []
        if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if condList: sql += " WHERE " + " AND ".join(condList)

        sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-largeComRank", dependencies=[Depends(chkApiKey)])
# @app.post(f"/api/sel-largeComRank")
def selLargeComRank(
        sgg: str = Query(None, description="시군구")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
    ):
    """
    기능\n
        TB_LARGE-COM-RANK 목록 조회\n
    테스트\n
        시군구: 서울특별시 강남구\n
        1쪽당 개수: 10\n
        현재 쪽: 1\n
    """

    try:
        def normalize_large_com_sgg(val):
            if val is None:
                return val

            val = str(val).strip()

            sgg_map = {
                "경기도 고양덕양구": "경기도 고양시",
                "경기도 고양일산동구": "경기도 고양시",
                "경기도 고양일산서구": "경기도 고양시",

                "경기도 성남분당구": "경기도 성남시",
                "경기도 성남수정구": "경기도 성남시",
                "경기도 성남중원구": "경기도 성남시",

                "경기도 수원권선구": "경기도 수원시",
                "경기도 수원영통구": "경기도 수원시",
                "경기도 수원장안구": "경기도 수원시",
                "경기도 수원팔달구": "경기도 수원시",

                "경기도 안산단원구": "경기도 안산시",
                "경기도 안산상록구": "경기도 안산시",

                "경기도 안양동안구": "경기도 안양시",
                "경기도 안양만안구": "경기도 안양시",

                "경기도 용인기흥구": "경기도 용인시",
                "경기도 용인수지구": "경기도 용인시",
                "경기도 용인처인구": "경기도 용인시",

                "경기도 부천시소사구": "경기도 부천시",
                "경기도 부천시오정구": "경기도 부천시",
                "경기도 부천시원미구": "경기도 부천시",
            }

            return sgg_map.get(val, val)   

        def get_large_com_sgg_search_list(val):
            if not val:
                return []

            val = str(val).strip()

            city_to_raw_map = {
                "경기도 고양시": [
                    "경기도 고양덕양구",
                    "경기도 고양일산동구",
                    "경기도 고양일산서구",
                ],
                "경기도 성남시": [
                    "경기도 성남분당구",
                    "경기도 성남수정구",
                    "경기도 성남중원구",
                ],
                "경기도 수원시": [
                    "경기도 수원권선구",
                    "경기도 수원영통구",
                    "경기도 수원장안구",
                    "경기도 수원팔달구",
                ],
                "경기도 안산시": [
                    "경기도 안산단원구",
                    "경기도 안산상록구",
                ],
                "경기도 안양시": [
                    "경기도 안양동안구",
                    "경기도 안양만안구",
                ],
                "경기도 용인시": [
                    "경기도 용인기흥구",
                    "경기도 용인수지구",
                    "경기도 용인처인구",
                ],
                "경기도 부천시": [
                    "경기도 부천시소사구",
                    "경기도 부천시오정구",
                    "경기도 부천시원미구",
                ],
            }

            return city_to_raw_map.get(val, [val])        

        # 기본 SQL
        sql = f"SELECT * FROM `{sysOpt['dbHostName']}.TB_LARGE-COM-RANK`"

        # 동적 SQL 파라미터
        condList = []
        # if sgg: condList.append(f"sgg LIKE '%{sgg}%'")
        if sgg:
            sgg_search_list = get_large_com_sgg_search_list(sgg)
            sggCond = [f"sgg LIKE '%{x}%'" for x in sgg_search_list]
            condList.append(f"({' OR '.join(sggCond)})")  
              
        if condList: sql += " WHERE " + " AND ".join(condList)

        # sql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        log.info(f"[CHECK] sql : {sql}")

        # 쿼리 실행
        baseQuery = client.query(sql)
        baseRes = baseQuery.result()

        fileList = [dict(row) for row in baseRes]
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, "검색 결과가 없습니다.", 0, 0, None)

        df = pd.DataFrame(fileList)

        if "sgg" in df.columns:
            df["sgg"] = df["sgg"].apply(normalize_large_com_sgg)

        if "numHouse" in df.columns:
            df["numHouse"] = pd.to_numeric(df["numHouse"], errors="coerce").fillna(0).astype(int)

        # 아파트 단지별 대단지 순위 유지
        # sgg로 합산하지 않음
        sort_cols = []
        if "numHouse" in df.columns:
            sort_cols.append("numHouse")

        if sort_cols:
            df = (
                df.sort_values(by="numHouse", ascending=False)
                .reset_index(drop=True)
            )
        else:
            df = df.reset_index(drop=True)

        allCnt = len(df)

        startIdx = (page - 1) * limit
        endIdx = startIdx + limit
        df = df.iloc[startIdx:endIdx].copy()

        result = df.to_dict(orient="records")

        return resResponse("succ", 200, "처리 완료", allCnt, len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# 전세가 영역
@app.post(f"/api/sel-statRentMeanBySggDong", dependencies=[Depends(chkApiKey)])
def selStatRentMeanBySggDong(
        sgg: str = Query(None, description="시군구")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능: TB_RENT(전세) 아파트 시군구 별로 평균 전세가 산출
    """
    try:
        # 1. 기본 SQL 설정 
        baseSql = f"SELECT sgg, dong, COUNT(*) AS cnt, AVG(deposit * 10000) AS mean_deposit FROM `{sysOpt['dbHostName']}.TB_RENT` WHERE monthlyRent = 0"

        # 2. 동적 SQL 파라미터
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")
        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            condSql = " AND " + " AND ".join(condList)
            baseSql += condSql

        # 3. 그룹핑 설정
        grpSql = " GROUP BY sgg, dong"
        baseSql += grpSql

        # 4. 정렬 설정
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql
        
        cntSql = baseSql  # 전체 건수 파악용 쿼리

        # 5. 페이징 설정
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        # 6. 쿼리 실행
        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cntList = [dict(row) for row in cntRes]
        cnt = len(cntList)

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRentSearchBySgg", dependencies=[Depends(chkApiKey)])
def statRentSearchBySgg(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """기능: 시군구별 전세 데이터 건수 요약"""
    try:
        baseSql = f"""
            SELECT sgg, COUNT(*) AS cnt 
            FROM `{sysOpt['dbHostName']}.TB_RENT`
            WHERE monthlyRent = 0 AND deposit > 0 AND NOT REGEXP_CONTAINS(apt, '임대|행복주택')
        """
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")
        if apt: condList.append(f"apt LIKE '%{apt}%'")
        if area: condList.append(f"area LIKE '%{area}%'")
        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")
        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList: baseSql += " AND " + " AND ".join(condList)
        baseSql += " GROUP BY sgg"
        
        if sort:
            sortParts = [f"{s.split('|')[0]} {s.split('|')[1].upper()}" for s in sort.split(',') if len(s.split('|')) == 2]
            if sortParts: baseSql += " ORDER BY " + ", ".join(sortParts)
        
        cntSql = f"SELECT COUNT(*) AS cnt FROM ({baseSql})"
        baseSql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"

        baseRes = [dict(row) for row in client.query(baseSql).result()]
        cnt = next(client.query(cntSql).result())['cnt'] if baseRes else 0
        return resResponse("succ", 200, "처리 완료", cnt, len(baseRes), baseRes)
    except Exception as e:
        log.error(f'Exception : {e}'); raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRentSearchByYear", dependencies=[Depends(chkApiKey)])
def statRentSearchByYear(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능: TB_RENT(전세) 연도별 거래 건수 요약 
    """
    try:
        # 1. 기본 SQL (연도별 그룹핑 및 건수 집계)
        baseSql = f"""
            SELECT year, COUNT(*) AS cnt 
            FROM `{sysOpt['dbHostName']}.TB_RENT`
            WHERE monthlyRent = 0 
              AND deposit > 0
              AND NOT REGEXP_CONTAINS(apt, '임대|행복주택')
        """

        # 2. 동적 SQL 파라미터 조립 
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")
        if apt:
            condList.append(f"apt LIKE '%{apt}%'")
        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")
        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")
        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            baseSql += " AND " + " AND ".join(condList)

        # 3. 그룹핑 및 정렬 
        baseSql += " GROUP BY year"
        
        if sort:
            sortParts = [f"{s.split('|')[0]} {s.split('|')[1].upper()}" for s in sort.split(',') if len(s.split('|')) == 2]
            if sortParts: baseSql += " ORDER BY " + ", ".join(sortParts)
        else:
            baseSql += " ORDER BY year ASC"  # 기본적으로 연도순 정렬
        
        cntSql = f"SELECT COUNT(*) AS cnt FROM ({baseSql})"
        baseSql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"

        # 4. 쿼리 실행
        baseRes = [dict(row) for row in client.query(baseSql).result()]
        if not baseRes:
            return resResponse("fail", 400, "검색 결과가 없습니다.", None)

        cntRes = client.query(cntSql).result()
        cnt = next(cntRes)['cnt']

        return resResponse("succ", 200, "처리 완료", cnt, len(baseRes), baseRes)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRentSearchByArea", dependencies=[Depends(chkApiKey)])
def statRentSearchByArea(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """기능: 평수별 전세 데이터 건수 요약"""
    try:
        baseSql = f"""
            SELECT area, COUNT(*) AS cnt 
            FROM `{sysOpt['dbHostName']}.TB_RENT`
            WHERE monthlyRent = 0 AND deposit > 0 AND NOT REGEXP_CONTAINS(apt, '임대|행복주택')
        """
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")
        if apt: condList.append(f"apt LIKE '%{apt}%'")
        if area: condList.append(f"area LIKE '%{area}%'")
        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")
        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList: baseSql += " AND " + " AND ".join(condList)
        baseSql += " GROUP BY area"
        
        if sort:
            sortParts = [f"{s.split('|')[0]} {s.split('|')[1].upper()}" for s in sort.split(',') if len(s.split('|')) == 2]
            if sortParts: baseSql += " ORDER BY " + ", ".join(sortParts)
        
        cntSql = f"SELECT COUNT(*) AS cnt FROM ({baseSql})"
        baseSql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"

        baseRes = [dict(row) for row in client.query(baseSql).result()]
        cnt = next(client.query(cntSql).result())['cnt'] if baseRes else 0
        return resResponse("succ", 200, "처리 완료", cnt, len(baseRes), baseRes)
    except Exception as e:
        log.error(f'Exception : {e}'); raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRentSearchByApt", dependencies=[Depends(chkApiKey)])
def statRentSearchByApt(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """기능: 아파트별 전세 데이터 건수 요약"""
    try:
        baseSql = f"""
            SELECT apt, COUNT(*) AS cnt 
            FROM `{sysOpt['dbHostName']}.TB_RENT`
            WHERE monthlyRent = 0 AND deposit > 0 AND NOT REGEXP_CONTAINS(apt, '임대|행복주택')
        """
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")
        if apt: condList.append(f"apt LIKE '%{apt}%'")
        if area: condList.append(f"area LIKE '%{area}%'")
        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")
        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList: baseSql += " AND " + " AND ".join(condList)
        baseSql += " GROUP BY apt"
        
        if sort:
            sortParts = [f"{s.split('|')[0]} {s.split('|')[1].upper()}" for s in sort.split(',') if len(s.split('|')) == 2]
            if sortParts: baseSql += " ORDER BY " + ", ".join(sortParts)
        
        cntSql = f"SELECT COUNT(*) AS cnt FROM ({baseSql})"
        baseSql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"

        baseRes = [dict(row) for row in client.query(baseSql).result()]
        cnt = next(client.query(cntSql).result())['cnt'] if baseRes else 0
        return resResponse("succ", 200, "처리 완료", cnt, len(baseRes), baseRes)
    except Exception as e:
        log.error(f'Exception : {e}'); raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRentMaxBySgg", dependencies=[Depends(chkApiKey)])
def selStatRentMaxBySgg(
        sgg: str = Query(None, description="시군구")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능: 시군구별(구 단위) 역대 최고 전세가(MAX) 조회
    """
    try:
        # 1. 기본 SQL 설정 (순수 전세 필터 + 임대 제외 + 원 단위 환산)
        baseSql = f"""
            SELECT sgg, COUNT(*) AS cnt, MAX(deposit * 10000) AS max_deposit 
            FROM `{sysOpt['dbHostName']}.TB_RENT`
            WHERE monthlyRent = 0 
              AND deposit > 0
              AND NOT REGEXP_CONTAINS(apt, '임대|행복주택')
        """

        # 2. 동적 SQL 파라미터 조립
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")
        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")
        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")
        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            baseSql += " AND " + " AND ".join(condList)

        # 3. 그룹핑 및 정렬
        baseSql += " GROUP BY sgg"
        
        if sort:
            sortParts = [f"{s.split('|')[0]} {s.split('|')[1].upper()}" for s in sort.split(',') if len(s.split('|')) == 2]
            if sortParts: baseSql += " ORDER BY " + ", ".join(sortParts)
        
        cntSql = f"SELECT COUNT(*) AS cnt FROM ({baseSql})"
        baseSql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"

        # 4. 쿼리 실행
        baseRes = [dict(row) for row in client.query(baseSql).result()]
        if not baseRes:
            return resResponse("fail", 400, "검색 결과가 없습니다.", None)

        cntRes = client.query(cntSql).result()
        cnt = next(cntRes)['cnt']

        return resResponse("succ", 200, "처리 완료", cnt, len(baseRes), baseRes)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRentMaxBySggApt", dependencies=[Depends(chkApiKey)])
def selStatRentMaxBySggApt(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , area: str = Query(None, description="평수")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능: 특정 아파트 단지 내 역대 최고 전세가(MAX) 조회
    """
    try:
        # 1. 기본 SQL 
        baseSql = f"""
            SELECT sgg, apt, COUNT(*) AS cnt, MAX(deposit * 10000) AS max_deposit 
            FROM `{sysOpt['dbHostName']}.TB_RENT`
            WHERE monthlyRent = 0 
              AND deposit > 0
              AND NOT REGEXP_CONTAINS(apt, '임대|행복주택')
        """

        # 2. 동적 파라미터 조립
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")
        if apt:
            condList.append(f"apt LIKE '%{apt}%'")
        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")
        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")
        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        if condList:
            baseSql += " AND " + " AND ".join(condList)

        # 3. 그룹핑 및 정렬
        baseSql += " GROUP BY sgg, apt"
        
        if sort:
            sortParts = [f"{s.split('|')[0]} {s.split('|')[1].upper()}" for s in sort.split(',') if len(s.split('|')) == 2]
            if sortParts: baseSql += " ORDER BY " + ", ".join(sortParts)
        
        cntSql = f"SELECT COUNT(*) AS cnt FROM ({baseSql})"
        baseSql += f" LIMIT {limit} OFFSET {(page - 1) * limit}"

        # 4. 쿼리 실행
        baseRes = [dict(row) for row in client.query(baseSql).result()]
        if not baseRes:
            return resResponse("fail", 400, "검색 결과가 없습니다.", None)

        cntRes = client.query(cntSql).result()
        cnt = next(cntRes)['cnt']

        return resResponse("succ", 200, "처리 완료", cnt, len(baseRes), baseRes)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-statRent", dependencies=[Depends(chkApiKey)])
def selStatRent(
        sgg: str = Query(None, description="시군구")
        , dong: str = Query(None, description="법정동")
        , area: str = Query(None, description="평수")
        , yyyymm: str = Query(None, description="연월")
    ):
    """
    기능: 전세가(deposit) 기준 전월 대비 최고상승/최고하락 아파트 목록
    """
    try:
        # 매매(amount) 대신 전세(deposit)를 사용하고, 원 단위(*10000)로 환산하여 비교.
        # 임대아파트로 인한 왜곡 방지를 위해 아파트명에 '임대'가 포함된 경우는 제외.
        baseSql = f"""
            WITH MONTHLYAVGPRICES AS (
                SELECT
                    APT, SGG, DONG, AREA,
                    DATE_TRUNC(DATE, MONTH) AS MONTH,
                    AVG(deposit * 10000) AS AVG_DEPOSIT
                FROM
                    `{sysOpt['dbHostName']}.TB_RENT`
                WHERE deposit > 0 AND monthlyRent = 0 AND NOT REGEXP_CONTAINS(apt, '임대')
                GROUP BY 1, 2, 3, 4, 5
            ),
            LAGGEDPRICES AS (
                SELECT
                    APT, SGG, DONG, AREA, MONTH, AVG_DEPOSIT,
                    LAG(AVG_DEPOSIT, 1, 0) OVER (PARTITION BY APT, SGG, DONG, AREA ORDER BY MONTH) AS PREV_AVG_DEPOSIT,
                    SAFE_DIVIDE(AVG_DEPOSIT - LAG(AVG_DEPOSIT, 1, 0) OVER (PARTITION BY APT, SGG, DONG, AREA ORDER BY MONTH), LAG(AVG_DEPOSIT, 1, 0) OVER (PARTITION BY APT, SGG, DONG, AREA ORDER BY MONTH)) * 100 AS RATE
                FROM MONTHLYAVGPRICES
            )
            SELECT * FROM (
                (SELECT '최고하락' AS GRP, SGG, DONG, DATE_ADD(MONTH, INTERVAL -1 MONTH) AS PREV_MONTH, MONTH AS CURR_MONTH, APT, AREA, PREV_AVG_DEPOSIT, AVG_DEPOSIT AS CURR_AVG_DEPOSIT, RATE
                FROM LAGGEDPRICES
                WHERE (SGG LIKE '%' || @sgg || '%' OR @sgg IS NULL)
                    AND (DONG LIKE '%' || @dong || '%' OR @dong IS NULL)
                    AND (AREA LIKE '%' || @area || '%' OR @area IS NULL)
                    AND (MONTH = DATE_TRUNC(PARSE_DATE('%Y%m', @yyyymm), MONTH) OR @yyyymm IS NULL)
                    AND PREV_AVG_DEPOSIT > 0
                ORDER BY RATE ASC LIMIT 30)
                UNION ALL
                (SELECT '최고상승' AS GRP, SGG, DONG, DATE_ADD(MONTH, INTERVAL -1 MONTH) AS PREV_MONTH, MONTH AS CURR_MONTH, APT, AREA, PREV_AVG_DEPOSIT, AVG_DEPOSIT AS CURR_AVG_DEPOSIT, RATE
                FROM LAGGEDPRICES
                WHERE (SGG LIKE '%' || @sgg || '%' OR @sgg IS NULL)
                    AND (DONG LIKE '%' || @dong || '%' OR @dong IS NULL)
                    AND (AREA LIKE '%' || @area || '%' OR @area IS NULL)
                    AND (MONTH = DATE_TRUNC(PARSE_DATE('%Y%m', @yyyymm), MONTH) OR @yyyymm IS NULL)
                    AND PREV_AVG_DEPOSIT > 0
                ORDER BY RATE DESC LIMIT 30)
            );
        """

        queryParam = [
            bigquery.ScalarQueryParameter("sgg", "STRING", sgg if sgg else None),
            bigquery.ScalarQueryParameter("dong", "STRING", dong if dong else None),
            bigquery.ScalarQueryParameter("area", "STRING", area if area else None),
            bigquery.ScalarQueryParameter("yyyymm", "STRING", yyyymm if yyyymm else None)
        ]

        queryJob = client.query(baseSql, job_config=bigquery.QueryJobConfig(query_parameters=queryParam))
        results = queryJob.result()
        fileList = [dict(row) for row in results]

        if not fileList:
            return resResponse("fail", 400, "검색 결과가 없습니다.", None)

        return resResponse("succ", 200, "처리 완료", len(fileList), len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-rent", dependencies=[Depends(chkApiKey)])
def selRent(
        sgg: str = Query(None, description="시군구")
        , apt: str = Query(None, description="도로명주소")
        , aptDtl: str = Query(None, description="도로명주소 상세")
        , key: str = Query(None, description="지번주소")
        , keyDtl: str = Query(None, description="지번주소 상세")
        , area: str = Query(None, description="평수")
        , minMonthlyRent: str = Query(None, description="최소 월세")
        , maxMonthlyRent: str = Query(None, description="최대 월세")
        , srtYear: int = Query(None, description="시작 연도")
        , endYear: int = Query(None, description="종료 연도")
        , srtDate: str = Query(None, description="시작일 %Y-%m-%d")
        , endDate: str = Query(None, description="종료일 %Y-%m-%d")
        , limit: int = Query(10, description="1쪽당 개수")
        , page: int = Query(1, description="현재 쪽")
        , sort: str = Query(None, description="정렬")
    ):
    """
    기능: TB_RENT 목록 조회 (매매 조회와 동일한 방식임)
    """
    try:
        # 1. 기본 SQL
        baseSql = f"SELECT * FROM `{sysOpt['dbHostName']}.TB_RENT`"
        cntSql = f"SELECT COUNT(*) AS CNT FROM `{sysOpt['dbHostName']}.TB_RENT`"

        # 2. 동적 SQL 파라미터 조립
        condList = []
        if sgg:
            sggList = [s.strip() for s in sgg.split(',')]
            sggCond = [f"sgg LIKE '%{s}%'" for s in sggList]
            condList.append(f"({' OR '.join(sggCond)})")

        if apt:
            aptList = [s.strip() for s in apt.split(',')]
            aptCond = [f"apt LIKE '%{s}%'" for s in aptList]
            condList.append(f"({' OR '.join(aptCond)})")

        if aptDtl:
            aptDtlList = [s.strip() for s in aptDtl.split(',')]
            aptDtlCond = [f"aptDtl LIKE '%{s}%'" for s in aptDtlList]
            condList.append(f"({' OR '.join(aptDtlCond)})")

        if key:
            keyList = [s.strip() for s in key.split(',')]
            keyCond = [f"key LIKE '%{s}%'" for s in keyList]
            condList.append(f"({' OR '.join(keyCond)})")

        if keyDtl:
            keyDtlList = [s.strip() for s in keyDtl.split(',')]
            keyDtlCond = [f"keyDtl LIKE '%{s}%'" for s in keyDtlList]
            condList.append(f"({' OR '.join(keyDtlCond)})")

        if area:
            areaList = [s.strip() for s in area.split(',')]
            areaCond = [f"area LIKE '%{s}%'" for s in areaList]
            condList.append(f"({' OR '.join(areaCond)})")

        if minMonthlyRent or maxMonthlyRent:
            condList.append(f"CAST(monthlyRent AS FLOAT64) BETWEEN {minMonthlyRent} AND {maxMonthlyRent}")

        if srtYear and endYear:
            condList.append(f"year BETWEEN {srtYear} AND {endYear}")

        if srtDate and endDate:
            condList.append(f"date BETWEEN DATE('{srtDate}') AND DATE('{endDate}')")

        # 3. WHERE 절 결합 
        if condList:
            condSql = " WHERE " + " AND ".join(condList)
            baseSql += condSql
            cntSql += condSql

        # 4. 정렬 로직 
        sortList = []
        if sort:
            for sortInfo in sort.split(','):
                sortPart = sortInfo.split('|')
                if sortPart is None or len(sortPart) != 2: continue
                sortCol = sortPart[0]
                sortOrd = sortPart[1].upper() if sortPart[1].upper() in ["ASC", "DESC"] else "ASC"
                sortList.append(f"{sortCol} {sortOrd}")
        if sortList:
            sortSql = " ORDER BY " + ", ".join(sortList)
            baseSql += sortSql

        # 5. 페이징 및 쿼리 실행 
        pageSql = f" LIMIT {limit} OFFSET {(page - 1) * limit}"
        baseSql += pageSql

        baseQuery = client.query(baseSql)
        baseRes = baseQuery.result()
        fileList = [dict(row) for row in baseRes]
        
        if fileList is None or len(fileList) < 1:
            return resResponse("fail", 400, f"검색 결과가 없습니다.", None)

        cntQuery = client.query(cntSql)
        cntRes = cntQuery.result()
        cnt = next(cntRes)['CNT']

        return resResponse("succ", 200, "처리 완료", cnt, len(fileList), fileList)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))


