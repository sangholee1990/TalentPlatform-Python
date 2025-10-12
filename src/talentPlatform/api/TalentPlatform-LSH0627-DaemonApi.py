# ============================================
# 요구사항
# ============================================
# LSH0627. Python을 이용한 AI 기반 랜딩 페이지 제작 및 API 연계 서비스

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api
# conda activate py39

# 운영 서버
# uvicorn TalentPlatform-LSH0627-DaemonApi:app --host=0.0.0.0 --port=9030
# nohup uvicorn TalentPlatform-LSH0627-DaemonApi:app --host=0.0.0.0 --port=9030 &
# tail -f nohup.out

# 테스트 서버
# uvicorn TalentPlatform-LSH0627-DaemonApi:app --reload --host=0.0.0.0 --port=9030

# 프로그램 종료
# ps -ef | grep "TalentPlatform-LSH0627-DaemonApi" | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9030
# lsof -i :9030 | awk '{print $2}' | xargs kill -9

# 명세1) http://49.247.41.71:9030/docs
# 명세2) http://49.247.41.71:9030/redoc

# ============================================
# 라이브러리
# ============================================
import glob
import os
import platform
import warnings
from datetime import timedelta
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Any, Dict
import configparser
import os
from urllib.parse import quote_plus
import requests
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from typing import List
from sqlalchemy.orm import Session
import os
import shutil
# from datetime import datetime
from fastapi import FastAPI, UploadFile, File
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
import requests
# from google.cloud import bigquery
# from google.oauth2 import service_account
# import db_dtypes
# from src.api.guest.router import router as guest_router
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import urllib
import os
import shutil
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import tempfile
import os
from enum import Enum
from pydantic import BaseModel, Field, constr, validator
from konlpy.tag import Okt
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from urllib import parse
import time
from urllib.parse import quote_plus, urlencode
import pytz
from pytrends.request import TrendReq
from bs4 import BeautifulSoup
from lxml import etree
import xml.etree.ElementTree as et
from pytrends.request import TrendReq
from fastapi.responses import StreamingResponse
from io import BytesIO
from google import genai
import configparser

# ============================================
# 유틸리티 함수
# ============================================
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

    # logger instance 생성
    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    # format 생성
    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    # handler 생성
    streamHandler = logging.StreamHandler()
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1, backupCount=30, encoding='utf-8')

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
def chkApiKey(api_key: str = Depends(APIKeyHeader(name="api"))):
    if api_key != '20251012-topbds':
        raise HTTPException(status_code=400, detail="API 인증 실패")

def resResponse(status: str, code: int, message: str, cnt: Any = None, data: Any = None) -> dict:
    return {
        "status": status
        , "code": code
        , "message": message
        , "cnt": cnt
        , "data": data
    }

def getPageDict(data, page=1, limit=10):
    result = {}
    start_index = (page - 1) * limit
    end_index = start_index + limit

    if data.empty:
        return result

    for (brand, type), list in zip(data.index, data.values):
        cnt = len(list)
        if cnt > 0:
            key = f"{brand}-{type}"
            result[key] = {
                'cnt': cnt,
                'item': list[start_index:end_index]
            }

    return result

# ============================================
# 주요 설정
# ============================================
env = 'local'
serviceName = 'LSH0627'
prjName = 'test'

# ctxPath = os.getcwd()
ctxPath = '/HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api'

log = initLog(env, ctxPath, prjName)

# 작업 경로 설정
# os.chdir(f"{ctxPath}")
# log.info(f"[CHECK] getcwd : {os.getcwd()}")

# 옵션 설정
sysOpt = {
    # CORS 설정
    'oriList': ['*'],

    # 입력 데이터
    'csvFile': '/HDD/DATA/OUTPUT/LSH0627/naverShop_prd.csv',

    # 설정 정보
    'cfgFile': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
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
    , allow_origins=sysOpt['oriList']
    , allow_credentials=True
    , allow_methods=["*"]
    , allow_headers=["*"]
)

# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')

# Gemini API키
config = configparser.ConfigParser()
config.read(sysOpt['cfgFile'], encoding='utf-8')
client = config.get(sysOpt['cfgKey'], sysOpt['cfgVal'])

# 설정 파일
try:
    csvFile = sysOpt['csvFile']
    csvList = sorted(glob.glob(csvFile))
    if csvList is None or len(csvList) < 1:
        log.error(f'설정 파일 없음, csvFile : {csvFile}')
        exit(1)

    csvInfo = csvList[0]
    csvData = pd.read_csv(csvInfo)

    # 주요 전처리
    csvData['title'] = csvData['title'].str.replace('<[^>]*>', '', regex=True).str.strip()
    csvData['isDlPrd'] = csvData['dlPrd'].notna()
    csvData['isMlPrd'] = csvData['mlPrd'].notna()
    csvData['isLprice'] = csvData['lprice'].notna()
    csvData['typeByTitle'] = csvData['typeByTitle'].replace({'전기자전거': '전기', '일반자전거': '일반'})
    csvData['brandByTitle'] = pd.Categorical(csvData['brandByTitle'], categories=['알톤 자전거', '삼천리 자전거', '스마트 자전거', '기타'], ordered=True)
    csvData['typeByTitle'] = pd.Categorical(csvData['typeByTitle'], categories=['전기', '하이브리드', 'MTB', '사이클', '일반', '미니벨로'], ordered=True)

    tmpData = csvData.copy()
    tmpDataL1 = tmpData.sort_values(by=['title', 'isDlPrd', 'isMlPrd', 'isLprice'], ascending=[True, False, False, False])
    csvDataL1 = tmpDataL1.drop_duplicates(subset=['title'], keep='first')


except Exception as e:
    log.error(f'설정 파일 실패, csvFile : {csvFile} : {e}')
    exit(1)

minYear = 2015
maxYear = 2025
selData = csvDataL1.loc[
    (csvDataL1['yearByTitle'] >= float(minYear)) & (csvDataL1['yearByTitle'] <= float(maxYear))
    ]
#
# selDataL2 = selData.groupby(['brandByTitle', 'typeByTitle'], observed=False)['title'].apply(list)
# selData = csvData[(csvData['title'] == '2020 알톤 스로틀 FS 전기자전거 앞뒤 서스펜션 20인치 미니벨로')]
# result = getPageDict(selDataL2, page=1, limit=10)


# ============================================
# 비즈니스 로직
# ============================================
class cfgBrandModel(BaseModel):
    year: str = Field(..., description='연식 (최소-최대)', examples=['2015-2025']),
    status: str = Query(default=..., description='자전거 상태', example='중', enum=['상', '중', '하']),
    limit: int = Query(10, description="1쪽당 개수"),
    page: int = Query(1, description="현재 쪽"),

class cfgPrd(BaseModel):
    year: str = Field(..., description='연식 (최소-최대)', examples=['2015-2025']),
    status: str = Query(default=..., description='자전거 상태', example='중', enum=['상', '중', '하']),
    model: str = Field(..., description='자전거 모델', examples=['2020 알톤 스로틀 FS 전기자전거 앞뒤 서스펜션 20인치 미니벨로']),

class cfgChatTypeCont(BaseModel):
    model: str = Query(default=..., description='생성형 AI 종류', example='gemini-2.5-pro', enum=['gemini-2.5-pro', 'gemini-1.5-flash-latest']),
    cont: str = Field(default=..., description='요청사항', example='''테스트
    '''),

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# @app.post(f"/api/sel-brandModel", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-brandModel")
async def selBrandModel(request: cfgBrandModel = Form(...)):
    try:
        year = request.year
        minYear, maxYear  = year.split('-')
        if year is None or len(year) < 1 or minYear is None or maxYear is None:
            return resResponse("fail", 400, f"연식 없음, year : {year}", None)

        status = request.status
        if status is None or len(status) < 1:
            return resResponse("fail", 400, f"자전거 상태 없음, status : {status}")

        page = request.page
        if page is None or len(page) < 1:
            return resResponse("fail", 400, f"현재 쪽 없음, page : {page}")

        limit = request.limit
        if limit is None or len(limit) < 1:
            return resResponse("fail", 400, f"1쪽당 개수 없음, limit : {limit}")

        selData = csvDataL1.loc[
            (csvDataL1['yearByTitle'] >= float(minYear)) & (csvDataL1['yearByTitle'] <= float(maxYear))
            ]

        if len(selData) < 1:
            return resResponse("fail", 400, f"브랜드/모델명 없음", None)

        selDataL2 = selData.groupby(['brandByTitle', 'typeByTitle'], observed=False)['title'].apply(list)
        result = getPageDict(selDataL2, page=page, limit=limit)
        log.info(f"result : {result}")

        return resResponse("succ", 200, "처리 완료", len(result), len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# @app.post(f"/api/sel-prd", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-prd")
async def selPrd(request: cfgPrd = Form(...)):
    try:
        year = request.year
        minYear, maxYear  = year.split('-')
        if year is None or len(year) < 1 or minYear is None or maxYear is None:
            return resResponse("fail", 400, f"연식 없음, year : {year}", None)

        status = request.status
        if status is None or len(status) < 1:
            return resResponse("fail", 400, f"자전거 상태 없음, status : {status}")

        model = request.status
        if model is None or len(model) < 1:
            return resResponse("fail", 400, f"자전거 모델 없음, model : {model}")

        selData = csvData.loc[
            (csvData['yearByTitle'] >= float(minYear)) & (csvData['yearByTitle'] <= float(maxYear))
            ]

        if len(selData) < 1:
            return resResponse("fail", 400, f"데이터 없음", None)

        result = getPageDict(selDataL2, page=page, limit=limit)
        log.info(f"result : {result}")

        return resResponse("succ", 200, "처리 완료", len(result), len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"/api/sel-chatTypeCont")
async def selChatTypeCont(request: cfgChatTypeCont = Form(...)):
    """
    기능\n
        생성형 AI 모델 기반 비교 리포트 (종합 성능, 상세 스펙, 종합 분석)\n
    테스트\n
        model: 생성형 AI 모델 종류\n
        cont: 요청사항\n
    """
    try:
        model = request.model
        if model is None or len(model) < 1:
            return resResponse("fail", 400, f"생성형 AI 모델 종류 없음, model : {model}")

        cont = request.cont
        if cont is None or len(cont) < 1:
            return resResponse("fail", 400, f"요청사항 없음, cont : {cont}")

        contTemplate = '''%type% 랜딩 페이지를 html 파일로 제작해조
            요청사항
                %cont%
           '''
        contents = contTemplate.replace('%cont%', cont)
        log.info(f"contents : {contents}")

        response = client.models.generate_content(
            model=model,
            contents=contents
        )
        result = response.text
        log.info(f"result : {result}")

        if result is None or len(result) < 1:
            return resResponse("fail", 400, "처리 실패")

        # return resResponse("succ", 200, "처리 완료", len(result), result)
        return Response(
            content=result,
            media_type="text/plain",
            headers={
                "Content-Disposition": "attachment; filename=generated_content.txt"
            }
        )

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))