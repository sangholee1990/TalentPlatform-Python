# ============================================
# 요구사항
# ============================================
# 모자이크앱  데이터 적재
# 명세1 http://49.247.41.71:9910/docs
# 인증키 20260221-bdwide

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2026
# cd /vol01/SYSTEMS/DMS02/PROG/PYTHON/DATA_BASE

# conda activate py39

# 운영 서버
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn TalentPlatform-INDI2026-DaemonApi-dataBaseMock:app --host=0.0.0.0 --port=9800 &
# nohup /home/indisystem/.conda/envs/py39/bin/uvicorn TalentPlatform-INDI2026-DaemonApi-dataBaseMock:app --reload --host=0.0.0.0 --port=9800 &
# tail -f nohup.out

# 테스트 서버
# /HDD/SYSTEMS/LIB/anaconda3/envs/py39/bin/uvicorn TalentPlatform-INDI2026-DaemonApi-dataBaseMock:app --reload --host=0.0.0.0 --port=9800
# /home/indisystem/.conda/envs/py39/bin/uvicorn TalentPlatform-INDI2026-DaemonApi-dataBaseMock:app --reload --host=0.0.0.0 --port=9800

# 프로그램 종료
# pkill -f TalentPlatform-INDI2026-DaemonApi-dataBaseMock
# ps -ef | grep "TalentPlatform-INDI2026-DaemonApi-dataBaseMock" | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9910
# lsof -i :9910 | awk '{print $2}' | xargs kill -9

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
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
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
import pytz
from fastapi.responses import StreamingResponse
from functools import lru_cache
warnings.filterwarnings('ignore')

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

# 인증키 검사
def chkKey(key: str = Depends(APIKeyHeader(name="key"))):
    if key != '20260221-bdwide':
        raise HTTPException(status_code=400, detail="API 인증 실패")

def resResponse(status: str, code: int, message: str, cnt: Any = None, data: Any = None) -> dict:
    return {
        "status": status
        , "code": code
        , "message": message
        , "cnt": cnt
        , "data": data
    }

@lru_cache(maxsize=None)
def loadJson(fileInfo: str):
    try:
        with open(fileInfo, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log.error(f'Mock 데이터 로드 실패, {e}')
        return resResponse("fail", 400, f"처리 실패, {e}")

# ============================================
# 주요 설정
# ============================================
env = 'dev'
serviceName = 'INDI2026'
prjName = 'dataBaseMock'

# ctxPath = os.getcwd()
# ctxPath = '/SYSTEMS/PROG/PYTHON/IDE/src/proj/indisystem/2026'
ctxPath = '/vol01/SYSTEMS/DMS02/PROG/PYTHON/DATA_BASE'

log = initLog(env, ctxPath, prjName)

# 작업 경로 설정
# os.chdir(f"{ctxPath}")
# log.info(f"[CHECK] getcwd : {os.getcwd()}")

# 옵션 설정
sysOpt = {
    # CORS 설정
    'oriList': ['*'],
    # 'file11': open('/DATA/INPUT/INDI2026/dataBaseMock/1-1__위성관측_대류운_response1.json', "r", encoding="utf-8"),
    # 'file12': open('/DATA/INPUT/INDI2026/dataBaseMock/1-2__위성관측_건조역_response.json', "r", encoding="utf-8"),
    # 'file21': open('/DATA/INPUT/INDI2026/dataBaseMock/2-1__기상관측_레이더_response1.json', "r", encoding="utf-8"),
    # 'file22': open('/DATA/INPUT/INDI2026/dataBaseMock/2-2__시정계_response1.json', "r", encoding="utf-8"),
    # 'file31': open('/DATA/INPUT/INDI2026/dataBaseMock/3-1__기상예보_특보현황_response.json', "r", encoding="utf-8"),
    # 'file41': open('/DATA/INPUT/INDI2026/dataBaseMock/4-1__기상분석_변수-온도_response.json', "r", encoding="utf-8"),
    # 'file42': open('/DATA/INPUT/INDI2026/dataBaseMock/4-2__기상분석_변수-바람_response.json', "r", encoding="utf-8"),
    # 'file51': open('/DATA/INPUT/INDI2026/dataBaseMock/5-1__태풍분석_적외태풍강조_response.json', "r", encoding="utf-8"),
    # 'file52': open('/DATA/INPUT/INDI2026/dataBaseMock/5-2__태풍분석_AMW해상풍_response.json', "r", encoding="utf-8"),
    'file11': '/vol01/SYSTEMS/DMS02/PROG/PYTHON/DATA_BASE/dataBaseMock/1-1__위성관측_대류운_response1.json',
    'file12': '/vol01/SYSTEMS/DMS02/PROG/PYTHON/DATA_BASE/dataBaseMock/1-2__위성관측_건조역_response.json',
    'file21': '/vol01/SYSTEMS/DMS02/PROG/PYTHON/DATA_BASE/dataBaseMock/2-1__기상관측_레이더_response1.json',
    'file22': '/vol01/SYSTEMS/DMS02/PROG/PYTHON/DATA_BASE/dataBaseMock/2-2__시정계_response1.json',
    'file31': '/vol01/SYSTEMS/DMS02/PROG/PYTHON/DATA_BASE/dataBaseMock/3-1__기상예보_특보현황_response.json',
    'file41': '/vol01/SYSTEMS/DMS02/PROG/PYTHON/DATA_BASE/dataBaseMock/4-1__기상분석_변수-온도_response.json',
    'file42': '/vol01/SYSTEMS/DMS02/PROG/PYTHON/DATA_BASE/dataBaseMock/4-2__기상분석_변수-바람_response.json',
    'file51': '/vol01/SYSTEMS/DMS02/PROG/PYTHON/DATA_BASE/dataBaseMock/5-1__태풍분석_적외태풍강조_response.json',
    'file52': '/vol01/SYSTEMS/DMS02/PROG/PYTHON/DATA_BASE/dataBaseMock/5-2__태풍분석_AMW해상풍_response.json',
}

app = FastAPI(
    title="데이터기반 모의 API",
    description="",
    version="1.0"
    ,openapi_url='/api'
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

# ============================================
# 비즈니스 로직
# ============================================
# 타임존 설정
tzKst = pytz.timezone('Asia/Seoul')
tzUtc = pytz.timezone('UTC')


mockData = {}
for key, path in sysOpt.items():
    if key == 'oriList': continue
    try:
        with open(path, 'r', encoding='utf-8') as f:
            mockData[key] = json.load(f)
        log.info(f"데이터 읽기, {path}")
    except Exception as e:
        log.error(f"데이터 실패, {e}")

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect():
    return RedirectResponse(url="/docs")

@app.get(f"/ping/", tags=['서버 동작 유무'])
async def ping():
    return "OK"

@app.get(f"/GetAPIReadFile/", tags=['1.1 위성관측 대류운, 2.2 기상관측 시정계, 3.1 기상예보 특보현황'])
async def getAPIReadFile(
        path: str = None,
        format: str = None,
):
    try:
        if path is None: return mockData['file11']
        elif "RDT" in path: return mockData['file11']
        elif "MIN_VIS2" in path or "VIS" in path: return mockData['file22']
        elif "WRN" in path: return mockData['file31']
    except Exception as e:
        log.error(f'Exception : {e}')
        return resResponse("fail", 400, f"처리 실패, {e}")

@app.get(f"/GetAPIGK2_WVDRY_data/", tags=['1.2 위성관측 건조역'])
async def getAPIGK2_WVDRY_data(
        imgtime: str = None,
        group: str = None,
        gap: str = None,
        req: str = None
):
    try:
        return mockData['file12']
    except Exception as e:
        log.error(f'Exception : {e}')
        return resResponse("fail", 400, f"처리 실패, {e}")

@app.get(f"/GetAPIRDR_data/", tags=['2.1 레이더'])
async def getAPIRDR_data(
        imgtime: str = None,
        req: str = None
):
    try:
        return mockData['file21']
    except Exception as e:
        log.error(f'Exception : {e}')
        return resResponse("fail", 400, f"처리 실패, {e}")

@app.get(f"/GetAPIKimncNwp/", tags=['4.1 기상분석 온도, 4.2 기상분석 바람'])
async def getAPIKimncNwp(
        name: str = None,
        group: str = None,
        nwp: str = None,
        data: str = None,
        map: str = None,
        tmfc: str = None,
        hf: str = None,
        disp: str = None,
        help: str = None,
        level: str = None,
        type: str = None,
        imgtime: str = None
):
    try:
        if name is None: return mockData['file41']
        if name == "T": return mockData['file41']
        elif name == "wind": return mockData['file42']
    except Exception as e:
        log.error(f'Exception : {e}')
        return resResponse("fail", 400, f"처리 실패, {e}")

@app.get(f"/GetAPIGK2A_EIR_C/", tags=['5.1 태풍분석 적외태풍강조'])
async def getAPIGK2A_EIR_C(
        area: str = None,
        imgtime: str = None,
        req: str = None
):
    try:
        return mockData['file51']
    except Exception as e:
        log.error(f'Exception : {e}')
        return resResponse("fail", 400, f"처리 실패, {e}")

@app.get(f"/GetAPIGK2A_ssw/", tags=['5.2 태풍분석 AMW해상풍'])
async def getAPIGK2A_ssw(
        sat: str = None,
        imgtime: str = None,
        req: str = None
):
    try:
        return mockData['file52']
    except Exception as e:
        log.error(f'Exception : {e}')
        return resResponse("fail", 400, f"처리 실패, {e}")


