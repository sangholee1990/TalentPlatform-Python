# ============================================
# 요구사항
# ============================================
# LSH0597. Python을 이용한 코딩 테스트 플랫폼 개발 (인텔마이티 이러닝 프로젝트)

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api
# conda activate py38
# uvicorn TalentPlatform-LSH0597-DaemonApi:app --reload --host=0.0.0.0 --port=9300
# nohup uvicorn TalentPlatform-LSH0597-DaemonApi:app --reload --host=0.0.0.0 --port=9300 &
# tail -f nohup.out

# ============================================
# 라이브러리
# ============================================
import glob
import os
import platform
import warnings
from datetime import timedelta
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import configparser
import os
from urllib.parse import quote_plus
import requests
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
import os
import shutil
# from datetime import datetime
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
# def chkApiKey(api_key: str = Depends(APIKeyHeader(name="api"))):
#     if api_key != '20241014.api':
#         raise HTTPException(status_code=400, detail=resRespone("fail", 400, "인증 실패"))

def resRespone(status: str, code: int, message: str, cnt: int = 0, data: Any = None) -> dict:
    return {
        "status": status
        , "code": code
        , "message": message
        , "cnt": cnt
        , "data": data
    }

# ============================================
# 주요 설정
# ============================================
env = 'local'
serviceName = 'LSH0597'
prjName = 'test'

ctxPath = os.getcwd()
# ctxPath = f"/SYSTEMS/PROG/PYTHON/IDE"

globalVar = {
    'ctxPath': f"{ctxPath}"
    , 'inpPath': f"/DATA/INPUT/{serviceName}"
    , 'outPath': f"/DATA/OUTPUT/{serviceName}"
    , 'figPath': f"/DATA/FIG/{serviceName}"
    , 'cfgPath': f"/SYSTEMS/PROG/PYTHON/IDE/resources/config"
}

for key, val in globalVar.items():
    if key.__contains__('Path'):
        os.makedirs(val, exist_ok=True)
        print(f"[CHECK] {key} : {val}")

# 작업 경로 설정
# os.chdir(f"{globalVar['ctxPath']}")
# print(f"[CHECK] getcwd : {os.getcwd()}")

log = initLog(env, ctxPath, prjName)

# 옵션 설정
sysOpt = {
    # 시작/종료 시간
    # 'srtDate': '2018-01-01'
    # , 'endDate': '2018-12-31'
}

app = FastAPI(
    openapi_url='/api'
    , docs_url='/docs'
    , redoc_url='/redoc'
)

# 공유 설정
# app.mount('/UPLOAD', StaticFiles(directory='/DATA/UPLOAD'), name='/DATA/UPLOAD')

# CORS 설정
oriList = [
    'http://localhost:9300'
    , 'http://49.247.41.71:9300'
]

app.add_middleware(
    CORSMiddleware
    # , allow_origins=["*"]
    , allow_origins=oriList
    , allow_credentials=True
    , allow_methods=["*"]
    , allow_headers=["*"]
)

# ============================================
# 비즈니스 로직
# ============================================

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")
