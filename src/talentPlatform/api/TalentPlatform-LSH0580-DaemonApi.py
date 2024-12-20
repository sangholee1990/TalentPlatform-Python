# ============================================
# 요구사항
# ============================================
# [완료] LSH0577. Python을 이용한 빅쿼리 기반으로 API 배포체계

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api
# conda activate py38
# uvicorn TalentPlatform-LSH0580-DaemonApi:app --reload --host=0.0.0.0 --port=9100
# nohup uvicorn TalentPlatform-LSH0580-DaemonApi:app --reload --host=0.0.0.0 --port=9100 &
# tail -f nohup.out

# 프로그램 종료
# ps -ef | grep "TalentPlatform-LSH0580-DaemonApi" | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9100
# lsof -i :9100 | awk '{print $2}' | xargs kill -9

# 명세1) http://49.247.41.71:9100/docs
# 명세2) http://49.247.41.71:9100/redoc

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
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Any, Dict
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

from google.cloud import bigquery
from google.oauth2 import service_account
import db_dtypes
# from src.api.guest.router import router as guest_router
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import urllib


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
def chkApiKey(api_key: str = Depends(APIKeyHeader(name="api"))):
    if api_key != '20241014-topbds':
        raise HTTPException(status_code=400, detail="API 인증 실패")


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
serviceName = 'LSH0580'
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
    'srtDate': '2018-01-01'
    , 'endDate': '2018-12-31'
    , 'api': {
        'nph-qpf_ana_img': 'https://apihub.kma.go.kr/api/typ03/cgi/dfs/nph-qpf_ana_img?eva=1&tm=%Y%m%d%H%M&qpf=B&ef=360&map=HR&grid=2&legend=1&size=600&zoom_level=0&zoom_x=0000000&zoom_y=0000000&stn=108&x1=470&y1=575&authKey=DMoNuRIXSjSKDbkSF_o0qg'
    }
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
    'http://localhost:8300'
    , 'http://localhost:3000'
    , 'http://49.247.41.71:8300'
]

app.add_middleware(
    CORSMiddleware
    # , allow_origins=["*"]
    , allow_origins=oriList
    , allow_credentials=True
    , allow_methods=["*"]
    , allow_headers=["*"]
)


# jsonFile = '{}/{}'.format(globalVar['cfgPath'], 'iconic-ruler-239806-7f6de5759012.json')
# jsonList = sorted(glob.glob(jsonFile))
# jsonInfo = jsonList[0]

# credentials = service_account.Credentials.from_service_account_file(jsonInfo)
# client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# base = declarative_base()

# ============================================
# 비즈니스 로직
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@app.get(f"/api/get/img")
def getImg(
        type: str = Query(None, description="이미지 종류")
):
    """
    기능\n
        이미지 검색\n
    테스트\n
        type: 이미지 종류 nph-qpf_ana_img\n
    """

    try:
        kst = pytz.timezone('Asia/Seoul')
        dtPreDate = datetime.now(kst)
        preMin = dtPreDate.minute
        if preMin % 10 >= 5:
            newMin = (preMin // 10) * 10 + 10
        else:
            newMin = (preMin // 10) * 10
        if newMin == 60:
            dtEndDate = dtPreDate.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            dtEndDate = dtPreDate.replace(minute=newMin, second=0, microsecond=0)

        dtEndDate = dtPreDate.replace(minute=newMin, second=0, microsecond=0)
        dtSrtDate = dtEndDate - timedelta(hours=24)
        dtDateList = pd.date_range(start=dtEndDate, end=dtSrtDate, freq='-10t')

        resData = None

        # dtDateInfo = dtDateList[0]
        for dtDateInfo in dtDateList:
            url = dtDateInfo.strftime(sysOpt['api'][type])
            res = urllib.request.urlopen(url)
            resCode = res.getcode()
            if resCode != 200: continue

            resCon = res.read()
            if len(resCon) < 5000: continue

            resData = {
                'url': url
                , 'dtDateInfo': dtDateInfo
                , 'tm': dtDateInfo.strftime('%Y%m%d%H%M')
                , 'size': len(resCon)
            }
            break

        if resData is None: raise Exception("검색 오류")

        return resRespone("succ", 200, "처리 완료", 1, resData)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))


@app.get(f"/api/set/img")
def getImg(
        url: str = Query(None, description="이미지 주소")
):
    """
    기능\n
        이미지 검색\n
    테스트\n
        url: 이미지 주소\n
    """

    try:
        resData = None
        res = urllib.request.urlopen(url)
        resCode = res.getcode()
        if resCode != 200: raise Exception("통신 오류")

        resCon = res.read()
        if len(resCon) < 5000: raise Exception("크기 오류")

        resData = {
            'url': url
            , 'size': len(resCon)
        }

        if resData is None: raise Exception("검색 오류")

        return resRespone("succ", 200, "처리 완료", 1, resData)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))
