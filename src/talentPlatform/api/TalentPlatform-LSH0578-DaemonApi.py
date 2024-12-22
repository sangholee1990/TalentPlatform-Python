# ============================================
# 요구사항
# ============================================
# LSH0578. Python을 이용한 생성형 AI 기반 블로그 포스팅 대필 및 API 연계 서비스

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api
# conda activate py39
# uvicorn TalentPlatform-LSH0578-DaemonApi:app --reload --host=0.0.0.0 --port=9200
# nohup uvicorn TalentPlatform-LSH0578-DaemonApi:app --host=0.0.0.0 --port=9200 &
# tail -f nohup.out

# 프로그램 종료
# ps -ef | grep "TalentPlatform-LSH0578-DaemonApi" | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9200
# lsof -i :9200 | awk '{print $2}' | xargs kill -9

# 명세1) http://49.247.41.71:9200/docs
# 명세2) http://49.247.41.71:9200/redoc

# 20241202 기술협의
# 금칙어 제공 서비스

# 프롬프트 선택지 추천 기능
# 후기성, 정보성
# 읽는사람이 어떤 목적
# 글쓴이의 업종 등등
# -> 정확한 프롬프트를 위한 서포트장치 필요

# 기획안 예시
# 특정 글감에 대해서 필요한 주제를 나열해 달라는 요구가 있을 수 있음
# 이에 다양한 주제를 받을 수 있는 기능이 필요함

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

# from google.cloud import bigquery
# from google.oauth2 import service_account
# import db_dtypes
# from src.api.guest.router import router as guest_router
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import urllib

import google.generativeai as genai
import os
import shutil
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import tempfile
import os
from enum import Enum
from pydantic import BaseModel, Field, constr, validator

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
serviceName = 'LSH0578'
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

genai.configure(api_key=None)
model = genai.GenerativeModel('gemini-1.5-pro')

csvFile = '{}/{}'.format(globalVar['inpPath'], '20241103_13개 분야 별로 대표 템플릿 생성형 AI 4종 결과 - 최종.csv')
csvList = sorted(glob.glob(csvFile))
csvInfo = csvList[0]
csvData = pd.read_csv(csvInfo)

# ============================================
# 비즈니스 로직
# ============================================
# class pdfToTxtData(BaseModel):
#     cont: str = Field(..., example='텍스트 추출', description='요청사항')
#     file: UploadFile = File(None, example=None, description='PDF 파일')

class blogTypePostData(BaseModel):
    type: str = Query(default=..., description='분야', example='뷰티화장품', enum=[
        "법률", "뷰티화장품", "병원", "부동산",
        "맛집", "반려동물", "자동차", "운동",
        "여행", "휴대폰성지", "설비", "청소", "인테리어"
    ])
    cont: str = Field(default=..., example='대표 블로그를 이용하여 필수 키워드 (알잘딱깔센)를 포함하여 블로그 포스팅을 작성해줘', description='요청사항')

class blogPostData(BaseModel):
    cont: str = Field(default=..., example='대표 블로그를 이용하여 필수 키워드 (알잘딱깔센)를 포함하여 블로그 포스팅을 작성해줘', description='요청사항')

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# @app.post(f"/api/sel-pdfToTxt", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-pdfToTxt")
# async def selPdfToTxt(request: pdfToTxtData = Form(...)):
async def selPdfToTxt(
        cont: str = Form('텍스트 추출', example='텍스트 추출', description='요청사항')
        ,  file: UploadFile = File(None, example=None, description='PDF 파일')
    ):
    """
    기능\n
        PDF 인쇄 파일로부터 텍스트 추출\n
    테스트\n
        cont: 요청사항\n
        file: PDF 인쇄 파일\n
    """

    tmpFileInfo = None

    try:
        if cont == None or len(cont) < 1:
            return resRespone("fail", 400, f"요청사항이 없습니다 ({cont}).", None)

        if file == None:
            return resRespone("fail", 400, f"PDF 파일이 없습니다 ({file}).", None)

        if file.content_type != 'application/pdf':
            return resRespone("fail", 400, "PDF 파일 없음", None)

        log.info(f"[CHECK] cont : {cont}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=globalVar['inpPath']) as tmpFile:
            tmpFile.write(file.file.read())
            tmpFileInfo = tmpFile.name
        log.info(f"[CHECK] tmpFileInfo : {tmpFile.name}")

        pdfFile = genai.upload_file(mime_type=file.content_type, path=tmpFileInfo, display_name=tmpFileInfo)
        res = model.generate_content([cont, pdfFile])
        result = res.candidates[0].content.parts[0].text
        log.info(f"[CHECK] result : {result}")

        return resRespone("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmpFileInfo and os.path.exists(tmpFileInfo):
            os.remove(tmpFileInfo)

# @app.post(f"/api/sel-blogTypePost", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-blogTypePost")
async def selBlogTypePost(request: blogTypePostData = Form(...)):
    """
    기능\n
        분야별 템플릿 및 요청사항을 기반으로 블로그 포스팅 대필\n
    테스트\n
        type: 분야\n
            20241103_13개 분야 별로 대표 템플릿 생성형 AI 4종 결과\n
            https://docs.google.com/spreadsheets/d/1KMVTWiPQ6AZA1F3CqBLv2tw3TvG8kOLcOL3ISnFBQLk/edit?gid=472867053#gid=472867053 \n
        cont: 요청사항\n
    """
    try:
        type = request.type
        if type == None or len(type) < 1:
            return resRespone("fail", 400, f"분야가 없습니다 ({type}).", None)

        cont = request.cont
        if cont == None or len(cont) < 1:
            return resRespone("fail", 400, f"요청사항이 없습니다 ({cont}).", None)

        csvDataL1 = csvData.loc[csvData['분야'] == type]
        if csvDataL1.empty:
            return resRespone("fail", 400, "템플릿 파일 없음", None)

        log.info(f"[CHECK] csvDataL1 : {csvDataL1}")

        contTemplate = csvDataL1['텍스트 추출'].iloc[0]

        contL1 = f"{contTemplate} \n {cont}"
        log.info(f"[CHECK] contL1 : {contL1}")

        res = model.generate_content(contL1)
        result = res.candidates[0].content.parts[0].text
        log.info(f"[CHECK] result : {result}")

        return resRespone("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# @app.post(f"/api/sel-blogPost", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-blogPost")
async def selBlogPost(request: blogPostData = Form(...)):
    """
    기능\n
        요청사항을 기반으로 블로그 포스팅 대필\n
    테스트\n
        cont: 요청사항\n
    """
    try:
        cont = request.cont
        if cont == None or len(cont) < 1:
            return resRespone("fail", 400, f"요청사항이 없습니다 ({cont}).", None)

        res = model.generate_content(cont)
        result = res.candidates[0].content.parts[0].text
        log.info(f"[CHECK] result : {result}")

        return resRespone("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))