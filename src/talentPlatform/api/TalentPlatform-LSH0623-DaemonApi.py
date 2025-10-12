# ============================================
# 요구사항
# ============================================
# LSH0623. Python을 이용한 AI 기반 랜딩 페이지 제작 및 API 연계 서비스

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api
# conda activate py39

# 운영 서버
# uvicorn TalentPlatform-LSH0623-DaemonApi:app --host=0.0.0.0 --port=9400
# nohup uvicorn TalentPlatform-LSH0623-DaemonApi:app --host=0.0.0.0 --port=9400 &
# tail -f nohup.out

# 테스트 서버
# uvicorn TalentPlatform-LSH0623-DaemonApi:app --reload --host=0.0.0.0 --port=9400

# 프로그램 종료
# ps -ef | grep "TalentPlatform-LSH0623-DaemonApi" | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9400
# lsof -i :9400 | awk '{print $2}' | xargs kill -9

# 명세1) http://49.247.41.71:9400/docs
# 명세2) http://49.247.41.71:9400/redoc

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
    if api_key != '20250811-topbds':
        raise HTTPException(status_code=400, detail="API 인증 실패")

def resResponse(status: str, code: int, message: str, cnt: Any = None, data: Any = None) -> dict:
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
serviceName = 'LSH0623'
prjName = 'test'

ctxPath = os.getcwd()
# ctxPath = f"/SYSTEMS/PROG/PYTHON/IDE"

log = initLog(env, ctxPath, prjName)

# 작업 경로 설정
# os.chdir(f"{ctxPath}")
# log.info(f"[CHECK] getcwd : {os.getcwd()}")

# 옵션 설정
sysOpt = {
    # CORS 설정
    'oriList': ['*'],
    'imageUrl': f'https://pixabay.com/api/?key=14824120-f852820b33507a12ff4cdd8bf&q=%keyword%&image_type=photo&pretty=true&lang=kr&per_page=200',
    'videoUrl': f'https://pixabay.com/api/videos/?key=14824120-f852820b33507a12ff4cdd8bf&q=%keyword%&pretty=true&lang=kr&per_page=200',

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

# ============================================
# 비즈니스 로직
# ============================================
class modelTypeContData(BaseModel):
    model: str = Query(default=..., description='생성형 AI 종류', example='gemini-2.5-pro', enum=['gemini-2.5-pro', 'gemini-1.5-flash-latest'])
    type: str = Query(default=..., description='축제', example='gemini-2.5-pro', enum=['축제', '행사', '광고'])
    cont: str = Field(default=..., example='''
이벤트명: 국립해양생물자원관 개관10주년 캠페인 (한산모시축제)
이벤트 유형: 축제
이벤트 개요: 이벤트 진행, 사은품 제공, 부스 운영
날짜: 2025년 6월 13일 (금) - 6월 15일 (일)
장소: 국립해양생물자원관
원하는 분위기: 밝음
    ''', description='요청사항')

class contData(BaseModel):
    cont: str = Field(default=..., example='없음', description='요청사항')

class imageData(BaseModel):
    keyword: str = Field(default=..., example='자연+바다', description='검색어')

class videoData(BaseModel):
    keyword: str = Field(default=..., example='자연+바다', description='검색어')

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# @app.post(f"/api/sel-blogTypePost", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-modelTypeContData")
async def selModelTypeContData(request: modelTypeContData = Form(...)):
    """
    기능\n
        생성형 AI 종류 및 분야별 랜딩 페이지 생성\n
    테스트\n
        model: 생성형 AI 종류\n
        type: 분야 (축제)\n
        cont: 요청사항\n
            이벤트명: 국립해양생물자원관 개관10주년 캠페인 (한산모시축제)\n
            이벤트 유형: 축제\n
            이벤트 개요: 이벤트 진행, 사은품 제공, 부스 운영\n
            날짜: 2025년 6월 13일 (금) - 6월 15일 (일)\n
            장소: 국립해양생물자원관\n
            원하는 분위기: 밝음\n
    """
    try:
        model = request.model
        if model is None or len(model) < 1:
            return resResponse("fail", 400, f"생성형 AI 모델 종류가 없습니다 : {model}")

        type = request.type
        if type is None or len(type) < 1:
            return resResponse("fail", 400, f"분야가 없습니다 : {type}")

        cont = request.cont
        if cont is None or len(cont) < 1:
            return resResponse("fail", 400, f"요청사항이 없습니다 : {cont}")

        contTemplate = '''%type% 랜딩 페이지를 html 파일로 제작해조
            요청사항
                %cont%
            고려사항
                스타일링 Tailwind CSS의 CDN 버전을 사용하여 전체적인 디자인을 구현
                에디터 TinyMCE 6의 커뮤니티 버전(CDN)을 사용하여 모든 콘텐츠 편집 기능을 구현
                이미지 pixabay 사용
            핵심기능
                편집모드 (TinyMCE 에디터 툴바를 통해 색상, 폰트 등 즉시 수정 지원)
                    <script src="https://cdn.jsdelivr.net/npm/tinymce@6/tinymce.min.js"></script>
                    tinymce.init({
                        selector: '.editable',
                        inline: true, // 인라인 모드 활성화 (핵심 기능)
                        plugins: 'lists link image table code help wordcount autoresize',
                        toolbar: 'undo redo | blocks fontfamily fontsize | bold italic underline strikethrough forecolor backcolor | alignleft aligncenter alignright alignjustify | bullist numlist outdent indent | link image media table hr charmap emoticons codesample | searchreplace visualblocks code fullscreen | removeformat | help',
                        menubar: false, // 상단 메뉴바 비활성화
                        language: 'ko_KR', // 언어 설정 (필요 시 cdn에서 추가 로드 필요)
                        autoresize_bottom_margin: 20,
                        automatic_uploads: true,
                        image_title: true,
                        file_picker_types: 'image',
                          file_picker_callback: (cb, value, meta) => {
                            const input = document.createElement('input');
                            input.setAttribute('type', 'file');
                            input.setAttribute('accept', 'image/*');
        
                            input.addEventListener('change', (e) => {
                                const file = e.target.files[0];
        
                                const reader = new FileReader();
                                reader.addEventListener('load', () => {
                                    // TinyMCE의 이미지 캐시에 파일을 등록합니다.
                                    const id = 'blobid' + (new Date()).getTime();
                                    const blobCache =  tinymce.activeEditor.editorUpload.blobCache;
                                    const base64 = reader.result.split(',')[1];
                                    const blobInfo = blobCache.create(id, file, base64);
                                    blobCache.add(blobInfo);
        
                                    // 콜백 함수를 호출하여 에디터에 이미지를 삽입합니다.
                                    cb(blobInfo.blobUri(), { title: file.name });
                                });
                                reader.readAsDataURL(file);
                            });
        
                            input.click();
                        },
                    });
                미리보기
           '''

        contents = contTemplate.replace('%type%', type).replace('%cont%', cont)
        log.info(f"[CHECK] contents : {contents}")

        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=contents
        )
        result = response.text
        log.info(f"[CHECK] result : {result}")

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

# @app.post(f"/api/sel-blogPost", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-contData")
async def selContData(request: contData = Form(...)):
    """
    기능\n
        요청사항을 기반으로 문의사항\n
    테스트\n
        cont: 요청사항\n
    """
    try:
        cont = request.cont
        if cont is None or len(cont) < 1:
            return resResponse("fail", 400, f"요청사항이 없습니다 : {cont}")

        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=cont
        )

        result = response.text
        # log.info(f"[CHECK] result : {result}")

        if result is None or len(result) < 1:
            return resResponse("fail", 400, "처리 실패")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# @app.post(f"/api/sel-blogPost", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-contData")
async def selContData(request: contData = Form(...)):
    """
    기능\n
        요청사항을 기반으로 문의사항\n
    테스트\n
        cont: 요청사항\n
    """
    try:
        cont = request.cont
        if cont is None or len(cont) < 1:
            return resResponse("fail", 400, f"요청사항이 없습니다 : {cont}")

        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=cont
        )

        result = response.text
        # log.info(f"[CHECK] result : {result}")

        if result is None or len(result) < 1:
            return resResponse("fail", 400, "처리 실패")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# @app.post(f"/api/sel-imageData", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-imageData")
async def selImageData(request: imageData = Form(...)):
    """
    기능\n
        검색어를 기반으로 오픈소스 이미지 가져오기\n
    테스트\n
        keyword: 검색어\n
    """
    try:
        keyword = request.keyword
        if keyword is None or len(keyword) < 1:
            return resResponse("fail", 400, f"검색어가 없습니다 : {keyword}")

        imageUrl = sysOpt['imageUrl'].replace('%keyword%', keyword)
        response = requests.get(imageUrl)
        response.raise_for_status()
        result = response.json()
        # log.info(f"[CHECK] result : {result}")

        if result is None or len(result) < 1:
            return resResponse("fail", 400, "처리 실패")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))

# @app.post(f"/api/sel-videoData", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-videoData")
async def selVideoData(request: videoData = Form(...)):
    """
    기능\n
        검색어를 기반으로 오픈소스 영상 가져오기\n
    테스트\n
        keyword: 검색어\n
    """
    try:
        keyword = request.keyword
        if keyword is None or len(keyword) < 1:
            return resResponse("fail", 400, f"검색어가 없습니다 : {keyword}")

        imageUrl = sysOpt['videoUrl'].replace('%keyword%', keyword)
        response = requests.get(imageUrl)
        response.raise_for_status()
        result = response.json()
        # log.info(f"[CHECK] result : {result}")

        if result is None or len(result) < 1:
            return resResponse("fail", 400, "처리 실패")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))