# ============================================
# 요구사항
# ============================================
# LSH0597. Python을 이용한 코딩 테스트 플랫폼 개발 (인텔마이티 이러닝 프로젝트)

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/talentPlatform/api
# conda activate py39
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
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import configparser
import os
from urllib.parse import quote_plus
import requests
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
import os
import shutil
# from datetime import datetime
from pydantic import BaseModel, Field, constr, validator
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
import re
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
from typing import List, Any, Dict, Optional
import uuid
import subprocess

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

sysOpt = {
    # 임시 경로
    'tmpPath': '{outPath}/%Y%m/%d/%H/%M/{uid}/main.{ext}',

    # 실행 시간
    'timeOut': 10,

    # 실행 정보
    'code': {
        'c': {
            'ext': 'c',
            'exe': '/usr/bin/gcc',
            'cmd': '{exe} {fileInfo} -o {filePath}/a.out && {filePath}/a.out',
        },
        'java': {
            'ext': 'java',
            'cmp': '/usr/bin/javac',
            'exe': '/usr/bin/java',
            'cmd': '{cmp} {fileInfo} && {exe} -cp {filePath} main',
        },
        'python3': {
            'ext': 'py',
            'exe': '/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python3.8',
            'cmd': '{exe} {fileInfo}',
        },
        'javascript': {
            'ext': 'js',
            'exe': '/usr/bin/node',
            'cmd': '{exe} {fileInfo}',
        },
    },
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
class cfgCodeProc(BaseModel):
    lang: str = Query(default=..., description='프로그래밍 언어', example='python3', enum=[
        "python3", "c", "java", "javascript"
    ])
    code: str = Field(default=..., description="코드", example="print('Hello, Python!')")

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

# @app.post(f"/api/sel-pdfToTxt", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-codeProc")
async def selCodeProc(request: cfgCodeProc = Form(...)):
    """
    기능\n
        프로그래밍 언어 및 소스코드를 통해 코딩 테스트 플랫폼 실행\n
    테스트\n
        lang: 프로그래밍 언어 (python3, javascript, java, c)\n
        code: 소스코드\n
    """

    try:
        lang = request.lang
        if lang is None or len(lang) < 1:
            raise HTTPException(status_code=400, detail=resRespone("fail", 400, f"프로그래밍 언어를 확인해주세요 ({lang}).", None))

        code = request.code
        if code is None or len(code) < 1:
            raise HTTPException(status_code=400, detail=resRespone("fail", 400, f"소스코드를 확인해주세요 ({code}).", None))

        sysInfo = sysOpt['code'][lang]
        if sysInfo is None or len(sysInfo) < 1:
            raise HTTPException(status_code=400, detail=resRespone("fail", 400, f"설정 정보를 확인해주세요 ({sysInfo}).", None))

        # 임시 파일 생성
        uid = str(uuid.uuid4())
        fileInfo = datetime.now().strftime(sysOpt['tmpPath']).format(outPath=globalVar['outPath'], exe=sysInfo['exe'], uid=uid, ext=sysInfo['ext'])
        log.info(f"[CHECK] fileInfo : {fileInfo}")

        filePath = os.path.dirname(fileInfo)
        log.info(f"[CHECK] filePath : {filePath}")

        cmd = None
        # if re.search('c', lang, re.IGNORECASE):
        #     cmd = sysInfo['cmd'].format(exe=sysInfo['exe'], fileInfo=fileInfo, filePath=filePath)
        # elif re.search('java', lang, re.IGNORECASE):
        #     cmd = sysInfo['cmd'].format(cmp=sysInfo['cmp'], exe=sysInfo['exe'], fileInfo=fileInfo, filePath=filePath)
        # elif re.search('python', lang, re.IGNORECASE):
        #     cmd = sysInfo['cmd'].format(exe=sysInfo['exe'], fileInfo=fileInfo)
        # elif re.search('javascript', lang, re.IGNORECASE):
        #     cmd = sysInfo['cmd'].format(exe=sysInfo['exe'], fileInfo=fileInfo)
        # if cmd is None or len(cmd) < 1:
        #     raise HTTPException(status_code=400, detail=resRespone("fail", 400, f"실행 명령어를 확인해주세요 ({cmd}).", None))

        try:
            cmd = sysInfo['cmd'].format(
                fileInfo=fileInfo, filePath=filePath, exe=sysInfo.get('exe'), cmp=sysInfo.get('cmp')
            )
        except KeyError:
            raise HTTPException(status_code=400, detail=resRespone("fail", 400, f"실행 명령어를 확인해주세요 ({cmd}).", None))
        log.info(f"[CHECK] cmd : {cmd}")



        # 코드 저장
        os.makedirs(filePath, exist_ok=True)
        with open(fileInfo, "w") as codeFile:
            codeFile.write(code.encode("utf-8").decode("unicode_escape").replace("\r\n", "\n"))

        # 코드 실행
        codeProcRun = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            shell=True,
            timeout=sysOpt['timeOut']
        )

        result = {
            "stdOut": codeProcRun.stdout.strip(),
            "stdErr": codeProcRun.stderr.strip(),
            "exitCode": codeProcRun.returncode
        }
        log.info(f"[CHECK] result : {result}")

        if result['exitCode'] == 0 and len(result['stdOut']) > 0 and len(result['stdErr']) < 1:
            return resRespone("succ", 200, "처리 완료", len(result), result)
        else:
            return resRespone("fail", 400, "처리 실패", None, result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=resRespone("fail", 400, "처리 실패",None, str(e)))
    finally:
        if lang == "c" and os.path.exists(os.path.join(filePath, "a.out")):
            os.remove(os.path.join(filePath, "a.out"))
        if lang == "java" and os.path.exists(os.path.join(filePath, "main.class")):
            os.remove(os.path.join(filePath, "main.class"))


