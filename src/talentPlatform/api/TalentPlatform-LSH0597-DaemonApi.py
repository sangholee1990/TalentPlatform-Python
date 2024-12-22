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
# nohup uvicorn TalentPlatform-LSH0597-DaemonApi:app --host=0.0.0.0 --port=9300 &
# tail -f nohup.out

# 프로그램 종료
# ps -ef | grep "TalentPlatform-LSH0597-DaemonApi" | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9300
# lsof -i :9300 | awk '{print $2}' | xargs kill -9

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
import google.generativeai as genai

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
    if api_key != '20241220-topbds':
        raise HTTPException(status_code=400, detail="API 인증 실패")

def resResponse(status: str, code: int, message: str, cnt: int = 0, data: Any = None) -> dict:
    return {
        "status": status,
        "code": code,
        "message": message,
        "cnt": cnt,
        "data": data,
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

    # 제한 시간
    'timeOut': 10,

    # 실행 정보
    'code': {
        'c': {
            'ext': 'c',
            'ver': 'gcc (GCC) 11.5.0 20240719 (Red Hat 11.5.0-2)',
            'exe': '/usr/bin/gcc',
            'cmd': '{exe} {fileInfo} -o {filePath}/a.out -O2 && {filePath}/a.out',
        },
        'java': {
            'ext': 'java',
            'ver': 'java 20.0.2 2023-07-18',
            'cmp': '/usr/bin/javac',
            'exe': '/usr/bin/java',
            'cmd': '{cmp} -g:none -O {fileInfo} -d {filePath} && {exe} -Xmx512m -XX:+UseG1GC -cp {filePath} main',
        },
        'python3': {
            'ext': 'py',
            'ver': 'Python 3.8.18 & conda 24.5.0',
            'exe': '/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python3.8',
            'cmd': '{exe} -O {fileInfo}',
        },
        'javascript': {
            'ext': 'js',
            'ver': 'v20.17.0',
            'exe': '/usr/bin/node',
            'cmd': '{exe} --optimize_for_size {fileInfo}',
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

genai.configure(api_key=None)
model = genai.GenerativeModel('gemini-1.5-pro')

# ============================================
# 비즈니스 로직
# ============================================
class cfgCodeProc(BaseModel):
    lang: str = Query(default=..., description='프로그래밍 언어', example='python3', enum=[
        "python3", "c", "java", "javascript"
    ])
    code: str = Field(default=..., description="코드", example="print('Hello, Python!')")

class cfgCodeHelp(BaseModel):
    cont: str = Field(default=..., description='헬프', example='코드를 수정해줘')

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

# @app.post(f"/api/sel-codeProc", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-codeProc")
async def selCodeProc(request: cfgCodeProc = Form(...)):
    """
    기능\n
        프로그래밍 언어 및 소스코드를 통해 코딩 테스트 플랫폼 실행\n
    요청 파라미터\n
        lang 프로그래밍 언어 (c, java, python3, javascript)\n
        code 소스코드\n
            - Escape 문자열 처리
                > # 줄바꿈 -> \\r\\n
                > " -> \\\"
                > ' -> \\\'
                > \\n -> \\\\n
                > etc

            - c 샘플코드
                > IDE 편집기
                    #include<stdio.h>
                    int main() {
                        char letter;

                        for (letter = 'A'; letter <= 'Z'; letter++) {
                            printf("%c", letter);
                        }

                        printf('\\n');
                        return 0;
                    }

                > Escape 문자열 처리
                    #include <stdio.h>\\r\\nint main() {\\r\\n    char letter;\\r\\n\\r\\n    for (letter = 'A'; letter <= 'Z'; letter++) {\\r\\n        printf("\\%c", letter);\\r\\n    }\\r\\n\\r\\n    printf("\\\\n");\\r\\n    return 0;\\r\\n}\\r\\n

            - java 샘플코드
                > IDE 편집기
                    public class main {
                        public static void main(String[] args) {
                            for (char letter = 'A'; letter <= 'Z'; letter++) {
                                System.out.print(letter);
                            }

                            System.out.println();
                        }
                    }

                > Escape 문자열 처리
                    public class main {\\r\\n    public static void main(String[] args) {\\r\\n        for (char letter = 'A'; letter <= 'Z'; letter++) {\\r\\n            System.out.print(letter);\\r\\n        }\\r\\n        System.out.println();\\r\\n    }\\r\\n}

            - python3 샘플코드
                > IDE 편집기
                    for letter in range(ord('A'), ord('Z') + 1):
                        print(chr(letter), end='')
                    print()


                > Escape 문자열 처리
                    for letter in range(ord('A'), ord('Z') + 1):\\r\\n    print(chr(letter), end='')\\r\\nprint()\\r\\n

            - javascript 샘플코드
                > IDE 편집기
                    for (let letter = 'A'.charCodeAt(0); letter <= 'Z'.charCodeAt(0); letter++) {
                        process.stdout.write(String.fromCharCode(letter));
                    }

                > Escape 문자열 처리
                    for (let letter = 'A'.charCodeAt(0); letter <= 'Z'.charCodeAt(0); letter++) {\\r\\n    process.stdout.write(String.fromCharCode(letter));\\r\\n}\\r\\n

    응답 결과\n
        설명서
            - status 처리상태 (succ, fail)
            - code HTTP 응답코드 (성공 200, 그 외)
            - message 처리 메시지 (처리 완료, 처리 실패, 에러 메시지)
            - cnt 세부결과 개수
            - data 세부결과
                > file 실행파일 위치
                > code 실행파일 내용
                > sysInfo 설정정보 (ext 확장자, ver 버전, exe 실행기, cmd 명령어)
                > stdOut 코드 실행 시 표준출력 (성공 출력결과, 그 외 "")
                > stdErr 코드 실행 시 에러출력 (에러 출력결과, 그 외 "")
                > exitCode 코드 실행 시 상태코드 (성공 0, 그 외)

        샘플결과
            {
              "status": "succ",
              "code": 200,
              "message": "처리 완료",
              "cnt": 6,
              "data": {
                "file": "/DATA/OUTPUT/LSH0597/202412/20/11/40/6d403d14-9efd-4fde-8054-f3b4fadd585d/main.py",
                "code": "print('Hello, Python!')",
                "sysInfo": {
                  "ext": "py",
                  "ver": "Python 3.8.18 & conda 24.5.0",
                  "exe": "/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python3.8",
                  "cmd": "{exe} -O {fileInfo}"
                },
                "stdOut": "Hello, Python!",
                "stdErr": "",
                "exitCode": 0
              }
            }

    """

    try:
        lang = request.lang
        if lang is None or len(lang) < 1:
            return resResponse("fail", 400, f"프로그래밍 언어를 확인해주세요 ({lang}).", None)

        code = request.code
        if code is None or len(code) < 1:
            return resResponse("fail", 400, f"소스코드를 확인해주세요 ({code}).", None)

        sysInfo = sysOpt['code'][lang]
        if sysInfo is None or len(sysInfo) < 1:
            return resResponse("fail", 400, f"설정 정보를 확인해주세요 ({sysInfo}).", None)

        # 임시 파일 생성
        uid = str(uuid.uuid4())
        fileInfo = datetime.now().strftime(sysOpt['tmpPath']).format(outPath=globalVar['outPath'], exe=sysInfo['exe'], uid=uid, ext=sysInfo['ext'])
        log.info(f"[CHECK] fileInfo : {fileInfo}")

        filePath = os.path.dirname(fileInfo)
        log.info(f"[CHECK] filePath : {filePath}")

        cmd = None
        try:
            cmd = sysInfo['cmd'].format(fileInfo=fileInfo, filePath=filePath, exe=sysInfo.get('exe'), cmp=sysInfo.get('cmp'))
        except ValueError as e:
            return resResponse("fail", 400, f"실행 명령어를 확인해주세요 ({cmd}).", None, str(e))
        log.info(f"[CHECK] cmd : {cmd}")

        # 코드 저장
        os.makedirs(filePath, exist_ok=True)
        codeData = code.encode("utf-8").decode("unicode_escape").replace("\r\n", "\n")
        with open(fileInfo, "w") as codeFile:
            codeFile.write(codeData)

        # 코드 실행
        try:
            codeProcRun = subprocess.run(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                shell=True,
                timeout=sysOpt['timeOut']
            )
        except subprocess.TimeoutExpired as e:
            return resResponse("fail", 400, f"제한시간 {sysOpt['timeOut']} 초를 초과하였습니다.", None, str(e))

        result = {
            "file": fileInfo,
            "code": codeData,
            "sysInfo": sysInfo,
            "stdOut": codeProcRun.stdout.strip(),
            "stdErr": codeProcRun.stderr.strip(),
            "exitCode": codeProcRun.returncode
        }
        log.info(f"[CHECK] result : {result}")

        if result['exitCode'] == 0 and len(result['stdOut']) > 0 and len(result['stdErr']) < 1:
            return resResponse("succ", 200, "처리 완료", len(result), result)
        else:
            return resResponse("fail", 400, "처리 실패", None, result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if lang == "c" and os.path.exists(os.path.join(filePath, "a.out")):
            os.remove(os.path.join(filePath, "a.out"))
        if lang == "java" and os.path.exists(os.path.join(filePath, "main.class")):
            os.remove(os.path.join(filePath, "main.class"))

# @app.post(f"/api/sel-codeHelp", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-codeHelp")
async def selCodeHelp(request: cfgCodeHelp = Form(...)):
    """
    기능\n
        소스코드 및 요청사항을 기반으로 헬퍼\n
    테스트\n
        cont: 요청사항\n
    """
    try:
        cont = request.cont
        if cont == None or len(cont) < 1:
            return resResponse("fail", 400, f"요청사항이 없습니다 ({cont}).", None)

        res = model.generate_content(cont)
        result = res.candidates[0].content.parts[0].text
        log.info(f"[CHECK] result : {result}")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))