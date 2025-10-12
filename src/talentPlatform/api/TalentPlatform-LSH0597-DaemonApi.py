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

# uvicorn TalentPlatform-LSH0597-DaemonApi:app --reload --host=0.0.0.0 --port=9400
# lsof -i :9400 | awk '{print $2}' | xargs kill -9

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
from typing import List, Any, Dict, Optional, Union
import uuid
import subprocess
import google.generativeai as genai
import json
import configparser

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
    'http://localhost:9300',
    'http://49.247.41.71:9300',
    'http://localhost:9400',
    'http://49.247.41.71:9400',
]

app.add_middleware(
    CORSMiddleware
    , allow_origins=oriList
    , allow_credentials=True
    , allow_methods=["*"]
    , allow_headers=["*"]
)

# Gemini API키
config = configparser.ConfigParser()
config.read(sysOpt['cfgFile'], encoding='utf-8')
apiKey = config.get(sysOpt['cfgKey'], sysOpt['cfgVal'])
genai.configure(api_key=apiKey)
model = genai.GenerativeModel('gemini-1.5-pro')
# model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06')

# ============================================
# 비즈니스 로직
# ============================================
class cfgCodeProc(BaseModel):
    lang: str = Query(default=..., description='프로그래밍 언어', example='python3', enum=["python3", "javascript", "c", "java"])
    code: str = Field(default=..., description="코드", example="print('Hello, Python!')")

class cfgCodeDtlProc(BaseModel):
    lang: str = Query(default=..., description='프로그래밍 언어', example="python3", enum=["python3", "c", "java", "javascript"])
    code: str = Field(default=..., description="코드", example='start, end = input().split()\\r\\n\\r\\nfor letter in range(ord(start), ord(end) + 1):\\r\\n    print(chr(letter), end="")\\r\\nprint()\\r\\n')
    inpList: str = Field(default=None, description="입력 목록", example='[["C", "Z"], ["X", "Z"], ["A", "Z"]]')
    expList: str = Field(default=None, description="예상 출력 목록", example='[["CDEFGHIJKLMNOPQRSTUVWXYZ"], ["XYZ"], ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"]]')
    timeOut: Optional[int] = Field(default=5, description="제한 시간 (초)", example=5)

class cfgCodeHelp(BaseModel):
    cont: str = Field(default=..., description='헬프', example='소스코드 ...n표준에러 ...\\n 요청사항 ...')

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

# @app.post(f"/api/sel-codeDtlProc", dependencies=[Depends(chkApiKey)])
@app.post(f"/api/sel-codeDtlProc")
async def codeDtlProc(request: cfgCodeDtlProc = Form(...)):
    """
    기능\n
        프로그래밍 언어 및 소스코드를 통해 코딩 테스트 플랫폼 실행\n
    요청 파라미터\n
        lang 프로그래밍 언어 (python3, c, java, javascript)\n
        inpList 입력 목록 [["C", "Z"], ["X", "Z"], ["A", "Z"]]
            - 입력1 ["C", "Z"] -> C Z
            - 입력2 ["X", "Z"] -> X Z
            - 입력3 ["A", "Z"] -> A Z\n
        expList 예상 출력 목록 [["CDEFGHIJKLMNOPQRSTUVWXYZ"], ["XYZ"], ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"]]
            - 예상 출력1 ["CDEFGHIJKLMNOPQRSTUVWXYZ"] -> CDEFGHIJKLMNOPQRSTUVWXYZ
            - 예상 출력2 ["XYZ"] -> XYZ
            - 예상 출력3 ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"] -> ABCDEFGHIJKLMNOPQRSTUVWXYZ\n
        timeOut 제한 시간 (초)\n
        code 소스코드
            - Escape 문자열 처리
                > 줄바꿈 -> \\r\\n
                > " -> \\\"
                > ' -> \\\'
                > \\n -> \\\\n
                > etc

            - c 샘플코드
                > IDE 편집기
                    #include <stdio.h>

                    int main() {
                        char start, end;

                        scanf("%c %c", &start, &end);

                        for (char letter = start; letter <= end; letter++) {
                            printf("%c", letter);
                        }

                        printf("\\n");
                        return 0;
                    }

                > Escape 문자열 처리
                    #include <stdio.h>\\r\\n\\r\\nint main() {\\r\\n    char start, end;\\r\\n\\r\\n    scanf(\"%c %c\", &start, &end);\\r\\n\\r\\n    for (char letter = start; letter <= end; letter++) {\\r\\n        printf(\"%c\", letter);\\r\\n    }\\r\\n\\r\\n    printf(\"\\\\n\");\\r\\n    return 0;\\r\\n}

            - java 샘플코드
                > IDE 편집기
                    import java.util.Scanner;

                    public class main {
                        public static void main(String[] args) {
                            Scanner scanner = new Scanner(System.in);

                            char start = scanner.next().charAt(0);
                            char end = scanner.next().charAt(0);

                            for (char letter = start; letter <= end; letter++) {
                                System.out.print(letter);
                            }

                            System.out.println();
                            scanner.close();
                        }
                    }

                > Escape 문자열 처리
                    import java.util.Scanner;\\r\\n\\r\\npublic class main {\\r\\n    public static void main(String[] args) {\\r\\n        Scanner scanner = new Scanner(System.in);\\r\\n\\r\\n        char start = scanner.next().charAt(0);\\r\\n        char end = scanner.next().charAt(0);\\r\\n\\r\\n        for (char letter = start; letter <= end; letter++) {\\r\\n            System.out.print(letter);\\r\\n        }\\r\\n\\r\\n        System.out.println();\\r\\n        scanner.close();\\r\\n    }\\r\\n}

            - python3 샘플코드
                > IDE 편집기
                    start, end = input().split()

                    for letter in range(ord(start), ord(end) + 1):
                        print(chr(letter), end="")
                    print()

                > Escape 문자열 처리
                    start, end = input().split()\\r\\n\\r\\nfor letter in range(ord(start), ord(end) + 1):\\r\\n    print(chr(letter), end="")\\r\\nprint()\\r\\n

            - javascript 샘플코드
                > IDE 편집기
                    process.stdin.setEncoding("utf8");

                    process.stdin.on("data", (data) => {
                        const [startChar, endChar] = data.trim().split(" ");

                        for (let letter = startChar.charCodeAt(0); letter <= endChar.charCodeAt(0); letter++) {
                            process.stdout.write(String.fromCharCode(letter));
                        }

                        console.log();
                        process.exit();
                    });

                > Escape 문자열 처리
                    process.stdin.setEncoding("utf8");\\r\\n\\r\\nprocess.stdin.on("data", (data) => {\\r\\n    const [startChar, endChar] = data.trim().split(" ");\\r\\n\\r\\n    for (let letter = startChar.charCodeAt(0); letter <= endChar.charCodeAt(0); letter++) {\\r\\n        process.stdout.write(String.fromCharCode(letter));\\r\\n    }\\r\\n\\r\\n   console.log();\\r\\n    process.exit();\\r\\n});

    응답 결과\n
        설명서
            - status 처리상태 (succ, fail)
            - code HTTP 응답코드 (성공 200, 그 외)
            - message 처리 메시지 (처리 완료, 처리 실패, 기타)
            - cnt 세부결과 개수
            - data 세부결과
                codeRun 코드실행 명령어
                    > file 실행파일 위치
                    > code 실행파일 내용
                    > timeOut 제한시간 (초)
                    > sysInfo 설정정보 (ext 확장자, ver 버전, exe 실행기, cmd 명령어 패턴)

                codeResult 코드실행 결과
                    [
                        {
                            > stdOut 표준출력 (성공 출력결과, 그 외 "")
                            > stdErr 에러출력 (에러 출력결과, 그 외 "")
                            > exitCode 상태코드 (성공 0, 그 외)
                            > cmd 코드실행 시 명령어
                            > inp 코드실행 시 입력 정보
                            > exp 코드실행 시 출력 정보
                            > flag 단위검사 (성공 succ, 실패 fail)
                        },
                        {
                            ...
                        },
                    ]

                codeFlag 코드실행 통합검사 (성공 succ, 실패 fail)

        샘플결과
            {
              "status": "succ",
              "code": 200,
              "message": "처리 완료",
              "cnt": 3,
              "data": {
                "codeRun": {
                  "file": "/DATA/OUTPUT/LSH0597/202502/05/04/41/70b43a31-bea2-46a4-9759-9c60edac1837/main.py",
                  "code": "start, end = input().split() ...",
                  "timeOut": 5,
                  "sysInfo": {
                    "ext": "py",
                    "ver": "Python 3.8.18 & conda 24.5.0",
                    "exe": "/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python3.8",
                    "cmd": "{exe} -O {fileInfo}"
                  }
                },
                "codeResult": [
                  {
                    "stdOut": "CDEFGHIJKLMNOPQRSTUVWXYZ",
                    "stdErr": "",
                    "exitCode": 0,
                    "cmd": "/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python3.8 -O /DATA/OUTPUT/LSH0597/202502/05/04/41/70b43a31-bea2-46a4-9759-9c60edac1837/main.py",
                    "inp": "C Z",
                    "exp": "CDEFGHIJKLMNOPQRSTUVWXYZ",
                    "flag": "succ"
                  },
                  {
                    "stdOut": "XYZ",
                    "stdErr": "",
                    "exitCode": 0,
                    "cmd": "/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python3.8 -O /DATA/OUTPUT/LSH0597/202502/05/04/41/70b43a31-bea2-46a4-9759-9c60edac1837/main.py",
                    "inp": "X Z",
                    "exp": "XYZ",
                    "flag": "succ"
                  },
                  {
                    "stdOut": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                    "stdErr": "",
                    "exitCode": 0,
                    "cmd": "/HDD/SYSTEMS/LIB/anaconda3/envs/py38/bin/python3.8 -O /DATA/OUTPUT/LSH0597/202502/05/04/41/70b43a31-bea2-46a4-9759-9c60edac1837/main.py",
                    "inp": "A Z",
                    "exp": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                    "flag": "succ"
                  }
                ],
                "codeFlag": "succ"
              }
            }
    """

    try:
        # log.info(f"[CHECK] request : {request}")
        lang = request.lang
        if lang is None or len(lang) < 1:
            return resResponse("fail", 400, f"프로그래밍 언어를 확인해주세요 ({lang}).", None)

        code = request.code
        if code is None or len(code) < 1:
            return resResponse("fail", 400, f"소스코드를 확인해주세요 ({code}).", None)

        timeOut = request.timeOut
        if timeOut is None:
            return resResponse("fail", 400, f"소스코드를 확인해주세요 ({timeOut}).", None)

        try:
            inpList = json.loads(request.inpList)
        except Exception:
            inpList = None
        if inpList is None or len(inpList) < 1:
            return resResponse("fail", 400, f"소스코드를 확인해주세요 ({inpList}).", None)
        log.info(f"[CHECK] inpList : {inpList}")

        try:
            expList = json.loads(request.expList)
        except Exception:
            expList = None
        if expList is None or len(expList) < 1:
            return resResponse("fail", 400, f"소스코드를 확인해주세요 ({expList}).", None)
        log.info(f"[CHECK] expList : {expList}")

        sysInfo = sysOpt['code'][lang]
        if sysInfo is None or len(sysInfo) < 1:
            return resResponse("fail", 400, f"설정 정보를 확인해주세요 ({sysInfo}).", None)

        # 임시 파일 생성
        uid = str(uuid.uuid4())
        fileInfo = datetime.now().strftime(sysOpt['tmpPath']).format(outPath=globalVar['outPath'], exe=sysInfo['exe'], uid=uid, ext=sysInfo['ext'])
        log.info(f"[CHECK] fileInfo : {fileInfo}")

        filePath = os.path.dirname(fileInfo)
        if filePath is None or len(filePath) < 1:
            return resResponse("fail", 400, f"임시 파일을 확인해주세요 ({filePath}).", None)
        log.info(f"[CHECK] filePath : {filePath}")

        # 코드 저장
        os.makedirs(filePath, exist_ok=True)
        codeData = code.encode("utf-8").decode("unicode_escape").replace("\r\n", "\n")
        with open(fileInfo, "w") as codeFile:
            codeFile.write(codeData)

        try:
            cmd = sysInfo['cmd'].format(fileInfo=fileInfo, filePath=filePath, exe=sysInfo.get('exe'), cmp=sysInfo.get('cmp'))
        except ValueError as e:
            cmd = None
        if cmd is None or len(cmd) < 1:
            return resResponse("fail", 400, f"코드 실행 명령어를 확인해주세요 ({cmd}).", None)
        log.info(f"[CHECK] cmd : {cmd}")

        codeResList = []
        for i, inpInfo in enumerate(inpList):
            inpInfoPar = ' '.join(inpInfo)
            expInfo = expList[i][0]
            log.info(f"[CHECK] inpInfoPar : {inpInfoPar}")
            log.info(f"[CHECK] expInfo : {expInfo}")

            # 코드 실행
            try:
                codeProcRun = subprocess.run(
                    cmd,
                    input=inpInfoPar,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    shell=True,
                    timeout=timeOut
                )

                stdOut = codeProcRun.stdout.strip()
                stdErr = codeProcRun.stderr.strip()
                exitCode = codeProcRun.returncode

                flag = "succ" if (
                        exitCode == 0 and  # 실행 성공 (exitCode 0)
                        len(stdOut) > 0 and  # 출력이 존재
                        len(stdErr) < 1 and  # 오류 출력 없음
                        stdOut == expInfo  # 예상 출력과 일치
                ) else "fail"

                codeResult = {
                    "stdOut": stdOut,
                    "stdErr": stdErr,
                    "exitCode": exitCode,
                    "cmd": cmd,
                    "inp": inpInfoPar,
                    "exp": expInfo,
                    "flag": flag,
                }

                log.info(f"[CHECK] codeResult : {codeResult}")
                codeResList.append(codeResult)

            except subprocess.TimeoutExpired as e:
                return resResponse("fail", 400, f"제한시간 {timeOut} 초를 초과하였습니다.", None, str(e))
            except Exception as e:
                return resResponse("fail", 400, f"코드 실행을 실패하였습니다.", None, str(e))

            # 최종 결과
            codeFlag = "succ" if all(codeResInfo["flag"] == "succ" for codeResInfo in codeResList) else "fail"

            result = {
                "codeRun": {
                    "file": fileInfo,
                    "code": codeData,
                    "timeOut": timeOut,
                    "sysInfo": sysInfo,
                },
                "codeResult": codeResList,
                "codeFlag": codeFlag,
            }

        if codeFlag == "succ":
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
        chatGPT 요청사항을 기반으로 헬퍼\n
    요청 파라미터\n
        cont 요청사항\n
            - 샘플코드
                > IDE 편집기
                    소스코드
                    print('Hello, Python!')2

                    표준에러
                    File \\"/DATA/OUTPUT/LSH0597/202412/24/18/23/30b03a92-622b-4958-8edc-64820ee75ecb/main.py\\", line 1\\n    print('Hello, Python!')2\\n                           ^nSyntaxError: invalid syntax

                    요청사항
                    소스코드 실행 시 표준에러가 발생되고 있어 이를 수정해조

                > 문자열 처리
                    소스코드\\n             print('Hello, Python!')2\\n              표준에러\\n             File \\"/DATA/OUTPUT/LSH0597/202412/24/18/23/30b03a92-622b-4958-8edc-64820ee75ecb/main.py\\", line 1\\n    print('Hello, Python!')2\\n                           ^\\nSyntaxError: invalid syntax\\n              요청사항\\n             소스코드 실행 시 표준에러가 발생되고 있어 이를 수정해조

    응답 결과\n
        설명서
            - status 처리상태 (succ, fail)
            - code HTTP 응답코드 (성공 200, 그 외)
            - message 처리 메시지 (처리 완료, 처리 실패, 에러 메시지)
            - cnt 세부결과 개수
            - data 세부결과

        샘플결과
            {
              "status": "succ",
              "code": 200,
              "message": "처리 완료",
              "cnt": 520,
              "data": "`print('Hello, Python!')2` 에서 `2`가 문제입니다.  `print()` 함수 호출 뒤에 붙은 `2`는 파이썬 인터프리터가 이해할 수 없는 구문입니다.  아마도 실수로 입력되었을 가능성이 큽니다.\\n\\n다음과 같이 수정하면 됩니다.\\n\\n```python\\nprint('Hello, Python!')\\n```\\n\\n`2`를 제거하고 `print()` 함수만 남겨두면 \\"Hello, Python!\\"이 정상적으로 출력됩니다.\\n\\n\\n만약 2를 곱하기 연산으로 사용하려고 했다면, 문자열과 숫자는 직접 곱할 수 없습니다.  문자열을 반복하려면 다음과 같이 수정해야 합니다.\\n\\n```python\\nprint('Hello, Python!' * 2)\\n```\\n\\n이렇게 수정하면 \\"Hello, Python!Hello, Python!\\"이 출력됩니다.\\n\\n\\n어떤 의도였는지에 따라  `2`를 삭제하거나 곱셈 연산자 `*`를 추가하는 두 가지 방법 중 하나를 선택해야 합니다.  대부분의 경우, 단순히 `2`를 삭제하는 것이 의도에 맞을 것입니다.\\n"
            }
    """
    try:
        cont = request.cont
        if cont is None or len(cont) < 1:
            return resResponse("fail", 400, f"요청사항이 없습니다 ({cont}).", None)

        res = model.generate_content(cont)
        result = res.candidates[0].content.parts[0].text
        log.info(f"[CHECK] result : {result}")

        return resResponse("succ", 200, "처리 완료", len(result), result)

    except Exception as e:
        log.error(f'Exception : {e}')
        raise HTTPException(status_code=400, detail=str(e))


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