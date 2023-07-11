from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import subprocess
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Dict, List
from fastapi.staticfiles import StaticFiles
import subprocess
import matplotlib.pyplot as plt
from fastapi import FastAPI
from io import BytesIO
from starlette.responses import StreamingResponse
import numpy as np
from typing import List, Dict, Union
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import glob
import re
from fastapi.middleware.cors import CORSMiddleware

# from fastapi.security import OAuth2PasswordBearer
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

origins = [
    "http://localhost:8080",
    "http://localhost:9000",
    "http://localhost:9100",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # 허용되는 출처 목록
    allow_credentials=True,
    allow_methods=["*"], # 허용되는 HTTP 메서드
    allow_headers=["*"]  # 허용되는 HTTP 헤더
)

UPLOAD_PATH = "/DATA/UPLOAD" 

# 공유폴더
#app.mount('/UPLOAD', StaticFiles(directory='/DATA/UPLOAD'), name='/DATA/UPLOAD')
app.mount('/UPLOAD', StaticFiles(directory=UPLOAD_PATH), name='/DATA/UPLOAD')


def makePlot():
    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    cos, sin = np.cos(x), np.sin(x)

    # 출력된 각 줄의 길이를 그래프로 그립니다.
    plt.figure(figsize=(10, 6))
    plt.plot(x, cos, color="blue", linewidth=2.5, linestyle="-", label="cosine")
    plt.plot(x, sin, color="red", linewidth=2.5, linestyle="-", label="sine")

    plt.legend(loc='upper left', frameon=False)

    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

    plt.yticks([-1, 0, +1],
               [r'$-1$', r'$0$', r'$+1$'])

    # 그래프를 이미지로 저장하고 그것을 바이트로 변환합니다.
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf

def cmdProc(args):
    cmd = []
    for section in ['cmd', 'input', 'output']:
        for key, value in args[section].items():
            if section != 'cmd':
                cmd.extend([f"--{key}", value])
            else:
                cmd.extend([f'{value}'])
    return ' '.join(cmd)

class DownloadResponse(BaseModel):
    filename: str


class ScriptRequest(BaseModel):
    input_param: str


class ScriptResponse(BaseModel):
    output_param: str

@app.get("/file_list", response_model=List[Dict[str, Union[str, float]]])
async def get_file_list():
    try:
        pathlist = glob.glob(UPLOAD_PATH + '/*.*')
        files = []
        for path in pathlist:
            if not re.search(r"\.(zip|rar)$", path, re.I):
                size_MB = round(os.path.getsize(path) / (1024.0 * 1024.0), 2)
                files.append({"name": os.path.basename(path), "size": f"{size_MB} MB"})

        return files
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        #file_path = f"/DATA/UPLOAD/{file.filename}"
        file_path = f"{UPLOAD_PATH}/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "content_type": file.content_type}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload_multiple/")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    errors = []
    for file in files:
        try:
            #file_path = f"/DATA/UPLOAD/{file.filename}"
            file_path = f"{UPLOAD_PATH}/{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append({"filename": file.filename, "content_type": file.content_type})
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
    return {"uploaded": uploaded_files, "errors": errors}

@app.post("/download/")
async def download_file(request: DownloadResponse):
    #file_path = f"/DATA/UPLOAD/{request.filename}"
    file_path = f"{UPLOAD_PATH}/{request.filename}"
    try:
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="application/octet-stream", filename=request.filename)
        else:
            return {"error": "File not found"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/run_cmd")
# async def run_script(script_path: str, args: Dict[str, Dict[str, str]]):
async def run_cmd(args: Dict[str, Dict[str, str]]):
    try:
        runCmd = cmdProc(args)
        print(f'runCmd : {runCmd}')

        result = subprocess.run(runCmd, shell=True, capture_output=True, text=True, check=True, encoding='utf-8')
        output = result.stdout
        error = result.stderr

        return {"output": output, "error": error}
    except Exception as e:
        return {"error": str(e)}

@app.get("/make_plot")
async def make_plot():
    try:
        # 그래프 생성
        buf = makePlot()

        # 그래프를 이미지로 반환합니다.
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}

