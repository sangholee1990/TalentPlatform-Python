import os

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List
import configparser
import pymysql
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



# cd /SYSTEMS/PROG/PYTHON/FAST-API
# cd /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/fast-api
# conda activate py38
# uvicorn TalentPlatform-bdwide-firmware:app --reload --host=0.0.0.0 --port=9998
# http://223.130.134.136:9998/docs

# {
#   "type": "esp-32-fota-https",
#   "version": "2",
#   "host": "192.168.x.xxx",
#   "port": 80,
#   "bin": "/test/http_test.bin"
# }


def getPubliIp():
    response = requests.get('https://api.ipify.org')
    return response.text


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# DB 정보
config = configparser.ConfigParser()
config.read('config/system.cfg', encoding='utf-8')
dbUser = config.get('mysql-clova-dms02', 'user')
dbPwd = quote_plus(config.get('mysql-clova-dms02', 'pwd'))
dbHost = config.get('mysql-clova-dms02', 'host')
dbHost = 'localhost' if dbHost == getPubliIp() else dbHost
dbPort = config.get('mysql-clova-dms02', 'port')
dbName = config.get('mysql-clova-dms02', 'dbName')

# DB 세션
SQLALCHEMY_DATABASE_URL = f'mysql+pymysql://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}'
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 공유 폴더
UPLOAD_PATH = "/DATA/UPLOAD"

app = FastAPI()
Base = declarative_base()
Base.metadata.create_all(bind=engine)


class Firmware(Base):
    __tablename__ = "TB_IOT_FIRM"

    ID = Column(Integer, primary_key=True, index=True)
    TYPE = Column(String(200), index=True)
    VER = Column(String(200), index=True)
    HOST = Column(String(200), index=True)
    PORT = Column(Integer, index=True)
    BIN = Column(String(200), index=True)
    REG_DATE = Column(DateTime, default=datetime.datetime.utcnow)

# class FirmwareBase(BaseModel):
#     TYPE: constr(max_length=200)
#     VER: constr(max_length=200)
#     HOST: constr(max_length=200)
#     PORT: int
    # BIN: str = Field(..., max_length=200)

class FirmwareBase(BaseModel):
    TYPE: str
    VER: str
    HOST: str
    PORT: str

class DownloadResponse(BaseModel):
    filename: str


# @app.post("/firm/")
# def create_firmware(firmware: FirmwareBase, db: Session = Depends(get_db)):
#     try:
#         db_firmware = Firmware(**firmware.dict())
#         db.add(db_firmware)
#         db.commit()
#         db.refresh(db_firmware)
#
#         return db_firmware
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=400, detail="Firmware creation failed")

# @app.post("/firm/upload")
# def create_firmware(firmware: FirmwareBase, BIN: UploadFile = File(...), db: Session = Depends(get_db)):
#     try:
#         # datetime.datetime.now().strftime("%d_%m_%Y %H%M%S")
#         dtDateTime = datetime.datetime.utcnow()
#         # Save the uploaded file
#         file_path = f"{UPLOAD_PATH}/{dtDateTime.strftime('%Y%m/%d/%H')}{BIN.filename}"
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, "wb") as f:
#             f.write(BIN.file.read())
#
#         # Create the firmware object in the database
#         db_firmware = Firmware(**firmware.dict(), REG_DATE=dtDateTime, BIN=file_path)
#         db.add(db_firmware)
#         db.commit()
#         db.refresh(db_firmware)
#
#         return db_firmware
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=400, detail="Firmware creation failed")

# @app.post("/upload/")
# async def upload_file(file: UploadFile = File(...)):
#     try:
#         # file_path = f"/DATA/UPLOAD/{file.filename}"
#         file_path = f"{UPLOAD_PATH}/{file.filename}"
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         return {"filename": file.filename, "content_type": file.content_type}
#     except Exception as e:
#         return {"error": str(e)}


# @app.post("/firm/upload")
# def create_firmware(firmware: FirmwareBase, file: UploadFile = File(...)):
#     print(f"File name from client: {firmware.TYPE}")
#     print(f"File version from client: {firmware.VER}")
#     print(f"Host from client: {firmware.HOST}")
#     print(f"Port from client: {firmware.PORT}")
#     print(f"Uploaded filename: {file.filename}")

@app.post("/firm/upload")
def create_firmware(
    file: UploadFile = File(...),
    TYPE: str = Form(...),
    VER: str = Form(...),
    HOST: str = Form(...),
    PORT: int = Form(...),
):
    print(f"File name from client: {TYPE}")
    print(f"File version from client: {VER}")
    print(f"Host from client: {HOST}")
    print(f"Port from client: {PORT}")
    print(f"Uploaded filename: {file.filename}")

# @app.post("/firm/upload")
# # def create_firmware(firmware: FirmwareBase, db: Session = Depends(get_db)):
# def create_firmware(firmware: FirmwareBase, file: UploadFile = File(...)):
# # def create_firmware(firmware: FirmwareBase, file: UploadFile = File(...), db: Session = Depends(get_db)):
# # def create_firmware(file: UploadFile = File(...), db: Session = Depends(get_db)):
#     # print(FirmwareBase)
#     # print(File(...))
#
#     try:
#         # # datetime.datetime.now().strftime("%d_%m_%Y %H%M%S")
#         # dtDateTime = datetime.datetime.utcnow()
#         # # print(dtDateTime)
#         # #
#         # # Save the uploaded file
#         # file_path = f"{UPLOAD_PATH}/{dtDateTime.strftime('%Y%m/%d/%H')}/{file.filename}"
#         # print(file_path)
#         # os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         # # with open(file_path, "wb") as f:
#         # #     f.write(firmware_file.file.read())
#         # with open(file_path, "wb") as buffer:
#         #     shutil.copyfileobj(file.file, buffer)
#
#         print(f"File name from client: {firmware.TYPE}")
#         print(f"File version from client: {firmware.VER}")
#         print(f"Host from client: {firmware.HOST}")
#         print(f"Port from client: {firmware.PORT}")
#         # print(f"Actual filename: {file.filename}")
#
#         # Create the firmware object in the database
#         # db_firmware = Firmware(**firmware.dict(), REG_DATE=dtDateTime, BIN=file_path)
#         # # db_firmware = Firmware(**firmware.dict(), BIN=file_path)
#         # db.add(db_firmware)
#         # db.commit()
#         # db.refresh(db_firmware)
#
#         # db_firmware = firmware.insert().values(TYPE=firmware.TYPE, VER=firmware.VER, HOST=firmware.HOST, PORT=firmware.PORT, REG_DATE=dtDateTime, BIN=file_path)
#         # db.execute(db_firmware)
#         # db.commit()
#
#         # return db_firmware
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=400, detail="Firmware creation failed")


@app.post("firm/down/")
async def download_file(request: DownloadResponse):
    # file_path = f"/DATA/UPLOAD/{request.filename}"
    file_path = f"{UPLOAD_PATH}/{request.filename}"
    try:
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="application/octet-stream", filename=request.filename)
        else:
            return {"error": "File not found"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/firm/file_info")
def file_info(db: Session = Depends(get_db)):
    return db.query(Firmware).order_by(Firmware.REG_DATE.desc()).limit(1).all()


@app.get("/firm/file_list")
def file_list(db: Session = Depends(get_db)):
    return db.query(Firmware).all()
