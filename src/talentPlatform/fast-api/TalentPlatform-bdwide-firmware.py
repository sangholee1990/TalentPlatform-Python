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

# cd /SYSTEMS/PROG/PYTHON/FAST-API
# cd /SYSTEMS/PROG/PYTHON/PyCharm/src/talentPlatform/fast-api
# conda activate py38
# uvicorn TalentPlatform-bdwide-firmware:app --reload --host=0.0.0.0 --port=9998
# http://223.130.134.136:9998

# {
#   "type": "esp-32-fota-https",
#   "version": "2",
#   "host": "192.168.x.xxx",
#   "port": 80,
#   "bin": "/test/http_test.bin"
# }

# DB 연결 정보
pymysql.install_as_MySQLdb()

# DB 정보
config = configparser.ConfigParser()
config.read('config/system.cfg', encoding='utf-8')
# dbUser = config.get('mysql-clova-dms02', 'user')
# dbPwd = config.get('mysql-clova-dms02', 'pwd')
# dbHost = config.get('mysql-clova-dms02', 'host')
# dbPort = config.get('mysql-clova-dms02', 'port')
# dbName = config.get('mysql-clova-dms02', 'dbName')

dbUser = config.get('mysql-dms05', 'user')
dbPwd = config.get('mysql-dms05', 'pwd')
dbHost = config.get('mysql-dms05', 'host')
dbPort = config.get('mysql-dms05', 'port')
dbName = config.get('mysql-dms05', 'dbName')

# SQLALCHEMY_DATABASE_URL = 'mysql://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName)
# SQLALCHEMY_DATABASE_URL = 'mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(dbUser, dbPwd, dbHost, dbPort, dbName)
SQLALCHEMY_DATABASE_URL = f'mysql+pymysql://{dbUser}:{quote_plus(dbPwd)}@{dbHost}:{dbPort}/{dbName}'
# SQLALCHEMY_DATABASE_URL = f'mysql+pymysql://{dbUser}:{quote_plus(dbPwd)}@{dbHost}:{dbPort}/{dbName}'
# SQLALCHEMY_DATABASE_URL = f'mysql://{dbUser}:{quote_plus(dbPwd)}@{dbHost}:{dbPort}/{dbName}'
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Firmware(Base):
    __tablename__ = "firmwares"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(50), index=True)
    version = Column(String(50), index=True)
    host = Column(String(200), index=True)
    port = Column(Integer, index=True)
    bin = Column(String(200), index=True)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)

class FirmwareBase(BaseModel):
    type: str = Field(..., max_length=50)
    version: str = Field(..., max_length=50)
    host: str = Field(..., max_length=200)
    port: int
    bin: str = Field(..., max_length=200)

@app.post("/firmwares/")
def create_firmware(firmware: FirmwareBase, db: Session = Depends(get_db)):
    try:
        db_firmware = Firmware(**firmware.dict())
        db.add(db_firmware)
        db.commit()
        db.refresh(db_firmware)
        return db_firmware
    except Exception as e:
        raise HTTPException(status_code=400, detail="Firmware creation failed")

@app.get("/firmwares/recent/")
def read_recent_firmwares(db: Session = Depends(get_db)):
    return db.query(Firmware).order_by(Firmware.upload_time.desc()).limit(5).all()

@app.get("/firmwares/all/")
def read_all_firmwares(db: Session = Depends(get_db)):
    return  db.query(Firmware).all()