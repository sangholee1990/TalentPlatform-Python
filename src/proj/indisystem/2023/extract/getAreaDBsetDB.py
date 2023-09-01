# -*- coding: utf-8 -*-
import sys

import pandas as pd
import psycopg2
import re
import yaml
import os
from sqlalchemy import create_engine
import numpy as np
from urllib.parse import quote_plus
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import common.initiator as common
from sqlalchemy import create_engine, text

# ===========================================================
# 실행 방법
# ===========================================================
# /home/guest_user1/SYSTEMS/KIER/LIB/py38/bin/python3 /home/guest_user1/SYSTEMS/KIER/PROG/PYTHON/extract/getAreaDBsetDB.py

# ===========================================================
# 주요 함수
# ===========================================================
def convFloatToIntList(val):
    scaleFactor = 10000
    addOffset = 0
    return ((np.around(val, 4) * scaleFactor) - addOffset).astype(int).tolist()

def invIntToFloat(val):
    scaleFactor = 10000
    addOffset = 0
    return (np.array(val) + addOffset) / scaleFactor

def dbMergeData(session, table, dataList, pkList=['MODEL_TYPE']):
    try:
        stmt = insert(table)
        setData = {key: getattr(stmt.excluded, key) for key in dataList.keys()}
        onConflictStmt = stmt.on_conflict_do_update(
            index_elements=pkList
            , set_=setData
        )
        session.execute(onConflictStmt, dataList)
        session.commit()

    except Exception as e:
        session.rollback()
        print(f'Exception : {e}')

    finally:
        session.close()

# ===========================================================
# 입력 정보
# ===========================================================
ctxPath = os.getcwd()

# 특정 영역 정보
minLon = 128
maxLon = 128.5
minLat = 33
maxLat = 33.5
print(f'[CHECK] minLon : {minLon}')
print(f'[CHECK] maxLon : {maxLon}')
print(f'[CHECK] minLat : {minLat}')
print(f'[CHECK] maxLat : {maxLat}')

# 시작일/종료일
srtDate = '2023-06-27 00:00'
endDate = '2023-07-01 00:00'
# srtDate = '2023-08-20 00:00'
# endDate = '2023-08-21 00:00'

# 년월일 시분초 변환
srtDt = pd.to_datetime(srtDate, format='%Y-%m-%d %H:%M').strftime("%Y%m%d%H%M%S")
endDt = pd.to_datetime(endDate, format='%Y-%m-%d %H:%M').strftime("%Y%m%d%H%M%S")
print(f'[CHECK] srtDt : {srtDt}')
print(f'[CHECK] endDt : {endDt}')

# 정수형 모델 정보 (TB_INT_MODEL)
modelType = 'KIER-LDAPS-2K'
# modelType = 'KIER-LDAPS-2K-30M'
# modelType = 'KIER-LDAPS-2K-60M'
# modelType = 'KIER-RDAPS-3K'
# modelType = 'KIER-RDAPS-3K-30M'
# modelType = 'KIER-RDAPS-3K-60M'
# modelType = 'KIER-WIND'
# modelType = 'KIER-WIND-30M'
# modelType = 'KIER-WIND-60M'
# modelType = 'KIM-3K'
# modelType = 'LDAPS-1.5K'
# modelType = 'RDAPS-12K'
print(f'[CHECK] modelType : {modelType}')

# 위경도 기본 정보 (TB_GEO), 위경도 상세 정보 (TB_GEO_DTL)
modelTypeToGeo = {
    'KIER-LDAPS-2K': 'KIER-LDAPS-2K'
    , 'KIER-LDAPS-2K-30M': 'KIER-LDAPS-2K'
    , 'KIER-LDAPS-2K-60M': 'KIER-LDAPS-2K'
    , 'KIER-RDAPS-3K': 'KIER-RDAPS-3K'
    , 'KIER-RDAPS-3K-30M': 'KIER-RDAPS-3K'
    , 'KIER-RDAPS-3K-60M': 'KIER-RDAPS-3K'
    , 'KIER-WIND': 'KIER-WIND'
    , 'KIER-WIND-30M': 'KIER-WIND'
    , 'KIER-WIND-60M': 'KIER-WIND'
    , 'KIM-3K': 'KIM-3K'
    , 'LDAPS-1.5K': 'LDAPS-1.5K'
    , 'RDAPS-12K': 'RDAPS-12K'
}

geoType = modelTypeToGeo.get(modelType)

# ===========================================================
# PostgreSQL 설정 정보
# ===========================================================
cfgInfo = f'{ctxPath}/config/config.yml'

with open(cfgInfo, 'rt', encoding='UTF-8') as file:
    cfgData = yaml.safe_load(file)['db_info']

sqlDbUrl = f'{cfgData["dbType"]}://{cfgData["dbUser"]}:{quote_plus(cfgData["dbPwd"])}@{cfgData["dbHost"]}:{cfgData["dbPort"]}/{cfgData["dbName"]}'
engine = create_engine(sqlDbUrl)
sessionMake = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = sessionMake()

# DB 연결 시 타임아웃 1시간 설정 : 60 * 60 * 1000
session.execute(text("SET statement_timeout = 3600000;"))

# 트랜잭션이 idle 상태 5분 설정 : 5 * 60 * 1000
session.execute(text("SET idle_in_transaction_session_timeout = 300000;"))

# 격리 수준 설정
session.execute(text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;"))

# 세션 커밋
session.commit()

# 테이블 정보
metaData = MetaData()

# 예보 모델 테이블
# 정수형
tbIntModel = Table('TB_INT_MODEL', metaData, autoload_with=engine, schema=cfgData["dbSchema"])
tbIntProc = Table('TB_INT_PROC', metaData, autoload_with=engine, schema=cfgData["dbSchema"])

# ===========================================================
# DB 정보 가져오기
# ===========================================================
# SQL 쿼리
# sql = f"""
# SELECT * FROM "DMS01"."TB_GEO_DTL" LIMIT 1;
# """

sql = f"""
WITH GET_SFC_INFO AS (SELECT MIN("ROW")     AS "MIN_ROW_SFC",
                             MAX("ROW")     AS "MAX_ROW_SFC",
                             MIN("COL")     AS "MIN_COL_SFC",
                             MAX("COL")     AS "MAX_COL_SFC",
                             MIN("LON_SFC") AS "MIN_LON_SFC",
                             MAX("LON_SFC") AS "MAX_LON_SFC",
                             MIN("LAT_SFC") AS "MIN_LAT_SFC",
                             MAX("LAT_SFC") AS "MAX_LAT_SFC"
                      FROM (SELECT "ROW", "COL", "LON_SFC", "LAT_SFC"
                            FROM "DMS01"."TB_GEO_DTL"
                            WHERE 1 = 1
                              AND "MODEL_TYPE" = '{geoType}'
                              AND "LON_SFC" BETWEEN {minLon} AND {maxLon}
                              AND "LAT_SFC" BETWEEN {minLat} AND {maxLat}) AS A)

   , GET_PRE_INFO AS (SELECT MIN("ROW")     AS "MIN_ROW_PRE",
                             MAX("ROW")     AS "MAX_ROW_PRE",
                             MIN("COL")     AS "MIN_COL_PRE",
                             MAX("COL")     AS "MAX_COL_PRE",
                             MIN("LON_PRE") AS "MIN_LON_PRE",
                             MAX("LON_PRE") AS "MAX_LON_PRE",
                             MIN("LAT_PRE") AS "MIN_LAT_PRE",
                             MAX("LAT_PRE") AS "MAX_LAT_PRE"
                      FROM (SELECT "ROW", "COL", "LON_PRE", "LAT_PRE"
                            FROM "DMS01"."TB_GEO_DTL"
                            WHERE 1 = 1
                              AND "MODEL_TYPE" = '{geoType}'
                              AND "LON_PRE" BETWEEN {minLon} AND {maxLon}
                              AND "LAT_PRE" BETWEEN {minLat} AND {maxLat}) AS B)

SELECT TO_CHAR(C."ANA_DT", 'YYYY-MM-DD HH24:MI:SS')                                                         AS "ANA_DT"
     , TO_CHAR(C."FOR_DT", 'YYYY-MM-DD HH24:MI:SS')                                                         AS "FOR_DT"
     , C."MODEL_TYPE"
     , D."LON_SFC"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)] [(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "LON_SFC"
     , D."LAT_SFC"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)] [(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "LAT_SFC"
     , D."LON_PRE"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "LON_PRE"
     , D."LAT_PRE"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "LAT_PRE"

    /* 지표면 */
     , C."SW_D"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)] [(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "SW_D"
     , C."SW_DC"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)] [(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "SW_DC"
     , C."SW_DDNI"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)] [(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "SW_DDNI"
     , C."SW_DDIF"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)] [(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "SW_DDIF"
     , C."SW_NET"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)] [(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "SW_NET"
     , C."SW_UC"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)] [(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "SW_UC"
     , C."SW_U"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)] [(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "SW_U"
     , C."U"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)] [(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "U"
     , C."V"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)] [(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "V"

        /* 상층 */
     , C."U850"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "U850"
     , C."U875"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "U875"
     , C."U900"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "U900"
     , C."U925"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "U925"
     , C."U975"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "U975"
     , C."U1000"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "U1000"
     , C."V850"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "V850"
     , C."V875"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "V875"
     , C."V900"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "V900"
     , C."V925"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "V925"
     , C."V975"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "V975"
     , C."V1000"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)] [(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "V1000"

    /* 최근접 인덱스 SFC/PRE */
     , A.*
     , B.*
FROM GET_SFC_INFO A
   , GET_PRE_INFO B
   , "DMS01"."TB_INT_MODEL" C
   , (SELECT * FROM "DMS01"."TB_GEO" WHERE "MODEL_TYPE" = '{geoType}') AS D
WHERE 1 = 1
  AND C."MODEL_TYPE" = '{modelType}'
  AND C."ANA_DT" BETWEEN TO_TIMESTAMP('{srtDt}', 'YYYYMMDDHH24MISS') AND TO_TIMESTAMP('{endDt}', 'YYYYMMDDHH24MISS');
"""

data = pd.read_sql(sql, engine)
print(f'[CHECK] data : {data}')

# ===========================================================
# DB 요소 계산
# ===========================================================
# DB 내 속도 향상을 위해서 정수형으로 적재
# 따라서 invIntToFloat를 통해 scaleFactor = 10000, addOffset = 0 고려

# 풍속 수식 : np.sqrt(U벡터**2 + V벡터**2)
data['WS80'] = data.apply(lambda item:
                np.sqrt(invIntToFloat(item['U']) ** 2 + invIntToFloat(item['V']) ** 2)
                , axis=1)

# 풍향 수식 : (180 + np.arctan2(U벡터, V벡터) * (180/np.pi)) % 360
data['WS100'] = data.apply(lambda item:
                (180 + np.arctan2(invIntToFloat(item['U']), invIntToFloat(item['V'])) * (180/np.pi)) % 360
                , axis=1)

# ===========================================================
# DB 적재
# ===========================================================
# DB 등록/수정
dbData = {}

# 필수 정보
dbData['ANA_DT'] = data['ANA_DT'][0]
dbData['FOR_DT'] = data['FOR_DT'][0]
dbData['MODEL_TYPE'] = data['MODEL_TYPE'][0]

# 선택 정보
dbData['WS80'] = convFloatToIntList(data['WS80'][0])
dbData['WS100'] = convFloatToIntList(data['WS100'][0])

if len(dbData) < 1:
    print(f'해당 파일에서 지표면 및 상층 데이터를 확인해주세요.')
    sys.exit(1)

print(f'[CHECK] dbData : {dbData.keys()} : {np.shape(dbData[list(dbData.keys())[3]])}')
dbMergeData(session, tbIntProc, dbData, pkList=['MODEL_TYPE', 'ANA_DT', 'FOR_DT'])