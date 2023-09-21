# -*- coding: utf-8 -*-
import sys

import pandas as pd
import psycopg2
import re
import yaml
import os

# ===========================================================
# 실행 방법
# ===========================================================
# /home/guest_user1/SYSTEMS/KIER/LIB/py38/bin/python3 /home/guest_user1/SYSTEMS/KIER/PROG/PYTHON/extract/getAreaDB.py

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
# modelType = 'GFS-25K'
print(f'[CHECK] modelType : {modelType}')

# ===========================================================
# 위경도 기본/상세 정보
# ===========================================================
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
    , 'GFS-25K': 'GFS-25K'
}

geoType = modelTypeToGeo.get(modelType)

# ===========================================================
# PostgreSQL 설정 정보
# ===========================================================
cfgInfo = f'{ctxPath}/config/config.yml'

with open(cfgInfo, 'rt', encoding='UTF-8') as file:
    cfgData = yaml.safe_load(file)['db_info']

# 데이터베이스 연결
conn = psycopg2.connect(
    dbname=cfgData['dbName']
    , user=cfgData['dbUser']
    , password=cfgData['dbPwd']
    , host=cfgData['dbHost']
)

# 커서 생성
cur = conn.cursor()

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

SELECT TO_CHAR(AA."ANA_DT", 'YYYY-MM-DD HH24:MI:SS') AS "ANA_DT"
     , TO_CHAR(AA."FOR_DT", 'YYYY-MM-DD HH24:MI:SS') AS "FOR_DT"
     , AA."MODEL_TYPE"

    /* 지표 인덱스 정보 */
     , AA."MIN_ROW_SFC"
     , AA."MAX_ROW_SFC"
     , AA."MIN_COL_SFC"
     , AA."MAX_COL_SFC"
    /* 지표 위경도 정보 */
     , AA."MIN_LON_SFC"
     , AA."MAX_LON_SFC"
     , AA."MIN_LAT_SFC"
     , AA."MAX_LAT_SFC"
    /* 상층 인덱스 정보 */
     , AA."MIN_ROW_PRE"
     , AA."MAX_ROW_PRE"
     , AA."MIN_COL_PRE"
     , AA."MAX_COL_PRE"
    /* 상층 위경도 정보 */
     , AA."MIN_LON_PRE"
     , AA."MAX_LON_PRE"
     , AA."MIN_LAT_PRE"
     , AA."MAX_LAT_PRE"
    /* 2차원을 단일값으로 변환 */
     , unnest(AA."LON_SFC")                          AS "LON_SFC"
     , unnest(AA."LAT_SFC")                          AS "LAT_SFC"
     , unnest(AA."LON_PRE")                          AS "LON_PRE"
     , unnest(AA."LAT_PRE")                          AS "LAT_PRE"

    /* 지표면 */
     , unnest(AA."SW_D")::Float / 100000             AS "SW_D"
     , unnest(AA."SW_DC")::Float / 100000            AS "SW_DC"
     , unnest(AA."SW_DDNI")::Float / 100000          AS "SW_DDNI"
     , unnest(AA."SW_DDIF")::Float / 100000          AS "SW_DDIF"
     , unnest(AA."SW_NET")::Float / 100000           AS "SW_NET"
     , unnest(AA."SW_UC")::Float / 100000            AS "SW_UC"
     , unnest(AA."SW_U")::Float / 100000             AS "SW_U"
     , unnest(AA."U")::Float / 100000                AS "U"
     , unnest(AA."V")::Float / 100000                AS "V"
    /* 상층 */
     , unnest(AA."U850")::Float / 100000             AS "U850"
     , unnest(AA."U875")::Float / 100000             AS "U875"
     , unnest(AA."U900")::Float / 100000             AS "U900"
     , unnest(AA."U925")::Float / 100000             AS "U925"
     , unnest(AA."U975")::Float / 100000             AS "U975"
     , unnest(AA."U1000")::Float / 100000            AS "U1000"
     , unnest(AA."V850")::Float / 100000             AS "V850"
     , unnest(AA."V875")::Float / 100000             AS "V875"
     , unnest(AA."V900")::Float / 100000             AS "V900"
     , unnest(AA."V925")::Float / 100000             AS "V925"
     , unnest(AA."V975")::Float / 100000             AS "V975"
     , unnest(AA."V1000")::Float / 100000            AS "V1000"
FROM (SELECT C."ANA_DT"                                                                                                                                                                        AS "ANA_DT"
           , C."FOR_DT"                                                                                                                                                                        AS "FOR_DT"
           , C."MODEL_TYPE"
          /* 위경도 */
           , D."LON_SFC"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)][(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "LON_SFC"
           , D."LAT_SFC"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)][(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "LAT_SFC"
           , D."LON_PRE"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "LON_PRE"
           , D."LAT_PRE"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)] AS "LAT_PRE"

          /* 지표면 */
           , C."SW_D"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)][(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)]    AS "SW_D"
           , C."SW_DC"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)][(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)]   AS "SW_DC"
           , C."SW_DDNI"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)][(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "SW_DDNI"
           , C."SW_DDIF"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)][(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)] AS "SW_DDIF"
           , C."SW_NET"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)][(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)]  AS "SW_NET"
           , C."SW_UC"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)][(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)]   AS "SW_UC"
           , C."SW_U"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)][(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)]    AS "SW_U"
           , C."U"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)][(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)]       AS "U"
           , C."V"[(SELECT "MIN_ROW_SFC" FROM GET_SFC_INFO):(SELECT "MAX_ROW_SFC" FROM GET_SFC_INFO)][(SELECT "MIN_COL_SFC" FROM GET_SFC_INFO):(SELECT "MAX_COL_SFC" FROM GET_SFC_INFO)]       AS "V"

          /* 상층 */
           , C."U850"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)]    AS "U850"
           , C."U875"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)]    AS "U875"
           , C."U900"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)]    AS "U900"
           , C."U925"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)]    AS "U925"
           , C."U975"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)]    AS "U975"
           , C."U1000"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)]   AS "U1000"
           , C."V850"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)]    AS "V850"
           , C."V875"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)]    AS "V875"
           , C."V900"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)]    AS "V900"
           , C."V925"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)]    AS "V925"
           , C."V975"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)]    AS "V975"
           , C."V1000"[(SELECT "MIN_ROW_PRE" FROM GET_PRE_INFO):(SELECT "MAX_ROW_PRE" FROM GET_PRE_INFO)][(SELECT "MIN_COL_PRE" FROM GET_PRE_INFO):(SELECT "MAX_COL_PRE" FROM GET_PRE_INFO)]   AS "V1000"

          /* 최근접 인덱스 SFC/PRE */
           , A.*
           , B.*
      FROM GET_SFC_INFO A
         , GET_PRE_INFO B
         , "DMS01"."TB_INT_MODEL" C
         , (SELECT * FROM "DMS01"."TB_GEO" WHERE "MODEL_TYPE" = '{geoType}') AS D
      WHERE 1 = 1
        AND C."MODEL_TYPE" = '{modelType}'
        AND C."ANA_DT" BETWEEN TO_TIMESTAMP('{srtDt}', 'YYYYMMDDHH24MISS') AND TO_TIMESTAMP('{endDt}', 'YYYYMMDDHH24MISS')
        ORDER BY C."ANA_DT", C."FOR_DT", C."MODEL_TYPE"
    ) AA;  
"""

# 쿼리 실행
cur.execute(sql)

# 결과 가져오기
results = cur.fetchall()

# 결과 출력
colNameList = [desc[0] for desc in cur.description]
data = pd.DataFrame(results, columns=colNameList)
print(data)

if len(data) > 0:
    saveFile = f'{ctxPath}/CSV/{modelType}_{minLon}-{maxLon}_{minLat}-{maxLat}_{srtDt}_{endDt}.csv'
    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
    data.to_csv(saveFile, index=False)
    print(f'[CHECK] saveFile : {saveFile}')

# 커서와 연결 종료
cur.close()
conn.close()