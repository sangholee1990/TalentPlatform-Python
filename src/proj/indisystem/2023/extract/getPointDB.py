# -*- coding: utf-8 -*-

import pandas as pd
import psycopg2
import re
import yaml
import os

# ===========================================================
# 실행 방법
# ===========================================================
# /home/guest_user1/SYSTEMS/KIER/LIB/py38/bin/python3 /home/guest_user1/SYSTEMS/KIER/PROG/PYTHON/extract/getPointDB.py

# ===========================================================
# 입력 정보
# ===========================================================
# /home/guest_user1/SYSTEMS/KIER/PROG/PYTHON

# 특정 지점 정보
lon = 126.0
lat = 35

# 시작일/종료일
srtDate = '2023-06-27 00:00'
endDate = '2023-07-01 00:00'
# srtDate = '2023-08-20 00:00'
# endDate = '2023-08-21 00:00'

# 년월시시분초 변환
srtDt = pd.to_datetime(srtDate, format='%Y-%m-%d %H:%M').strftime("%Y%m%d%H%M%S")
endDt = pd.to_datetime(endDate, format='%Y-%m-%d %H:%M').strftime("%Y%m%d%H%M%S")

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
ctxPath = os.getcwd()
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
WITH GET_SFC_INFO AS (SELECT "ROW" AS "ROW_SFC", "COL" AS "COL_SFC", "LON_SFC", "LAT_SFC"
							  FROM "DMS01"."TB_GEO_DTL"
							  WHERE 1 = 1
							  AND "MODEL_TYPE" = '{geoType}'
							  ORDER BY (-- Haversine 공식
											   6371 * acos(
											       sin(radians(@"LAT_SFC")) * sin(radians({lat})) +
											       cos(radians(@"LAT_SFC")) * cos(radians({lat})) *
											       cos(radians({lon}) - radians(@"LON_SFC"))
											   )
										   )
							  LIMIT 1)
		   , GET_PRE_INFO AS (SELECT "ROW" AS "ROW_PRE", "COL" AS "COL_PRE", "LON_PRE", "LAT_PRE"
							  FROM "DMS01"."TB_GEO_DTL"
							  WHERE 1 = 1
							  AND "MODEL_TYPE" = '{geoType}'
							  ORDER BY (-- Haversine 공식
									   6371 * acos(
										   sin(radians(@"LAT_PRE")) * sin(radians({lat})) +
										   cos(radians(@"LAT_PRE")) * cos(radians({lat})) *
										   cos(radians({lon}) - radians(@"LON_PRE"))
									   )
								   )
							  LIMIT 1)

				SELECT TO_CHAR(C."ANA_DT", 'YYYY-MM-DD HH24:MI:SS') AS "ANA_DT"
					 , TO_CHAR(C."FOR_DT", 'YYYY-MM-DD HH24:MI:SS') AS "FOR_DT"
					 , C."MODEL_TYPE"

				     /* 지표면 */
					 , C."SW_D"[(SELECT "ROW_SFC" FROM GET_SFC_INFO)][(SELECT "COL_SFC" FROM GET_SFC_INFO)]::FLOAT / 10000    AS "SW_D"
					 , C."SW_DC"[(SELECT "ROW_SFC" FROM GET_SFC_INFO)][(SELECT "COL_SFC" FROM GET_SFC_INFO)]::FLOAT / 10000   AS "SW_DC"
					 , C."SW_DDNI"[(SELECT "ROW_SFC" FROM GET_SFC_INFO)][(SELECT "COL_SFC" FROM GET_SFC_INFO)]::FLOAT / 10000 AS "SW_DDNI"
					 , C."SW_DDIF"[(SELECT "ROW_SFC" FROM GET_SFC_INFO)][(SELECT "COL_SFC" FROM GET_SFC_INFO)]::FLOAT / 10000 AS "SW_DDIF"
					 , C."SW_NET"[(SELECT "ROW_SFC" FROM GET_SFC_INFO)][(SELECT "COL_SFC" FROM GET_SFC_INFO)]::FLOAT / 10000  AS "SW_NET"
					 , C."SW_UC"[(SELECT "ROW_SFC" FROM GET_SFC_INFO)][(SELECT "COL_SFC" FROM GET_SFC_INFO)]::FLOAT / 10000   AS "SW_UC"
					 , C."SW_U"[(SELECT "ROW_SFC" FROM GET_SFC_INFO)][(SELECT "COL_SFC" FROM GET_SFC_INFO)]::FLOAT / 10000    AS "SW_U"
					 , C."U"[(SELECT "ROW_SFC" FROM GET_SFC_INFO)][(SELECT "COL_SFC" FROM GET_SFC_INFO)]::FLOAT / 10000       AS "U"
					 , C."V"[(SELECT "ROW_SFC" FROM GET_SFC_INFO)][(SELECT "COL_SFC" FROM GET_SFC_INFO)]::FLOAT / 10000       AS "V"

-- 					 /* 상층 */
					 , C."U850"[(SELECT "ROW_PRE" FROM GET_PRE_INFO)][(SELECT "COL_PRE" FROM GET_PRE_INFO)]::FLOAT / 10000    AS "U850"
					 , C."U875"[(SELECT "ROW_PRE" FROM GET_PRE_INFO)][(SELECT "COL_PRE" FROM GET_PRE_INFO)]::FLOAT / 10000    AS "U875"
					 , C."U900"[(SELECT "ROW_PRE" FROM GET_PRE_INFO)][(SELECT "COL_PRE" FROM GET_PRE_INFO)]::FLOAT / 10000    AS "U900"
					 , C."U925"[(SELECT "ROW_PRE" FROM GET_PRE_INFO)][(SELECT "COL_PRE" FROM GET_PRE_INFO)]::FLOAT / 10000    AS "U925"
					 , C."U975"[(SELECT "ROW_PRE" FROM GET_PRE_INFO)][(SELECT "COL_PRE" FROM GET_PRE_INFO)]::FLOAT / 10000    AS "U975"
					 , C."U1000"[(SELECT "ROW_PRE" FROM GET_PRE_INFO)][(SELECT "COL_PRE" FROM GET_PRE_INFO)]::FLOAT / 10000   AS "U1000"
					 , C."V850"[(SELECT "ROW_PRE" FROM GET_PRE_INFO)][(SELECT "COL_PRE" FROM GET_PRE_INFO)]::FLOAT / 10000    AS "V850"
					 , C."V875"[(SELECT "ROW_PRE" FROM GET_PRE_INFO)][(SELECT "COL_PRE" FROM GET_PRE_INFO)]::FLOAT / 10000    AS "V875"
					 , C."V900"[(SELECT "ROW_PRE" FROM GET_PRE_INFO)][(SELECT "COL_PRE" FROM GET_PRE_INFO)]::FLOAT / 10000    AS "V900"
					 , C."V925"[(SELECT "ROW_PRE" FROM GET_PRE_INFO)][(SELECT "COL_PRE" FROM GET_PRE_INFO)]::FLOAT / 10000    AS "V925"
					 , C."V975"[(SELECT "ROW_PRE" FROM GET_PRE_INFO)][(SELECT "COL_PRE" FROM GET_PRE_INFO)]::FLOAT / 10000    AS "V975"
					 , C."V1000"[(SELECT "ROW_PRE" FROM GET_PRE_INFO)][(SELECT "COL_PRE" FROM GET_PRE_INFO)]::FLOAT / 10000   AS "V1000"

					/* 최근접 인덱스 SFC/PRE */
					 , A.*
					 , B.*
				FROM GET_SFC_INFO A
				   , GET_PRE_INFO B
				   , "DMS01"."TB_INT_MODEL" C
				WHERE 1 = 1
				  AND C."MODEL_TYPE" = '{modelType}'
	     		  AND C."ANA_DT" BETWEEN TO_TIMESTAMP('{srtDt}', 'YYYYMMDDHH24MISS') AND TO_TIMESTAMP('{endDt}', 'YYYYMMDDHH24MISS');
"""

# 쿼리 실행
cur.execute(sql)

# 결과 가져오기
results = cur.fetchall()

# 결과 출력
colNameList = [desc[0] for desc in cur.description]
data = pd.DataFrame(results, columns=colNameList)
print(data)

# 커서와 연결 종료
cur.close()
conn.close()