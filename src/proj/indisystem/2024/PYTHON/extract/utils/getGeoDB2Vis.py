# -*- coding: utf-8 -*-

import pandas as pd
import psycopg2
import re
import yaml
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import re

# ===========================================================
# 실행 방법
# ===========================================================
# /home/guest_user1/SYSTEMS/KIER/LIB/py38/bin/python3 /home/guest_user1/SYSTEMS/KIER/PROG/PYTHON/extract/getPointDB.py

# ===========================================================
# 입력 정보
# ===========================================================
# /home/guest_user1/SYSTEMS/KIER/PROG/PYTHON

# ===========================================================
# PostgreSQL 설정 정보
# ===========================================================
ctxPath = os.getcwd()
# cfgInfo = f'{ctxPath}/config/config.yml'
cfgInfo = f'/SYSTEMS/PROG/PYTHON/PyCharm/src/proj/indisystem/2024/PYTHON/extract/config/config.yml'

with open(cfgInfo, 'rt', encoding='UTF-8') as file:
    cfgData = yaml.safe_load(file)['db_info']

# 데이터베이스 연결
conn = psycopg2.connect(
    dbname=cfgData['dbName']
    , user=cfgData['dbUser']
    , password=cfgData['dbPwd']
    , host='localhost' if cfgData['dbHost'] == cfgData['serverHost'] else cfgData['dbHost']
)

# 커서 생성
cur = conn.cursor()

# SQL 쿼리
sql = f"""
SELECT * FROM "DMS01"."TB_GEO_DTL";
"""

# 쿼리 실행
cur.execute(sql)

# 결과 가져오기
result = cur.fetchall()

# 결과 출력
colNameList = [desc[0] for desc in cur.description]
data = pd.DataFrame(result, columns=colNameList)
# print(data)

modelTypeList = set(data['MODEL_TYPE'])
for modelTypeInfo in modelTypeList:

    # if not (modelTypeInfo == 'GFS-25K'): continue
    if not (modelTypeInfo == 'KIER-LDAPS-0.6K'): continue
    print(f'[CHECK] modelTypeInfo : {modelTypeInfo}')

    dataL1 = data.loc[(data['MODEL_TYPE'] == modelTypeInfo)]
    saveImg = '{}/{}.png'.format('/DATA/FIG/INDI2023/MODEL', modelTypeInfo)

    meanLon = np.nanmean(dataL1['LON_SFC'])
    meanLat = np.nanmean(dataL1['LAT_SFC'])

    minLon = np.nanmin(dataL1['LON_SFC'])
    maxLon = np.nanmax(dataL1['LON_SFC'])
    minLat = np.nanmin(dataL1['LAT_SFC'])
    maxLat = np.nanmax(dataL1['LAT_SFC'])

    # mainTitle =  f'{minLon:.2f} ~ {maxLon:.2f} / {minLat:.2f} ~ {maxLat:.2f}'
    mainTitle =  f'{meanLon:.2f} ({minLon:.2f} ~ {maxLon:.2f}) / {meanLat:.2f} ({minLat:.2f} ~ {maxLat:.2f})'

    plt.figure(dpi=600, figsize=(9/1.5, 10/1.5))
    map = Basemap(projection='cyl', resolution='f', llcrnrlon=minLon, urcrnrlon=maxLon, llcrnrlat=minLat, urcrnrlat=maxLat)
    cs = map.scatter(dataL1['LON_SFC'], dataL1['LAT_SFC'], c=dataL1['LAT_SFC'] * 0 + 1, s=1, vmin = 0, vmax = 2, marker='s', cmap='Greys_r')

    map.drawcoastlines()
    map.drawmapboundary()
    map.drawcountries(linewidth=1, linestyle='solid', color='k')

    if re.search('KIER-RDAPS|KIER-LDAPS|KIER-WIND|LDAPS', modelTypeInfo, re.IGNORECASE):
        map.drawmeridians(range(-180, 180, 1), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 0, 1])
        map.drawparallels(range(-90, 90, 1), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 0, 0, 0])
    elif re.search('RDAPS', modelTypeInfo, re.IGNORECASE):
        map.drawmeridians(range(-180, 180, 10), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 0, 1])
        map.drawparallels(range(-90, 90, 5), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 0, 0, 0])
    elif re.search('KIM', modelTypeInfo, re.IGNORECASE):
        map.drawmeridians(range(-180, 180, 4), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 0, 1])
        map.drawparallels(range(-90, 90, 2), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 0, 0, 0])
    elif re.search('GFS', modelTypeInfo, re.IGNORECASE):
        map.drawmeridians(range(-180, 180, 5), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 0, 1])
        map.drawparallels(range(-90, 90, 4), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 0, 0, 0])
    else:
        pass

    plt.ylabel(None)
    plt.xlabel(None)
    # cbar2 = plt.colorbar(cs, orientation='horizontal')
    # cbar2.set_label(None, fontsize=13)
    plt.title(mainTitle)
    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=False)
    plt.tight_layout()
    # plt.show()
    plt.close()

    print(f'[CHECK] saveImg : {saveImg}')

# 커서와 연결 종료
cur.close()
conn.close()