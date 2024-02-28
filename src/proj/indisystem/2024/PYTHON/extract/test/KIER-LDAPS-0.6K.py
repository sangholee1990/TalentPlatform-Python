import netCDF4 as nc
import glob
import numpy as np
import sys
from mod_find_nearest import find_nearest_grid_point
import xarray as xr
import pandas as pd
import xarray as xr
from pandas.tseries.offsets import Hour
from sqlalchemy import MetaData, Table
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker
import yaml
from urllib.parse import quote_plus


def initCfgInfo(sysOpt):
    result = None

    try:
        with open(sysOpt['cfgInfo'], "rt", encoding="UTF-8") as stream:
            cfgInfo = yaml.safe_load(stream)

        dbInfo = cfgInfo['db_info']
        dbType = dbInfo['dbType']
        dbUser = dbInfo['dbUser']
        dbPwd = quote_plus(dbInfo['dbPwd'])
        dbHost = 'localhost' if dbInfo['dbHost'] == dbInfo['serverHost'] else dbInfo['dbHost']
        dbPort = dbInfo['dbPort']
        dbName = dbInfo['dbName']
        dbSchema = dbInfo['dbSchema']

        sqlDbUrl = f'{dbType}://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}'

        # 커넥션 풀 관리
        # engine = create_engine(sqlDbUrl)
        engine = create_engine(sqlDbUrl, pool_size=20, max_overflow=0)
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
        tbIntModel = Table('TB_INT_MODEL', metaData, autoload_with=engine, schema=dbSchema)

        # 기본 위경도 테이블
        tbGeo = Table('TB_GEO', metaData, autoload_with=engine, schema=dbSchema)

        # 상세 위경도 테이블
        tbGeoDtl = Table('TB_GEO_DTL', metaData, autoload_with=engine, schema=dbSchema)

        result = {
            'engine': engine
            , 'session': session
            , 'cfgInfo': cfgInfo
            , 'tbIntModel': tbIntModel
            , 'tbGeo': tbGeo
            , 'tbGeoDtl': tbGeoDtl
        }

        return result

    except Exception as e:
        return result

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

    finally:
        session.close()

def calcWsdWdr(U, V, H, alt):

    nt, nz, nlat, nlon = U.shape
    U_int = U[nt - 1, :, :, :]
    V_int = V[nt - 1, :, :, :]
    H_int = H[nt - 1, :, :, :]

    WSP0 = np.sqrt(U_int[0] ** 2 + V_int[0] ** 2)
    WSP = np.log(np.sqrt(U_int ** 2 + V_int ** 2)) - np.log(WSP0)
    LHS = np.log(H_int / H_int[0])

    WSP00 = np.full((nlat, nlon), np.nan)
    WDR00 = np.full((nlat, nlon), np.nan)
    for i in range(nlon):
        for j in range(nlat):
            alp, _, _, _ = np.linalg.lstsq(LHS[:, j, i, np.newaxis], WSP[:, j, i], rcond=None)
            WSP00[j, i] = WSP0[j, i] * (alt / H_int[0, j, i]) ** alp[0]

            k = np.argmax(H_int[:, j, i] > alt)
            aa = (H_int[k + 1, j, i] - alt)
            bb = (alt - H_int[k, j, i])
            uEle = (U_int[k, j, i] * aa + U_int[k + 1, j, i] * bb) / (aa + bb)
            vEle = (V_int[k, j, i] * aa + V_int[k + 1, j, i] * bb) / (aa + bb)

            WDR00[j, i] = (np.arctan2(-uEle, -vEle) * 180.0 / np.pi) % 360.0

    result = {
        'WSP': WSP00
        , 'WDR': WDR00
        , 'alt': alt
    }

    return result


sysOpt = {
    'cfgInfo': '/vol01/SYSTEMS/KIER/PROG/PYTHON/extract/config/config.yml'
}

# *********************************************
# DB 세선 정보 및 테이블 메타정보 가져오기
# *********************************************
cfgOpt = initCfgInfo(sysOpt)
if cfgOpt is None or len(cfgOpt) < 1:
    exit(1)

# Case configuration
fcode = '2024-02-05_12_vg_cub'

# HMS01
# ref_file   = 'def/wrfout_d01_2024-02-05_12:00:00'
ref_file   = '/DATA/INPUT/INDI2024/DATA/KIER-LDAPS-0.6K/wrfout_d01_2024-01-01_12_00_00'
target_lat = 35.465325
target_lon = 126.12869444444443

# hms_idx = find_nearest_grid_point(ref_file, target_lat, target_lon)
# j_hms, i_hms = hms_idx # j, i index of WRF file at target point

# Use glob to find all files matching the pattern 'wrfwind*.nc' and sort them
# file_list = sorted(glob.glob('cub/wrfwind*'))
file_list = sorted(glob.glob('/DATA/INPUT/INDI2024/DATA/KIER-LDAPS-0.6K/wrfout_d01_2024-01-01_12_00_00*'))


file_name = file_list[0]
# for file_name in file_list:
#     print('=== Reading file', file_name, '...')

maxK = 9
# Open the NetCDF file
# ds = xr.open_dataset(file_name)
# with nc.Dataset(file_name) as ds:
with xr.open_dataset(file_name) as ds:
    # 동적 최소 차원 계산
    shapeList = [ds.variables[var].shape for var in ['U', 'V', 'PH', 'PHB']]

    minShape = [min(dim) for dim in zip(*shapeList)]
    mt, mz, mlat, nlon = minShape

    # Extract the lower 5 levels for each variable and append to the respective lists
    # Assuming the dimensions are in the order (time, level, y, x)
    U = ds.variables['U'][:mt, :maxK, :mlat, :nlon]
    V = ds.variables['V'][:mt, :maxK, :mlat, :nlon]
    PH = ds.variables['PH'][:mt, :maxK + 1, :mlat, :nlon]
    PHB = ds.variables['PHB'][:mt, :maxK + 1, :mlat, :nlon]

    H_s = ( PH + PHB ) / 9.80665
    H = 0.5 * ( H_s[:,:-1] + H_s[:,1:])

    # 특정 고도에 따른 풍향/풍속 계산
    result = calcWsdWdr(U, V, H, alt=86)

    ROW = 107
    COL = 61
    print(result['WSP'][ROW, COL], result['WDR'][ROW, COL])


# # ROW                           107
#
#
# # 위경도
# lat2d = ds.variables['XLAT'][0]
# lon2d = ds.variables['XLONG'][0]
#
# # Calculate squared distances
# dist_squared = (lat2d - target_lat) ** 2 + (lon2d - target_lon) ** 2

# *********************************************
# [템플릿] 기본 위경도 정보를 DB 삽입
# *********************************************
dbData = {}
# modelType = 'KIER-LDAPS'
# modelType = 'KIER-RDAPS'
modelType = 'KIER-LDAPS-0.6K'
dbData['MODEL_TYPE'] = modelType

# 지표
# orgData = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS/wrfsolar_d02.2023-06-30_03:00:00.nc')
# orgData = xr.open_mfdataset('/vol01/DATA/MODEL/KIER-LDAPS/wrfsolar_d02.2023-06-30_03:00:00.nc')
# orgData = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfsolar_d02.2023-06-30_04:00:00.nc')
# orgData = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfsolar_d02.2023-06-30_04:00:00.nc')
orgData = xr.open_mfdataset('/DATA/INPUT/INDI2024/DATA/KIER-LDAPS-0.6K/wrfout_d01_2024-01-01_12_00_00')
data = orgData['SWDOWN'].isel(Time=0)

dbData['LON_SFC'] = data['XLONG'].values.tolist() if len(data['XLONG'].values) > 0 else None
dbData['LAT_SFC'] = data['XLAT'].values.tolist() if len(data['XLAT'].values) > 0 else None

# 상층
# orgData2 = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS/wrfout_d02_2023-06-30_03:00:00.nc')
# orgData2 = xr.open_mfdataset('/vol01/DATA/MODEL/KIER-LDAPS/wrfout_d02_2023-06-30_03:00:00.nc')
# orgData2 = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfout_d02_2023-06-30_04:00:00.nc')
# orgData2 = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfout_d02_2023-06-30_04:00:00.nc')
orgData2 = xr.open_mfdataset('/DATA/INPUT/INDI2024/DATA/KIER-LDAPS-0.6K/wrfout_d01_2024-01-01_12_00_00')
data2 = orgData2['U'].isel(Time = 0, bottom_top = 0)
dbData['LON_PRE'] = data2['XLONG_U'].values.tolist() if len(data2['XLONG_U'].values) > 0 else None
dbData['LAT_PRE'] = data2['XLAT_U'].values.tolist() if len(data2['XLAT_U'].values) > 0 else None

dbMergeData(session=cfgOpt['session'], table=cfgOpt['tbGeo'], dataList=dbData, pkList=['MODEL_TYPE'])

# *********************************************
# [템플릿] 상세 위경도 정보를 DB 삽입
# *********************************************
sfcData = orgData['SWDOWN'].isel(Time=0).to_dataframe().reset_index(drop=False).rename(
    columns={
        'south_north': 'ROW'
        , 'west_east': 'COL'
        , 'XLAT': 'LAT_SFC'
        , 'XLONG': 'LON_SFC'
    }
).drop(['SWDOWN'], axis='columns')

preData = orgData2['U'].isel(Time = 0, bottom_top = 0).to_dataframe().reset_index(drop=False).rename(
    columns={
        'south_north': 'ROW'
        , 'west_east_stag': 'COL'
        , 'XLAT_U': 'LAT_PRE'
        , 'XLONG_U': 'LON_PRE'
    }
).drop(['U', 'XTIME'], axis='columns')

dataL2 = pd.merge(left=sfcData, right=preData, how='inner', left_on=['ROW', 'COL'], right_on=['ROW', 'COL'])
dataL2['MODEL_TYPE'] = modelType


# dataL2['dist'] = (dataL2['LAT_PRE'] - target_lat) ** 2 + (dataL2['LON_PRE'] - target_lon) ** 2

# Find the index of the minimum distance
# min_dist_idx = np.unravel_index(np.argmin(dataL2['dist']), dataL2['dist'].shape)
# dataL2[16112]
# dataL2.iloc[min_dist_idx[0]]
# ROW                           107
# COL                            62

# # (107, 61)

dataList = dataL2.to_dict(orient='records')
dbMergeData(cfgOpt['session'], cfgOpt['tbGeoDtl'], dataList=dataList, pkList=['MODEL_TYPE', 'ROW', 'COL'])
