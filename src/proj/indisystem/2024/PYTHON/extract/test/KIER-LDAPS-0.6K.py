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

        if isinstance(dataList, dict): dataList = [dataList]

        stmt = insert(table)
        # setData = {key: getattr(stmt.excluded, key) for key in dataList.keys()}
        setData = {key: getattr(stmt.excluded, key) for key in dataList[0].keys()}

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
# # with xr.open_dataset(file_name) as ds:
#     # 동적 최소 차원 계산
#     shapeList = [ds.variables[var].shape for var in ['U', 'V', 'PH', 'PHB']]
#
#     minShape = [min(dim) for dim in zip(*shapeList)]
#     mt, mz, mlat, mlon = minShape
#
#     # Extract the lower 5 levels for each variable and append to the respective lists
#     # Assuming the dimensions are in the order (time, level, y, x)
#     U = ds.variables['U'][:mt, :maxK, :mlat, :mlon]
#     V = ds.variables['V'][:mt, :maxK, :mlat, :mlon]
#     PH = ds.variables['PH'][:mt, :maxK + 1, :mlat, :mlon]
#     PHB = ds.variables['PHB'][:mt, :maxK + 1, :mlat, :mlon]
#
#     H_s = ( PH + PHB ) / 9.80665
#     H = 0.5 * ( H_s[:,:-1] + H_s[:,1:])
#
#     # 특정 고도에 따른 풍향/풍속 계산
#     result = calcWsdWdr(U, V, H, alt=86)
#
#     ROW = 107
#     COL = 61
#     print(result['WSP'][ROW, COL], result['WDR'][ROW, COL])


# # # 위경도
# # lat2d = ds.variables['XLAT'][0]
# # lon2d = ds.variables['XLONG'][0]
# #
# # # Calculate squared distances
# # dist_squared = (lat2d - target_lat) ** 2 + (lon2d - target_lon) ** 2
#
# *********************************************
# [템플릿] 기본 위경도 정보를 DB 삽입
# *********************************************
dbData = {}
# modelType = 'KIER-LDAPS'
# modelType = 'KIER-RDAPS'
modelType = 'KIER-LDAPS-0.6K'
dbData['MODEL_TYPE'] = modelType

# # 지표
# # orgData = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS/wrfsolar_d02.2023-06-30_03:00:00.nc')
# # orgData = xr.open_mfdataset('/vol01/DATA/MODEL/KIER-LDAPS/wrfsolar_d02.2023-06-30_03:00:00.nc')
# # orgData = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfsolar_d02.2023-06-30_04:00:00.nc')
# # orgData = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfsolar_d02.2023-06-30_04:00:00.nc')
orgData = xr.open_mfdataset('/DATA/INPUT/INDI2024/DATA/KIER-LDAPS-0.6K/wrfout_d01_2024-01-01_12_00_00')
data = orgData['SWDOWN'].isel(Time=0)

dbData['LON_SFC'] = data['XLONG'].values.tolist() if len(data['XLONG'].values) > 0 else None
dbData['LAT_SFC'] = data['XLAT'].values.tolist() if len(data['XLAT'].values) > 0 else None

# # 상층
# # orgData2 = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-LDAPS/wrfout_d02_2023-06-30_03:00:00.nc')
# # orgData2 = xr.open_mfdataset('/vol01/DATA/MODEL/KIER-LDAPS/wrfout_d02_2023-06-30_03:00:00.nc')
# # orgData2 = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfout_d02_2023-06-30_04:00:00.nc')
# # orgData2 = xr.open_mfdataset('/DATA/INPUT/INDI2023/MODEL/KIER-RDAPS/wrfout_d02_2023-06-30_04:00:00.nc')
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
).drop(['SWDOWN', 'XTIME'], axis='columns')

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


# import netCDF4 as nc
# import glob
# import numpy as np
# import sys
# from mod_find_nearest import find_nearest_grid_point
# import xarray as xr
# import pandas as pd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
#
# def calcWsdWdr(U, V, H, alt):
#     nt, nz, nlat, nlon = U.shape
#     U_int = U[nt - 1, :, :, :]
#     V_int = V[nt - 1, :, :, :]
#     H_int = H[nt - 1, :, :, :]
#
#     WSP0 = np.sqrt(U_int[0] ** 2 + V_int[0] ** 2)
#     WSP = np.log(np.sqrt(U_int ** 2 + V_int ** 2)) - np.log(WSP0)
#     LHS = np.log(H_int / H_int[0])
#
#     WSP00 = np.full((nlat, nlon), np.nan)
#     WDR00 = np.full((nlat, nlon), np.nan)
#     for i in range(nlon):
#         for j in range(nlat):
#             alp, _, _, _ = np.linalg.lstsq(LHS[:, j, i, np.newaxis], WSP[:, j, i], rcond=None)
#             WSP00[j, i] = WSP0[j, i] * (alt / H_int[0, j, i]) ** alp[0]
#
#             k = np.argmax(H_int[:, j, i] > alt)
#             aa = (H_int[k + 1, j, i] - alt)
#             bb = (alt - H_int[k, j, i])
#             uEle = (U_int[k, j, i] * aa + U_int[k + 1, j, i] * bb) / (aa + bb)
#             vEle = (V_int[k, j, i] * aa + V_int[k + 1, j, i] * bb) / (aa + bb)
#
#             WDR00[j, i] = (np.arctan2(-uEle, -vEle) * 180.0 / np.pi) % 360.0
#
#     result = {
#         'WSP': WSP00
#         , 'WDR': WDR00
#         , 'alt': alt
#
#     }
#
#     return result
#
# # Case configuration
# fcode = '2024-02-05_12_vg_cub'
#
# # HMS01
# # ref_file   = 'def/wrfout_d01_2024-02-05_12:00:00'
# ref_file   = '/DATA/INPUT/INDI2024/DATA/KIER-LDAPS-0.6K/wrfout_d01_2024-01-01_12_00_00'
# target_lat = 35.465325
# target_lon = 126.12869444444443
#
# hms_idx = find_nearest_grid_point(ref_file, target_lat, target_lon)
# j_hms, i_hms = hms_idx # j, i index of WRF file at target point
#
# # Use glob to find all files matching the pattern 'wrfwind*.nc' and sort them
# # file_list = sorted(glob.glob('cub/wrfwind*'))
# file_list = sorted(glob.glob('/DATA/INPUT/INDI2024/DATA/KIER-LDAPS-0.6K/wrfout_d01_2024-01-01_12_00_00*'))
#
# # Initialize empty lists to store the data from all files
# U_all, V_all, PH_all, PHB_all = [], [], [], []
# k_max = 9
#
#
# # Loop through each file in the list
# for file_name in file_list:
#     print('=== Reading file',file_name,'...')
#
#     # Open the NetCDF file
#     # ds = nc.Dataset(file_name)
#     # ds = nc.Dataset(file_name)
#
#     # Extract the lower 5 levels for each variable and append to the respective lists
#     # Assuming the dimensions are in the order (time, level, y, x)
#     # U_all.append(ds.variables['U'][:, :k_max, j_hms, i_hms])
#     # V_all.append(ds.variables['V'][:, :k_max, j_hms, i_hms])
#     # PH_all.append(ds.variables['PH'][:, :k_max+1, j_hms, i_hms])
#     # PHB_all.append(ds.variables['PHB'][:, :k_max+1, j_hms, i_hms])
#
#     ds = xr.open_dataset(file_name)
#
#     # 시간 인덱스를 예보 시간 기준으로 변환
#     timeByteList = ds['Times'].values
#     timeList = [timeInfo.decode('UTF-8').replace('_', ' ') for timeInfo in timeByteList]
#     ds['Time'] = pd.to_datetime(timeList)
#
#     # data = ds.resample(Time='60T').mean(dim='Time', skipna=True)
#     data = ds
#
#     maxK = 9
#     shapeList = [data.variables[var].shape for var in ['U', 'V', 'PH', 'PHB']]
#     minShape = [min(dim) for dim in zip(*shapeList)]
#     mt, mz, mlat, mlon = minShape
#
#     # data['U'].isel(Time = 0, bottom_top = 0).plot()
#     # a = data['U'].isel(Time = 0, bottom_top = slice(0, maxK)).values[np.newaxis]
#     # data['U'].isel(Time = 0, bottom_top = 43).plot()
#     # plt.show()
#
#     # data['U'].isel(Time = 0)
#     U = data['U'][0, :maxK, :mlat, :mlon].values[np.newaxis]
#     V = data['V'][0, :maxK, :mlat, :mlon].values[np.newaxis]
#     PH = data['PH'][0, :maxK + 1, :mlat, :mlon].values[np.newaxis]
#     PHB = data['PHB'][0, :maxK + 1, :mlat, :mlon].values[np.newaxis]
#
#     # U = data['U'].isel(Time=0, bottom_top=slice(0, maxK), south_north=slice(0, mlat), west_east_stag=slice(0, mlon)).values[np.newaxis]
#     # V = data['V'].isel(Time=0, bottom_top=slice(0, maxK), south_north_stag=slice(0, mlat), west_east=slice(0, mlon)).values[np.newaxis]
#     # PH = data['PH'].isel(Time=0, bottom_top_stag=slice(0, maxK + 1), south_north=slice(0, mlat), west_east=slice(0, mlon)).values[np.newaxis]
#     # PHB = data['PHB'].isel(Time=0, bottom_top_stag=slice(0, maxK + 1), south_north=slice(0, mlat), west_east=slice(0, mlon)).values[np.newaxis]
#
#     H_s = ( PH + PHB ) / 9.80665
#     H = 0.5 * ( H_s[:,:-1] + H_s[:,1:])
#
#     result = calcWsdWdr(U, V, H, alt=86)
#     # result = calcWsdWdr(U, V, H, alt=99)
#     # result = calcWsdWdr(U, V, H, alt=90)
#
#     print(i_hms, j_hms)
#     print(result['WSP'][j_hms, i_hms], result['WDR'][j_hms, i_hms])
#     # print(result['WSP'][i_hms, j_hms], result['WDR'][i_hms, j_hms])
#
#
#     # 0.5063842652569039 327.92485027166236
#     ROW = 107
#     COL = 61
#     print(result['WSP'][ROW, COL], result['WDR'][ROW, COL])
#
#     # [0.52078629] [327.92485027]
#     # print(WSP99, WDR99)
#
# # data.variables['U'].shape
# # U.shape
# # Concatenate the lists along the time dimension to create single arrays for each variable
# # U = np.concatenate(U_all, axis=0)
# # V = np.concatenate(V_all, axis=0)
# # PH = np.concatenate(PH_all, axis=0)
# # PHB = np.concatenate(PHB_all, axis=0)
#
#
# # nt, nz = U.shape
# # WSP86 = np.zeros(nt)
# # WSP99 = np.zeros(nt)
# # WDR86 = np.zeros(nt)
# # WDR99 = np.zeros(nt)
# # for i in range(nt):
# #     U_int = U[i, :]
# #     V_int = V[i, :]
# #     H_int = H[i, :]
# #
# #     # Wind speed using power-law
# #     WSP0 = np.sqrt( U_int[0]**2 + V_int[0]**2 )
# #     WSP = np.log( np.sqrt( U_int**2 + V_int**2 ) ) - np.log( WSP0 )
# #     LHS = np.log( H_int / H_int[0] )
# #     alp, res, rank, s = np.linalg.lstsq(LHS[:, np.newaxis], WSP, rcond=None )
# #     WSP86[i] = WSP0 * ( 86.0 / H_int[0] )**alp
# #     WSP99[i] = WSP0 * ( 99.0 / H_int[0] )**alp
# #
# #     # Wind direction using linear interpolation
# #     k86 = np.argmax( H_int > 86.0 )
# #     aa = ( H_int[k86+1] - 86.0 )
# #     bb = ( 86.0 - H_int[k86] )
# #     U86 = ( U_int[k86] * aa + U_int[k86+1] * bb ) / ( aa + bb )
# #     V86 = ( V_int[k86] * aa + V_int[k86+1] * bb ) / ( aa + bb )
# #
# #     k99 = np.argmax( H_int > 99.0 )
# #     aa = ( H_int[k99+1] - 99.0 )
# #     bb = ( 99.0 - H_int[k99] )
# #     U99 = ( U_int[k99] * aa + U_int[k99+1] * bb ) / ( aa + bb )
# #     V99 = ( V_int[k99] * aa + V_int[k99+1] * bb ) / ( aa + bb )
# #
# #     WDR86[i] = ( np.arctan2( -U86, -V86 )*180.0 / np.pi )%360.0
# #     WDR99[i] = ( np.arctan2( -U99, -V99 )*180.0 / np.pi )%360.0
#
# # j, i
# # (107, 61)
# #
# # # [0.50638425] [327.92485027]
# # print(WSP86, WDR86)
# #
# # # [0.52078629] [327.92485027]
# # print(WSP99, WDR99)
#
# # np.save('tmp_np_arr/wsp86_'+fcode+'.npy', WSP86)
# # np.save('tmp_np_arr/wsp99_'+fcode+'.npy', WSP99)
# # np.save('tmp_np_arr/wdr86_'+fcode+'.npy', WDR86)
# # np.save('tmp_np_arr/wdr99_'+fcode+'.npy', WDR99)
#
