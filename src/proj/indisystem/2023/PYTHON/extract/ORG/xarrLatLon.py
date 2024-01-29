import xarray as xr
import Nio
from urllib.parse import quote_plus
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.pool import QueuePool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import pandas as pd



#import matplotlib.pyplot as plt

def main():
	dbData={}

	modelType='RDAPS'
	dbData['MODEL_TYPE']=modelType

#	unisFile="/vol01/DATA/MODEL/KIM/r030_v040_ne36_unis_h001.2023063000.gb2"
#	presFile="/vol01/DATA/MODEL/KIM/r030_v040_ne36_pres_h006.2023063000.gb2"

#	unisFile="/vol01/DATA/MODEL/LDAPS/l015_v070_erlo_unis_h024.2023062918.gb2"
#	presFile="/vol01/DATA/MODEL/LDAPS/l015_v070_erlo_pres_h024.2023062918.gb2"

	unisFile="/vol01/DATA/MODEL/RDAPS/g120_v070_erea_unis_h024.2023062918.gb2"
	presFile="/vol01/DATA/MODEL/RDAPS/g120_v070_erea_pres_h024.2023062918.gb2"

	cfgOpt = initCfgInfo()

	orgPData=xr.open_mfdataset(presFile, engine="pynio")
	orgSData=xr.open_mfdataset(unisFile, engine="pynio")
	dbData['LAT_PRE']=orgPData['gridlat_0'].values.tolist()
	dbData['LON_PRE']=orgPData['gridlon_0'].values.tolist()
	dbData['LAT_SFC']=orgSData['gridlat_0'].values.tolist()
	dbData['LON_SFC']=orgSData['gridlon_0'].values.tolist()
	print("Starting Insert tbGeo ")
	dbMergeData(cfgOpt['session'], cfgOpt['tbGeo'], dbData, pkList=['MODEL_TYPE'])
	print("End Insert tbGeo ")
	sfcData = orgSData['gridrot_0'].to_dataframe().reset_index(drop=False).rename(
    	columns={
       	 'ygrid_0': 'ROW'
       	 , 'xgrid_0': 'COL'
       	 , 'gridlat_0': 'LAT_SFC'
       	 , 'gridlon_0': 'LON_SFC'
     	}
 	).drop(['gridrot_0'], axis='columns')
#	print(sfcData)

#	print(orgPData)
	presData = orgPData['gridrot_0'].to_dataframe().reset_index(drop=False).rename(
    	columns={
       	 'ygrid_0': 'ROW'
       	 , 'xgrid_0': 'COL'
       	 , 'gridlat_0': 'LAT_PRE'
       	 , 'gridlon_0': 'LON_PRE'
     	}
 	).drop(['gridrot_0'], axis='columns')
#	print(presData)
	dataL2 = pd.merge(left=sfcData, right=presData, how='inner', left_on=['ROW', 'COL'], right_on=['ROW', 'COL'])
	dataL2['MODEL_TYPE']=modelType
	dataList = dataL2.to_dict(orient='records')
#	print(dataList)
	print("Starting Insert tbGeoDtl ")
	dbMergeData(cfgOpt['session'], cfgOpt['tbGeoDtl'], dataList, pkList=['MODEL_TYPE', 'ROW', 'COL'])
	print("End Insert tbGeoDtl ")

	'''
	f= Nio.open_file(inFile,"r")

	dbData['LAT_SFC']=f.variables['gridlat_0'][:].tolist()
	dbData['LON_SFC']=f.variables['gridlon_0'][:].tolist()

	inFile1="/vol01/DATA/MODEL/LDAPS/l015_v070_erlo_pres_h024.2023062918.gb2"
	f1= Nio.open_file(inFile1,"r")

	dbData['LAT_PRE']=f1.variables['gridlat_0'][:].tolist()
	dbData['LON_PRE']=f1.variables['gridlon_0'][:].tolist()
	if 'unis' in inFile:
		dbData['LAT_SFC']=f.variables['gridlat_0'][:].tolist()
		dbData['LON_SFC']=f.variables['gridlon_0'][:].tolist()
	else:
		dbData['LAT_PRE']=f.variables['gridlat_0'][:].tolist()
		dbData['LON_PRE']=f.variables['gridlon_0'][:].tolist()
	print(f.variables['gridlat_0'][:])
	'''

def initCfgInfo():
    result = None

    try:
        '''
        dbInfo = sysOpt['dbInfo']
        dbType = dbInfo['dbType']
        dbUser = dbInfo['dbUser']
        dbPwd = quote_plus(dbInfo['dbPwd'])
        dbHost = 'localhost' if dbInfo['dbHost'] == dbInfo['serverHost'] else dbInfo['dbHost']
        dbPort = dbInfo['dbPort']
        dbName = dbInfo['dbName']
        dbSchema = dbInfo['dbSchema']
        '''
        dbType='postgresql'
        dbUser='kier'
        dbPwd=quote_plus('kier20230707!@#')
        dbHost='dev3.indisystem.co.kr'
        dbPort='55432'
        dbName='kier'
        dbSchema='DMS01'

        sqlDbUrl = f'{dbType}://{dbUser}:{dbPwd}@{dbHost}:{dbPort}/{dbName}'
        # 커넥션 풀 관
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
        # 실수형
        tbModel = Table('TB_MODEL', metaData, autoload_with=engine, schema=dbSchema)
        # 바이트형
        tbByteModel = Table('TB_BYTE_MODEL', metaData, autoload_with=engine, schema=dbSchema)
        # 정수형
        tbIntModel = Table('TB_INT_MODEL', metaData, autoload_with=engine, schema=dbSchema)
        # 기본 위경도 테이블
        tbGeo = Table('TB_GEO', metaData, autoload_with=engine, schema=dbSchema)
        # 상세 위경도 테이블
        tbGeoDtl = Table('TB_GEO_DTL', metaData, autoload_with=engine, schema=dbSchema)

        result = {
            'engine': engine
            , 'session': session
            , 'sessionMake': sessionMake
            , 'tbModel': tbModel
            , 'tbByteModel': tbByteModel
            , 'tbIntModel': tbIntModel
            , 'tbGeo': tbGeo
            , 'tbGeoDtl': tbGeoDtl
        }

        return result

    except Exception as e:
        print(f'Exception : {e}')
        return result

    finally:
        print(f'[END] initCfgInfo')

#def dbMergeData(session, table, dataList, pkList=['ANA_DT', 'FOR_DT', 'MODEL_TYPE']):
def dbMergeData(session, table, dataList, pkList=['MODEL_TYPE']):
    try:
        stmt = insert(table)
        onConflictStmt = stmt.on_conflict_do_update(
            index_elements=pkList
            , set_=stmt.excluded
        )
        session.execute(onConflictStmt, dataList)
        session.commit()

    except Exception as e:
        session.rollback()
        print(f'Exception : {e}')

    finally:
        session.close()
if __name__ =='__main__':
    main()
