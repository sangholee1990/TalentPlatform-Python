# -*- coding: utf-8 -*-

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
import common.initiator as common

class ManageDB:
    def __init__(self, config):
        self.dbType = config['dbType']
        self.dbUser = config['dbUser']
        self.dbPwd = quote_plus(config['dbPwd'])
        # self.dbHost = config['dbHost']
        self.dbHost = 'localhost' if config['dbHost'] == config['serverHost'] else config['dbHost']
        self.dbPort = config['dbPort']
        self.dbName = config['dbName']
        self.dbSchema = config['dbSchema']

    def initCfgInfo(self):

        common.logger.info(f'[START] initCfgInfo')

        result = None

        try:
            '''
            '''

            sqlDbUrl = f'{self.dbType}://{self.dbUser}:{self.dbPwd}@{self.dbHost}:{self.dbPort}/{self.dbName}'

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
            # 정수형
            tbIntModel = Table('TB_INT_MODEL', metaData, autoload_with=engine, schema=self.dbSchema)

            # 기본 위경도 테이블
            tbGeo = Table('TB_GEO', metaData, autoload_with=engine, schema=self.dbSchema)

            # 상세 위경도 테이블
            tbGeoDtl = Table('TB_GEO_DTL', metaData, autoload_with=engine, schema=self.dbSchema)

            result = {
                'engine': engine
                , 'session': session
                , 'sessionMake': sessionMake
                , 'tbIntModel': tbIntModel
                , 'tbGeo': tbGeo
                , 'tbGeoDtl': tbGeoDtl
            }

            return result

        except Exception as e:
            common.logger.error(f'Exception : {e}')
            return result

        finally:
            common.logger.info(f'[END] initCfgInfo')

    # def dbMergeData(session, table, dataList, pkList=['ANA_DT', 'FOR_DT', 'MODEL_TYPE']):
    def dbMergeData(self, session, table, dataList, pkList=['MODEL_TYPE']):
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
            common.logger.error(f'Exception : {e}')

        finally:
            session.close()
