from sqlalchemy import MetaData, Table
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker

class ConnectionDB :
	
	def __init__(self,config):
		self.dbType=""
		self.dbUser=""
		self.dbPwd=""
		self.dbHost=""
		self.dbPort=""
		self.dbName=""
		self.dbSchema=""
	
	def initCfgInfo(self):
		result= None
		
		try:
			dbInfo=
	def getConInfo(self):
		conn_info=self.config['db_info']
		self.dbType=conn_info['dbtype']
		self.dbUser=conn_info['dbUser']
		self.dbPwd=conn_info['dbPwd']
		self.dbHost=conn_info['dbHost']
		self.dbPort=conn_info['dbPort']
		self.dbName=conn_info['dbName']
		self.dbSchema=conn_info['dbSchema']
		
	
