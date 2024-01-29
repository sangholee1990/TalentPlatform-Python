# import xarray as xr
# import Nio
# import pandas as pd
# import psycopg2
#
#
# class ManageDB :
# 	def __init__(self,config):
# 		self.dbType=config['dbType']
# 		self.dbUser=config['dbUser']
# 		self.dbPwd=quote_plus(config['dbPwd'])
# 		self.dbHost=config['dbHost']
# 		self.dbPort=config['dbPort']
# 		self.dbName=config['dbName']
# 		self.dbSchema=config['dbSchema']
# 		self.db=""
# 		self.cursor=""
#
# 	def __del__(self):
# 		self.db.close()
# 		self.cursor.close()
#
# 	def connectionDB(self):
# 		self.db = psycopg2.connect(host=self.dbHost, dbname=self.dbName,user=self.dbUser,password=self.dbPwd,port=self.dbPort)
# 		self.cursor = self.db.cursor()
#
# 	def execute(self,query,args={}):
# 		self.cursor.execute(query,args)
# 		row = self.cursor.fetchall()
# 		return row
#
# 	def commit(self):
# 		self.cursor.commit()
#
#
# 	def insertDB(self,schema,table,colum,data):
# 		sql = "INSERT INTO {schema}.{table}({colum}) VALUES ('{data}') ;".format(schema=schema,table=table,colum=colum,data=data)
# 		try:
# 			self.cursor.execute(sql)
# 			self.db.commit()
# 		except Exception as e :
# 			print(" insert DB err ",e)
#
# 	def readDB(self,schema,table,colum):
# 		sql = " SELECT {colum} from {schema}.{table}".format(colum=colum,schema=schema,table=table)
# 		try:
# 			self.cursor.execute(sql)
# 			result = self.cursor.fetchall()
# 		except Exception as e :
# 			result = (" read DB err",e)
#
# 		return result
#
# 	def updateDB(self,schema,table,colum,value,condition):
# 		sql = " UPDATE {schema}.{table} SET {colum}='{value}' WHERE {colum}='{condition}' ".format(schema=schema , table=table , colum=colum ,value=value,condition=condition )
# 		try :
# 			self.cursor.execute(sql)
# 			self.db.commit()
# 		except Exception as e :
# 			print(" update DB err",e)
#
#     def deleteDB(self,schema,table,condition):
# 		sql = " delete from {schema}.{table} where {condition} ; ".format(schema=schema,table=table,condition=condition)
# 		try :
# 			self.cursor.execute(sql)
# 			self.db.commit()
# 		except Exception as e:
# 			print( "delete DB err", e)
