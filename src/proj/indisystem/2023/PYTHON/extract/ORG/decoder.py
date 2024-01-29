# -*- coding: utf-8 -*-
import common.initiator as common
import os
import Nio
#from draw import DrawSatelliteImage

class Decoder:
	def __init__(self,inFile,modelName,varNameLists,locNameLists) :
		self.inFile = inFile
		self.modelName = modelName
		self.varNameLists = varNameLists
		self.locNameLists = locNameLists

	def openFile(self):
		gribFP = Nio.open_file(self.inFile)
		return gribFP
		
	def run(self):
		if self.modelName in ['ECMWF','GFS','KIM','LDAPS','RDAPS'] :
			self.decodeGrib()
		else :
			common.logger.error("[ERROR]Check the Model Name - ECMWF, GFS, KIM, LDAPS, RDAPS"+self.modelName)
			os.exit()          
    
	def decodeGrib(self) :
		gribFP = openFile()
		for varName in self.varNameLists :
			data=gribP.variables[varName][1][:]
			print(data)	
			print(data.shape)	
			print(data.size)	
#                dI=DrawSatelliteImage(data,lats,lons)
#                dI.run()
#                print(data.shape, lats.shape, lats.min(), lats.max(), lons.min(), lons.max())
#                print(data)
#                print(lats)
#                fileName="D:\myProject\gribDB\lon.txt"
#                f=open(fileName,"w")
#                for d in lons :
#                    f.write(f"{d}\n")
#        dI=DrawSatelliteImage(data,lats,lons)
#        dI.run()
