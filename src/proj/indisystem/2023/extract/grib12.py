# -*- coding: utf-8 -*-
import common.initiator as common
import os
import Nio

class Grib12:
	def __init__(self,inFile) :
		self.inFile = inFile
		self.gribFP=''

#Open the file
	def openFile(self):
		self.gribFP = Nio.open_file(self.inFile)
		return self.gribFP

#Show variable names and coordinates
	def getVariableName(self):
		return self.gribFP.variables.keys()
	def getCoordinate(self):
		return self.gribFP.dimensions.keys()

#Select variable and coordinate variables
	def getVariable(self,varName):
		data=self.gribFP.variables[varName][:]
		return data
	def getVariable3(self,varName):
		data=self.gribFP.variables[varName][:,:,:]
		return data
	def getVariable31(self,varName,level):
		data=self.gribFP.variables[varName][level][:]
		return data

#Dimensions, shape and size
	def getShapeSize(self,varName):
		return varName.shape,varName.size,len(varName)

#Variable attributes
	def getAttributes(self,varName):
		return list(self.gribFP.variables[varName].attributes.keys())
	def getAttrValue(self,varName,attName):
		return self.gribFP.variables[varName].attributes[attName]
	def getxxx(self,varName):
		return self.gribFP.variables[varName]
