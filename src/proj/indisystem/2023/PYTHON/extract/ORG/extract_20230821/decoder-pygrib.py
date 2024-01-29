# -*- coding: utf-8 -*-
import common.initiator as common
import os
import pygrib
from draw import DrawSatelliteImage
#from netCDF4 import Dataset

class Decoder:
    def __init__(self,inFile,modelName,paramIdLists) :
        self.inFile = inFile
        self.modelName = modelName
        self.paramIdLists = paramIdLists
    
    def run(self) :
        if self.modelName in ['ECMWF','GFS','KIM','LDAPS','RDAPS'] :
            self.decodeGrib()
        else :
            common.logger.error("[ERROR]Check the Model Name - ECMWF, GFS, KIM, LDAPS, RDAPS"+self.modelName)
            os.exit()          
    
    def decodeGrib(self) :
        gribDatas = pygrib.open(self.inFile)
        gribDatas.seek(0)
        print(">>>")
        print(self.paramIdLists)
        for gribData in gribDatas :
            if gribData['paramId'] in self.paramIdLists :
#                print(gribData)
#                print(">>>"+str(gribData['paramId'])+"::"+str(gribData['name'])+"::"+str(gribData['shortName'])+">>"+str(gribData['units'])+">>"+str(gribData['level'])+">>"+str(gribData['validityDate'])+">>"+str(gribData['validityTime']))
#                print("==="+str(gribData['year'])+"::"+str(gribData['month'])+"::"+str(gribData['day'])+"::"+str(gribData['hour'])+"::"+str(gribData['minute']))
                data, lats, lons = gribData.data(lat1=32.97114181518555,lat2=38.83988571166992,lon1=124.09576416015625,lon2=130.054931640625)
#                dI=DrawSatelliteImage(data,lats,lons)
#                dI.run()
#                print(data.shape, lats.shape, lats.min(), lats.max(), lons.min(), lons.max())
#                print(data)
#                print(lats)
#                fileName="D:\myProject\gribDB\lon.txt"
#                f=open(fileName,"w")
#                for d in lons :
#                    f.write(f"{d}\n")
        dI=DrawSatelliteImage(data,lats,lons)
        dI.run()
        """
        fileName="D:\myProject\gribDB\data.txt"
        f=open(fileName,"w")
        for d in data :
            f.write(f"{d}\n")



        pLon=0
        count=0
        for lon in lons :
            nLon=lon
            if pLon < nLon :
                pLon = nLon
                count +=1
            else :    
#               print(str(count))
                count = 1
                pLon = 0

        print("++++++++++++++++++++++++++++++")        
        pLat=0
        count=0
        for lat in lats :
            nLat=lat
            if pLat < nLat :
                pLat = nLat
                count +=1
            else :    
#                print(str(count))
                count = 1
                pLat = 0
        print(data.shape, lats.shape, lats.min(), lats.max(), lons.min(), lons.max())
#        print(data)
        print(lats)
        print(lons)
        """        
