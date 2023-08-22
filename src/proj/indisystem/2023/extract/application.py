import datetime
import numpy as np
from grib12 import Grib12
from manageDB import ManageDB
import re
import sys
import xarray as xr
import pandas as pd

class Application:

    def __init__(self, inFile, modelName, modelName1, config):
        self.inFile = inFile
        self.modelName = modelName
        self.modelName1 = modelName1
        self.config = config
        self.dbconfig = config['db_info']
        self.varNameLists = ""
        self.locNameLists = ""
        self.levelNameLists = ""
        self.dbData = {}

    def run(self):
        if re.search('KIER-LDAPS|KIER-RDAPS', self.modelName, re.IGNORECASE):
            self.processKIER()

        elif re.search('KIM|LDAPS|RDAPS', self.modelName, re.IGNORECASE):
            gribApp = Grib12(self.inFile)
            gribApp.openFile()
            self.processKMA(gribApp)
        else:
            print(f'[ERROR] 모델 종류 ({self.modelName})를 확인해주세요.')
            sys.exit(1)

    def processKMA(self, gribApp):
        self.getVar()
        dbapp = ManageDB(self.dbconfig)
        initDB = dbapp.initCfgInfo()
        dbapp.dbMergeData

        tmpADate = gribApp.getAttrValue(self.varNameLists[0]['name'], 'initial_time')
        analDate = datetime.datetime.strptime(tmpADate, '%m/%d/%Y (%H:%M)')
        tmpFDate = gribApp.getAttrValue(self.varNameLists[0]['name'], 'forecast_time')
        tmpFHour = tmpFDate[0]
        forcDate = analDate + datetime.timedelta(hours=int(tmpFHour))

        self.dbData['ANA_DT'] = analDate
        self.dbData['FOR_DT'] = forcDate
        self.dbData['MODEL_TYPE'] = self.modelName
        for vlist in self.varNameLists:
            #			print (vlist['name'])
            for idx, level in enumerate(vlist['level'], 0):
                if level == '-1':
                    self.dbData[vlist['colName'][idx]] = self.convFloatToIntList(gribApp.getVariable(vlist['name']))
                else:
                    self.dbData[vlist['colName'][idx]] = self.convFloatToIntList(
                        gribApp.getVariable31(vlist['name'], idx))
        #		print(self.dbData)

        dbapp.dbMergeData(initDB['session'], initDB['tbIntModel'], self.dbData, pkList=['MODEL_TYPE', 'ANA_DT', 'FOR_DT'])

    def processKIER(self):

        # DB 가져오기
        dbapp = ManageDB(self.dbconfig)
        initDB = dbapp.initCfgInfo()

        # NetCDF 파일 읽기
        fileInfo = self.inFile
        orgData = xr.open_mfdataset(self.inFile)
        print(f'[CHECK] fileInfo : {fileInfo}')

        # 분석시간
        anaDate = pd.to_datetime(orgData.START_DATE, format='%Y-%m-%d_%H:%M:%S')

        # 시간 인덱스를 예보 시간 기준으로 변환
        timeByteList = orgData['Times'].values
        timeList = [timeInfo.decode('UTF-8').replace('_', ' ') for timeInfo in timeByteList]
        orgData['Time'] = pd.to_datetime(timeList)

        # NREML의 경우  KIER-LDAPS/KIER-RDAPS 30분, 1시간 평균값을 저장
        # Solar 서버의 KIER-WIND 10분(순간), 30분, 1시간 평균값을 저장
        # 60M : 1시간 평균, 30M : 30분 평균, 10M : 10분 순간
        if re.search('60M', self.modelName, re.IGNORECASE):
            data = orgData.resample(Time='1H').mean(dim='Time', skipna=True)
        elif re.search('30M', self.modelName, re.IGNORECASE):
            data = orgData.resample(Time='30T').mean(dim='Time', skipna=True)
        elif re.search('10M', self.modelName, re.IGNORECASE):
            data = orgData.resample(Time='10T').asfreq()
        else:
            data = orgData

        # 예보 시간
        forDateList = data['Time'].values
        for idx, forDateInfo in enumerate(forDateList):
            forDate = pd.to_datetime(forDateInfo, format='%Y-%m-%d_%H:%M:%S')
            print(f'[CHECK] anaDate : {anaDate} / forDate : {forDate}')

            for i, modelKey in enumerate(['UNIS', 'PRES']):
                modelType = 'KIER-LDAPS' if  re.search('KIER-LDAPS', self.modelName, re.IGNORECASE) else 'KIER-RDAPS'
                modelInfo = self.config['modelName'].get(f'{modelType}_{modelKey}')
                if modelInfo is None:
                    print(f'[WARN] 설정 파일 (config.yml)에서 설정 정보 (KIER-LDAPS/RDAPS, UNIS/PRES)를 확인해주세요.')
                    continue

                # *********************************************************
                # DB 등록/수정
                # *********************************************************
                # 필수 컬럼
                dbData = {}
                dbData['ANA_DT'] = anaDate
                dbData['FOR_DT'] = forDate
                dbData['MODEL_TYPE'] = self.modelName

                # 선택 컬럼
                for j, varInfo in enumerate(modelInfo['varName']):
                    name = varInfo['name']
                    for level, colName in zip(varInfo['level'], varInfo['colName']):
                        try:
                            if re.search('unis', modelKey, re.IGNORECASE):
                                if len(data[name].isel(Time=idx).values) < 1: continue
                                dbData[colName] = self.convFloatToIntList(data[name].isel(Time=idx).values)
                            else:
                                if len(data[name].isel(Time=idx, bottom_top=int(level)).values) < 1: continue
                                dbData[colName] = self.convFloatToIntList(data[name].isel(Time=idx, bottom_top=int(level)).values)

                        except Exception as e:
                            print(f'Exception : {e}')

                if len(dbData) < 1:
                    print(f'[WARN] 해당 파일 ({fileInfo})에서 지표면 및 상층 데이터를 확인해주세요.')
                    sys.exit(1)

                print(f'[CHECK] dbData : {dbData.keys()}')
                dbapp.dbMergeData(initDB['session'], initDB['tbIntModel'], dbData, pkList=['MODEL_TYPE', 'ANA_DT', 'FOR_DT'])

    def getVar(self):
        self.varNameLists = self.config['modelName'][self.modelName1]['varName']

    def convFloatToIntList(self, val):
        scaleFactor = 10000
        addOffset = 0
        return ((np.around(val, 4) * scaleFactor) - addOffset).astype(int).tolist()

    """
    def insertData(self,):
        #decoding grib data
        # insert DB
    def getConfig(self) :
        return 	
    """
