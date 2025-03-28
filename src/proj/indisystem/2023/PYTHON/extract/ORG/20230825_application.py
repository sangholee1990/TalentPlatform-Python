# -*- coding: utf-8 -*-

import datetime
import numpy as np
from grib12 import Grib12
from manageDB import ManageDB
import re
import sys
import xarray as xr
import pandas as pd
import common.initiator as common
import matplotlib.pyplot as plt

class Application:

    def __init__(self, inFile, modelName, modelKey, config):
        self.inFile = inFile
        self.modelName = modelName
        self.modelKey = modelKey
        self.config = config
        self.dbconfig = config['db_info']
        self.varNameLists = ""
        self.locNameLists = ""
        self.levelNameLists = ""
        self.dbData = {}

    def run(self):
        if re.search('KIER-LDAPS|KIER-RDAPS|KIER-WIND', self.modelName, re.IGNORECASE):
            self.processKIER()
        elif re.search('KIM|LDAPS|RDAPS', self.modelName, re.IGNORECASE):
            gribApp = Grib12(self.inFile)
            gribApp.openFile()
            self.processKMA(gribApp)
        else:
            common.logger.error(f'모델 종류 ({self.modelName})를 확인해주세요.')
            sys.exit(1)

    def processKMA(self, gribApp):
        self.getVar()
        dbapp = ManageDB(self.dbconfig)
        initDB = dbapp.initCfgInfo()

        tmpADate = gribApp.getAttrValue(self.varNameLists[0]['name'], 'initial_time')
        analDate = datetime.datetime.strptime(tmpADate, '%m/%d/%Y (%H:%M)')
        tmpFDate = gribApp.getAttrValue(self.varNameLists[0]['name'], 'forecast_time')
        tmpFHour = tmpFDate[0]
        forcDate = analDate + datetime.timedelta(hours=int(tmpFHour))
        common.logger.info(f'[CHECK] anaDate : {analDate} / forDate : {forcDate}')

        # DB 등록/수정
        self.dbData['ANA_DT'] = analDate
        self.dbData['FOR_DT'] = forcDate
        self.dbData['MODEL_TYPE'] = self.modelName

        for vlist in self.varNameLists:
            for idx, level in enumerate(vlist['level'], 0):
                try:
                    if level == '-1':
                        if len(gribApp.getVariable(vlist['name'])) < 1: continue
                        self.dbData[vlist['colName'][idx]] = self.convFloatToIntList(gribApp.getVariable(vlist['name']))
                    else:
                        if len(gribApp.getVariable31(vlist['name'], idx)) < 1: continue
                        self.dbData[vlist['colName'][idx]] = self.convFloatToIntList(gribApp.getVariable31(vlist['name'], idx))

                except Exception as e:
                    common.logger.error(f'Exception : {e}')

        if len(self.dbData) < 1:
            common.logger.error(f'해당 파일 ({self.inFile})에서 지표면 및 상층 데이터를 확인해주세요.')
            sys.exit(1)

        common.logger.info(f'[CHECK] dbData : {self.dbData.keys()} : {np.shape(self.dbData[list(self.dbData.keys())[3]])}')
        dbapp.dbMergeData(initDB['session'], initDB['tbIntModel'], self.dbData, pkList=['MODEL_TYPE', 'ANA_DT', 'FOR_DT'])

    def processKIER(self):

        # DB 가져오기
        dbapp = ManageDB(self.dbconfig)
        initDB = dbapp.initCfgInfo()

        # NetCDF 파일 읽기
        orgData = xr.open_mfdataset(self.inFile)
        common.logger.info(f'[CHECK] inFile : {self.inFile}')

        # 분석시간
        anaDate = pd.to_datetime(orgData.START_DATE, format='%Y-%m-%d_%H:%M:%S')

        # 시간 인덱스를 예보 시간 기준으로 변환
        timeByteList = orgData['Times'].values
        timeList = [timeInfo.decode('UTF-8').replace('_', ' ') for timeInfo in timeByteList]
        orgData['Time'] = pd.to_datetime(timeList)

        # 해당 조건 시 처리 필요
        # KIER-LDAPS-2K-60M : 60분 평균 (지표 평균)
        # KIER-RDAPS-3K-60M : 60분 평균 (지표 평균)
        # KIER-WIND-60M : 60분 평균 (지표 평균, 상층 평균)

        # KIER-LDAPS-2K-30M : 30분 평균 (지표 평균)
        # KIER-RDAPS-3K-30M : 30분 평균 (지표 평균)
        # KIER-WIND-30M : 30분 평균 (지표 평균, 상층 평균)

        # KIER-LDAPS-2K : 1시간 순간 (지표 순간, 상층 순간)
        # KIER-RDAPS-3K : 1시간 순간 (지표 순간, 상층 순간)

        # KIER-WIND : 10분 단위 (지표 순간, 상층 순간)

        if re.search('60M', self.modelName, re.IGNORECASE):
            data = orgData.resample(Time='60T').mean(dim='Time', skipna=True)
        elif re.search('30M', self.modelName, re.IGNORECASE):
            data = orgData.resample(Time='30T').mean(dim='Time', skipna=True)
        elif re.search('KIER-LDAPS|KIER-RDAPS', self.modelName, re.IGNORECASE):
            data = orgData.resample(Time='60T').asfreq()
        elif re.search('KIER-WIND', self.modelName, re.IGNORECASE):
            data = orgData.resample(Time='10T').asfreq()
        else:
            common.logger.error(f'모델 종류 ({self.modelName})를 확인해주세요.')
            sys.exit(1)

        # 예보 시간
        forDateList = data['Time'].values
        for idx, forDateInfo in enumerate(forDateList):
            forDate = pd.to_datetime(forDateInfo, format='%Y-%m-%d_%H:%M:%S')

            # modelType = 'KIER-LDAPS-2K' if re.search('KIER-LDAPS', self.modelName, re.IGNORECASE) else 'KIER-RDAPS-3K'
            # modelInfo = self.config['modelName'].get(f'{modelType}_{self.modelKey}')
            modelInfo = self.config['modelName'].get(f'{self.modelName}_{self.modelKey}')
            if modelInfo is None:
                common.logger.warn(f'설정 파일 (config.yml)에서 설정 정보 (KIER-LDAPS/RDAPS, UNIS/PRES/ALL)를 확인해주세요.')
                continue

            common.logger.info(f'[CHECK] anaDate : {anaDate} / forDate : {forDate}')

            # DB 등록/수정
            self.dbData = {}
            self.dbData['ANA_DT'] = anaDate
            self.dbData['FOR_DT'] = forDate
            self.dbData['MODEL_TYPE'] = self.modelName

            # 선택 컬럼
            for j, varInfo in enumerate(modelInfo['varName']):
                name = varInfo['name']
                for level, colName in zip(varInfo['level'], varInfo['colName']):
                    if data.get(name) is None: continue

                    try:
                        if level == '-1':
                            if len(data[name].isel(Time=idx).values) < 1: continue
                            self.dbData[colName] = self.convFloatToIntList(data[name].isel(Time=idx).values)
                        else:
                            if len(data[name].isel(Time=idx, bottom_top=int(level)).values) < 1: continue
                            self.dbData[colName] = self.convFloatToIntList(data[name].isel(Time=idx, bottom_top=int(level)).values)
                    except Exception as e:
                        common.logger.error(f'Exception : {e}')

            if len(self.dbData) < 1:
                common.logger.error(f'해당 파일 ({self.inFile})에서 지표면 및 상층 데이터를 확인해주세요.')
                sys.exit(1)

            common.logger.info(f'[CHECK] dbData : {self.dbData.keys()} : {np.shape(self.dbData[list(self.dbData.keys())[3]])}')
            dbapp.dbMergeData(initDB['session'], initDB['tbIntModel'], self.dbData, pkList=['MODEL_TYPE', 'ANA_DT', 'FOR_DT'])

    def getVar(self):
        self.varNameLists = self.config['modelName'][f'{self.modelName}_{self.modelKey}']['varName']

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