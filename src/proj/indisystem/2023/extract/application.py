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
import pygrib
import wrf
from netCDF4 import Dataset
import os

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
            self.processWrfKIER()
            # self.processXarrayKIER()
        elif re.search('KIM|LDAPS|RDAPS', self.modelName, re.IGNORECASE):
            gribApp = Grib12(self.inFile)
            gribApp.openFile()
            self.processKMA(gribApp)
        elif re.search('GFS', self.modelName, re.IGNORECASE):
            self.processGFS()
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

        common.logger.info(f'[CHECK] dbData : {self.dbData.keys()} : {np.shape(self.dbData[list(self.dbData.keys())[3]])} : {len(self.dbData.keys())}')
        dbapp.dbMergeData(initDB['session'], initDB['tbIntModel'], self.dbData, pkList=['MODEL_TYPE', 'ANA_DT', 'FOR_DT'])

    def processWrfKIER(self):

        # DB 가져오기
        dbapp = ManageDB(self.dbconfig)
        initDB = dbapp.initCfgInfo()

        # NetCDF 파일 읽기
        orgData = Dataset(self.inFile, 'r')
        common.logger.info(f'[CHECK] inFile : {self.inFile}')

        # 분석시간
        anaDate = pd.to_datetime(orgData.START_DATE, format='%Y-%m-%d_%H:%M:%S')

        # 시간 인덱스를 예보 시간 기준으로 변환
        timeByteList = wrf.getvar(orgData, "times", timeidx=wrf.ALL_TIMES).values
        timeList = pd.to_datetime(timeByteList)

        # 그룹핑
        grpData = pd.Series(range(len(timeList)), index=timeList)

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
            grpDataL1 = grpData.resample('60T')
        elif re.search('30M', self.modelName, re.IGNORECASE):
            grpDataL1 = grpData.resample('30T')
        elif re.search('KIER-LDAPS|KIER-RDAPS', self.modelName, re.IGNORECASE):
            grpDataL1 = grpData.resample('60T')
        elif re.search('KIER-WIND', self.modelName, re.IGNORECASE):
            grpDataL1 = grpData.resample('10T')
        else:
            common.logger.error(f'모델 종류 ({self.modelName})를 확인해주세요.')
            sys.exit(1)

        for forDate, group in grpDataL1:
            if re.search('60M', self.modelName, re.IGNORECASE):
                timeIdxList = group.values
            elif re.search('30M', self.modelName, re.IGNORECASE):
                timeIdxList = group.values
            elif re.search('KIER-LDAPS|KIER-RDAPS', self.modelName, re.IGNORECASE):
                timeIdxList = [group.values[0]] if forDate == grpData.index[group.values[0]] else None
            elif re.search('KIER-WIND', self.modelName, re.IGNORECASE):
                timeIdxList = [group.values[0]] if forDate == grpData.index[group.values[0]] else None
            else:
                common.logger.error(f'모델 종류 ({self.modelName})를 확인해주세요.')
                sys.exit(1)

            if timeIdxList is None: continue

            # DB 등록/수정
            self.dbData = {}
            self.dbData['ANA_DT'] = anaDate
            self.dbData['FOR_DT'] = forDate
            self.dbData['MODEL_TYPE'] = self.modelName

            modelInfo = self.config['modelName'].get(f'{self.modelName}_{self.modelKey}')
            if modelInfo is None:
                common.logger.warn(f'설정 파일 (config.yml)에서 설정 정보 (KIER-LDAPS/RDAPS, UNIS/PRES/ALL)를 확인해주세요.')
                continue

            # common.logger.info(f'[CHECK] anaDate : {anaDate} / forDate : {forDate} / timeIdxList : {timeIdxList} / timeList : {timeList[timeIdxList]}')
            common.logger.info(f'[CHECK] anaDate : {anaDate} / forDate : {forDate} / timeIdxList : {timeIdxList}')

            # 선택 컬럼
            for j, varInfo in enumerate(modelInfo['varName']):
                name = varInfo['name']
                for level, colName in zip(varInfo['level'], varInfo['colName']):
                    try:
                        valList = []
                        for timeIdx in timeIdxList:
                            if level == '-1':
                                val = wrf.getvar(orgData, name, timeidx=timeIdx)
                            else:
                                pressure = wrf.getvar(orgData, "pressure", timeidx=timeIdx)
                                # selVal = wrf.getvar(orgData, name, units="kt", timeidx=timeIdx)
                                selVal = wrf.getvar(orgData, name, units="m/s", timeidx=timeIdx)

                                # level = 1000
                                # bottom_top = 0
                                # level = 975
                                # bottom_top = 1
                                # level = 950
                                # bottom_top = 2
                                # level = 925
                                # bottom_top = 3
                                # level = 900
                                # bottom_top = 4
                                # level = 875
                                # bottom_top = 5
                                level = 850
                                bottom_top = 6

                                # NEW 방법
                                # val = wrf.interplevel(selVal, pressure, int(level))
                                # NEW2 방법
                                val = wrf.vinterp(orgData, field = selVal, vert_coord = 'pressure', interp_levels = [int(level)], extrapolate=True, timeidx=timeIdx)

                                # mainTitle = f'NEW (wrf-python) / anaDate = {anaDate} \n forDate = {forDate} / level = {int(level)} hPa'
                                mainTitle = f'NEW2 (wrf-python) / anaDate = {anaDate} \n forDate = {forDate} / level = {int(level)} hPa'
                                val.plot()
                                plt.title(mainTitle)
                                # saveImg = '{}/{}.png'.format('/DATA/FIG/INDI2023/TEST', f'NEW-{int(level)}')
                                saveImg = '{}/{}.png'.format('/DATA/FIG/INDI2023/TEST', f'NEW2-{int(level)}')
                                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                                plt.show()
                                plt.close()

                                aa = xr.open_dataset(self.inFile)
                                aa2 = aa['U'].isel(Time = 0, bottom_top=bottom_top)

                                mainTitle = f'ORG (xarray) / anaDate = {anaDate} \n forDate = {forDate} / level = {int(level)} hPa'
                                aa2.plot()
                                plt.title(mainTitle)
                                saveImg = '{}/{}.png'.format('/DATA/FIG/INDI2023/TEST', f'ORG-{int(level)}')
                                os.makedirs(os.path.dirname(saveImg), exist_ok=True)
                                plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
                                plt.show()
                                plt.close()

                            if len(val) < 1: continue
                            valList.append(val)

                        if len(valList) < 1: continue
                        meanVal = np.nanmean(np.nan_to_num(valList), axis=0)
                        self.dbData[colName] = self.convFloatToIntList(meanVal)
                    except Exception as e:
                        common.logger.error(f'Exception : {e}')

            if len(self.dbData) < 1:
                common.logger.error(f'해당 파일 ({self.inFile})에서 지표면 및 상층 데이터를 확인해주세요.')
                sys.exit(1)

            common.logger.info(f'[CHECK] dbData : {self.dbData.keys()} : {np.shape(self.dbData[list(self.dbData.keys())[3]])} : {len(self.dbData.keys())}')
            dbapp.dbMergeData(initDB['session'], initDB['tbIntModel'], self.dbData, pkList=['MODEL_TYPE', 'ANA_DT', 'FOR_DT'])

    def processXarrayKIER(self):

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

            common.logger.info(f'[CHECK] dbData : {self.dbData.keys()} : {np.shape(self.dbData[list(self.dbData.keys())[3]])} : {len(self.dbData.keys())}')
            dbapp.dbMergeData(initDB['session'], initDB['tbIntModel'], self.dbData, pkList=['MODEL_TYPE', 'ANA_DT', 'FOR_DT'])

    def processGFS(self):

        # DB 가져오기
        dbapp = ManageDB(self.dbconfig)
        initDB = dbapp.initCfgInfo()

        # 관심영역 설정
        roi = {'minLat': 22.0, 'maxLat': 49.0, 'minLon': 108.0, 'maxLon': 147.0}

        # GFS 파일 읽기
        data = xr.open_dataset(self.inFile, engine='pynio').sel(lat_0=slice(roi['maxLat'], roi['minLat']), lon_0=slice(roi['minLon'], roi['maxLon']))
        common.logger.info(f'[CHECK] inFile : {self.inFile}')

        attrInfo = data[list(data.dtypes)[0]].attrs
        anaDate = pd.to_datetime(attrInfo['initial_time'], format="%m/%d/%Y (%H:%M)")
        forDate = anaDate + pd.DateOffset(hours = int(attrInfo['forecast_time'][0]))

        modelInfo = self.config['modelName'].get(f'{self.modelName}_{self.modelKey}')
        if modelInfo is None:
            common.logger.warn(f'설정 파일 (config.yml)에서 설정 정보 (GFS-25K, ALL)를 확인해주세요.')
            sys.exit(1)

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
                        if len(data[name].values) < 1: continue
                        self.dbData[colName] = self.convFloatToIntList(data[name].values)
                    else:
                        if len(data[name].isel(lv_ISBL0=int(level)).values) < 1: continue
                        self.dbData[colName] = self.convFloatToIntList(data[name].isel(lv_ISBL0=int(level)).values)
                except Exception as e:
                    common.logger.error(f'Exception : {e}')

        if len(self.dbData) < 1:
            common.logger.error(f'해당 파일 ({self.inFile})에서 지표면 및 상층 데이터를 확인해주세요.')
            sys.exit(1)

        common.logger.info(f'[CHECK] dbData : {self.dbData.keys()} : {np.shape(self.dbData[list(self.dbData.keys())[3]])} : {len(self.dbData.keys())}')
        dbapp.dbMergeData(initDB['session'], initDB['tbIntModel'], self.dbData, pkList=['MODEL_TYPE', 'ANA_DT', 'FOR_DT'])

    def getVar(self):
        self.varNameLists = self.config['modelName'][f'{self.modelName}_{self.modelKey}']['varName']

    def convFloatToIntList(self, val):
        scaleFactor = 10000
        addOffset = 0
        result = np.where(~np.isnan(val), ((np.around(val, 4) * scaleFactor) - addOffset).astype(int), np.nan)
        return result.tolist()

    """
    def insertData(self,):
        #decoding grib data
        # insert DB
    def getConfig(self) :
        return 	
    """

    # dtSrtDate = pd.to_datetime('2022-01-01', format='%Y-%m-%d')
    # dtEndDate = pd.to_datetime('2022-01-02', format='%Y-%m-%d')
    # timeList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='10T')

    # lat1D = wrf.getvar(orgData, 'XLAT', timeidx=0)
    # lon1D = wrf.getvar(orgData, 'XLONG', timeidx=0)
    # val1D = val
    # val1D = meanVal

    # val1D.plot()
    # plt.show()

    # plt.scatter(lon1D, lat1D, c=val1D)
    # plt.colorbar()
    # plt.show()