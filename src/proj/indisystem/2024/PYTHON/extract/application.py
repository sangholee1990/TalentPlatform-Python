# -*- coding: utf-8 -*-

import datetime
import numpy as np
from common.grib12 import Grib12
from manageDB import ManageDB
import re
import sys
import xarray as xr
import pandas as pd
import common.initiator as common
import wrf
from netCDF4 import Dataset
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units
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
        if re.search('KIER-LDAPS-0.6K', self.modelName, re.IGNORECASE):
            self.processWrfWindKIER()
        elif re.search('KIER-LDAPS|KIER-RDAPS|KIER-WIND', self.modelName, re.IGNORECASE):
            self.processWrfKIER()
            # self.processXarrayKIER()
        elif re.search('KIM|LDAPS|RDAPS', self.modelName, re.IGNORECASE):
            gribApp = Grib12(self.inFile)
            gribApp.openFile()
            self.processKMA(gribApp)
        elif re.search('GFS', self.modelName, re.IGNORECASE):
            self.processGFS()
        elif re.search('REANALY-ERA5', self.modelName, re.IGNORECASE):
            self.processReanalyEra5()
        elif re.search('SAT-SENT1', self.modelName, re.IGNORECASE):
            self.processSatSent1()
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
        # orgData.variables
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
        elif re.search('KIER-LDAPS-2K-ORG|KIER-RDAPS-3K-ORG', self.modelName, re.IGNORECASE):
            grpDataL1 = grpData.resample('3T')
        elif re.search('KIER-LDAPS|KIER-RDAPS', self.modelName, re.IGNORECASE):
            grpDataL1 = grpData.resample('60T')
        elif re.search('KIER-WINDre', self.modelName, re.IGNORECASE):
            grpDataL1 = grpData.resample('30T')
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
            elif re.search('KIER-LDAPS-2K-ORG|KIER-RDAPS-3K-ORG', self.modelName, re.IGNORECASE):
                timeIdxList = [group.values[0]] if forDate == grpData.index[group.values[0]] else None
            elif re.search('KIER-LDAPS|KIER-RDAPS', self.modelName, re.IGNORECASE):
                timeIdxList = [group.values[0]] if forDate == grpData.index[group.values[0]] else None
            elif re.search('KIER-WINDre', self.modelName, re.IGNORECASE):
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
                                if re.search('RH', name, re.IGNORECASE):
                                    val = wrf.g_rh.get_rh_2m(orgData, timeidx=timeIdx)
                                else:
                                    val = wrf.getvar(orgData, name, timeidx=timeIdx)
                            else:
                                # pressure = wrf.getvar(orgData, 'pressure', timeidx=timeIdx)
                                selVal = wrf.getvar(orgData, name, units=varInfo['unit'], timeidx=timeIdx)
                                # val = wrf.interplevel(selVal, pressure, int(level))
                                val = wrf.vinterp(orgData, field = selVal, vert_coord = 'pressure', interp_levels = [int(level)], extrapolate=True, timeidx=timeIdx).isel(interp_level = 0)
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
                        if re.search('ALB', name, re.IGNORECASE):
                            self.dbData[colName] = self.convFloatToIntList(data[name].values / 100)
                        else:
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

    def processReanalyEra5(self):

        # DB 가져오기
        dbapp = ManageDB(self.dbconfig)
        initDB = dbapp.initCfgInfo()

        # REANALY-ERA5 파일 읽기
        orgData = xr.open_dataset(self.inFile, engine='pynio')
        common.logger.info(f'[CHECK] inFile : {self.inFile}')

        modelInfo = self.config['modelName'].get(f'{self.modelName}_{self.modelKey}')
        if modelInfo is None:
            common.logger.warn(f'설정 파일 (config.yml)에서 설정 정보 (REANALY-ERA5-25K, UNIS|PRES)를 확인해주세요.')
            sys.exit(1)

        timeList = orgData['time'].values
        for timeInfo in timeList:

            data = orgData.sel(time=timeInfo)
            anaDate = pd.to_datetime(timeInfo, format='%Y-%m-%d_%H:%M:%S')
            forDate = anaDate

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
                            if re.search('t2m', name, re.IGNORECASE):
                                # 상대습도 계산 (t2m 기온 K, d2m 이슬점온도 K)
                                rh = relative_humidity_from_dewpoint(data[name].values * units.kelvin, data['d2m'].values * units.kelvin).to('percent')
                                self.dbData[colName] = self.convFloatToIntList(rh.magnitude)
                            elif re.search('ssrd|ssrdc|fdir|ssr', name, re.IGNORECASE):
                                # 단위 변환 (Jm-2 -> Wm-2)
                                self.dbData[colName] = self.convFloatToIntList(data[name].values / 3600)
                            else:
                                self.dbData[colName] = self.convFloatToIntList(data[name].values)
                        else:
                            if len(data[name].sel(level=int(level)).values) < 1: continue
                            self.dbData[colName] = self.convFloatToIntList(data[name].sel(level=int(level)).values)
                    except Exception as e:
                        common.logger.error(f'Exception : {e}')

            if len(self.dbData) < 1:
                common.logger.error(f'해당 파일 ({self.inFile})에서 지표면 및 상층 데이터를 확인해주세요.')
                sys.exit(1)

            common.logger.info(f'[CHECK] dbData : {self.dbData.keys()} : {np.shape(self.dbData[list(self.dbData.keys())[3]])} : {len(self.dbData.keys())}')
            dbapp.dbMergeData(initDB['session'], initDB['tbIntModel'], self.dbData, pkList=['MODEL_TYPE', 'ANA_DT', 'FOR_DT'])

    def processSatSent1(self):

        # DB 가져오기
        dbapp = ManageDB(self.dbconfig)
        initDB = dbapp.initCfgInfo()

        # SAT-SENT1 파일 읽기
        isExist = os.path.exists(self.inFile)
        if not isExist:
            common.logger.warn(f'입력 파일 ({self.inFile}을 확인해주세요.')
            sys.exit(1)

        modelInfo = self.config['modelName'].get(f'{self.modelName}_{self.modelKey}')
        if modelInfo is None:
            common.logger.warn(f'설정 파일 (config.yml)에서 설정 정보 (SAT-SENT1, ALL)를 확인해주세요.')
            sys.exit(1)

        # 정규 표현식을 사용하여 날짜와 시간 부분 추출
        sDateTime = re.search(r'_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})', self.inFile).group(1)

        # 추출된 문자열을 datetime 객체로 변환
        dtDateTime = pd.to_datetime(sDateTime, format='%Y_%m_%d_%H_%M_%S')

        anaDate = dtDateTime
        forDate = dtDateTime

        common.logger.info(f'[CHECK] anaDate : {anaDate} / forDate : {forDate}')

        # DB 등록/수정
        self.dbData = {}
        self.dbData['ANA_DT'] = anaDate
        self.dbData['FOR_DT'] = forDate
        self.dbData['MODEL_TYPE'] = self.modelName

        if len(self.dbData) < 1:
            common.logger.error(f'해당 파일 ({self.inFile})에서 지표면 및 상층 데이터를 확인해주세요.')
            sys.exit(1)

        common.logger.info(f'[CHECK] dbData : {self.dbData.keys()} : {np.shape(self.dbData[list(self.dbData.keys())[2]])} : {len(self.dbData.keys())}')
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
        elif re.search('KIER-WINDre', self.modelName, re.IGNORECASE):
            data = orgData.resample(Time='30T').asfreq()
        elif re.search('KIER-WIND', self.modelName, re.IGNORECASE):
            data = orgData.resample(Time='10T').asfreq()
        else:
            common.logger.error(f'모델 종류 ({self.modelName})를 확인해주세요.')
            sys.exit(1)

        # 예보 시간
        forDateList = data['Time'].values
        for idx, forDateInfo in enumerate(forDateList):
            forDate = pd.to_datetime(forDateInfo, format='%Y-%m-%d_%H:%M:%S')

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

    def processWrfWindKIER(self):

        # DB 가져오기
        dbapp = ManageDB(self.dbconfig)
        initDB = dbapp.initCfgInfo()

        # NetCDF 파일 읽기
        orgData = xr.open_dataset(self.inFile)
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
        elif re.search('10M', self.modelName, re.IGNORECASE):
            data = orgData.resample(Time='10T').mean(dim='Time', skipna=True)
        elif re.search('KIER-LDAPS-0.6K-ORG', self.modelName, re.IGNORECASE):
            # data = orgData.resample(Time='60T').asfreq()
            data = orgData
        else:
            common.logger.error(f'모델 종류 ({self.modelName})를 확인해주세요.')
            sys.exit(1)

        # 예보 시간
        forDateList = data['Time'].values
        for forDateIdx, forDateInfo in enumerate(forDateList):
            forDate = pd.to_datetime(forDateInfo, format='%Y-%m-%d_%H:%M:%S')

            modelInfo = self.config['modelName'].get(f'{self.modelName}_{self.modelKey}')
            if modelInfo is None:
                common.logger.warn(f'설정 파일 (config.yml)에서 설정 정보 (KIER-LDAPS-0.6K, ALL)를 확인해주세요.')
                continue

            common.logger.info(f'[CHECK] anaDate : {anaDate} / forDate : {forDate}')

            # DB 등록/수정
            self.dbData = {}
            self.dbData['ANA_DT'] = anaDate
            self.dbData['FOR_DT'] = forDate
            self.dbData['MODEL_TYPE'] = self.modelName

            maxK = 9
            shapeList = [data.variables[var].shape for var in ['U', 'V', 'PH', 'PHB']]
            minShape = [min(dim) for dim in zip(*shapeList)]
            mt, mz, mlat, mlon = minShape

            # 선택 컬럼
            for j, varInfo in enumerate(modelInfo['varName']):
                name = varInfo['name']

                for level, colName in zip(varInfo['level'], varInfo['colName']):
                    try:
                        if not re.search('WSP', name, re.IGNORECASE): continue

                        # 2024.03.04 KIER 소스코드 참조 및 최적화
                        U = data['U'][forDateIdx, :maxK, :mlat, :mlon].values[np.newaxis]
                        V = data['V'][forDateIdx, :maxK, :mlat, :mlon].values[np.newaxis]
                        PH = data['PH'][forDateIdx, :maxK + 1, :mlat, :mlon].values[np.newaxis]
                        PHB = data['PHB'][forDateIdx, :maxK + 1, :mlat, :mlon].values[np.newaxis]

                        H_s = ( PH + PHB ) / 9.80665
                        H = 0.5 * ( H_s[:,:-1] + H_s[:,1:])

                        # 특정 고도에 따른 풍향/풍속 계산
                        result = self.calcWsdWdr(U, V, H, alt=int(level))
                        if len(result) < 1: continue

                        self.dbData[colName] = self.convFloatToIntList(result['WSP'])
                        # self.dbData[colName] = self.convFloatToIntList(result['WDR'])

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

        # result = np.where(~np.isnan(val), ((np.around(val, 4) * scaleFactor) - addOffset).astype(int), np.nan)
        result = ((np.around(val, 4) * scaleFactor) - addOffset).astype(int)

        return result.tolist()

    def revIntToFloatList(self, val):
        scaleFactor = 10000
        addOffset = 0

        result = (np.array(val) + addOffset) / scaleFactor

        return result.tolist()

    def calcWsdWdr(self, U, V, H, alt):

        nt, nz, nlat, nlon = U.shape
        U_int = U[nt - 1, :, :, :]
        V_int = V[nt - 1, :, :, :]
        H_int = H[nt - 1, :, :, :]

        WSP0 = np.sqrt(U_int[0] ** 2 + V_int[0] ** 2)
        WSP = np.log(np.sqrt(U_int ** 2 + V_int ** 2)) - np.log(WSP0)
        LHS = np.log(H_int / H_int[0])

        WSP00 = np.full((nlat, nlon), np.nan)
        WDR00 = np.full((nlat, nlon), np.nan)
        for i in range(nlon):
            for j in range(nlat):
                alp, _, _, _ = np.linalg.lstsq(LHS[:, j, i, np.newaxis], WSP[:, j, i], rcond=None)
                WSP00[j, i] = WSP0[j, i] * (alt / H_int[0, j, i]) ** alp[0]

                k = np.argmax(H_int[:, j, i] > alt)
                aa = (H_int[k + 1, j, i] - alt)
                bb = (alt - H_int[k, j, i])
                uEle = (U_int[k, j, i] * aa + U_int[k + 1, j, i] * bb) / (aa + bb)
                vEle = (V_int[k, j, i] * aa + V_int[k + 1, j, i] * bb) / (aa + bb)

                WDR00[j, i] = (np.arctan2(-uEle, -vEle) * 180.0 / np.pi) % 360.0

        # 문턱값 검사 (DB 내 결측값 -99999 표시)
        WSP00 = np.where(WSP00 < 100, WSP00, -9.9999)

        result = {
            'WSP': WSP00
            , 'WDR': WDR00
            , 'alt': alt
        }

        return result
