# -*- coding: utf-8 -*-

from src.talentPlatform.unitSysHelper.InitConfig import *
import logging
import logging.handlers
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import glob
import platform
import xarray as xr
import cfgrib
import pandas as pd
from datetime import timedelta, date
import numpy as np

warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
sns.set(font="Malgun Gothic", rc={"axes.unicode_minus":False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# =================================================
# Set Env
# =================================================
# 작업환경 경로 설정
contextPath = '/SYSTEMS/PROG/PYTHON/GRIB2'
prjName = 'test'
serviceName = 'LSH0175'

log = initLog(contextPath, prjName)
globalVar = initGlobalVar(contextPath, prjName)

# =================================================
# Option
# =================================================
inLat = 37.56083
inLon = 129.2172
inVars = ['shww', 'wvdir', 'pp1d', 'u', 'v']

# =================================================
# Main
# =================================================
try:
    log.info('[START] {}'.format('Main'))

    fileList = glob.glob('{}/{}'.format(globalVar['inpPath'], '*.gb2'))

    for fileInfo in fileList:
        log.info('[CHECK] fileInfo : {}'.format(fileInfo))
        
        ds = cfgrib.open_dataset(fileInfo)
        dtDateTime = pd.to_datetime(ds.time.values).strftime('%Y%m%d%H%M')
        log.info('[CHECK] ds : {}'.format(ds))

        for inVar in inVars:
          log.info('[CHECK] inVar : {}'.format(inVar))

          # 오랜 시간이 소요
          # df = ds[inVar].to_dataframe()
          # saveFile = '{}/{}_{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, inVar, 'TOTAL_DATA', dtDateTime)
          # df.to_csv(saveFile, index=None)

          dataL2 = pd.DataFrame()
          for i in ds[inVar]:
            dtDateTimeFore = pd.to_datetime((ds.time + i.step).values).strftime('%Y%m%d%H%M')
            nearVal = np.nan_to_num(i.sel(latitude = inLat, longitude = inLon, method='nearest'))
            interpVal = np.nan_to_num(i.interp(latitude = inLat, longitude = inLon))
      
            dict = {
              'inVar': [inVar]
              , 'dtDateTime': [dtDateTime]
              , 'dtDateTimeFore': [dtDateTimeFore]
              , 'nearVal': [nearVal]
              , 'interpVal': [interpVal]
            }
            log.info('[CHECK] dict : {}'.format(dict))
  
            dataL2 = dataL2.append(pd.DataFrame.from_dict(dict))

          saveFile = '{}/{}_{}_{}_{}.csv'.format(globalVar['outPath'], serviceName, inVar, 'SEL_DATA', dtDateTime)
          dataL2.to_csv(saveFile, index=None)
      
except Exception as e:
    log.error("Exception : {}".format(e))
    # traceback.print_exc()
    # sys.exit(1)

finally:
    log.info('[END] {}'.format('Main'))
