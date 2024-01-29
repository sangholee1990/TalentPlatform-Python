import glob
# import seaborn as sns
import logging
import logging.handlers
import logging.handlers
import os
import platform
import sys
import argparse
import traceback
import warnings
# import datetime as dt
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.cm as cm

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import pandas as pd
import math
from scipy import spatial
from pandas.tseries.offsets import Day, Hour, Minute, Second
from scipy.interpolate import Rbf
from numpy import zeros, newaxis
from urllib.parse import quote_plus
import pytz
#import requests
from sqlalchemy import create_engine
import re
import configparser
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import re
import configparser
from sqlalchemy.ext.declarative import declarative_base
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
import psycopg2
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.pool import QueuePool

            # *********************************************
            # [템플릿] 기본 위경도 정보를 DB 삽입
            # *********************************************
dbData = {}
modelType = 'KIER-LDAPS'
dbData['MODEL_TYPE'] = modelType
            #
            # # 지표
orgData = xr.open_mfdataset('/vol01/DATA/MODEL/KIER-LDAPS/wrfsolar_d02.2023-06-30_03:00:00.nc')
print(orgData)
data = orgData['SWDOWN'].isel(Time=0)
            #
dbData['LON_SFC'] = data['XLONG'].values.tolist() if len(data['XLONG'].values) > 0 else None
dbData['LAT_SFC'] = data['XLAT'].values.tolist() if len(data['XLAT'].values) > 0 else None
            #
            # # 상층
orgData2 = xr.open_mfdataset('/vol01/DATA/MODEL/KIER-LDAPS/wrfout_d02_2023-06-30_03:00:00.nc')
data2 = orgData2['U'].isel(Time = 0, bottom_top = 0)
dbData['LON_PRE'] = data2['XLONG_U'].values.tolist() if len(data2['XLONG_U'].values) > 0 else None
dbData['LAT_PRE'] = data2['XLAT_U'].values.tolist() if len(data2['XLAT_U'].values) > 0 else None
            # dbMergeData(cfgOpt['sessionMake'], cfgOpt['tbGeo'], dbData, pkList=['MODEL_TYPE'])

            # *********************************************
            # [템플릿] 상세 위경도 정보를 DB 삽입
            # *********************************************
sfcData = orgData['SWDOWN'].isel(Time=0).to_dataframe().reset_index(drop=False).rename(
    columns={
        'south_north': 'ROW'
        , 'west_east': 'COL'
        , 'XLAT': 'LAT_SFC'
        , 'XLONG': 'LON_SFC'
     }
 ).drop(['SWDOWN'], axis='columns')
            #
preData = orgData2['U'].isel(Time = 0, bottom_top = 0).to_dataframe().reset_index(drop=False).rename(
    columns={
        'south_north': 'ROW'
        , 'west_east_stag': 'COL'
        , 'XLAT_U': 'LAT_PRE'
        , 'XLONG_U': 'LON_PRE'
    }
 ).drop(['U', 'XTIME'], axis='columns')
dataL2 = pd.merge(left=sfcData, right=preData, how='inner', left_on=['ROW', 'COL'], right_on=['ROW', 'COL'])
dataL2['MODEL_TYPE'] = modelType
            #
dataList = dataL2.to_dict(orient='records')
#print(dataList)
            # dbMergeData(cfgOpt['sessionMake'], cfgOpt['tbGeoDtl'], dataList, pkList=['MODEL_TYPE', 'ROW', 'COL'])


