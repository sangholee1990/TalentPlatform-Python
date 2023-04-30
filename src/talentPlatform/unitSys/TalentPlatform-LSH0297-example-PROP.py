# -*- coding: utf-8 -*-

###########################
# load packages
###########################
import glob
import xarray as xr
from datetime import datetime
import os
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

print('[START] {}'.format("TalentPlatform-LSH0297-example-PROP.py"))

###########################
# TEST
###########################
dtSrtDate = pd.to_datetime('2020-04-05', format='%Y-%m-%d')
dtEndDate = pd.to_datetime('2020-04-10', format='%Y-%m-%d')
# dtSrtDate = pd.to_datetime('2015-06-01', format='%Y-%m-%d')
# dtEndDate = pd.to_datetime('2015-06-03', format='%Y-%m-%d')
# dtSrtDate = pd.to_datetime('2020-07-02', format='%Y-%m-%d')
# dtEndDate = pd.to_datetime('2020-07-08', format='%Y-%m-%d')
dtIncDateList = pd.date_range(start=dtSrtDate, end=dtEndDate, freq='1D')

for i, dtIncDateInfo in enumerate(dtIncDateList):

    dtYmd = dtIncDateInfo.strftime('%Y%m%d')
    dtYmdFmt = dtIncDateInfo.strftime('%Y-%m-%d')

    # inpFile = '/home/sbpark/data/Satellite/SMAP/NEE/SMAP_L4_C_mdl_{}*_Vv6042_001.h5'.format(dtYmd)
    inpFile = 'DATA/SMAP_L4_C_mdl_{}*_Vv6042_001.h5'.format(dtYmd)
    fileList = sorted(glob.glob(inpFile))
    if fileList is None or len(fileList) < 1:
        print('[ERROR] inpFile : {} / {}'.format(inpFile, '입력 자료를 확인해주세요.'))
        continue

    fileInfo = fileList[0]
    print("[CHECK] fileInfo : ", fileInfo)

    # **********************************************
    # QA 데이터 전처리
    # **********************************************
    qaData = xr.open_mfdataset(fileInfo, engine = 'netcdf4', group='QA')
    qaFlag = qaData['carbon_model_bitflag']
    qaFlagData = qaFlag.to_dataframe().dropna().reset_index()
    qaFlagData['bit'] = list(map('{0:016b}'.format, qaFlagData['carbon_model_bitflag'].astype('int')))

    # carbon_model_bitflag : 빅엔디안 방식으로서 우측을 기준으로 정보 추출 (큰 주소 > 작은 주소)
    qaFlagData['isFill'] = qaFlagData['bit'].str[0:1].apply(lambda x: int(x, 2))
    qaFlagData['ftMethod'] = qaFlagData['bit'].str[1:2].apply(lambda x: int(x, 2))
    qaFlagData['ndviMethod'] = qaFlagData['bit'].str[2:3].apply(lambda x: int(x, 2))
    qaFlagData['gppMethod'] = qaFlagData['bit'].str[3:4].apply(lambda x: int(x, 2))
    qaFlagData['qaScore'] = qaFlagData['bit'].str[4:8].apply(lambda x: int(x, 2))
    qaFlagData['pftDom'] = qaFlagData['bit'].str[8:12].apply(lambda x: int(x, 2))
    qaFlagData['socBit'] = qaFlagData['bit'].str[12:13].apply(lambda x: int(x, 2))
    qaFlagData['rhBit'] = qaFlagData['bit'].str[13:14].apply(lambda x: int(x, 2))
    qaFlagData['gppBit'] = qaFlagData['bit'].str[14:15].apply(lambda x: int(x, 2))
    qaFlagData['neeBit'] = qaFlagData['bit'].str[15:16].apply(lambda x: int(x, 2))

    # 요약 통계량
    # qaFlagSummary = qaFlagData.describe()

    # 2. 격자별로 자료의 PFT code가 1인 경우의 변수값(NEE)의 최대값을 찾아 mapping함.
    # 년도가 2015-2020년 일경우 각 년도별로 그림이 6장이 나올 수 있어야 함.
    # 이 최대값을 원하는 위경도,년도,월에 대해서 2015-2020년까지 출력하여 csv로 저장

    # **********************************************
    # GEO 데이터 전처리
    # **********************************************
    geoData = xr.open_mfdataset(fileInfo, engine = 'netcdf4', group='GEO')
    geoDataL1 = geoData.to_dataframe().reset_index()

    # **********************************************
    # NEE 데이터 전처리
    # **********************************************
    neeData = xr.open_mfdataset(fileInfo, engine = 'netcdf4', group='NEE')
    neeMean = neeData['nee_mean']
    neePft1 = neeData['nee_pft1_mean']
    neePft2 = neeData['nee_pft2_mean']
    neePft3 = neeData['nee_pft3_mean']
    neePft4 = neeData['nee_pft4_mean']
    neePft5 = neeData['nee_pft5_mean']
    neePft6 = neeData['nee_pft6_mean']
    neePft7 = neeData['nee_pft7_mean']
    neePft8 = neeData['nee_pft8_mean']

#    neePftData = pd.concat(
#        [
#            neePft1.to_dataframe().reset_index()[['nee_pft1_mean']]
#            , neePft2.to_dataframe().reset_index()[['nee_pft2_mean']]
#            , neePft3.to_dataframe().reset_index()[['nee_pft3_mean']]
#            , neePft4.to_dataframe().reset_index()[['nee_pft4_mean']]
#            , neePft5.to_dataframe().reset_index()[['nee_pft5_mean']]
#            , neePft6.to_dataframe().reset_index()[['nee_pft6_mean']]
#            , neePft7.to_dataframe().reset_index()[['nee_pft7_mean']]
#            , neePft8.to_dataframe().reset_index()[['nee_pft8_mean']]
#        ]
#        , axis=1
#    )

    neePftData = pd.concat(
        [
            neePft2.to_dataframe().reset_index()[['nee_pft2_mean']]
            , neePft5.to_dataframe().reset_index()[['nee_pft5_mean']]
        ]
        , axis=1
    )

    maxData = neePftData.min(axis=1)	# missing values are excluded????
    # maxDataL1 = pd.concat([neePftData, maxData], axis=1)
    meanData = neeMean.to_dataframe().reset_index()

    # statData = pd.concat([meanData, maxData], axis=1).rename(
    statData = pd.concat([meanData, neePftData], axis=1).rename(
        columns={
            0: 'nee_max'
        }
    )

    # **********************************************
    # 데이터 병합
    # **********************************************
    dataL1 = statData.merge(right=qaFlagData, left_on = ['x', 'y'], right_on = ['x', 'y'], how='left')\
            .merge(right=geoDataL1, left_on = ['x', 'y'], right_on = ['x', 'y'], how='left')

    xEle = geoData['x']
    yEle = geoData['y'][::-1]

    saveData = xr.Dataset(
        {
            'lat': (('time', 'y', 'x'),  (dataL1['latitude'].values.reshape(1, len(yEle), len(xEle))))
            , 'lon': (('time', 'y', 'x'),  (dataL1['longitude'].values.reshape(1, len(yEle), len(xEle))))
            , 'meanNee': (('time', 'y', 'x'),  (dataL1['nee_mean'].values.reshape(1, len(yEle), len(xEle))))
            # 2022.04.24 각각 neePftData 변경
            # , 'maxNee': (('time', 'y', 'x'), (dataL1['nee_max'].values.reshape(1, len(yEle), len(xEle))))
            , 'meanNeePft2': (('time', 'y', 'x'), (dataL1['nee_pft2_mean'].values.reshape(1, len(yEle), len(xEle))))
            , 'meanNeePft5': (('time', 'y', 'x'), (dataL1['nee_pft5_mean'].values.reshape(1, len(yEle), len(xEle))))
            , 'neeBit': (('time', 'y', 'x'), (dataL1['neeBit'].values.reshape(1, len(yEle), len(xEle))))
            , 'qaScore': (('time', 'y', 'x'), (dataL1['qaScore'].values.reshape(1, len(yEle), len(xEle))))
            , 'pftDom': (('time', 'y', 'x'), (dataL1['pftDom'].values.reshape(1, len(yEle), len(xEle))))
            , 'ndviMethod': (('time', 'y', 'x'), (dataL1['ndviMethod'].values.reshape(1, len(yEle), len(xEle))))
            , 'ftMethod': (('time', 'y', 'x'), (dataL1['ftMethod'].values.reshape(1, len(yEle), len(xEle))))
            , 'isFill': (('time', 'y', 'x'), (dataL1['isFill'].values.reshape(1, len(yEle), len(xEle))))
        }
        , coords={
            'time': pd.date_range(dtIncDateInfo, periods=1)
            , 'x': xEle
            , 'y': yEle
        }
    )
    ###########################
    # save
    ###########################
    # saveNcFile = './SMAP_L4_ORG_{}.nc'.format(dtYmd)
    # os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
    # saveData.to_netcdf(saveNcFile)
    #
    #saveCsvFile = './SMAP_L4_ORG_{}.csv'.format(dtYmd)
    #os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)
    #saveData.to_dataframe().reset_index().to_csv(saveCsvFile)

    ###########################
    # masking
    ###########################
    saveDataL1 = saveData.where(
        (
                (saveData['neeBit'] == 0)
                & (saveData['qaScore'] == 0) | (saveData['qaScore'] == 1)
                & (saveData['ndviMethod'] == 0)
                & (saveData['ftMethod'] == 0)
#                & (saveData['isFill'] == 0)
         )
    )

    ###########################
    # save
    ###########################
    saveNcFile = 'TMP2/SMAP_L4_MASK_{}.nc'.format(dtYmd)
    os.makedirs(os.path.dirname(saveNcFile), exist_ok=True)
    saveDataL1.to_netcdf(saveNcFile)

    saveCsvFile = 'TMP2/SMAP_L4_MASK_{}.csv'.format(dtYmd)
    os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)
    saveDataL1.to_dataframe().reset_index().to_csv(saveCsvFile)

    ###########################
    # Fig
    ###########################
    # xEle1D = saveData['x'].values
    # yEle1D = saveData['y'].values
    # time1D = saveDataL1['time'].values
    # val2D = saveDataL1.sel(time = time1D[0])['maxNee'].values
    #
    # saveImg = './SMAP_L4_ORG_{}.png'.format(dtYmd)
    # plt.contourf(xEle1D, yEle1D, val2D, cmap=plt.cm.RdBu_r, vmin=-10.0, vmax=10.0)
    # plt.title( os.path.basename(saveImg))
    # plt.colorbar()
    # plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    # plt.show()

    xEle1D = saveDataL1['x'].values
    yEle1D = saveDataL1['y'].values
    time1D = saveDataL1['time'].values
    val2D = saveDataL1.sel(time = time1D[0])['meanNeePft2'].values

    saveImg = 'FIG/SMAP_L4_MASK_PFT2_{}.png'.format(dtYmd)
    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
    plt.contourf(xEle1D, yEle1D, val2D, cmap=plt.cm.RdBu_r, vmin=-10.0, vmax=10.0)
    plt.title(os.path.basename(saveImg))
    plt.colorbar()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.close()
    plt.show()


    xEle1D = saveDataL1['x'].values
    yEle1D = saveDataL1['y'].values
    time1D = saveDataL1['time'].values
    val2D = saveDataL1.sel(time = time1D[0])['meanNeePft5'].values

    saveImg = 'FIG/SMAP_L4_MASK_PFT5_{}.png'.format(dtYmd)
    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
    plt.contourf(xEle1D, yEle1D, val2D, cmap=plt.cm.RdBu_r, vmin=-10.0, vmax=10.0)
    plt.title(os.path.basename(saveImg))
    plt.colorbar()
    plt.savefig(saveImg, dpi=600, bbox_inches='tight')
    plt.close()
    plt.show()

print('[END] {}'.format("TalentPlatform-LSH0297-example-PROP.py"))