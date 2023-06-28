# -*- coding: utf-8 -*-

# 상호님에게 전달해주시면 감사하겠습니다
# 안에 3 모델이 있고 이 중에서 위 두 모델은 기존 climate data 고해상도 작업에 사용이 된 적이 없는 모델입니다.
# 연구 목표는 위 두 모델을 이용한 climate data (강우량, wind, solar 등이지만 현재는 배출량 데이터로 생각 중입니다)의 해상도 증가이며 기존의 srcnn이나 선형보간법의 performance를 넘기는 겁니다
#
# 강우량말고 solar wind데이터도
# 비슷하게 해보신 적은 없으신거지요?
#
# 아키텍처가 gan 같은 경우는 가능한 게 상상이 가는데 제가 transformer쪽 아키텍처를 잘 몰라서요 ㅠㅠ
#
# SWINIR
# https://arxiv.org/abs/2108.10257
# https://github.com/JingyunLiang/SwinIR
#


# 위 해당 모델 structure가 저희처럼 emission gridded data에도 적용이 되나요?
#
# DRLN
# https://arxiv.org/pdf/1906.12021v2.pdf
# https://paperswithcode.com/paper/densely-residual-laplacian-super-resolution#code
# https://github.com/saeed-anwar/DRLN
#
# 위 해당 모델 structure가 저희처럼 emission gridded data에도 적용이 되나요?
#
#
# SRCNN
# https://arxiv.org/pdf/1501.00092v3.pdf
# 요거는 기존 baseline 비교용으로 많이 사용된 거라서 타당성은 문제 없을 거 같습니다.
#
#
#
# 이 딥러닝 모델을 통해 고해상도를 만드는 거는 단독 프로젝트입니다!
#
# 결국 이런 resolution 연구는 validation data에서 고 해상도 데이터가 필요하잖아요
# 데이터가 필요하잖아요
# 22.09.14
# 18:05
# 그래서 제 생각은 예시로 저희 graced 데이터의 일부를( 특정 지역, 바다가 아닌 값이 많은)을 합치는 계산을 통해(interpolation이 아닌) 0.2도 0.4도 1도(적당한 큰 숫자)로 바꾸고 이게 training data가 되는 거는 어떨까요?
# 대부분 논문을 보면 내삽이랑 비하더라고요
# https://www.pnas.org/doi/10.1073/pnas.1918964117#data-availability
#
#
# https://github.com/NREL/PhIRE
#
# 그 머신러닝 모델 관련 논문입니다. 전에 보내드렸는지 기억이 잘 안나는데 그 당시의 가장 퍼포먼스가 좋던 모델인 ESRGAN을 활용하여 solar wind data 해상도를 증가한 연구입니다. 밑의 github링크에는 그 당시 학습에 사용했던 wind solar data가 있습니다! 가장 이상적인 거는 저희 GRACED 데이터셋으로 test하는 거지만 혹시나 몰라서 보냅니다!
#
#
# # NSRDB 입력 태양광 데이터
# https://nsrdb.nrel.gov/data-viewer/download/intro
#
# 1000x1000 그리드
# 이거를 200x200 저해상도 이미지 생성
# - swingIR,
#
# 1000x1000 NetCDF 파일 받아서...
# 미국이 좋음
# 위성데이터를 받을 수 있음.
# Daily 저해상도 낮추고
#
# ............ 5분, 10분 1년치 7000개,
# 교차검증 수행해서 모델학습 테스트/검증 데이터
#
# RMSE
#
# swingIR (주)
# , DR모델 (주)
# , PHIRE (PH 모델)
# , 기본 CNN모델
#  선형보간법


# conda activate py38
# cp ../../PhIRE/TEST/h08-test_200.png .

# 003 Real-World Image Super-Resolution (use --tile 400 if you run out-of-memory)
# (middle size)
# python main_test_swinir.py --task real_sr --scale 4 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq testsets/RealSRSet+5images --tile
# python main_test_swinir.py --task real_sr --scale 4 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq TEST --tile

# (larger size + trained on more datasets)
# python main_test_swinir.py --task real_sr --scale 4 --large_model --model_path model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq TEST

# 중해상도
# python main_test_swinir.py --task real_sr --scale 2 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth --folder_lq input &
# python main_test_swinir.py --task real_sr --scale 4 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq input &

# X
# python main_test_swinir.py --task real_sr --scale 1 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq input
# python main_test_swinir.py --task real_sr --scale 8 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq input

# 고해상도
# python main_test_swinir.py --task real_sr --scale 4  --large_model --model_path model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq input &

# X
# python main_test_swinir.py --task real_sr --scale 1  --large_model --model_path model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq input
# python main_test_swinir.py --task real_sr --scale 2  --large_model --model_path model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x2_GAN.pth --folder_lq input
# python main_test_swinir.py --task real_sr --scale 4  --large_model --model_path model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq input
# python main_test_swinir.py --task real_sr --scale 8  --large_model --model_path model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq input

import os
import sys
import glob

ctxPath = '/SYSTEMS/PROG/PYTHON/HIGH-IMG/PhIRE'

os.chdir(ctxPath)
sys.path.append(ctxPath)

from PhIREGANs import *

# WIND - LR-MR
#-------------------------------------------------------------

# data_type = 'wind'
# data_path = 'example_data/wind_LR-MR.tfrecord'
# model_path = 'models/wind_lr-mr/trained_gan/gan'
# r = [2, 5]
# mu_sig=[[0.7684, -0.4575], [4.9491, 5.8441]]


# WIND - MR-HR
#-------------------------------------------------------------
'''
data_type = 'wind'
data_path = 'example_data/wind_MR-HR.tfrecord'
model_path = 'models/wind_mr-hr/trained_gan/gan'
r = [5]
mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]]
'''

# SOLAR - LR-MR
#-------------------------------------------------------------
'''
data_type = 'solar'
data_path = 'example_data/solar_LR-MR.tfrecord'
model_path = 'models/solar_lr-mr/trained_gan/gan'
r = [5]
mu_sig=[[344.3262, 113.7444], [370.8409, 111.1224]]
'''

data_type = 'solar'
data_path = 'example_data/solar_LR-MR.tfrecord'
model_path = 'models/solar_lr-mr/trained_gan/gan'
r = [5]
# r = [2, 5]
mu_sig=[[344.3262, 113.7444], [370.8409, 111.1224]]

# SOLAR - MR-HR
#-------------------------------------------------------------
'''
data_type = 'solar'
data_path = 'example_data/solar_MR-HR.tfrecord'
model_path = 'models/solar_mr-hr/trained_gan/gan'
r = [5]
mu_sig = [[344.3262, 113.7444], [386.9283, 117.9627]]
'''

# data_type = 'solar'
# data_path = 'example_data/solar_MR-HR.tfrecord'
# model_path = 'models/solar_mr-hr/trained_gan/gan'
# r = [5]
# mu_sig = [[344.3262, 113.7444], [386.9283, 117.9627]]

if __name__ == '__main__':

    import h5pyd

    # hs_endpoint = 'https://developer.nrel.gov/api/hsds'
    # hs_username = 'sangho.lee.1990@gmail.com'
    # hs_password =
    # hs_api_key = 'a2MKg6w6iZg5aM9bBlXL49oDjkt7sagl4Li3LBMS'

    # C:\Users\%USERNAME%\\.pyNRSDB
    # from pyNSRDB.requests import PSM_TMY_request
    # location = (-93.1567288182409, 45.15793882400205)
    # data = PSM_TMY_request(location)
    # data.head()

    # curl 'https://developer.nrel.gov/api/alt-fuel-stations/v1.json?limit=1&api_key=a2MKg6w6iZg5aM9bBlXL49oDjkt7sagl4Li3LBMS'
   # https://developer.nrel.gov/api/alt-fuel-stations/v1.json?limit=1&api_key=a2MKg6w6iZg5aM9bBlXL49oDjkt7sagl4Li3LBMS
   # https://developer.nrel.gov/api/solar/data_query/v1.json?api_key=a2MKg6w6iZg5aM9bBlXL49oDjkt7sagl4Li3LBMS&lat=40&lon=-105&radius=50&all=1

    # from rex import NSRDBX

    # nsrdb_file = '/nrel/nsrdb/v3/nsrdb_2018.h5'
    # with NSRDBX(nsrdb_file, hsds=True) as f:
    #     meta = f.meta
    #     time_index = f.time_index
    #     dni = f['dni', :, ::1000]

    # Return all but first 2 lines of csv to get data:
    # df = pd.read_csv('https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key,
    #                                                                                                                                                                                                                                                                                                       attr=attributes), skiprows=2)
    #
    # # Set the time index in the pandas dataframe:
    # df = df.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval + 'Min', periods=525600 / int(interval)))
    #
    # # take a look
    # print('shape:', df.shape)
    # df.head()

    # import site
    #
    # site.addsitedir('/Applications/sam-sdk-2015-6-30-r3/languages/python/')
    # import PySAM.PySSC as pssc
    #
    # # Download PySAM here: https://pypi.org/project/NREL-PySAM/
    #
    # ssc = pssc.PySSC()

    # import pandas as pd
    #
    # # Declare all variables as strings. Spaces must be replaced with '+', i.e., change 'John Smith' to 'John+Smith'.
    # # Define the lat, long of the location and the year
    # lat, lon, year = 33.2164, -97.1292, 2010
    # # You must request an NSRDB api key from the link above
    # api_key = 'a2MKg6w6iZg5aM9bBlXL49oDjkt7sagl4Li3LBMS'
    # # Set the attributes to extract (e.g., dhi, ghi, etc.), separated by commas.
    # attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'
    # # Choose year of data
    # year = '2010'
    # # Set leap year to true or false. True will return leap day data if present, false will not.
    # leap_year = 'false'
    # # Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
    # interval = '30'
    # # Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
    # # NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
    # # local time zone.
    # utc = 'false'
    # # Your full name, use '+' instead of spaces.
    # your_name = 'lee+sangho'
    # # Your reason for using the NSRDB.
    # reason_for_use = 'beta+testing'
    # # Your affiliation
    # your_affiliation = 'my+institution'
    # # Your email address
    # your_email = 'sangho.lee.1990@gmail.com'
    # # Please join our mailing list so we can keep you up-to-date on new developments.
    # mailing_list = 'true'
    #
    # # Declare url string
    # url = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key,attr=attributes)
    # # Return just the first 2 lines to get metadata:
    # info = pd.read_csv(url)
    # # See metadata for specified properties, e.g., timezone and elevation
    # timezone, elevation = info['Local Time Zone'], info['Elevation']
    #
    # url = 'https://developer.nrel.gov/api/solar/nsrdb_data_query.json?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key,attr=attributes)
    # # https://developer.nrel.gov/api/solar/nsrdb_data_query.json?api_key=a2MKg6w6iZg5aM9bBlXL49oDjkt7sagl4Li3LBMS&wkt=POINT(91.287+23.832)
    #
    # url2 = 'https://developer.nrel.gov/api/nsrdb/v2/solar/himawari-download.csv?names=2016&wkt=POINT%2891.287+23.832%29&interval=10&api_key=a2MKg6w6iZg5aM9bBlXL49oDjkt7sagl4Li3LBMS&email=sangho.lee.1990@gmail.com'
    # info2 = pd.read_csv(url2)
    # r : 해상도
    # data_type : 종류
    # mu_sig : 평균 및 표준편차

    # 1.5 TB
    # https://data.openei.org/s3_viewer?bucket=nrel-pds-nsrdb&prefix=v3%2F

    # 300 GB
    # https://data.openei.org/s3_viewer?bucket=nrel-pds-nsrdb&prefix=himawari%2F

    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import ImageShow

    # inpFile = '{}/{}'.format(globalVar['inpPath'], 'TEST/SAT/GK2A_*.nc')
    inpFile = '{}/{}'.format('/SYSTEMS/PROG/PYTHON/PyCharm/resources/config/satInfo', 'H08_20211123_0000_RFL010_FLDK.02401_02401.nc')
    fileList = sorted(glob.glob(inpFile))

    # 0.05 == 5 km
    h8Data = xr.open_mfdataset(fileList)

    plt.clf()
    plt.figure()
    h8Data['SWR'].plot()
    # plt.axis('off')
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('../TEST/h08-org_100.png', dpi=600, bbox_inches='tight')
    # plt.show()
    # PIL.Image

    from PIL import Image

    # with Image.open("hopper.jpg") as im:
    #     im.rotate(45).show()

    # from datashader.transfer_functions import Images, Image
    # img = PIL.Image(h8Data['SWR'].values)

    # im = Image.fromarray(np.uint8(h8Data['SWR'].values * 255))
    # im.show()

    # np.uint8(h8Data['SWR'].to_numpy())

    # maxVal = np.nanmax(h8Data['SWR'])
    # minVal = np.nanmin(h8Data['SWR'])
    # h8Data['SWR'].values

    from sklearn.preprocessing import minmax_scale
    foo_norm = minmax_scale(h8Data['SWR'], feature_range=(0, 255), axis=0)
    plt.imshow(foo_norm, interpolation ='bilinear')
    # LSH0353

    plt.figure(figsize=(4.00, 4.00), dpi=100)
    plt.imshow(foo_norm, interpolation ='nearest')
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('../TEST/h08-val_100.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()



    # 2401 * 600
    # plt.figure(figsize=(24.01, 24.01), dpi=100)
    plt.figure(figsize=(24.00, 24.00), dpi=400)
    # plt.figure(figsize=(24.01, 24.01), dpi=600)

    plt.imshow(foo_norm, interpolation ='nearest')
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('../TEST/h08-sample_100.png', dpi=100, bbox_inches='tight', pad_inches=0)
    # plt.savefig('TEST/h08-sample_200.png', dpi=200, bbox_inches='tight', pad_inches=0)
    # plt.savefig('TEST/h08-sample_400.png', dpi=400, bbox_inches='tight', pad_inches=0)
    # plt.savefig('TEST/h08-sample_600.png', dpi=600, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()

    # 2400 5km
    # 1200 2.5 km

    # 2400/2.0
    # 2400/4.0 = 600
    # 2400/8.0 = 300
    im = Image.open('../TEST/h08-sample_100.png')
    width, height = im.size

    # plt.figure(figsize=(3.00, 3.00), dpi=100)
    # # plt.figure(figsize=(6.00, 6.00), dpi=400)
    # plt.imshow(im.resize((300,300)), interpolation ='nearest')
    # # plt.imshow(im.resize((600,600)), interpolation ='nearest')
    # plt.axis('off')
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    # plt.savefig('TEST/h08-test_300.png', dpi=100, bbox_inches='tight', pad_inches=0)
    # # plt.savefig('TEST/h08-test_600.png', dpi=100, bbox_inches='tight', pad_inches=0)
    # plt.close()
    #
    #
    # plt.figure(figsize=(2.00, 2.00), dpi=100)
    # # plt.figure(figsize=(6.00, 6.00), dpi=400)
    # plt.imshow(im.resize((200,200)), interpolation ='nearest')
    # plt.axis('off')
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    # plt.savefig('TEST/h08-test_200.png', dpi=100, bbox_inches='tight', pad_inches=0)
    # plt.close()
    #
    #
    #
    # im = Image.open('TEST/h08-test_200.png')

    plt.figure(figsize=(6.00, 6.00), dpi=100)
    plt.imshow(im.resize((600,600)), interpolation ='nearest')
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('../TEST/h08-res-nearest_600.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure(figsize=(6.00, 6.00), dpi=100)
    plt.imshow(im.resize((600,600)), interpolation ='bilinear')
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('../TEST/h08-res-bilinear_600.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()


    plt.figure(figsize=(4.00, 4.00), dpi=100)
    plt.imshow(im.resize((400,400)), interpolation ='nearest')
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('TEST/h08-res-nearest_400.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure(figsize=(4.00, 4.00), dpi=100)
    plt.imshow(im.resize((400,400)), interpolation ='bilinear')
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('TEST/h08-res-bilinear_400.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure(figsize=(3.00, 3.00), dpi=100)
    plt.imshow(im.resize((300,300)), interpolation ='nearest')
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('../TEST/h08-res-nearest_300.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure(figsize=(3.00, 3.00), dpi=100)
    plt.imshow(im.resize((300,300)), interpolation ='bilinear')
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('../TEST/h08-res-bilinear_300.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()


    plt.figure(figsize=(2.00, 2.00), dpi=100)
    plt.imshow(im.resize((200,200)), interpolation ='nearest')
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('../TEST/h08-res-nearest_200.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure(figsize=(2.00, 2.00), dpi=200)
    plt.imshow(im.resize((200,200)), interpolation ='bilinear')
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('../TEST/h08-res-bilinear_200.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()








    # selH8NearData = h8Data.sel(latitude=lat1D, longitude=lon1D, method='nearest')
    # selH8IntpData = h8Data.interp(latitude=lat1D, longitude=lon1D)


    phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)

    model_dir = phiregans.pretrain(r=r,
                                   data_path=data_path,
                                   model_path=model_path,
                                   batch_size=1)

    model_dir = phiregans.train(r=r,
                                data_path=data_path,
                                model_path=model_dir,
                                batch_size=1)

    phiregans.test(r=r,
                   data_path=data_path,
                   model_path=model_dir,
                   batch_size=1, plot_data=True)

    print('[END]')
#
# inpFile = '{}/data_out/solar-*/{}'.format(ctxPath, 'dataSR.npy')
# inpFile = '{}/data_out/solar-20220927-001021/{}'.format(ctxPath, 'dataSR.npy')
# fileList = sorted(glob.glob(inpFile))
#
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np
#
# # data = np.load(fileList[1])
# data = np.load(fileList[0])
#
# data.shape
# data[0, :, :, :].shape
#
# selData = data[2, :, :, 0]
# # data[5, 100, 100].shape
# # plt.imshow(selData, cmap='gray')
# plt.imshow(selData)
# plt.show()
#

import pandas as pd
from pandas_tfrecords import pd2tf, tf2pd

inpFile = '{}/example_data/{}'.format(ctxPath, '*.tfrecord')
fileList = sorted(glob.glob(inpFile))

# data = np.load(fileList[0])
#
# filenames = [filename]
# raw_dataset = tf.data.TFRecordDataset(fileList[0])
# raw_dataset

my_df = tf2pd(fileList[0])
# my_df = tf2pd(fileList[2])
# #
# # dataset = tf.data.TFRecordDataset("train.tfrecord").map(deserialize_example).batch(4)
# # for x in dataset:
# #     print(x)
#
#
# import tensorflow as tf
# raw_dataset = tf.data.TFRecordDataset(fileList[0])
#
# for raw_record in raw_dataset.take(1):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)
#








import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ds = tf.data.TFRecordDataset([fileList[0]])
# for batch in ds.take(1):
#   example = tf.train.Example()
#   example.ParseFromString(batch.numpy())
#   print(example)

# dataset =  tf.io.decode_raw[fileList[0])
# dataset = dataset.map(_parse_train_).batch(1)
raw_dataset = tf.data.TFRecordDataset(fileList[0])

def decode(serialized_example):
    """
    Parses an image and label from the given `serialized_example`.
    It is used as a map function for `dataset.map`
    """
    IMAGE_SIZE = 28
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

    # 1. define a parser
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # 2. Convert the data
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    # 3. reshape
    image.set_shape((IMAGE_PIXELS))
    return image, label

# dataset = raw_dataset.map(decode)


# N_batch, height, width, [ua, va]
# N_batch, 높이, 너비, [DNI, DHI]
# 스크립트는 TFRecords를 사용하도록 설계되었습니다. numpy 배열을 호환 가능한 TFRecord로 변환하는 방법은 utils.py.


tfrecords_path = fileList[0]
# Make some test records

# with tf.io.TFRecordWriter(tfrecords_path) as writer:
#     for i in range(10):
#         example = tf.train.Example(
#             features=tf.train.Features(
#                 feature={
#                     # Fixed length
#                     'id': tf.train.Feature(
#                         int64_list=tf.train.Int64List(value=[i])),
#                     # Variable length
#                     'data': tf.train.Feature(
#                         float_list=tf.train.FloatList(value=range(i))),
#                 }))
#         writer.write(example.SerializeToString())
# # Print extracted feature information
features = list_record_features(tfrecords_path)
print(*features.items(), sep='\n')
# I am facing the following output:
# dataset
#
# parsed_tf_records = dataset.map(parse_ac_element)
# df = pd.DataFrame(
#     parsed_tf_records.as_numpy_iterator(),
#     columns=['Name', 'age', 'sex']
# )