# -*- coding: utf-8 -*-
import os
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from glob import glob
import time
import copy
import shutil
import cv2
from PIL import Image
import json
import requests
import gzip, pickle

def save_model(obj, filename):
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(obj, file)

def load_model(filename):
    file = gzip.GzipFile(filename, 'rb')
    obj = pickle.load(file)
    return obj

def recommends_cf(payload):
    try:
        response = requests.request("POST", REST_API_URL_CF, data=payload, verify=False)
        res_json = response.json().get('recommends')
        return res_json
    except Exception as ex:
        print(str(ex))

def recommends_simil(payload):
    try:
        response = requests.request("POST", REST_API_URL_SIMIL, data=payload, verify=False)
        res_json = response.json().get('recommends')
        return res_json
    except Exception as ex:
        print(str(ex))

if __name__ == '__main__':

    ctxPath = os.getcwd()
    print(f'[CHECK] ctxPath : {ctxPath}')

    with open(f'{ctxPath}/config_server.json', 'rb') as f:
        config = json.loads(f.read().decode())

    SERVER_IP = config['server_ip']
    PORT = config['server_port']
    REST_API_URL_CF = 'http://{}:{}/recommends_cf'.format(SERVER_IP, PORT)
    REST_API_URL_SIMIL = 'http://{}:{}/recommends_simil'.format(SERVER_IP, PORT)

    # -----------------------------------------------------
    # 사용자 선택
    PD_USERS = pd.read_excel(f'{ctxPath}/dataset/tbl_users.xlsx')

    # 희망 가격 6억원대 이상 고객군에 한함
    PD_USERS = PD_USERS[PD_USERS.price_from >= config.get('target_cus_price')][PD_USERS.area_from >= 60]
    # print(PD_USERS)

    # -------------------------------------------------
    USER_GU_LIST = PD_USERS['gu_name'].drop_duplicates().tolist()
    USER_GU_LIST.sort()
    # USER_LIST = PD_USERS['user_id'].drop_duplicates().tolist()
    # USER_LIST.sort()

    # -------------------------------------------------
    PD_APTS = pd.read_excel(f'{ctxPath}/dataset/tbl_apts.xlsx')
    APT_LIST = PD_APTS['apt_idx'].astype('str').tolist()
    # print(PD_APTS)


    # 아파트 환경 정보
    PD_APT_SURR = pd.read_excel(f'{ctxPath}/dataset/surroundings_scores_total.xlsx')
    PD_EVALS = load_model(f'{ctxPath}/dataset/tbl_user_apt_evals.dat')
    # print(PD_EVALS)

    # 양천구 user_00025
    # 양천구 user_00043
    userList = ['user_00025', 'user_00043']
    for userInfo in userList:
        # 본인이 조회한 아파트 찾기 (eval점수 4점 이상)
        # pd_evals = PD_EVALS[PD_EVALS.user_id == 'user_00025'][PD_EVALS.evals >= 4].reset_index(drop=True)
        # pd_evals = PD_EVALS[PD_EVALS.user_id == 'user_00025'].reset_index(drop=True)
        pd_evals = PD_EVALS[PD_EVALS.user_id == userInfo].reset_index(drop=True)
        print('pd_evals:', pd_evals)
        print(f'[CHECK] pd_evals : {pd_evals}')

        # *********************************************
        # CF 추천
        # *********************************************
        cfData = pd.DataFrame()
        for idx, item in pd_evals.iterrows():

            payload = {
                'user_id': item.user_id
                , 'apt_idx': item.apt_idx
            }

            # API 요청
            cfRes = recommends_cf(payload)
            for key, rcmd_list in cfRes.items():
                for rcmd in rcmd_list:
                    apiIdx, corr = rcmd

                    selPdAptsData = PD_APTS[PD_APTS.apt_idx == apiIdx]
                    apt = selPdAptsData.get('apt').values[0]
                    lat = selPdAptsData.get('latitude').values[0]  # 위도 (apt_y)
                    lon = selPdAptsData.get('longitude').values[0]  # 경도 (apt_x)

                    selPdRcmdData = PD_APT_SURR[
                        (PD_APT_SURR.apt_addr == apt) &
                        (PD_APT_SURR.apt_x == lon) &
                        (PD_APT_SURR.apt_y == lat)
                        ].drop_duplicates()
                    # print('selPdRcmdData:', selPdRcmdData)

                    selData = pd.merge(selPdAptsData, PD_APT_SURR, left_on=['apt', 'latitude', 'longitude'], right_on=['apt_addr', 'apt_y', 'apt_x'], suffixes=('_x', ''), how='inner').drop_duplicates()
                    colDelList = [col for col in selData.columns if '_x' in col]
                    selDataL1 = selData.drop(columns=colDelList).reset_index(drop=True)

                    itemData = pd.DataFrame([item]).reset_index(drop=True)
                    itemDataL1 = itemData.rename({'apt_idx': 'user_apt_idx'}, axis=1)

                    selDataL2 = pd.concat([itemDataL1, selDataL1], axis=1)
                    cfData = pd.concat([cfData, selDataL2], axis=0)

        cfDataL1 = cfData
        cateList = ['교통', '교육', '주거환경', '편의시설']
        for cate in cateList:
            cfDataL1[f'{cate}'] = cfData[f'{cate}'].round(0)
            cfDataL1[f'{cate}_rel'] = (cfData[f'{cate}_rel'] * 100).round(0)
            cfDataL1[f'{cate}_rel_total'] = (cfData[f'{cate}_rel_total'] * 100).round(0)

        # 파일 저장
        saveCsvFile = '{}/{}/{}_{}.csv'.format(ctxPath, 'OUTPUT', 'CF', userInfo)
        os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)
        cfDataL1.to_csv(saveCsvFile, index=False, encoding='UTF-8')
        print('[CHECK] saveCsvFile : {}'.format(saveCsvFile))

        # *********************************************
        # 유사 아파트 추천
        # *********************************************
        simData = pd.DataFrame()
        for idx, item in pd_evals.iterrows():

            payload = {
                'user_id': item.user_id
                , 'apt_idx': item.apt_idx
            }

            # API 요청
            simRes = recommends_simil(payload)
            for key, rcmd_list in simRes.items():
                for rcmd in rcmd_list:
                    apiIdx, corr = rcmd

                    selPdAptsData = PD_APTS[PD_APTS.apt_idx == apiIdx]
                    apt = selPdAptsData.get('apt').values[0]
                    lat = selPdAptsData.get('latitude').values[0]  # 위도 (apt_y)
                    lon = selPdAptsData.get('longitude').values[0]  # 경도 (apt_x)

                    selPdRcmdData = PD_APT_SURR[
                        (PD_APT_SURR.apt_addr == apt) &
                        (PD_APT_SURR.apt_x == lon) &
                        (PD_APT_SURR.apt_y == lat)
                        ].drop_duplicates()
                    # print('selPdRcmdData:', selPdRcmdData)

                    selData = pd.merge(selPdAptsData, PD_APT_SURR, left_on=['apt', 'latitude', 'longitude'],
                                       right_on=['apt_addr', 'apt_y', 'apt_x'], suffixes=('_x', ''),
                                       how='inner').drop_duplicates()
                    colDelList = [col for col in selData.columns if '_x' in col]
                    selDataL1 = selData.drop(columns=colDelList).reset_index(drop=True)

                    itemData = pd.DataFrame([item]).reset_index(drop=True)
                    itemDataL1 = itemData.rename({'apt_idx': 'user_apt_idx'}, axis=1)

                    selDataL2 = pd.concat([itemDataL1, selDataL1], axis=1)
                    simData = pd.concat([simData, selDataL2], axis=0)

        simDataL1 = simData
        cateList = ['교통', '교육', '주거환경', '편의시설']
        for cate in cateList:
            simDataL1[f'{cate}'] = simData[f'{cate}'].round(0)
            simDataL1[f'{cate}_rel'] = (simData[f'{cate}_rel'] * 100).round(0)
            simDataL1[f'{cate}_rel_total'] = (simData[f'{cate}_rel_total'] * 100).round(0)

        # 파일 저장
        saveCsvFile = '{}/{}/{}_{}.csv'.format(ctxPath, 'OUTPUT', 'SIM', userInfo)
        os.makedirs(os.path.dirname(saveCsvFile), exist_ok=True)
        simDataL1.to_csv(saveCsvFile, index=False, encoding='UTF-8')
        print('[CHECK] saveCsvFile : {}'.format(saveCsvFile))
