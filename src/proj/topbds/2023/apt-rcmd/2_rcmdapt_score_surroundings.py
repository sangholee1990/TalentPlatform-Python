

import os
import numpy as np
import pandas as pd
import requests
import xmltodict




# =====================================================================================
# 아파트별 주변시설 개수

data_dir = 'dataset/'
gu_list = [
    '강남구', '강동구', '강북구', '중구',                 # 456, 396, 114, 102
    '강서구', '과천시', '관악구', '광진구', '종로구',       # 518, 19, 226, 186, 103
    '구로구', '중랑구', '금천구', '노원구',                # 446, 232, 135, 305
    '도봉구', '동대문구', '동작구', '마포구',              # 205, 248, 210, 286,
    '서대문구', '서초구', '성동구', '성북구',              # 233, 488, 149, 171,
    '송파구', '양천구', '영등포구',                       # 327, 513, 255,
    '용산구', '은평구'                                  # 183, 424,
]



results = []
for gu_name in gu_list:
    # 시설 카운팅
    pd_surr = pd.read_excel(data_dir + 'surroundings_{}.xlsx'.format(gu_name))
    pd_apts = pd.read_excel(data_dir + '{}실거래가.xlsx'.format(gu_name))
    pd_apts = pd_apts[['아파트', 'latitude', 'longitude']]
    pd_apts = pd_apts.drop_duplicates()
    #
    for index, row in pd_apts.iterrows():
        apt = row['아파트']
        y = row['latitude']
        x = row['longitude']
        # area = row['전용면적(㎡)']
        # print(apt, y, x)
        #
        #
        pd_info = pd_surr[pd_surr.apt_x==x][pd_surr.apt_y==y]
        if len(pd_info) == 0:
            continue
        # ------------------------------------------------------
        cnt_car = len(pd_info[pd_info.category == '교통'])
        cnt_edu = len(pd_info[pd_info.category == '교육'])
        cnt_infra = len(pd_info[pd_info.category == '주거환경'])
        cnt_conv = len(pd_info[pd_info.category == '편의시설'])
        if (cnt_car + cnt_edu + cnt_infra + cnt_conv) == 0:
            continue
        # ------------------------------------------------------
        results.append([gu_name, apt, x, y, cnt_car, cnt_edu, cnt_infra, cnt_conv])
    print(gu_name, '합계:', len(results))



pd_results = pd.DataFrame(results)
pd_results.columns = ['gu_name', 'apt_addr', 'apt_x', 'apt_y', '교통', '교육', '주거환경', '편의시설']
pd_results.shape
pd_results.to_excel('dataset/surroundings_counts_total.xlsx', index=False)




# ======================================================================================
# 아파트별 주변시설 개수(상대점수)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


data_dir = 'dataset/'
gu_list = [
    '강남구', '강동구', '강북구', '중구',                 # 456, 396, 114, 102
    '강서구', '과천시', '관악구', '광진구', '종로구',       # 518, 19, 226, 186, 103
    '구로구', '중랑구', '금천구', '노원구',                # 446, 232, 135, 305
    '도봉구', '동대문구', '동작구', '마포구',              # 205, 248, 210, 286,
    '서대문구', '서초구', '성동구', '성북구',              # 233, 488, 149, 171,
    '송파구', '양천구', '영등포구',                       # 327, 513, 255,
    '용산구', '은평구'                                  # 183, 424,
]


# 아파트별 주변시설 개수 정보
pd_counts = pd.read_excel('dataset/surroundings_counts_total.xlsx')
cols = pd_counts.columns.tolist()



data_total = []
for gu_name in gu_list:
    pd_gu = pd_counts[pd_counts.gu_name==gu_name]# .reset_index()
    data_gu = pd_gu.values
    #
    #
    # 표준화 및 0~1척도 (gu)
    gu_counts = pd_gu[['교통', '교육', '주거환경', '편의시설']].values
    gu_scaled = scaler.fit_transform(gu_counts)
    for i in range(4):
        gu_scaled[:, i] = (gu_scaled[:, i] - np.min(gu_scaled[:, i])) / (np.max(gu_scaled[:, i]) - np.min(gu_scaled[:, i]))
    #
    #
    # 기존 데이터 병합 (gu_name, apt 정보까지)
    data_gu_new = np.concatenate([data_gu, gu_scaled], axis=1)
    if len(data_total) == 0:
        data_total = data_gu_new
    else:
        data_total = np.concatenate([data_total, data_gu_new], axis=0)



cols = cols + ['교통_rel', '교육_rel', '주거환경_rel', '편의시설_rel']



# 표준화 및 0~1척도 (total)
total_counts = pd_counts[['교통', '교육', '주거환경', '편의시설']].values
total_scaled = scaler.fit_transform(total_counts)
for i in range(4):
    total_scaled[:, i] = (total_scaled[:, i] - np.min(total_scaled[:, i])) / (
            np.max(total_scaled[:, i]) - np.min(total_scaled[:, i]))



# 기존 데이터 병합
data_total2 = np.concatenate([data_total, total_scaled], axis=1)
pd_total2 = pd.DataFrame(data_total2)
pd_total2.columns = cols + ['교통_rel_total', '교육_rel_total', '주거환경_rel_total', '편의시설_rel_total']
pd_total2.to_excel('dataset/surroundings_scores_total.xlsx', index=False)

